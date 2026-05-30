"""Conservative regridding for grids that aren't 1D-separable.

The existing ``conservative`` method uses axis-factored 1D overlap — fast and
elegant but strictly rectilinear. This module computes the full 2D cell
intersection via shapely, so it handles:

- curvilinear grids (2D ``lat[i, j]`` / ``lon[i, j]`` coordinate variables)
- unstructured meshes (arbitrary polygon cells, via
  :meth:`ConservativeRegridder.from_polygons`)
- grid-to-polygon aggregation (e.g. gridded data → country shapes)

For rectilinear grids a cheap analytic fast-path is used, but this module is
still slower and more memory-intensive than ``conservative``; prefer the
axis-factored path when your grid is 1D-separable.

Requires ``shapely >= 2.0``. If ``sparse`` is available, the weight matrix is
stored as ``sparse.COO``; otherwise a dense numpy matrix is used.
"""

import math
import os
import warnings
from abc import ABC, abstractmethod
from collections.abc import Callable, Hashable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Any, ClassVar, Literal, cast

import numpy as np
import scipy.sparse as sp
import xarray as xr

from xarray_regrid import utils
from xarray_regrid.methods._conservative_2d_serialization import (
    _coo_components,
    _coo_from_components,
    _metadata_attrs,
    _metadata_from_attrs,
)
from xarray_regrid.methods._conservative_2d_spec import Manifold, RegridSpec
from xarray_regrid.methods.conservative import get_valid_threshold

NetcdfEngine = Literal["netcdf4", "scipy", "h5netcdf"] | None


try:
    import shapely
    from shapely import affinity
    from shapely.strtree import STRtree

    _HAS_SHAPELY = True
except ImportError:  # pragma: no cover
    shapely = None
    affinity = None
    STRtree = None
    _HAS_SHAPELY = False

try:
    import sparse

    _HAS_SPARSE = True
except ImportError:  # pragma: no cover
    sparse = None
    _HAS_SPARSE = False

try:
    import spherely

    _HAS_SPHERELY = True
except ImportError:  # pragma: no cover
    spherely = None
    _HAS_SPHERELY = False


# We fill NaNs with 0 ourselves before matmul (see `_apply_core`), so sparse's
# "NaN will not be propagated" warning is spurious. A module-level filter
# avoids `warnings.catch_warnings` inside the hot matmul path, which is not
# thread-safe — dask threads would race on the global `warnings.filters` list.
warnings.filterwarnings(
    "ignore",
    message="Nan will not be propagated in matrix multiplication",
    category=RuntimeWarning,
)


SHAPELY_IMPORT_ERROR = (
    "polygon conservative regridding requires shapely >= 2.0; "
    "install with `pip install shapely`."
)


def _check_shapely() -> None:
    if not _HAS_SHAPELY:
        raise ImportError(SHAPELY_IMPORT_ERROR)


SPHERELY_IMPORT_ERROR = (
    "manifold='s2' requires the optional `spherely` package. "
    "Install with `pip install spherely` or "
    "`pip install xarray-regrid[spherical]`."
)


def _check_spherely() -> None:
    if not _HAS_SPHERELY:
        raise ImportError(SPHERELY_IMPORT_ERROR)


class _Direction:
    """Lazy weights, apply matrix, and coverage mask for one regrid direction.

    Holds the raw ``(n_dst, n_src)`` area matrix in this direction's orientation
    and derives, on first access:

    - ``weights``: row-normalized weight matrix
    - ``apply_matrix``: the weights as a scipy CSR matrix, used by
      ``_apply_core``'s ``(W @ data.T).T`` matmul (see :func:`_to_csr`)
    - ``coverage``: which output cells have any source overlap

    A regridder holds two of these (forward, backward); transposing the
    regridder swaps them with no recomputation.
    """

    def __init__(self, areas: "sparse.COO | np.ndarray") -> None:
        self.areas = areas

    @cached_property
    def weights(self) -> "sparse.COO | np.ndarray":
        return _row_normalize(self.areas)

    @cached_property
    def apply_matrix(self) -> "sp.csr_matrix":
        return _to_csr(self.weights)

    @cached_property
    def coverage(self) -> np.ndarray:
        return _coverage_mask(self.areas)


class ConservativeRegridder:
    """Reusable conservative regridder for grids that aren't 1D-separable:
    curvilinear (2D ``lat``/``lon``), unstructured (via :meth:`from_polygons`),
    or arbitrary polygon-to-polygon aggregation. For purely 1D-separable
    rectilinear grids, the ``.regrid.conservative`` accessor is faster.

    Build once, apply to many fields via :meth:`regrid` (or by calling the
    regridder); ``.T`` gives the backward regridder. Forward and backward
    weight matrices are cached lazily. Requires ``shapely >= 2.0``.

    The unnormalized cell-intersection area matrix
    ``A[i, j] = area(target_i ∩ source_j)`` is exposed as ``self.areas``
    (sparse ``(n_dst, n_src)`` if the ``sparse`` package is available, dense
    otherwise) — useful for conservation diagnostics and per-cell coverage
    analysis.
    """

    def __init__(
        self,
        source: xr.DataArray | xr.Dataset,
        target: xr.Dataset,
        x_coord: str = "longitude",
        y_coord: str = "latitude",
        manifold: Manifold = "planar",
        n_threads: int | None = None,
    ) -> None:
        _check_shapely()
        _check_manifold(manifold)
        source_grid, target_grid, src_x_sort_idx = _normalize_longitude_coords(
            source, target, x_coord
        )
        src_dims = _spatial_dims(source_grid, x_coord, y_coord)
        dst_dims = _spatial_dims(target_grid, x_coord, y_coord)
        if not src_dims:
            msg = f"source has no dims for coords {x_coord!r}, {y_coord!r}"
            raise ValueError(msg)
        if not dst_dims:
            msg = f"target has no dims for coords {x_coord!r}, {y_coord!r}"
            raise ValueError(msg)
        spec = RegridSpec(
            src_dims=src_dims,
            dst_dims=dst_dims,
            src_shape=tuple(int(source.sizes[d]) for d in src_dims),
            dst_shape=tuple(int(target.sizes[d]) for d in dst_dims),
            x_coord=x_coord,
            y_coord=y_coord,
            manifold=manifold,
        )
        backend = _BACKENDS[manifold]
        src_grid = backend.grid_from_coords(source_grid, x_coord, y_coord, src_dims)
        dst_grid = backend.grid_from_coords(target_grid, x_coord, y_coord, dst_dims)
        areas = backend.area_matrix(src_grid, dst_grid, n_threads=n_threads)
        if src_x_sort_idx is not None:
            x_dim_index = src_dims.index(source[x_coord].dims[0])
            areas = _remap_columns_for_axis_sort(
                areas, src_x_sort_idx, spec.src_shape, x_dim_index
            )
        self._init_state(
            areas=areas,
            source_coords=source.coords.to_dataset(),
            target_coords=target.coords.to_dataset(),
            spec=spec,
        )

    def _init_state(
        self,
        *,
        areas: "sparse.COO | np.ndarray",
        source_coords: xr.Dataset,
        target_coords: xr.Dataset,
        spec: RegridSpec,
    ) -> None:
        self.areas = areas
        self._source_coords = source_coords
        self._target_coords = target_coords
        self._spec = spec

    @property
    def spec(self) -> RegridSpec:
        """Canonical metadata describing this regridder's layout."""
        return self._spec

    @property
    def x_coord(self) -> str:
        return self._spec.x_coord

    @property
    def y_coord(self) -> str:
        return self._spec.y_coord

    @property
    def manifold(self) -> Manifold:
        return self._spec.manifold

    @cached_property
    def _forward(self) -> _Direction:
        return _Direction(self.areas)

    @cached_property
    def _backward(self) -> _Direction:
        return _Direction(_transpose_weights(self.areas))

    @property
    def forward_weights(self) -> "sparse.COO | np.ndarray":
        """The row-normalized forward weight matrix (source → target)."""
        return self._forward.weights

    @property
    def backward_weights(self) -> "sparse.COO | np.ndarray":
        """The row-normalized backward weight matrix (target → source)."""
        return self._backward.weights

    @property
    def target_areas(self) -> np.ndarray:
        """Area of each target cell overlapped by the source domain."""
        return _sum_matrix_axis_1d(self.areas, axis=1)

    @property
    def source_coverage_areas(self) -> np.ndarray:
        """Area of each source cell covered by target cells."""
        return _sum_matrix_axis_1d(self.areas, axis=0)

    def regrid(
        self,
        data: xr.DataArray | xr.Dataset,
        skipna: bool = True,
        nan_threshold: float = 1.0,
    ) -> xr.DataArray | xr.Dataset:
        """Regrid ``data`` forward (source → target)."""
        return _apply_stored_weights(
            data,
            direction=self._forward,
            spec=self.spec,
            target_coords=self._target_coords,
            skipna=skipna,
            nan_threshold=nan_threshold,
        )

    def __call__(
        self,
        data: xr.DataArray | xr.Dataset,
        skipna: bool = True,
        nan_threshold: float = 1.0,
    ) -> xr.DataArray | xr.Dataset:
        return self.regrid(data, skipna=skipna, nan_threshold=nan_threshold)

    def transpose(self) -> "ConservativeRegridder":
        """Return the backward regridder (target → source). The original's
        forward Direction becomes the new's backward (and vice versa), so any
        already-computed weight matrices are reused, not recomputed."""
        new = type(self)._from_state(
            areas=_transpose_weights(self.areas),
            source_coords=self._target_coords,
            target_coords=self._source_coords,
            spec=self._spec.transposed(),
        )
        if "_forward" in self.__dict__:
            new.__dict__["_backward"] = self.__dict__["_forward"]
        if "_backward" in self.__dict__:
            new.__dict__["_forward"] = self.__dict__["_backward"]
        return new

    @property
    def T(self) -> "ConservativeRegridder":  # noqa: N802
        """Alias for :meth:`transpose` (numpy-style transpose)."""
        return self.transpose()

    def __repr__(self) -> str:
        nnz = getattr(self.areas, "nnz", None)
        shape = getattr(self.areas, "shape", (None, None))
        nnz_str = f"nnz={nnz}" if nnz is not None else "dense"
        return (
            f"ConservativeRegridder(src_dims={self._spec.src_dims}, "
            f"dst_dims={self._spec.dst_dims}, {shape[0]}x{shape[1]}, {nnz_str})"
        )

    def to_netcdf(self, path: str | Path, engine: NetcdfEngine = None) -> None:
        """Save the weight matrix and reproducibility metadata to a netCDF file.
        Requires a group-aware engine (``netcdf4`` or ``h5netcdf``); ``engine``
        is forwarded to :func:`xarray.Dataset.to_netcdf`."""
        path = Path(path)
        row, col, data, shape = _coo_components(self.areas)
        ds_weights = xr.Dataset(
            {
                "_coo_row": (("nnz",), row),
                "_coo_col": (("nnz",), col),
                "_coo_data": (("nnz",), data),
            },
            attrs={
                **_metadata_attrs(self.spec, self._source_coords, self._target_coords),
                "n_dst": int(shape[0]),
                "n_src": int(shape[1]),
            },
        )
        ds_weights.to_netcdf(path, mode="w", engine=engine)
        self._source_coords.to_netcdf(
            path, mode="a", group="source_coords", engine=engine
        )
        self._target_coords.to_netcdf(
            path, mode="a", group="target_coords", engine=engine
        )

    @classmethod
    def from_netcdf(
        cls, path: str | Path, engine: NetcdfEngine = None
    ) -> "ConservativeRegridder":
        """Reload a regridder previously written with :meth:`to_netcdf`.

        Validates ``schema_version``; raises :class:`ValueError` if the file
        was written by an incompatible version.
        """
        path = Path(path)
        with xr.open_dataset(path, engine=engine) as ds_weights:
            attrs = dict(ds_weights.attrs)
            n_dst = int(attrs.pop("n_dst"))
            n_src = int(attrs.pop("n_src"))
            row = np.asarray(ds_weights["_coo_row"].values)
            col = np.asarray(ds_weights["_coo_col"].values)
            data = np.asarray(ds_weights["_coo_data"].values)
        meta = _metadata_from_attrs(attrs, path)

        with xr.open_dataset(path, group="source_coords", engine=engine) as g:
            source_coords = g.load()
        with xr.open_dataset(path, group="target_coords", engine=engine) as g:
            target_coords = g.load()

        return cls._from_state(
            areas=_coo_from_components(row, col, data, (n_dst, n_src)),
            source_coords=source_coords,
            target_coords=target_coords,
            spec=meta,
        )

    @classmethod
    def _from_state(
        cls,
        *,
        areas: "sparse.COO | np.ndarray",
        source_coords: xr.Dataset,
        target_coords: xr.Dataset,
        spec: RegridSpec,
    ) -> "ConservativeRegridder":
        """Construct a regridder directly from its canonical state. Shared
        bypass of ``__init__`` used by :meth:`from_netcdf` and
        :meth:`from_polygons`."""
        instance = object.__new__(cls)
        instance._init_state(
            areas=areas,
            source_coords=source_coords,
            target_coords=target_coords,
            spec=spec,
        )
        return instance

    @classmethod
    def from_polygons(
        cls,
        source_polygons: np.ndarray,
        target_polygons: np.ndarray,
        source_dim: str = "cell",
        target_dim: str = "cell",
        target_coords: xr.Dataset | None = None,
        periodic: bool = False,
        n_threads: int | None = None,
        predicate_filter: bool = True,
    ) -> "ConservativeRegridder":
        """Build a regridder from explicit shapely polygon arrays — for
        unstructured meshes (MPAS, ICON), arbitrary polygon targets (countries,
        watersheds), or any non-rectilinear combination.

        Args:
            source_polygons, target_polygons: 1D arrays of shapely Polygons.
            source_dim, target_dim: Dim names for source/target cells on the
                input and output arrays.
            target_coords: Optional Dataset of coord variables along
                ``target_dim`` to reattach on the output (else: integer index).
            periodic: Unwrap polygons that cross the antimeridian (treats x
                as longitude on a 360-degree periodic axis).
            n_threads: Thread count for parallel GEOS intersection.
            predicate_filter: If True, filter STRtree candidates with GEOS
                ``intersects``. Set False for tight-bbox grid cells to skip
                the predicate (faster on that case, pathological otherwise).

        The regridder applies along ``source_dim``: input data must be 1D
        along that dim with element order matching ``source_polygons`` (output
        is along ``target_dim``, matching ``target_polygons``). When source
        polygons come from a structured grid via :func:`polygons_from_coords`,
        flatten the data to match — those polygons are row-major in ``(y, x)``,
        so transpose to ``(y, x)`` before ``.values.ravel()``.

        Geometry is planar in the polygons' own coordinate space. For lat/lon
        cells, project into an equal-area CRS first or use the structured
        path with ``manifold="cea"``.
        """
        _check_shapely()
        src_polys = np.asarray(source_polygons)
        dst_polys = np.asarray(target_polygons)
        if src_polys.ndim != 1 or dst_polys.ndim != 1:
            msg = "source_polygons and target_polygons must be 1D arrays"
            raise ValueError(msg)
        if periodic:
            src_polys = _normalize_periodic_polygons(src_polys)
            dst_polys = _normalize_periodic_polygons(
                dst_polys, reference=_polygon_reference_x(src_polys)
            )

        src_grid = _PLANAR.grid_from_polygons(src_polys)
        dst_grid = _PLANAR.grid_from_polygons(dst_polys)
        n_src = int(src_polys.size)
        n_dst = int(dst_polys.size)
        tgt_ds = (
            target_coords
            if target_coords is not None
            else xr.Dataset(coords={target_dim: np.arange(n_dst)})
        )
        return cls._from_state(
            areas=_PLANAR.area_matrix(
                src_grid,
                dst_grid,
                n_threads=n_threads,
                predicate_filter=predicate_filter,
            ),
            source_coords=xr.Dataset(coords={source_dim: np.arange(n_src)}),
            target_coords=tgt_ds,
            spec=RegridSpec(
                src_dims=(source_dim,),
                dst_dims=(target_dim,),
                src_shape=(n_src,),
                dst_shape=(n_dst,),
                x_coord="",
                y_coord="",
                manifold="planar",
            ),
        )


def polygons_from_coords(
    x: np.ndarray,
    y: np.ndarray,
    manifold: Manifold = "planar",
    periodic: bool = False,
) -> np.ndarray:
    """Build a 1D row-major (y, x) array of shapely cell polygons from 1D or
    2D center coords. Convenience for mixing structured and unstructured paths
    via :meth:`ConservativeRegridder.from_polygons`. ``manifold="cea"``
    projects 1D lat/lon (degrees) into Lambert cylindrical equal-area space;
    ``periodic=True`` unwraps antimeridian-crossing cells."""
    _check_shapely()
    backend = _resolve_manifold(manifold)
    x = np.asarray(x)
    y = np.asarray(y)
    if periodic:
        x = _unwrap_longitude(x)
    return backend.polys_from_arrays(x, y)


def _apply_stored_weights(
    data: xr.DataArray | xr.Dataset,
    direction: _Direction,
    spec: RegridSpec,
    target_coords: xr.Dataset,
    skipna: bool,
    nan_threshold: float,
) -> xr.DataArray | xr.Dataset:
    """Apply ``direction``'s cached weight matrix to ``data`` via
    ``xr.apply_ufunc``.

    The apply matrix is a scipy CSR of shape ``(n_dst, n_src)``; ``_apply_core``
    evaluates ``(W @ flat.T).T`` (CSR · dense), which is markedly faster than a
    ``sparse.COO`` dense matmul (see :func:`_to_csr`).
    """
    actual_src_shape = tuple(
        int(data.sizes[d]) for d in spec.src_dims if d in data.sizes
    )
    if actual_src_shape != spec.src_shape:
        msg = (
            f"source spatial shape {actual_src_shape} on dims {spec.src_dims} does "
            f"not match the regridder's expected shape {spec.src_shape}"
        )
        raise ValueError(msg)

    src_tokens = tuple(f"__src_{d}" for d in spec.src_dims)
    data_renamed = data.rename(dict(zip(spec.src_dims, src_tokens, strict=True)))

    output_dtype = _result_dtype(data)
    coverage = direction.coverage
    result = xr.apply_ufunc(
        _apply_core,
        data_renamed,
        kwargs={
            "apply_weights": direction.apply_matrix,
            "coverage": coverage,
            "coverage_all": bool(coverage.all()),
            "src_shape": spec.src_shape,
            "dst_shape": spec.dst_shape,
            "skipna": skipna,
            "nan_threshold": nan_threshold,
            "output_dtype": output_dtype,
        },
        input_core_dims=[list(src_tokens)],
        output_core_dims=[list(spec.dst_dims)],
        exclude_dims=set(src_tokens),
        dask="parallelized",
        output_dtypes=[output_dtype],
        dask_gufunc_kwargs={
            "output_sizes": {d: int(target_coords.sizes[d]) for d in spec.dst_dims},
            "allow_rechunk": True,
        },
        keep_attrs=True,
    )

    return _assign_target_coords(result, target_coords, spec)


def _coverage_mask(areas: "sparse.COO | np.ndarray") -> np.ndarray:
    """Return a boolean ``(n_dst,)`` mask of target cells with any source
    overlap, derived from the raw area matrix."""
    if _HAS_SPARSE and isinstance(areas, sparse.COO):
        # COO sparse: row_sum is 0 iff row has no nonzero entries.
        n_dst = int(areas.shape[0])
        mask = np.zeros(n_dst, dtype=bool)
        mask[areas.coords[0]] = True
        return mask
    arr = np.asarray(areas)
    return np.asarray((arr > 0).any(axis=1))


def _sum_matrix_axis_1d(areas: "sparse.COO | np.ndarray", axis: int) -> np.ndarray:
    summed = areas.sum(axis=axis)
    if hasattr(summed, "todense"):
        summed = summed.todense()
    return np.asarray(summed, dtype=np.float64).reshape(-1)


def _normalize_longitude_coords(
    source: xr.DataArray | xr.Dataset,
    target: xr.Dataset,
    x_coord: str,
) -> tuple[xr.DataArray | xr.Dataset, xr.Dataset, np.ndarray | None]:
    """Unwrap x coordinates across the antimeridian so source and target share
    a contiguous longitude frame. No-op when the coord isn't present on both
    objects or doesn't look like a longitude.

    For 1D rectilinear longitudes on both sides this mirrors the per-value
    wrap done by :func:`xarray_regrid.utils.format_lon` for the axis-factored
    path, which is what makes a source on ``[0, 360]`` align with a target on
    ``[-180, 180]`` (and vice versa). If wrapping breaks source monotonicity
    the source coord is sorted in place; the caller is expected to remap the
    area-matrix columns by the returned ``src_x_sort_idx`` so the final matrix
    columns line up with the user's original data layout. For 2D / curvilinear
    coords the existing uniform-shift fallback is kept.
    """
    if x_coord not in source.coords or x_coord not in target.coords:
        return source, target, None

    source_x = np.asarray(source[x_coord].values)
    target_x = np.asarray(target[x_coord].values)
    if not _looks_like_longitude(source_x) and not _looks_like_longitude(target_x):
        return source, target, None

    source_x = _unwrap_longitude(source_x)
    target_x = _unwrap_longitude(target_x)
    src_finite = source_x[np.isfinite(source_x)]
    tgt_finite = target_x[np.isfinite(target_x)]

    src_x_sort_idx: np.ndarray | None = None
    if (
        source_x.ndim == 1
        and target_x.ndim == 1
        and src_finite.size
        and tgt_finite.size
    ):
        # Per-value wrap source into target's 360° window — mirrors format_lon
        # so source [0, 360] vs target [-180, 180] (and either reversed)
        # aligns. A uniform offset can't reconcile cross-convention grids
        # (mean diff is exactly 180° and round() is banker's-rounded to 0).
        source_x = utils.wrap_longitudes_to_target_window(
            source_x, float(tgt_finite[0]), float(tgt_finite[-1])
        )
        diffs = np.diff(source_x)
        if not (np.all(diffs > 0) or np.all(diffs < 0)):
            src_x_sort_idx = np.argsort(source_x, kind="stable")
            source_x = source_x[src_x_sort_idx]
    elif src_finite.size and tgt_finite.size:
        # 2D / curvilinear: fall back to uniform shift of target into source's
        # window. Doesn't handle cross-convention but preserves the existing
        # antimeridian-crossing behavior for 2D coords.
        target_x = target_x + _periodic_offset(
            float(src_finite.mean()), float(tgt_finite.mean())
        )
    return (
        utils.update_coord(source, x_coord, source_x),
        cast(xr.Dataset, utils.update_coord(target, x_coord, target_x)),
        src_x_sort_idx,
    )


def _looks_like_longitude(values: np.ndarray) -> bool:
    finite = values[np.isfinite(values)]
    return bool(finite.size and finite.min() >= -360.0 and finite.max() <= 360.0)


def _unwrap_longitude(values: np.ndarray) -> np.ndarray:
    """Unwrap longitudes along the trailing axis (CF convention). For 2D
    coords, latitude doesn't wrap mod 360°, so unwrapping it is incorrect."""
    radians = np.deg2rad(np.asarray(values, dtype=float))
    return np.rad2deg(np.unwrap(radians, axis=-1))


def _normalize_periodic_polygons(
    polygons: np.ndarray, reference: float | None = None
) -> np.ndarray:
    """Unwrap each polygon across the antimeridian, then shift each into the
    same 360-degree window as ``reference`` (or as the first finite center if
    ``reference`` is None)."""
    unwrapped = [_unwrap_polygon(p) for p in polygons]
    if reference is None:
        for poly in unwrapped:
            center = _polygon_center_x(poly)
            if np.isfinite(center):
                reference = center
                break
    if reference is None:
        return np.array(unwrapped, dtype=object)

    out = []
    for poly in unwrapped:
        offset = _periodic_offset(reference, _polygon_center_x(poly))
        out.append(affinity.translate(poly, xoff=offset) if offset != 0.0 else poly)
    return np.array(out, dtype=object)


def _polygon_reference_x(polygons: np.ndarray) -> float | None:
    """Mean polygon-center x across an array of polygons, or None if all
    polygons have non-finite centers."""
    bounds = shapely.bounds(polygons)
    centers = 0.5 * (bounds[:, 0] + bounds[:, 2])
    finite = centers[np.isfinite(centers)]
    return float(finite.mean()) if finite.size else None


def _polygon_center_x(polygon: Any) -> float:
    minx, _, maxx, _ = polygon.bounds
    return 0.5 * (float(minx) + float(maxx))


def _periodic_offset(reference: float, value: float) -> float:
    """Smallest multiple of 360° that brings ``value`` close to ``reference``.

    Half-up symmetric rounding (not Python's banker's ``round()``): for
    antipodal pairs (``reference - value`` exactly ±180°) we always shift by
    ±360° rather than tying to zero, so the polygon ends up adjacent to
    ``reference`` instead of being silently left antipodal.
    """
    if not np.isfinite(reference) or not np.isfinite(value):
        return 0.0
    diff = (reference - value) / 360.0
    return 360.0 * math.copysign(math.floor(abs(diff) + 0.5), diff)


def _unwrap_polygon(polygon: Any) -> Any:
    if polygon.is_empty:
        return polygon
    if polygon.geom_type == "Polygon":
        exterior = _unwrap_ring(np.asarray(polygon.exterior.coords))
        holes = [_unwrap_ring(np.asarray(ring.coords)) for ring in polygon.interiors]
        return shapely.Polygon(exterior, holes)
    if polygon.geom_type == "MultiPolygon":
        return shapely.MultiPolygon([_unwrap_polygon(part) for part in polygon.geoms])
    return polygon


def _unwrap_ring(ring: np.ndarray) -> np.ndarray:
    new_ring = np.asarray(ring, dtype=float).copy()
    new_ring[:, 0] = np.unwrap(new_ring[:, 0], period=360.0)
    return new_ring


def _remap_columns_for_axis_sort(
    areas: "sparse.COO | np.ndarray",
    sort_idx: np.ndarray,
    src_shape: tuple[int, ...],
    axis_index: int,
) -> "sparse.COO | np.ndarray":
    """Relabel the column indices of an ``(n_dst, prod(src_shape))`` area
    matrix so that columns appear in the user's original source-data order
    after ``sort_idx`` was applied along ``axis_index`` of ``src_shape``.

    For sparse, we permute the existing ``coords[1]`` column indices forward
    via ``sort_idx``. For dense, we fancy-index columns by the inverse
    permutation. Both express the same unravel/permute/ravel operation.
    """
    sort_idx = np.asarray(sort_idx, dtype=np.int64)
    permute = _axis_index_permuter(src_shape, axis_index)

    if _HAS_SPARSE and isinstance(areas, sparse.COO):
        new_col = permute(np.asarray(areas.coords[1], dtype=np.int64), sort_idx)
        return sparse.COO(
            coords=np.stack([np.asarray(areas.coords[0], dtype=np.int64), new_col]),
            data=np.asarray(areas.data),
            shape=areas.shape,
            has_duplicates=False,
            sorted=False,
        )

    arr = np.asarray(areas)
    inv_sort = np.empty_like(sort_idx)
    inv_sort[sort_idx] = np.arange(sort_idx.size, dtype=sort_idx.dtype)
    inv_perm = permute(np.arange(arr.shape[1], dtype=np.int64), inv_sort)
    return arr[:, inv_perm]


def _axis_index_permuter(
    src_shape: tuple[int, ...], axis_index: int
) -> "Callable[[np.ndarray, np.ndarray], np.ndarray]":
    """Return ``permute(flat_indices, lookup)`` that permutes the
    ``axis_index`` component of each row-major flat index via ``lookup``,
    leaving all other components untouched."""
    nx = int(src_shape[axis_index])
    inner = int(np.prod(src_shape[axis_index + 1 :]))
    stride = nx * inner

    def permute(flat: np.ndarray, lookup: np.ndarray) -> np.ndarray:
        outer = (flat // stride) * stride
        within = flat % stride
        return outer + lookup[within // inner] * inner + within % inner

    return permute


def _transpose_weights(
    w: "sparse.COO | np.ndarray",
) -> "sparse.COO | np.ndarray":
    """Materialize a transposed weight matrix.

    ``sparse.COO.T`` is a lazy view, so callers relying on ``.coords[0]``
    being row indices (e.g. the backward direction's area matrix) should
    materialize once here.
    """
    if _HAS_SPARSE and isinstance(w, sparse.COO):
        t = w.T
        return sparse.COO(
            coords=np.asarray(t.coords),
            data=np.asarray(t.data),
            shape=t.shape,
            has_duplicates=False,
            sorted=False,
        )
    return np.asarray(w).T.copy()


def _to_csr(weights: "sparse.COO | np.ndarray") -> "sp.csr_matrix":
    """Convert row-normalized weights to a scipy CSR matrix for the apply path.

    scipy's C SpMM (CSR · dense) is markedly faster than ``sparse.COO @
    ndarray``, whose numba kernel never converts COO to CSR (unlike sparse's
    own ``COO @ COO`` and ``GCXS @ ndarray`` paths). scipy is a hard
    dependency, so this is available regardless of whether ``sparse`` is
    installed (and also speeds up the dense fallback).
    """
    if _HAS_SPARSE and isinstance(weights, sparse.COO):
        return sp.csr_matrix(
            (weights.data, (weights.coords[0], weights.coords[1])),
            shape=weights.shape,
        )
    return sp.csr_matrix(np.asarray(weights))


def _is_rectilinear_pair(xc: xr.DataArray, yc: xr.DataArray) -> bool:
    """True when x and y are 1D coords on distinct dims (axis-aligned
    rectangles — the separable fast path); False for curvilinear 2D coords."""
    return bool(xc.ndim == 1 and yc.ndim == 1 and xc.dims[0] != yc.dims[0])


def _spatial_dims(
    obj: xr.DataArray | xr.Dataset, x_coord: str, y_coord: str
) -> tuple[Hashable, ...]:
    """Return the spatial dim order to feed to apply_ufunc.

    For 1D rectilinear coords (x and y ride on separate dims), canonicalize to
    ``(y_dim, x_dim)`` so the flattened cell order matches the polygon order
    emitted by the fast-path in ``_build_grid``. For curvilinear (2D) coords,
    preserve the dim order already present on ``obj``.
    """
    if x_coord not in obj.coords or y_coord not in obj.coords:
        return ()
    xc = obj[x_coord]
    yc = obj[y_coord]
    if _is_rectilinear_pair(xc, yc):
        return (yc.dims[0], xc.dims[0])
    dims = set(xc.dims) | set(yc.dims)
    return tuple(d for d in obj.dims if d in dims)


class GeometryBackend(ABC):
    """Per-manifold geometry strategy: how a grid's cells are built, how
    candidate ``(dst, src)`` overlap pairs are found, and how their intersection
    areas are computed.

    Bundling all three per manifold keeps the candidate search consistent with
    the coordinate system the cells actually live in: a planar lon/lat bbox
    filter is only sound for cells that *are* planar boxes, so each manifold
    owns a candidate search valid for its own geometry rather than sharing one
    assumption. A new manifold is one new subclass + one registry entry.
    """

    name: ClassVar[str]

    def check_deps(self) -> None:
        """Raise if an optional dependency for this manifold is missing."""

    @abstractmethod
    def grid_from_coords(
        self,
        obj: xr.DataArray | xr.Dataset,
        x_coord: str,
        y_coord: str,
        dims: tuple[Hashable, ...],
    ) -> "_Grid":
        """Build the cell geometry from the object's x/y coordinates."""

    @abstractmethod
    def polys_from_arrays(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Row-major ``(y, x)`` shapely cell polys for ``polygons_from_coords``."""

    @abstractmethod
    def area_matrix(
        self, src: "_Grid", dst: "_Grid", n_threads: int | None = None
    ) -> "sparse.COO | np.ndarray":
        """Unnormalized ``(n_dst, n_src)`` area-intersection matrix
        ``A[i, j] = area(dst_i ∩ src_j)``. (``predicate_filter`` is a
        planar-only candidate-search knob widened in by :class:`PlanarBackend`.)"""


class PlanarBackend(GeometryBackend):
    """Raw shapely geometry in the user's coordinate space. Rectilinear grids
    take the analytic axis-aligned box-overlap fast path; curvilinear and
    arbitrary-polygon grids go through GEOS clipping. Also serves the
    ``from_polygons`` path."""

    name = "planar"

    def grid_from_coords(
        self,
        obj: xr.DataArray | xr.Dataset,
        x_coord: str,
        y_coord: str,
        dims: tuple[Hashable, ...],
    ) -> "_Grid":
        xd = obj[x_coord]
        yd = obj[y_coord]
        if _is_rectilinear_pair(xd, yd):
            return _build_grid(np.asarray(xd.values), np.asarray(yd.values))
        xc, yc = xr.broadcast(xd, yd)
        return _build_grid(
            np.asarray(xc.transpose(*dims).values),
            np.asarray(yc.transpose(*dims).values),
        )

    def polys_from_arrays(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        return _build_grid(x, y).polys

    @staticmethod
    def grid_from_polygons(polys: np.ndarray) -> "_Grid":
        """Non-rectilinear _Grid from explicit shapely cell polygons (the
        ``from_polygons`` path), keeping all planar _Grid construction here."""
        return _Grid(polys=polys, bounds=shapely.bounds(polys), rectilinear=False)

    def area_matrix(
        self,
        src: "_Grid",
        dst: "_Grid",
        n_threads: int | None = None,
        *,
        predicate_filter: bool = False,
    ) -> "sparse.COO | np.ndarray":
        n_dst = len(dst.polys)
        n_src = len(src.polys)
        dst_idx, src_idx = _bbox_candidate_pairs(src.polys, dst.polys, predicate_filter)
        if dst_idx.size == 0:
            return _empty_weights(n_dst, n_src)
        if src.rectilinear and dst.rectilinear:
            areas = _analytic_box_areas(src.bounds[src_idx], dst.bounds[dst_idx])
        else:
            areas = _intersection_areas_threaded(
                dst.polys[dst_idx], src.polys[src_idx], n_threads=n_threads
            )
        return _assemble_area_matrix(dst_idx, src_idx, areas, n_dst, n_src)


class CeaBackend(PlanarBackend):
    """Lambert cylindrical equal-area: 1D rectilinear lat/lon (degrees) are
    projected to ``(lon_rad, sin(lat_rad))`` before building cells, giving
    mass-conservative weights on the sphere at the planar fast path's cost.
    Cells are axis-aligned boxes in projected space, so the candidate search
    and intersection reuse :class:`PlanarBackend`. Rectilinear-only."""

    name = "cea"

    def grid_from_coords(
        self,
        obj: xr.DataArray | xr.Dataset,
        x_coord: str,
        y_coord: str,
        dims: tuple[Hashable, ...],  # noqa: ARG002 — dispatched signature
    ) -> "_Grid":
        xd = obj[x_coord]
        yd = obj[y_coord]
        if not _is_rectilinear_pair(xd, yd):
            msg = 'manifold="cea" is only supported for rectilinear (1D lat/lon) coords'
            raise NotImplementedError(msg)
        return _build_cea_grid(np.asarray(xd.values), np.asarray(yd.values))

    def polys_from_arrays(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        if x.ndim != 1 or y.ndim != 1:
            msg = 'manifold="cea" requires 1D lat/lon arrays'
            raise ValueError(msg)
        return _build_cea_grid(x, y).polys


class S2Backend(GeometryBackend):
    """True great-circle geometry on the sphere via the optional ``spherely``
    package (s2geometry). Cells carry spherely Geographies in ``s2_polys`` and
    intersection areas are exact great-circle steradians. The candidate search
    is s2-specific — a bulge-faithful planar shadow plus antimeridian-seam
    handling — because a planar lon/lat bbox does not bound a great-circle cell.
    Rectilinear-only."""

    name = "s2"

    def check_deps(self) -> None:
        _check_spherely()

    def grid_from_coords(
        self,
        obj: xr.DataArray | xr.Dataset,
        x_coord: str,
        y_coord: str,
        dims: tuple[Hashable, ...],  # noqa: ARG002 — dispatched signature
    ) -> "_Grid":
        xd = obj[x_coord]
        yd = obj[y_coord]
        if not _is_rectilinear_pair(xd, yd):
            msg = 'manifold="s2" is only supported for rectilinear (1D lat/lon) coords'
            raise NotImplementedError(msg)
        return _build_s2_grid(np.asarray(xd.values), np.asarray(yd.values))

    def polys_from_arrays(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:  # noqa: ARG002
        msg = (
            "polygons_from_coords supports manifold 'planar' or 'cea'; for s2 "
            "use ConservativeRegridder(..., manifold='s2') directly."
        )
        raise ValueError(msg)

    def area_matrix(
        self, src: "_Grid", dst: "_Grid", n_threads: int | None = None
    ) -> "sparse.COO | np.ndarray":
        s2_src = cast("np.ndarray", src.s2_polys)
        s2_dst = cast("np.ndarray", dst.s2_polys)
        n_dst = len(s2_dst)
        n_src = len(s2_src)
        dst_idx, src_idx = self._candidate_pairs(src, dst)
        if dst_idx.size == 0:
            return _empty_weights(n_dst, n_src)
        areas = _s2_intersection_areas(
            s2_dst[dst_idx], s2_src[src_idx], n_threads=n_threads
        )
        return _assemble_area_matrix(dst_idx, src_idx, areas, n_dst, n_src)

    @staticmethod
    def _candidate_pairs(src: "_Grid", dst: "_Grid") -> tuple[np.ndarray, np.ndarray]:
        """Candidate (dst, src) pairs from the bulge-faithful planar shadow.

        Input coordinates are already convention-reconciled and antimeridian-
        unwrapped by :func:`_normalize_longitude_coords` (in ``__init__``); this
        loop solves a different, s2-only problem: the *planar shadow's* STRtree
        can't see that two cells on opposite sides of ±180° genuinely overlap on
        the sphere. Querying the dst boxes shifted by 0 / ±360° in longitude
        recovers those seam pairs — spherely then computes their true
        great-circle overlap (and the ``area > 0`` filter drops the rest). With
        the faithful shadow already bounding each cell's poleward bulge, the
        candidate set is a conservative superset on the sphere."""
        _check_shapely()
        tree = STRtree(np.asarray(src.polys))
        db = dst.bounds
        parts_d, parts_s = [], []
        for shift in (0.0, 360.0, -360.0):
            q = (
                dst.polys
                if shift == 0.0
                else shapely.box(db[:, 0] + shift, db[:, 1], db[:, 2] + shift, db[:, 3])
            )
            pairs = tree.query(np.asarray(q))
            parts_d.append(np.asarray(pairs[0]))
            parts_s.append(np.asarray(pairs[1]))
        dst_idx = np.concatenate(parts_d)
        src_idx = np.concatenate(parts_s)
        if dst_idx.size:
            n_src = len(src.polys)
            flat = dst_idx.astype(np.int64) * n_src + src_idx.astype(np.int64)
            _, uniq = np.unique(flat, return_index=True)
            dst_idx, src_idx = dst_idx[uniq], src_idx[uniq]
        return dst_idx, src_idx


# Manifold registry. Each backend owns its grid construction, candidate-pair
# search, and intersection kernel; new manifolds plug in via a single insert.
# ``_PLANAR`` is also held concretely — ``from_polygons`` uses it directly so its
# planar-only ``grid_from_polygons`` / ``predicate_filter`` are statically visible.
_PLANAR = PlanarBackend()
_BACKENDS: dict[str, GeometryBackend] = {
    b.name: b for b in (_PLANAR, CeaBackend(), S2Backend())
}


def _resolve_manifold(manifold: str) -> GeometryBackend:
    """Look up the backend for ``manifold``, raising a clear error on an unknown
    name (one source of truth for the message)."""
    if manifold not in _BACKENDS:
        valid = ", ".join(repr(m) for m in sorted(_BACKENDS))
        msg = f"manifold must be one of {{{valid}}}; got {manifold!r}"
        raise ValueError(msg)
    return _BACKENDS[manifold]


def _check_manifold(manifold: str) -> None:
    _resolve_manifold(manifold).check_deps()


def _build_cea_grid(lon_centers: np.ndarray, lat_centers: np.ndarray) -> "_Grid":
    """Build a rectilinear :class:`_Grid` whose cell polygons are in Lambert
    cylindrical equal-area coordinates (x' = lon_rad, y' = sin(lat_rad)).

    Projecting *edges* analytically — rather than projecting centers and then
    re-midpointing — is required because ``sin()`` is nonlinear: the projected
    midpoint of two lat centers is not the same as the midpoint of two
    projected lat edges.
    """
    _check_shapely()
    if lon_centers.size < 2 or lat_centers.size < 2:
        msg = 'manifold="cea" requires at least two cells per dimension'
        raise ValueError(msg)
    lat_edges_deg = np.clip(utils.infer_1d_edges(lat_centers), -90.0, 90.0)
    lon_edges_deg = utils.infer_1d_edges(lon_centers)
    return _rect_grid_from_edges(
        np.deg2rad(lon_edges_deg),
        np.sin(np.deg2rad(lat_edges_deg)),
    )


def _build_s2_grid(lon_centers: np.ndarray, lat_centers: np.ndarray) -> "_Grid":
    """Rectilinear _Grid carrying spherely great-circle cell polygons in
    ``s2_polys``, plus a planar shapely shadow (``polys``/``bounds``) used by
    :meth:`S2Backend._candidate_pairs` as the STRtree candidate filter (spherely
    has no spatial index). The shadow's latitude bounds are extended to each
    cell's great-circle apex so the bbox is a conservative superset of the s2
    cell — see :func:`_s2_shadow_boxes`."""
    _check_shapely()
    _check_spherely()
    if lon_centers.size < 2 or lat_centers.size < 2:
        msg = 'manifold="s2" requires at least two cells per dimension'
        raise ValueError(msg)
    lon_edges_deg = utils.infer_1d_edges(lon_centers)
    lat_edges_deg = np.clip(utils.infer_1d_edges(lat_centers), -90.0, 90.0)
    shadow_polys, shadow_bounds = _s2_shadow_boxes(lon_edges_deg, lat_edges_deg)
    return _Grid(
        polys=shadow_polys,
        bounds=shadow_bounds,
        rectilinear=True,
        s2_polys=_s2_cell_polys(lon_edges_deg, lat_edges_deg),
    )


def _s2_shadow_boxes(
    lon_edges_deg: np.ndarray, lat_edges_deg: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Planar shadow boxes that are a conservative superset of the s2 cells.

    A constant-latitude cell edge is *not* a great circle: its geodesic bows
    poleward, reaching a maximum latitude (the apex)

        apex(phi, dlon) = sign(phi) * atan( |tan(phi)| / cos(dlon / 2) )

    so a planar ``[lat0, lat1]`` box under-bounds the cell and the STRtree could
    drop a genuinely-overlapping pair. We extend each cell's poleward latitude
    bound to that apex (longitude edges are meridians — themselves great circles
    — so they do not bulge). The resulting bbox bounds the cell, making the
    candidate search lossless. Returns row-major ``(y, x)`` polys + bounds.
    """
    xlo = np.minimum(lon_edges_deg[:-1], lon_edges_deg[1:])  # (nx,)
    xhi = np.maximum(lon_edges_deg[:-1], lon_edges_deg[1:])
    ylo = np.minimum(lat_edges_deg[:-1], lat_edges_deg[1:])  # (ny,)
    yhi = np.maximum(lat_edges_deg[:-1], lat_edges_deg[1:])
    dlon = np.abs(lon_edges_deg[1:] - lon_edges_deg[:-1])  # (nx,) cell widths
    cos_half = np.cos(np.deg2rad(dlon) / 2.0)  # (nx,); <= 0 once dlon >= 180

    def apex(lat_deg: np.ndarray) -> np.ndarray:  # (ny,) -> (ny, nx)
        phi = np.deg2rad(lat_deg)[:, None]
        # arctan2 keeps cos_half <= 0 finite; sign(phi) zeroes the equator edge.
        mag = np.arctan2(np.abs(np.tan(phi)), cos_half[None, :])
        return np.rad2deg(np.sign(phi) * mag)

    lat_lo = np.clip(np.minimum(ylo[:, None], apex(ylo)), -90.0, 90.0)  # (ny, nx)
    lat_hi = np.clip(np.maximum(yhi[:, None], apex(yhi)), -90.0, 90.0)
    lon_lo = np.broadcast_to(xlo[None, :], lat_lo.shape).ravel()
    lon_hi = np.broadcast_to(xhi[None, :], lat_hi.shape).ravel()
    y0f, y1f = lat_lo.ravel(), lat_hi.ravel()
    polys = shapely.box(lon_lo, y0f, lon_hi, y1f)
    bounds = np.stack([lon_lo, y0f, lon_hi, y1f], axis=1)
    return polys, bounds


def _s2_cell_polys(lon_edges_deg: np.ndarray, lat_edges_deg: np.ndarray) -> np.ndarray:
    """Build an (n_cells,) object array of spherely.Geography polygons.

    Each cell's corners are taken as ``(min, max)`` of its lon/lat edge pair, so
    the ring is counter-clockwise in (lon, lat) **regardless of whether the
    source lat/lon run ascending or descending** (the CF ``90 → -90`` latitude
    convention is descending). ``spherely.create_polygon(..., oriented=True)``
    trusts the winding: a clockwise ring would be read as the cell's
    hemisphere-sized *complement*, silently corrupting every weight. Edges are
    monotonic and contiguous after longitude unwrapping, so no single cell
    straddles the antimeridian and the per-axis ``min/max`` is exact. Row-major
    ``(y, x)`` order matches the planar shadow from :func:`_rect_grid_from_edges`.
    """
    xlo = np.minimum(lon_edges_deg[:-1], lon_edges_deg[1:])
    xhi = np.maximum(lon_edges_deg[:-1], lon_edges_deg[1:])
    ylo = np.minimum(lat_edges_deg[:-1], lat_edges_deg[1:])
    yhi = np.maximum(lat_edges_deg[:-1], lat_edges_deg[1:])
    if hasattr(spherely, "polygons"):  # pragma: no cover — unreleased spherely#52
        # Vectorized constructor (benbovy/spherely#52, not yet released). When
        # it lands we build shells as (n, 4, 2) and skip the Python loop.
        x0, y0 = np.meshgrid(xlo, ylo, indexing="xy")
        x1, y1 = np.meshgrid(xhi, yhi, indexing="xy")
        shells = np.stack(
            [
                np.stack([x0, y0], axis=-1),
                np.stack([x1, y0], axis=-1),
                np.stack([x1, y1], axis=-1),
                np.stack([x0, y1], axis=-1),
            ],
            axis=-2,
        ).reshape(-1, 4, 2)
        return np.asarray(spherely.polygons(shells, oriented=True))
    # Per-cell fallback: `spherely.create_polygon` wants an iterable of tuples,
    # and a Python loop over pre-slicing an ndarray is slower than building
    # the tuple list inline.
    nx = xlo.size
    ny = ylo.size
    polys = np.empty(ny * nx, dtype=object)
    for j in range(ny):
        y0f, y1f = float(ylo[j]), float(yhi[j])
        for i in range(nx):
            x0f, x1f = float(xlo[i]), float(xhi[i])
            shell = [(x0f, y0f), (x1f, y0f), (x1f, y1f), (x0f, y1f)]
            polys[j * nx + i] = spherely.create_polygon(shell, oriented=True)
    return polys


def _infer_2d_corners(a: np.ndarray) -> np.ndarray:
    """Infer (ny+1, nx+1) cell corners from a 2D cell-center array. Interior
    corners are the mean of the 4 surrounding centers; boundary corners are
    reflected from the adjacent interior row/column."""
    a = np.asarray(a, dtype=float)
    ny, nx = a.shape
    pad = np.empty((ny + 2, nx + 2), dtype=a.dtype)
    pad[1:-1, 1:-1] = a
    pad[0, 1:-1] = 2 * a[0, :] - a[1, :]
    pad[-1, 1:-1] = 2 * a[-1, :] - a[-2, :]
    pad[1:-1, 0] = 2 * a[:, 0] - a[:, 1]
    pad[1:-1, -1] = 2 * a[:, -1] - a[:, -2]
    pad[0, 0] = 2 * pad[0, 1] - pad[0, 2]
    pad[0, -1] = 2 * pad[0, -2] - pad[0, -3]
    pad[-1, 0] = 2 * pad[-1, 1] - pad[-1, 2]
    pad[-1, -1] = 2 * pad[-1, -2] - pad[-1, -3]
    return 0.25 * (pad[:-1, :-1] + pad[1:, :-1] + pad[:-1, 1:] + pad[1:, 1:])


@dataclass
class _Grid:
    """Cached cell geometry consumed by a :class:`GeometryBackend`.

    ``polys`` is a flat (n_cells,) object array of shapely Polygons used by the
    STRtree candidate-pair search. ``bounds`` is the matching (n_cells, 4)
    ``(minx, miny, maxx, maxy)`` array. ``rectilinear`` is True when both the
    source x and y were 1D coordinate arrays (axis-aligned rectangles) — the
    :class:`PlanarBackend` uses this to compute intersection areas analytically
    from the bounds instead of via GEOS clipping.

    ``s2_polys`` is an optional (n_cells,) object array of spherely Geography
    polygons (great-circle cells on the sphere), set only for ``manifold="s2"``.
    There ``polys``/``bounds`` hold the *bulge-faithful* planar shadow (its
    latitude bounds reach each cell's great-circle apex, see
    :func:`_s2_shadow_boxes`) so the STRtree candidate set is a conservative
    superset of the s2 cells; :class:`S2Backend` computes the actual areas from
    ``s2_polys`` via spherely.
    """

    polys: np.ndarray
    bounds: np.ndarray
    rectilinear: bool
    s2_polys: np.ndarray | None = None


def _rect_grid_from_edges(xe: np.ndarray, ye: np.ndarray) -> _Grid:
    """Build a rectilinear :class:`_Grid` from already-prepared edge arrays.

    Shared by the raw-planar (:func:`_build_grid` 1D branch) and the analytic
    equal-area (:func:`_build_cea_grid`) paths.
    """
    x0, y0 = np.meshgrid(xe[:-1], ye[:-1], indexing="xy")
    x1, y1 = np.meshgrid(xe[1:], ye[1:], indexing="xy")
    x0f, y0f, x1f, y1f = x0.ravel(), y0.ravel(), x1.ravel(), y1.ravel()
    polys = shapely.box(x0f, y0f, x1f, y1f)
    bounds = np.stack([x0f, y0f, x1f, y1f], axis=1)
    return _Grid(polys=polys, bounds=bounds, rectilinear=True)


def _build_grid(xc: np.ndarray, yc: np.ndarray) -> _Grid:
    """Return a _Grid of cell geometry for a structured grid.

    Accepts 1D (rectilinear, separate x and y vectors) or 2D (curvilinear,
    co-shaped center arrays) inputs. Output order is row-major in the input dim
    order: for 2D inputs of shape (ny, nx) the polygons correspond to cells
    reshaped as ``(ny, nx)``.
    """
    _check_shapely()
    if xc.ndim == 1 and yc.ndim == 1:
        xe = utils.infer_1d_edges(xc.astype(float))
        ye = utils.infer_1d_edges(yc.astype(float))
        return _rect_grid_from_edges(xe, ye)
    if xc.ndim == 2 and yc.ndim == 2 and xc.shape == yc.shape:
        xcorn = _infer_2d_corners(xc)
        ycorn = _infer_2d_corners(yc)
        ny, nx = xc.shape
        c00 = np.stack([xcorn[:-1, :-1], ycorn[:-1, :-1]], axis=-1)
        c10 = np.stack([xcorn[:-1, 1:], ycorn[:-1, 1:]], axis=-1)
        c11 = np.stack([xcorn[1:, 1:], ycorn[1:, 1:]], axis=-1)
        c01 = np.stack([xcorn[1:, :-1], ycorn[1:, :-1]], axis=-1)
        rings = np.stack([c00, c10, c11, c01, c00], axis=2).reshape(ny * nx, 5, 2)
        polys = shapely.polygons(rings)
        return _Grid(polys=polys, bounds=shapely.bounds(polys), rectilinear=False)
    msg = "x and y coordinate arrays must both be 1D or both 2D"
    raise ValueError(msg)


def _bbox_candidate_pairs(
    src_polys: np.ndarray, dst_polys: np.ndarray, predicate_filter: bool
) -> tuple[np.ndarray, np.ndarray]:
    """Candidate ``(dst, src)`` cell pairs from a planar STRtree bbox query.

    ``predicate_filter=False`` (default) is a bbox-only query and relies on the
    downstream ``area > 0`` filter to drop bbox false-positives — a large win
    for tight quad bboxes, where the GEOS ``intersects`` predicate inside
    STRtree costs more than the no-op intersections it avoids.
    ``predicate_filter=True`` runs the predicate, which pays off for loose
    bboxes (long, thin diagonals).
    """
    _check_shapely()
    tree = STRtree(np.asarray(src_polys))
    if predicate_filter:
        pairs = tree.query(np.asarray(dst_polys), predicate="intersects")
    else:
        pairs = tree.query(np.asarray(dst_polys))
    return np.asarray(pairs[0]), np.asarray(pairs[1])


def _analytic_box_areas(src_bounds: np.ndarray, dst_bounds: np.ndarray) -> np.ndarray:
    """Per-pair overlap area of axis-aligned boxes from their ``(minx, miny,
    maxx, maxy)`` bounds — exact for rectilinear cells, no GEOS clipping."""
    dx = np.minimum(src_bounds[:, 2], dst_bounds[:, 2]) - np.maximum(
        src_bounds[:, 0], dst_bounds[:, 0]
    )
    dy = np.minimum(src_bounds[:, 3], dst_bounds[:, 3]) - np.maximum(
        src_bounds[:, 1], dst_bounds[:, 1]
    )
    return np.maximum(dx, 0.0) * np.maximum(dy, 0.0)


def _assemble_area_matrix(
    dst_idx: np.ndarray,
    src_idx: np.ndarray,
    areas: np.ndarray,
    n_dst: int,
    n_src: int,
) -> "sparse.COO | np.ndarray":
    """Drop zero-area candidate pairs and assemble the (sparse or dense)
    ``(n_dst, n_src)`` unnormalized area-intersection matrix. Row-normalize via
    :func:`_row_normalize` to get forward weights; transpose first for backward
    (target → source)."""
    keep = areas > 0
    return _coo_or_dense(
        dst_idx[keep], src_idx[keep], areas[keep].astype(np.float64), (n_dst, n_src)
    )


def _row_normalize(
    areas: "sparse.COO | np.ndarray",
) -> "sparse.COO | np.ndarray":
    """Normalize rows of an area matrix so each row sums to 1 (rows with no
    overlap stay all-zero, which produces NaN output under the apply path)."""
    if _HAS_SPARSE and isinstance(areas, sparse.COO):
        n_dst = areas.shape[0]
        dst_idx = areas.coords[0]
        src_idx = areas.coords[1]
        data = areas.data
        if data.size == 0:
            return areas
        row_sum = np.bincount(dst_idx, weights=data, minlength=n_dst)
        new_data = data / row_sum[dst_idx]
        return sparse.COO(
            coords=np.stack([dst_idx, src_idx]),
            data=new_data,
            shape=areas.shape,
            has_duplicates=False,
            sorted=False,
        )
    row_sum = areas.sum(axis=1, keepdims=True)
    row_sum = np.where(row_sum == 0, 1.0, row_sum)
    return areas / row_sum


def _threaded_pairwise(
    a: np.ndarray,
    b: np.ndarray,
    kernel: "Callable[[np.ndarray, np.ndarray], np.ndarray]",
    *,
    n_threads: int | None,
    serial_below: int,
    max_threads: int,
) -> np.ndarray:
    """Apply a vectorized pairwise ``kernel(a, b) -> areas`` over numpy arrays of
    geometries, chunked across a ``ThreadPoolExecutor`` when worthwhile.

    Both GEOS (shapely) and s2 (spherely) release the GIL during the boolean op,
    so threading scales near-linearly without pickling. Serial below
    ``serial_below`` pairs (pool spin-up dominates) or when ``n_threads <= 1``;
    ``None`` auto-selects ``min(cpu_count, max_threads)``.
    """
    n = len(a)
    if n_threads is None:
        n_threads = 1 if n < serial_below else min(os.cpu_count() or 1, max_threads)
    if n_threads <= 1 or n == 0:
        return kernel(a, b)

    splits = np.array_split(np.arange(n), n_threads)

    def _work(idx: np.ndarray) -> np.ndarray:
        return kernel(a[idx], b[idx])

    with ThreadPoolExecutor(max_workers=n_threads) as pool:
        parts = list(pool.map(_work, splits))
    return np.concatenate(parts)


def _intersection_areas_threaded(
    a: np.ndarray, b: np.ndarray, n_threads: int | None
) -> np.ndarray:
    """Per-pair planar intersection areas over shapely geometries (GEOS)."""
    _check_shapely()
    return _threaded_pairwise(
        a,
        b,
        lambda x, y: shapely.area(shapely.intersection(x, y)),
        n_threads=n_threads,
        serial_below=1_000,
        max_threads=16,
    )


def _s2_intersection_areas(
    dst_geog: np.ndarray, src_geog: np.ndarray, n_threads: int | None = None
) -> np.ndarray:
    """Per-pair great-circle intersection areas in steradians (spherely).

    ``radius=1.0`` gives areas on the unit sphere — fine for row-normalized
    weights, where the Earth radius cancels out.
    """
    _check_spherely()
    return _threaded_pairwise(
        dst_geog,
        src_geog,
        lambda x, y: np.asarray(spherely.area(spherely.intersection(x, y), radius=1.0)),
        n_threads=n_threads,
        serial_below=50_000,
        max_threads=4,
    )


def _empty_weights(n_dst: int, n_src: int) -> "sparse.COO | np.ndarray":
    empty = np.zeros(0, dtype=np.int64)
    return _coo_or_dense(empty, empty, np.zeros(0, dtype=np.float64), (n_dst, n_src))


def _coo_or_dense(
    rows: np.ndarray,
    cols: np.ndarray,
    data: np.ndarray,
    shape: tuple[int, int],
) -> "sparse.COO | np.ndarray":
    """Build a ``sparse.COO`` if available, else a dense ndarray."""
    if _HAS_SPARSE:
        return sparse.COO(
            coords=np.stack([rows, cols]),
            data=data,
            shape=shape,
            has_duplicates=False,
            sorted=False,
        )
    arr = np.zeros(shape, dtype=data.dtype if data.size else np.float64)
    if data.size:
        arr[rows, cols] = data
    return arr


def _apply_core(
    arr: np.ndarray,
    apply_weights: Any,
    coverage: np.ndarray,
    coverage_all: bool,
    src_shape: tuple[int, ...],
    dst_shape: tuple[int, ...],
    skipna: bool,
    nan_threshold: float,
    output_dtype: np.dtype,
) -> np.ndarray:
    """Apply the CSR weight matrix along the trailing spatial dims.

    ``arr`` has shape ``(..., *src_shape)``; ``apply_weights`` is a scipy CSR
    of shape ``(n_dst, n_src)`` and the matmul is ``(W @ flat.T).T`` —
    returns ``(..., *dst_shape)``.

    ``coverage`` is a boolean ``(n_dst,)`` mask: target cells with any source
    overlap. ``coverage_all`` is precomputed so every block skips the
    ``coverage.all()`` scan. Uncovered cells are always masked to NaN in the
    output, regardless of ``skipna`` — domain boundaries and polygon holes
    produce NaN (matches the axis-factored ``conservative`` method).
    """
    n_spatial = len(src_shape)
    leading_shape = arr.shape[:-n_spatial] if n_spatial > 0 else arr.shape
    n_src = int(np.prod(src_shape))
    flat = arr.reshape(-1, n_src) if leading_shape else arr.reshape(1, n_src)

    if skipna and np.issubdtype(flat.dtype, np.floating):
        nan_mask = np.isnan(flat)
        has_nan = nan_mask.any()
    else:
        has_nan = False

    if has_nan:
        mask = (~nan_mask).astype(flat.dtype)
        filled = np.where(nan_mask, flat.dtype.type(0.0), flat)
        numerator = np.asarray((apply_weights @ filled.T).T)
        fraction = np.asarray((apply_weights @ mask.T).T)
        threshold = get_valid_threshold(nan_threshold)
        with np.errstate(invalid="ignore", divide="ignore"):
            result = numerator / fraction
        valid = fraction >= threshold
    else:
        result = np.asarray((apply_weights @ flat.T).T)
        valid = None
    # Always mask domain-uncovered cells. With NaNs the uncovered rows already
    # have fraction=0 (so the threshold check below catches them), but ANDing
    # coverage in keeps the contract explicit and robust to threshold tweaks.
    if not coverage_all:
        cov = coverage[np.newaxis, :]
        valid = cov if valid is None else (valid & cov)
    if valid is not None:
        result = np.where(valid, result, np.nan)

    # The CSR matmul promotes to float64 (weights are float64) regardless of
    # the input dtype — cast back to the requested output dtype so float32-in
    # really produces float32-out (halves memory for float32 pipelines).
    # astype(copy=False) is a no-op when the dtype already matches.
    result = result.astype(output_dtype, copy=False)

    out_shape = (*leading_shape, *dst_shape) if leading_shape else dst_shape
    return result.reshape(out_shape)


def _result_dtype(obj: xr.DataArray | xr.Dataset) -> np.dtype:
    if isinstance(obj, xr.DataArray):
        return utils.min_weight_dtype(obj.dtype)
    dtypes = [v.dtype for v in obj.data_vars.values()]
    if not dtypes:
        return np.dtype(np.float64)
    return utils.min_weight_dtype(*dtypes)


def _assign_target_coords(
    obj: xr.DataArray | xr.Dataset,
    target_ds: xr.Dataset,
    spec: RegridSpec,
) -> xr.DataArray | xr.Dataset:
    """Attach target coordinates that name a spatial axis or live on the
    output spatial dims. Scalar coords (``dims == ()``) ride along too, so
    pinned target metadata (e.g. a fixed timestamp) is preserved."""
    dst_dim_set = set(spec.dst_dims)
    spatial = (spec.x_coord, spec.y_coord)
    new_coords = {
        name: coord
        for name, coord in target_ds.coords.items()
        if name in spatial or set(coord.dims).issubset(dst_dim_set)
    }
    return obj.assign_coords(new_coords) if new_coords else obj
