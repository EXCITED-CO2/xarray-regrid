"""Conservative regridding implementation."""

from collections.abc import Hashable
from typing import overload

import numpy as np
import xarray as xr

try:
    import sparse
except ImportError:
    sparse = None

from xarray_regrid import utils


@overload
def conservative_regrid(
    data: xr.DataArray,
    target_ds: xr.Dataset,
    latitude_coord: str | None,
    skipna: bool = True,
    nan_threshold: float = 1.0,
    output_chunks: dict[Hashable, int] | None = None,
) -> xr.DataArray: ...


@overload
def conservative_regrid(
    data: xr.Dataset,
    target_ds: xr.Dataset,
    latitude_coord: str | None,
    skipna: bool = True,
    nan_threshold: float = 1.0,
    output_chunks: dict[Hashable, int] | None = None,
) -> xr.Dataset: ...


def conservative_regrid(
    data: xr.DataArray | xr.Dataset,
    target_ds: xr.Dataset,
    latitude_coord: str | Hashable | None,
    skipna: bool = True,
    nan_threshold: float = 1.0,
    output_chunks: dict[Hashable, int] | None = None,
) -> xr.DataArray | xr.Dataset:
    """Refine a dataset using conservative regridding.

    The method implementation is based on a post by Stephan Hoyer; "For the case of
    interpolation between rectilinear grids (even on the sphere), you can factorize
    regridding along each axis. This is less general but makes the entire calculation
    much simpler, because its feasible to store interpolation weights as dense matrices
    and to use dense matrix multiplication."
    https://discourse.pangeo.io/t/conservative-region-aggregation-with-xarray-geopandas-and-sparse/2715/3

    Args:
        data: Input dataset.
        target_ds: Dataset which coordinates the input dataset should be regrid to.
        latitude_coord: Name of the latitude coordinate. If not provided, attempt to
            infer it from the first coordinate equaling the string 'lat' or 'latitude'.
        skipna: If True, enable handling for NaN values. This adds some overhead,
            so should be disabled for optimal performance on data without NaNs.
        nan_threshold: Threshold value that will retain any output points containing
            at least this many non-null input points. The default value is 1.0,
            which will keep output points containing any non-null inputs. The threshold
            is applied sequentially to each dimension, and may produce different results
            than a threshold applied concurrently to all regridding dimensions.
        output_chunks: Optional dictionary of explicit chunk sizes for the output data.
            If not provided, the output will be chunked the same as the input data.

    Returns:
        Regridded input dataset
    """
    # Attempt to infer the latitude coordinate
    if latitude_coord is None:
        for coord in data.coords:
            if str(coord).lower() in ["lat", "latitude"]:
                latitude_coord = coord
                break

    # Make sure the regridding coordinates are sorted
    coord_names = [coord for coord in target_ds.coords if coord in data.coords]
    target_ds_sorted = xr.Dataset(coords=target_ds.coords)
    for coord_name in coord_names:
        target_ds_sorted = utils.ensure_monotonic(target_ds_sorted, coord_name)
        data = utils.ensure_monotonic(data, coord_name)
    coords = {name: target_ds_sorted[name] for name in coord_names}

    regridded_data = utils.call_on_dataset(
        conservative_regrid_dataset,
        data,
        coords,
        latitude_coord,
        skipna,
        nan_threshold,
        output_chunks,
    )

    regridded_data = regridded_data.reindex_like(target_ds, copy=False)

    return regridded_data


def conservative_regrid_dataset(
    data: xr.Dataset,
    coords: dict[Hashable, xr.DataArray],
    latitude_coord: Hashable,
    skipna: bool,
    nan_threshold: float,
    output_chunks: dict[Hashable, int] | None = None,
) -> xr.Dataset:
    """Dataset implementation of the conservative regridding method."""
    data_vars = dict(data.data_vars)
    data_coords = dict(data.coords)
    data_attrs = {v: data_vars[v].attrs for v in data_vars}
    coord_attrs = {c: data_coords[c].attrs for c in data_coords}
    ds_attrs = data.attrs

    # Create weights array and coverage mask for each regridding dim
    weights = {}
    covered = {}
    for coord in coords:  # noqa: PLC0206
        covered[coord] = (coords[coord] <= data[coord].max()) & (
            coords[coord] >= data[coord].min()
        )

        target_coords = coords[coord].to_numpy()
        source_coords = data[coord].to_numpy()
        nd_weights = get_weights(source_coords, target_coords)

        da_weights = utils.create_dot_dataarray(
            nd_weights, str(coord), target_coords, source_coords
        )
        # Modify weights to correct for latitude distortion
        if coord == latitude_coord:
            da_weights = apply_spherical_correction(da_weights, latitude_coord)
        weights[coord] = da_weights

    # Apply the weights, using a unique set that matches chunking of each array
    for array in data_vars.keys():  # noqa: PLC0206
        var_weights = {}
        for coord, weight_array in weights.items():
            var_input_chunks = data_vars[array].chunksizes.get(coord)
            var_output_chunks = output_chunks.get(coord) if output_chunks else None
            var_weights[coord] = format_weights(
                weight_array,
                coord,
                data_vars[array].dtype,
                var_input_chunks,
                var_output_chunks,
            )

        data_vars[array] = apply_weights(
            da=data_vars[array],
            weights=var_weights,
            skipna=skipna,
            nan_threshold=nan_threshold,
        )
        # Mask out any regridded points outside the original domain
        # Limit to dims present on this array otherwise .where broadcasts
        var_covered = xr.DataArray(True)
        for coord in var_weights.keys():
            var_covered = var_covered & covered[coord]
        data_vars[array] = data_vars[array].where(var_covered)

    # Rebuild the results ensuring we preserve attributes and other coordinates
    for array, attrs in data_attrs.items():
        data_vars[array].attrs = attrs

    ds_regridded = xr.Dataset(data_vars=data_vars, attrs=ds_attrs)

    for coord, attrs in coord_attrs.items():
        if coord not in ds_regridded.coords:
            # Add back any additional coordinates from the original dataset
            ds_regridded[coord] = data_coords[coord]
        ds_regridded[coord].attrs = attrs

    return ds_regridded


def apply_weights(
    da: xr.DataArray,
    weights: dict[Hashable, xr.DataArray],
    skipna: bool,
    nan_threshold: float,
) -> xr.DataArray:
    """Apply the weights over all regridding dimensions simultaneously with `xr.dot`."""
    coords = list(weights.keys())
    weight_arrays = list(weights.values())

    if skipna:
        valid_frac = xr.dot(
            da.notnull(), *weight_arrays, dim=list(weights.keys()), optimize=True
        )

    da_regrid: xr.DataArray = xr.dot(
        da.fillna(0), *weight_arrays, dim=list(weights.keys()), optimize=True
    )

    if skipna:
        da_regrid /= valid_frac
        da_regrid = da_regrid.where(valid_frac >= get_valid_threshold(nan_threshold))

    # Rename temporary coordinates and ensure original dimension order
    coord_map = {f"target_{coord}": coord for coord in coords}
    da_regrid = da_regrid.rename(coord_map).transpose(*da.dims)

    return da_regrid


def get_valid_threshold(nan_threshold: float) -> float:
    """Invert the nan_threshold and coerce it to just above zero and below
    one to handle numerical precision limitations in the weight sum."""
    # This matches xesmf where na_thresh=0 keeps points with any valid data
    valid_threshold: float = 1 - np.clip(nan_threshold, 1e-6, 1.0 - 1e-6)
    return valid_threshold


def get_weights(source_coords: np.ndarray, target_coords: np.ndarray) -> np.ndarray:
    """Determine the weights to map from the old coordinates to the new coordinates.

    Args:
        source_coords: Source coordinates (center points)
        target_coords Target coordinates (center points)

    Returns:
        Weights, which can be used with a dot product to apply the conservative regrid.
    """
    target_intervals = utils.to_intervalindex(target_coords)
    source_intervals = utils.to_intervalindex(source_coords)

    overlap = utils.overlap(source_intervals, target_intervals)
    return utils.normalize_overlap(overlap)


def apply_spherical_correction(
    dot_array: xr.DataArray, latitude_coord: Hashable
) -> xr.DataArray:
    """Apply a sperical earth correction on the prepared dot product weights."""
    da = dot_array.copy()
    latitude_res = np.median(np.diff(dot_array[latitude_coord].to_numpy(), 1))
    lat_weights = lat_weight(dot_array[latitude_coord].to_numpy(), latitude_res)
    da.values = utils.normalize_overlap(dot_array.values * lat_weights[:, np.newaxis])
    return da


def lat_weight(latitude: np.ndarray, latitude_res: float) -> np.ndarray:
    """Return the weight of gridcells based on their latitude.

    Args:
        latitude: (Center) latitude values of the gridcells, in degrees.
        latitude_res: Resolution/width of the grid cells, in degrees.

    Returns:
        Weights, same shape as latitude input.
    """
    dlat: float = np.radians(latitude_res)
    lat = np.radians(latitude)
    h = np.sin(lat + dlat / 2) - np.sin(lat - dlat / 2)
    return h * dlat / (np.pi * 4)  # type: ignore


def format_weights(
    weights: xr.DataArray,
    coord: Hashable,
    input_dtype: np.dtype,
    input_chunks: tuple[int, ...] | None,
    output_chunks: tuple[int, ...] | int | None,
) -> xr.DataArray:
    """Format the raw weights array such that:

    1. Weights match the dtype of the input data
    1. Weights are chunked 1:1 with the source data
    2. Weights are chunked as requested in the target grid. If no chunks are
        provided, the same chunksize as the source grid will be used.
        See: https://github.com/dask/dask/issues/2225
    3. Weights are converted to a sparse representation (on a per chunk basis)
        if the `sparse` package is available.
    """
    weights_dtype = utils.min_weight_dtype(input_dtype)
    new_weights = weights.copy().astype(weights_dtype)

    chunks: dict[Hashable, tuple[int, ...] | int] = {}
    if input_chunks is not None:
        chunks[coord] = input_chunks
        if output_chunks is None:
            # Set output chunking to match input, but precise chunks won't match shape,
            # so take the max in case of uneven chunks
            output_chunks = max(input_chunks)

    if output_chunks is not None:
        chunks[f"target_{coord}"] = output_chunks

    if chunks:
        new_weights = new_weights.chunk(chunks)
        if sparse is not None:
            new_weights.data = new_weights.data.map_blocks(sparse.COO)
    elif sparse is not None:
        new_weights.data = sparse.COO(weights.data)

    return new_weights
