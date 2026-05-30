"""Tests for conservative_2d."""

import numpy as np
import pytest
import xarray as xr

import xarray_regrid  # noqa: F401  (registers the accessor)
from xarray_regrid import ConservativeRegridder, polygons_from_coords
from xarray_regrid.methods.conservative_2d import _build_s2_grid

shapely = pytest.importorskip("shapely")


def _rect_da(ny=60, nx=120, nt=2, seed=0):
    x = np.linspace(-180, 180, nx, endpoint=False) + 180 / nx
    y = np.linspace(-90, 90, ny, endpoint=False) + 90 / ny
    rng = np.random.default_rng(seed)
    return xr.DataArray(
        rng.normal(size=(nt, ny, nx)).astype(np.float64),
        dims=("time", "y", "x"),
        coords={"time": np.arange(nt), "y": y, "x": x},
        name="var",
    )


def _rect_target(ny=24, nx=47):
    x = np.linspace(-180, 180, nx, endpoint=False) + 180 / nx
    y = np.linspace(-90, 90, ny, endpoint=False) + 90 / ny
    return xr.Dataset(coords={"y": y, "x": x})


def test_polygon_matches_factored_planar():
    """On a rectilinear grid with no spherical correction, the polygon path
    should reproduce the axis-factored path to machine precision."""
    da = _rect_da()
    target = _rect_target()
    ref = da.regrid.conservative(target, latitude_coord=None)
    got = da.regrid.conservative_2d(target, x_coord="x", y_coord="y")
    got = got.transpose(*ref.dims)
    np.testing.assert_allclose(got.values, ref.values, atol=1e-12)


def test_polygon_dask_time_chunks():
    da = _rect_da(nt=4).chunk({"time": 2})
    target = _rect_target()
    got = da.regrid.conservative_2d(target, x_coord="x", y_coord="y")
    assert got.chunks is not None
    got = got.compute()
    ref = _rect_da(nt=4).regrid.conservative(target, latitude_coord=None)
    np.testing.assert_allclose(got.transpose(*ref.dims).values, ref.values, atol=1e-12)


def test_polygon_rechunks_spatial():
    """Spatially-chunked input should be accepted (rechunked internally)."""
    da = _rect_da().chunk({"time": 1, "y": 30, "x": 40})
    target = _rect_target()
    out = da.regrid.conservative_2d(target, x_coord="x", y_coord="y")
    out.compute()


def test_polygon_nan_threshold():
    """Stricter nan_threshold produces more NaN output cells when partial
    overlaps exist."""
    da = _rect_da()
    da.values[:, 21:29, :] = np.nan
    target = _rect_target(ny=23, nx=47)
    out1 = da.regrid.conservative_2d(
        target, x_coord="x", y_coord="y", skipna=True, nan_threshold=1.0
    )
    out0 = da.regrid.conservative_2d(
        target, x_coord="x", y_coord="y", skipna=True, nan_threshold=0.0
    )
    assert int(np.isnan(out0.values).sum()) > int(np.isnan(out1.values).sum())


def test_polygon_curvilinear_target():
    """Curvilinear target (2D lat/lon corners) returns finite values."""
    da = _rect_da()
    ny_t, nx_t = 20, 30
    xi, yi = np.meshgrid(
        np.linspace(-120, 120, nx_t),
        np.linspace(-60, 60, ny_t),
        indexing="xy",
    )
    th = np.deg2rad(30)
    x2d = xi * np.cos(th) - yi * np.sin(th)
    y2d = xi * np.sin(th) + yi * np.cos(th)
    target = xr.Dataset(coords={"x": (("ny", "nx"), x2d), "y": (("ny", "nx"), y2d)})
    out = da.regrid.conservative_2d(target, x_coord="x", y_coord="y")
    assert out.shape == (2, 20, 30)
    assert np.isfinite(out.values).mean() > 0.9


def test_antimeridian_rectilinear_constant():
    da = xr.DataArray(
        np.full((2, 4), 2.5),
        dims=("latitude", "longitude"),
        coords={
            "latitude": np.array([-2.5, 2.5]),
            "longitude": np.array([167.5, 172.5, -177.5, -172.5]),
        },
    )
    target = xr.Dataset(
        coords={
            "latitude": np.array([-2.5, 2.5]),
            "longitude": np.array([170.0, -170.0]),
        }
    )

    out = da.regrid.conservative_2d(target, x_coord="longitude", y_coord="latitude")
    np.testing.assert_allclose(out.values, 2.5, atol=1e-12)


def test_cross_convention_longitude_alignment():
    """Source on [0, 360] with target on [-180, 180] (and vice versa) must
    align — a uniform shift can't reconcile the two conventions, so per-value
    wrap of source longitudes is required. Regression: previously yielded
    NaN on half the target cells because banker's rounding on the exact-180°
    mean diff produced a zero offset."""
    src_vals = np.array([1.0, 2.0, 3.0, 4.0])
    tgt_vals_neg = np.array([-135.0, -45.0, 45.0, 135.0])
    src_vals_neg_x = np.array([45.0, 135.0, 225.0, 315.0])
    da = xr.DataArray(
        np.broadcast_to(src_vals, (2, 4)).copy(),
        dims=("latitude", "longitude"),
        coords={"latitude": [-30.0, 30.0], "longitude": src_vals_neg_x},
    )
    target = xr.Dataset(coords={"latitude": [-30.0, 30.0], "longitude": tgt_vals_neg})
    expected = da.regrid.conservative(target).transpose("latitude", "longitude")
    out_planar = da.regrid.conservative_2d(
        target, x_coord="longitude", y_coord="latitude"
    ).transpose("latitude", "longitude")
    out_cea = da.regrid.conservative_2d(
        target, x_coord="longitude", y_coord="latitude", manifold="cea"
    ).transpose("latitude", "longitude")
    np.testing.assert_allclose(out_planar.values, expected.values, atol=1e-12)
    np.testing.assert_allclose(out_cea.values, expected.values, atol=1e-12)

    # Reverse: source on [-180, 180], target on [0, 360].
    da_rev = xr.DataArray(
        np.broadcast_to(src_vals, (2, 4)).copy(),
        dims=("latitude", "longitude"),
        coords={"latitude": [-30.0, 30.0], "longitude": tgt_vals_neg},
    )
    target_rev = xr.Dataset(
        coords={"latitude": [-30.0, 30.0], "longitude": src_vals_neg_x}
    )
    expected_rev = da_rev.regrid.conservative(target_rev).transpose(
        "latitude", "longitude"
    )
    out_rev = da_rev.regrid.conservative_2d(
        target_rev, x_coord="longitude", y_coord="latitude"
    ).transpose("latitude", "longitude")
    np.testing.assert_allclose(out_rev.values, expected_rev.values, atol=1e-12)


def test_polygon_nan_threshold_invalid():
    da = _rect_da()
    with pytest.raises(ValueError):
        da.regrid.conservative_2d(
            _rect_target(), x_coord="x", y_coord="y", nan_threshold=1.5
        )


def test_invalid_manifold_raises():
    da = _rect_da()
    with pytest.raises(ValueError, match="manifold must be one of"):
        da.regrid.conservative_2d(
            _rect_target(), x_coord="x", y_coord="y", manifold="bogus"
        )


def test_polygon_dataset_input():
    """A Dataset input with multiple variables should regrid all of them."""
    da = _rect_da()
    ds = xr.Dataset({"a": da, "b": da * 2.0})
    target = _rect_target()
    out = ds.regrid.conservative_2d(target, x_coord="x", y_coord="y")
    assert set(out.data_vars) == {"a", "b"}
    np.testing.assert_allclose(
        out["b"].transpose(*out["a"].dims).values,
        out["a"].transpose(*out["a"].dims).values * 2.0,
        atol=1e-12,
    )


# --- ConservativeRegridder (reusable) ------------------------------------------


def test_regridder_reusable_matches_oneshot():
    """Reusing a single ConservativeRegridder on multiple fields matches the
    one-shot `conservative_2d_regrid` call."""
    da1 = _rect_da(seed=1)
    da2 = _rect_da(seed=2)
    target = _rect_target()
    regridder = ConservativeRegridder(da1, target, x_coord="x", y_coord="y")
    ref1 = da1.regrid.conservative_2d(target, x_coord="x", y_coord="y")
    ref2 = da2.regrid.conservative_2d(target, x_coord="x", y_coord="y")
    out1 = regridder.regrid(da1)
    out2 = regridder(da2)  # __call__ alias
    np.testing.assert_allclose(out1.values, ref1.values, atol=1e-12)
    np.testing.assert_allclose(out2.values, ref2.values, atol=1e-12)


def test_regridder_weight_cache():
    """Forward weight matrix is built lazily and then reused across calls."""
    da = _rect_da()
    target = _rect_target()
    regridder = ConservativeRegridder(da, target, x_coord="x", y_coord="y")
    # _forward is a cached_property; not yet materialized before first regrid.
    assert "_forward" not in regridder.__dict__
    regridder.regrid(da)
    assert "_forward" in regridder.__dict__
    w1 = regridder.forward_weights
    regridder.regrid(da)
    assert regridder.forward_weights is w1  # same object, not rebuilt


def test_regridder_transpose_roundtrip_rectilinear_aligned():
    """If target cells are aligned unions of source cells, forward followed by
    backward reproduces a constant field exactly."""
    # 120 source cells along each axis, target is a 4x coarsening (exact union
    # of source cells). A constant source field survives the roundtrip to
    # itself because every source cell is fully covered.
    ns = 120
    nt = 30  # exactly ns / 4
    x_s = np.linspace(-180, 180, ns, endpoint=False) + 180 / ns
    y_s = np.linspace(-90, 90, ns, endpoint=False) + 90 / ns
    x_t = np.linspace(-180, 180, nt, endpoint=False) + 180 / nt
    y_t = np.linspace(-90, 90, nt, endpoint=False) + 90 / nt
    da = xr.DataArray(
        np.full((ns, ns), 3.5, dtype=np.float64),
        dims=("y", "x"),
        coords={"y": y_s, "x": x_s},
    )
    target = xr.Dataset(coords={"y": y_t, "x": x_t})
    regridder = ConservativeRegridder(da, target, x_coord="x", y_coord="y")
    coarse = regridder.regrid(da)
    back = regridder.T.regrid(coarse)
    # Every value should match the original constant.
    np.testing.assert_allclose(back.values, 3.5, atol=1e-12)


def test_regridder_T_preserves_weights():  # noqa: N802
    """regridder.T.T should share the raw area matrix with the original."""
    da = _rect_da()
    target = _rect_target()
    r = ConservativeRegridder(da, target, x_coord="x", y_coord="y")
    rr = r.T.T
    # Same shape, same coords, same data.
    assert r.areas.shape == rr.areas.shape
    if hasattr(r.areas, "data"):
        np.testing.assert_array_equal(r.areas.data, rr.areas.data)


def test_regridder_shape_mismatch_raises():
    """Applying the regridder to data whose spatial shape differs from the
    source it was built for should raise."""
    da = _rect_da()
    target = _rect_target()
    regridder = ConservativeRegridder(da, target, x_coord="x", y_coord="y")
    smaller = _rect_da(ny=30, nx=60)
    with pytest.raises(ValueError, match="spatial shape"):
        regridder.regrid(smaller)


def test_cea_manifold_matches_factored():
    """Polygon path with ``manifold="cea"`` should match the axis-factored
    sin-weighted conservative path to within a tight tolerance on lat/lon
    grids (both are analytically equivalent for cylindrical equal-area)."""
    lon_s = np.linspace(-180, 180, 180, endpoint=False) + 1.0
    lat_s = np.linspace(-90, 90, 90, endpoint=False) + 1.0
    lon_t = np.linspace(-180, 180, 60, endpoint=False) + 3.0
    lat_t = np.linspace(-90, 90, 30, endpoint=False) + 3.0
    vals = np.cos(np.deg2rad(lat_s))[:, None] ** 2 * np.sin(np.deg2rad(lon_s))[None, :]
    da = xr.DataArray(
        vals,
        dims=("latitude", "longitude"),
        coords={"latitude": lat_s, "longitude": lon_s},
    )
    target = xr.Dataset(coords={"latitude": lat_t, "longitude": lon_t})

    factored = da.regrid.conservative(target, latitude_coord="latitude")
    polygon = da.regrid.conservative_2d(
        target, x_coord="longitude", y_coord="latitude", manifold="cea"
    )
    # Both methods should agree to the grid's own quadrature accuracy. Near the
    # poles the factored path's median-dlat approximation introduces a small
    # discrepancy; 1e-3 absolute is well below the planar raw error floor
    # (~1e-2 at this resolution, tested separately).
    np.testing.assert_allclose(
        polygon.transpose(*factored.dims).values,
        factored.values,
        atol=1e-3,
    )


def test_cea_conserves_integral():
    """Mass conservation check on the sphere. For cos^2(lat), true integral is
    8*pi/3; the regridder on a 2-to-6-degree grid should keep the
    spherical-area-weighted sum within the grid quadrature floor when
    manifold="cea", and miss it by ~17x more with planar."""
    lon_s = np.linspace(-180, 180, 180, endpoint=False) + 1.0
    lat_s = np.linspace(-90, 90, 90, endpoint=False) + 1.0
    lon_t = np.linspace(-180, 180, 60, endpoint=False) + 3.0
    lat_t = np.linspace(-90, 90, 30, endpoint=False) + 3.0
    da = xr.DataArray(
        np.cos(np.deg2rad(lat_s))[:, None] ** 2 * np.ones(lon_s.size)[None, :],
        dims=("latitude", "longitude"),
        coords={"latitude": lat_s, "longitude": lon_s},
    )
    target = xr.Dataset(coords={"latitude": lat_t, "longitude": lon_t})

    out_cea = da.regrid.conservative_2d(
        target, x_coord="longitude", y_coord="latitude", manifold="cea"
    )
    out_planar = da.regrid.conservative_2d(
        target, x_coord="longitude", y_coord="latitude", manifold="planar"
    )

    # True target spherical cell areas
    dlon_arr = np.full(lon_t.size, np.deg2rad(np.mean(np.diff(lon_t))))
    lat_r = np.deg2rad(lat_t)
    dlat_r = np.gradient(lat_r)
    dlat_bands = np.sin(lat_r + dlat_r / 2) - np.sin(lat_r - dlat_r / 2)
    a_tgt = dlat_bands[:, None] * dlon_arr[None, :]

    true_val = 8 * np.pi / 3
    cea_vals = out_cea.transpose("latitude", "longitude").values
    planar_vals = out_planar.transpose("latitude", "longitude").values
    err_cea = abs(float((cea_vals * a_tgt).sum()) - true_val)
    err_planar = abs(float((planar_vals * a_tgt).sum()) - true_val)
    # CEA should be at least 10x more accurate than raw planar here.
    assert err_cea < 0.1 * err_planar, (
        f"err_cea={err_cea:.2e} err_planar={err_planar:.2e}"
    )


# --- from_polygons (unstructured mesh) ----------------------------------------


def _box_polygons():
    rng = np.random.default_rng(1)
    n = 50
    cx = rng.uniform(-170, 170, n)
    cy = rng.uniform(-80, 80, n)
    return shapely.box(cx - 5, cy - 5, cx + 5, cy + 5)


def test_polygons_from_coords_periodic():
    polys = polygons_from_coords(
        np.array([167.5, 172.5, -177.5, -172.5]),
        np.array([-2.5, 2.5]),
        periodic=True,
    )
    bounds = shapely.bounds(polys)
    widths = bounds[:, 2] - bounds[:, 0]
    assert np.all(widths < 10.1)


def test_from_polygons_basic():
    src_polys = _box_polygons()
    tgt_polys = polygons_from_coords(
        np.linspace(-180, 180, 30, endpoint=False) + 6,
        np.linspace(-90, 90, 15, endpoint=False) + 6,
    )
    rgr = ConservativeRegridder.from_polygons(
        src_polys, tgt_polys, source_dim="face", target_dim="cell"
    )
    da = xr.DataArray(
        np.arange(src_polys.size, dtype=np.float64),
        dims=("face",),
    )
    out = rgr.regrid(da)
    assert out.dims == ("cell",)
    assert out.sizes["cell"] == tgt_polys.size


def test_from_polygons_attaches_target_aux_coords():
    src_polys = _box_polygons()
    tgt_polys = polygons_from_coords(
        np.linspace(-180, 180, 12, endpoint=False) + 15,
        np.linspace(-90, 90, 6, endpoint=False) + 15,
    )
    region_id = np.arange(tgt_polys.size) + 100
    target_coords = xr.Dataset(
        coords={
            "cell": np.arange(tgt_polys.size),
            "region_id": ("cell", region_id),
        }
    )
    rgr = ConservativeRegridder.from_polygons(
        src_polys,
        tgt_polys,
        source_dim="face",
        target_dim="cell",
        target_coords=target_coords,
    )
    da = xr.DataArray(np.arange(src_polys.size, dtype=np.float64), dims=("face",))
    out = rgr.regrid(da)
    assert "region_id" in out.coords
    np.testing.assert_array_equal(out["region_id"].values, region_id)


def test_from_polygons_periodic_antimeridian():
    src_polys = np.array(
        [shapely.Polygon([(175, -5), (-175, -5), (-175, 5), (175, 5)])],
        dtype=object,
    )
    tgt_polys = np.array(
        [
            shapely.box(160, -5, 170, 5),
            shapely.box(175, -5, 185, 5),
            shapely.box(-170, -5, -160, 5),
        ],
        dtype=object,
    )
    rgr = ConservativeRegridder.from_polygons(
        src_polys,
        tgt_polys,
        source_dim="src",
        target_dim="tgt",
        periodic=True,
    )
    out = rgr.regrid(xr.DataArray([7.0], dims=("src",)))

    assert np.isnan(out.values[0])
    assert out.values[1] == pytest.approx(7.0)
    assert np.isnan(out.values[2])


def test_from_polygons_mass_conservation():
    """Sum of intersected mass should match the direct A·s calculation to
    machine precision for any source field."""

    src_polys = _box_polygons()
    tgt_polys = polygons_from_coords(
        np.linspace(-180, 180, 36, endpoint=False) + 5,
        np.linspace(-90, 90, 18, endpoint=False) + 5,
    )
    rgr = ConservativeRegridder.from_polygons(
        src_polys, tgt_polys, source_dim="face", target_dim="cell"
    )
    rng = np.random.default_rng(3)
    s = rng.normal(size=src_polys.size)
    da = xr.DataArray(s, dims=("face",))
    out = rgr.regrid(da).values
    # Direct mass = sum_i s_i * source_coverage_i. Matches output if we
    # multiply output by target-covered area.
    tgt_covered = rgr.target_areas
    valid = tgt_covered > 0
    direct = float((s * rgr.source_coverage_areas).sum())
    via_regrid = float((out[valid] * tgt_covered[valid]).sum())
    rel = abs(direct - via_regrid) / max(abs(direct), 1e-12)
    assert rel < 1e-12, f"rel err {rel:.2e}"


def test_from_polygons_transpose_roundtrip():
    """Roundtrip mesh ↔ mesh of a constant field returns the constant."""
    src_polys = polygons_from_coords(
        np.linspace(-180, 180, 24, endpoint=False) + 7.5,
        np.linspace(-90, 90, 12, endpoint=False) + 7.5,
    )
    tgt_polys = polygons_from_coords(
        np.linspace(-180, 180, 12, endpoint=False) + 15,
        np.linspace(-90, 90, 6, endpoint=False) + 15,
    )
    rgr = ConservativeRegridder.from_polygons(
        src_polys, tgt_polys, source_dim="src", target_dim="tgt"
    )
    da = xr.DataArray(np.full(src_polys.size, 3.5), dims=("src",))
    out = rgr.regrid(da)
    back = rgr.T.regrid(out)
    # Inner source cells (fully covered by target cells they map to) must be 3.5.
    # Edge cells may be NaN if target domain doesn't cover them.
    finite = np.isfinite(back.values)
    np.testing.assert_allclose(back.values[finite], 3.5, atol=1e-12)


def test_from_polygons_nan_propagation():
    """NaN source cells propagate through skipna=True correctly."""
    src_polys = polygons_from_coords(
        np.linspace(-180, 180, 24, endpoint=False) + 7.5,
        np.linspace(-90, 90, 12, endpoint=False) + 7.5,
    )
    tgt_polys = polygons_from_coords(
        np.linspace(-180, 180, 12, endpoint=False) + 15,
        np.linspace(-90, 90, 6, endpoint=False) + 15,
    )
    rgr = ConservativeRegridder.from_polygons(
        src_polys, tgt_polys, source_dim="src", target_dim="tgt"
    )
    vals = np.full(src_polys.size, 1.0)
    vals[:5] = np.nan
    da = xr.DataArray(vals, dims=("src",))
    out_keep = rgr.regrid(da, nan_threshold=1.0)
    out_strict = rgr.regrid(da, nan_threshold=0.0)
    # Strict should have at least as many NaNs.
    strict_nans = int(np.isnan(out_strict.values).sum())
    keep_nans = int(np.isnan(out_keep.values).sum())
    assert strict_nans >= keep_nans


def test_from_polygons_target_outside_source_is_nan():
    """Target cells entirely outside the source domain must be NaN regardless
    of skipna (regression test for a bug where the "skip mask matmul when no
    NaNs" optimization returned zeros for uncovered cells)."""

    src = np.array([shapely.box(0, 0, 1, 1)], dtype=object)
    tgt = polygons_from_coords(
        np.linspace(100, 110, 4, endpoint=False) + 1.25,
        np.linspace(100, 110, 4, endpoint=False) + 1.25,
    )
    rgr = ConservativeRegridder.from_polygons(src, tgt, source_dim="src")
    # Source has no NaNs → used to wrongly return 0 here.
    da = xr.DataArray(np.array([5.0]), dims=("src",))
    out = rgr.regrid(da, skipna=True)
    assert np.isnan(out.values).all()
    out_no = rgr.regrid(da, skipna=False)
    assert np.isnan(out_no.values).all()


def test_from_polygons_hole_is_nan():
    """A target cell fully inside a source-polygon hole should be NaN, not 0."""

    ring_with_hole = shapely.Polygon(
        [(0, 0), (10, 0), (10, 10), (0, 10)],
        [[(3, 3), (7, 3), (7, 7), (3, 7)]],
    )
    src = np.array([ring_with_hole], dtype=object)
    tgt = polygons_from_coords(
        np.linspace(0, 10, 5, endpoint=False) + 1,
        np.linspace(0, 10, 5, endpoint=False) + 1,
    )
    rgr = ConservativeRegridder.from_polygons(src, tgt, source_dim="src")
    out = rgr.regrid(xr.DataArray([7.0], dims=("src",)))
    # The cell centered at (5,5), edges [4,6]x[4,6], lies entirely in the hole.
    assert int(np.isnan(out.values).sum()) >= 1


def test_from_polygons_input_validation():
    # 2D polygons array rejected
    p = shapely.box(0, 0, 1, 1)
    arr_2d = np.array([[p, p], [p, p]], dtype=object)
    with pytest.raises(ValueError, match="1D"):
        ConservativeRegridder.from_polygons(arr_2d, np.array([p]))


# --- netCDF save / load -------------------------------------------------------


def test_regrid_preserves_input_dtype():
    """Float32 in → float32 out; float64 in → float64 out. Sparse promotes
    to float64 internally, so we rely on an explicit cast at the end of
    ``_apply_core``."""
    da = _rect_da()
    target = _rect_target()
    rgr = ConservativeRegridder(da, target, x_coord="x", y_coord="y")
    assert rgr.regrid(da.astype(np.float32)).dtype == np.float32
    assert rgr.regrid(da.astype(np.float64)).dtype == np.float64
    # Integer inputs promote (float32 can't hold int32 without precision loss).
    assert np.issubdtype(rgr.regrid((da * 10).astype(np.int32)).dtype, np.floating)


def test_to_netcdf_roundtrip_structured(tmp_path):
    """Save, reload, regrid → identical output to the original regridder."""
    da = _rect_da()
    target = _rect_target()
    rgr = ConservativeRegridder(da, target, x_coord="x", y_coord="y")
    out_before = rgr.regrid(da).values

    path = tmp_path / "regridder.nc"
    rgr.to_netcdf(path)
    rgr2 = ConservativeRegridder.from_netcdf(path)

    np.testing.assert_array_equal(rgr2.regrid(da).values, out_before)
    assert rgr2.spec == rgr.spec


def test_to_netcdf_preserves_manifold(tmp_path):
    lat_s = np.linspace(-90, 90, 30, endpoint=False) + 3
    lon_s = np.linspace(-180, 180, 60, endpoint=False) + 3
    lat_t = np.linspace(-90, 90, 15, endpoint=False) + 6
    lon_t = np.linspace(-180, 180, 30, endpoint=False) + 6
    da = xr.DataArray(
        np.cos(np.deg2rad(lat_s))[:, None] ** 2 * np.ones(lon_s.size)[None, :],
        dims=("latitude", "longitude"),
        coords={"latitude": lat_s, "longitude": lon_s},
    )
    target = xr.Dataset(coords={"latitude": lat_t, "longitude": lon_t})

    rgr = ConservativeRegridder(
        da, target, x_coord="longitude", y_coord="latitude", manifold="cea"
    )
    before = rgr.regrid(da).values
    path = tmp_path / "r.nc"
    rgr.to_netcdf(path)
    rgr2 = ConservativeRegridder.from_netcdf(path)
    assert rgr2.manifold == "cea"
    np.testing.assert_allclose(rgr2.regrid(da).values, before, atol=1e-15)


def test_to_netcdf_transpose_works_after_reload(tmp_path):
    """A reloaded regridder can still take .T for backward regridding."""
    da = _rect_da()
    target = _rect_target()
    rgr = ConservativeRegridder(da, target, x_coord="x", y_coord="y")
    path = tmp_path / "r.nc"
    rgr.to_netcdf(path)
    rgr2 = ConservativeRegridder.from_netcdf(path)

    fwd = rgr.regrid(da)
    back_original = rgr.T.regrid(fwd).values
    back_reloaded = rgr2.T.regrid(fwd).values
    np.testing.assert_array_equal(back_original, back_reloaded)


def test_to_netcdf_unstructured_roundtrip(tmp_path):
    """from_polygons regridder roundtrips too (single spatial dim each side)."""
    rng = np.random.default_rng(1)
    cx = rng.uniform(-170, 170, 30)
    cy = rng.uniform(-80, 80, 30)
    src_polys = shapely.box(cx - 5, cy - 5, cx + 5, cy + 5)
    tgt_polys = polygons_from_coords(
        np.linspace(-180, 180, 24, endpoint=False) + 7.5,
        np.linspace(-90, 90, 12, endpoint=False) + 7.5,
    )
    rgr = ConservativeRegridder.from_polygons(
        src_polys, tgt_polys, source_dim="face", target_dim="cell"
    )
    da = xr.DataArray(rng.normal(size=30), dims=("face",))
    out_before = rgr.regrid(da).values

    path = tmp_path / "r.nc"
    rgr.to_netcdf(path)
    rgr2 = ConservativeRegridder.from_netcdf(path)
    np.testing.assert_array_equal(rgr2.regrid(da).values, out_before)


def test_to_netcdf_metadata_fields(tmp_path):
    """Metadata captures grid ranges, version, created timestamp."""
    da = _rect_da()
    target = _rect_target()
    rgr = ConservativeRegridder(da, target, x_coord="x", y_coord="y")
    path = tmp_path / "r.nc"
    rgr.to_netcdf(path)

    with xr.open_dataset(path) as ds:
        attrs = dict(ds.attrs)

    assert attrs["x_coord"] == "x"
    assert attrs["y_coord"] == "y"
    assert str(attrs["manifold"]) == "planar"
    assert tuple(int(size) for size in attrs["src_shape"]) == rgr.spec.src_shape
    assert tuple(int(size) for size in attrs["dst_shape"]) == rgr.spec.dst_shape
    # Grid ranges captured when the coord is present in source/target.
    assert "source_x_range" in attrs
    assert "target_x_range" in attrs
    assert attrs["source_x_range"][0] <= attrs["source_x_range"][1]
    assert attrs["created"]
    assert int(attrs["schema_version"]) == 1


def test_from_netcdf_rejects_unknown_schema(tmp_path):
    """Loading a file written with a future schema version raises cleanly."""
    h5py = pytest.importorskip("h5py")
    da = _rect_da()
    target = _rect_target()
    rgr = ConservativeRegridder(da, target, x_coord="x", y_coord="y")
    path = tmp_path / "r.nc"
    rgr.to_netcdf(path)

    # Bump the on-disk schema_version so the loader should reject it.
    with h5py.File(path, "a") as f:
        f.attrs["schema_version"] = 999

    with pytest.raises(ValueError, match="schema version"):
        ConservativeRegridder.from_netcdf(path)


def test_regridder_transpose_curvilinear():
    """Transpose works when the target is a curvilinear grid with different
    dim names from the source."""
    da = _rect_da(ny=40, nx=80)
    ny_t, nx_t = 20, 30
    xi, yi = np.meshgrid(
        np.linspace(-120, 120, nx_t),
        np.linspace(-60, 60, ny_t),
        indexing="xy",
    )
    th = np.deg2rad(15)
    x2 = xi * np.cos(th) - yi * np.sin(th)
    y2 = xi * np.sin(th) + yi * np.cos(th)
    target = xr.Dataset(coords={"x": (("ny", "nx"), x2), "y": (("ny", "nx"), y2)})
    regridder = ConservativeRegridder(da, target, x_coord="x", y_coord="y")
    fwd = regridder.regrid(da)
    assert fwd.dims == ("time", "ny", "nx")
    # Going backward we should land back on (time, y, x)
    back = regridder.T.regrid(fwd)
    assert "y" in back.dims and "x" in back.dims
    assert back.sizes["y"] == da.sizes["y"]
    assert back.sizes["x"] == da.sizes["x"]


# --- s2 manifold (optional, requires spherely) --------------------------------

# NB: a MODULE-level importorskip would skip this whole file (incl. the 35
# planar/cea/polygon tests above) when spherely is absent — which is the
# default CI env, since spherely lives only in the `spherical` extra. Gate just
# the s2 tests with a per-test marker instead.
try:
    import spherely

    _HAS_SPHERELY = True
except ImportError:  # pragma: no cover
    spherely = None
    _HAS_SPHERELY = False

needs_spherely = pytest.mark.skipif(
    not _HAS_SPHERELY, reason="s2 manifold tests require spherely"
)


def _latlon_da(ny=24, nx=36, lat_max=87.5, lon_max=175, fill=None, seed=0):
    lat = np.linspace(-lat_max, lat_max, ny)
    lon = np.linspace(-lon_max, lon_max, nx)
    if fill is None:
        vals = np.random.default_rng(seed).normal(size=(ny, nx))
    else:
        vals = np.full((ny, nx), fill)
    return xr.DataArray(
        vals,
        dims=("latitude", "longitude"),
        coords={"latitude": lat, "longitude": lon},
    )


def _latlon_target(ny=9, nx=18, lat_max=80, lon_max=170):
    return xr.Dataset(
        coords={
            "latitude": np.linspace(-lat_max, lat_max, ny),
            "longitude": np.linspace(-lon_max, lon_max, nx),
        }
    )


def _s2_regridder(da, target):
    return ConservativeRegridder(
        da, target, x_coord="longitude", y_coord="latitude", manifold="s2"
    )


@needs_spherely
def test_s2_conserves_known_integral():
    """Gold-standard conservation: a field with a KNOWN spherical integral
    (cos^2(lat) * (1.5 + sin(lon)) integrates to 4*pi over the sphere) regridded
    with manifold="s2" onto a co-extensive global grid must conserve mass to
    machine precision when measured with INDEPENDENT great-circle cell areas
    (spherely.area) — not the regridder's own matrix.

    This is the non-tautological form. A self-area check (row/col sums of
    rgr.areas) is an algebraic identity that "conserves" *any* matrix, and a
    constant field conserves under row-normalization even with dropped/wrong
    overlaps. A spatially-varying field + independent areas + matched domains is
    what actually exercises the geometry. (The all-pairs coverage tests pin the
    candidate search locally; this pins the whole pipeline globally, against a
    known analytic value.)
    """

    def glat(n):
        return np.linspace(-90, 90, n, endpoint=False) + 90 / n  # edges hit ±90

    def glon(n):
        return np.linspace(-180, 180, n, endpoint=False) + 180 / n

    def s2_cell_areas(lon, lat):  # INDEPENDENT areas, not from the regridder
        grid = _build_s2_grid(np.asarray(lon), np.asarray(lat))
        return np.asarray(spherely.area(grid.s2_polys, radius=1.0)).reshape(
            lat.size, lon.size
        )

    lat_s, lon_s = glat(120), glon(240)
    lat_t, lon_t = glat(40), glon(80)
    grid_lat, grid_lon = np.meshgrid(lat_s, lon_s, indexing="ij")
    field = np.cos(np.deg2rad(grid_lat)) ** 2 * (1.5 + np.sin(np.deg2rad(grid_lon)))
    da = xr.DataArray(
        field,
        dims=("latitude", "longitude"),
        coords={"latitude": lat_s, "longitude": lon_s},
    )
    target = xr.Dataset(coords={"latitude": lat_t, "longitude": lon_t})

    out = (
        ConservativeRegridder(
            da, target, x_coord="longitude", y_coord="latitude", manifold="s2"
        )
        .regrid(da)
        .values
    )
    assert np.isfinite(out).all()  # co-extensive global grids -> full coverage

    i_src = float((da.values * s2_cell_areas(lon_s, lat_s)).sum())
    i_tgt = float((out * s2_cell_areas(lon_t, lat_t)).sum())
    np.testing.assert_allclose(i_tgt, i_src, rtol=1e-9)  # conserved (machine prec.)
    np.testing.assert_allclose(i_src, 4 * np.pi, rtol=2e-3)  # ~ the known value


@needs_spherely
def test_s2_preserves_constant_field():
    """Roundtrip of a constant field through the s2 path reproduces the
    constant to machine precision."""
    da = _latlon_da(fill=7.3)
    rgr = _s2_regridder(da, _latlon_target())
    out = rgr.regrid(da).values
    finite = np.isfinite(out)
    np.testing.assert_allclose(out[finite], 7.3, atol=1e-12)


@needs_spherely
def test_s2_netcdf_roundtrip(tmp_path):
    """A saved s2 regridder reloads back to an s2 regridder and produces
    identical output."""
    da = _latlon_da(seed=1)
    rgr = _s2_regridder(da, _latlon_target())
    before = rgr.regrid(da).values
    path = tmp_path / "s2.nc"
    rgr.to_netcdf(path)
    rgr2 = ConservativeRegridder.from_netcdf(path)
    assert rgr2.manifold == "s2"
    np.testing.assert_array_equal(rgr2.regrid(da).values, before)


@needs_spherely
def test_s2_handles_pole_touching_cells():
    """A global grid whose outer rows reach ±90° exercises the `oriented=True`
    branch of `spherely.create_polygon` — without an explicit orientation, s2
    would silently pick the complementary (hemisphere-sized) interpretation
    of a near-pole cell."""
    lat = np.linspace(-89.5, 89.5, 24)  # inferred outer edges clip to ±90°
    lon = np.linspace(-175, 175, 36)
    vals = (np.cos(np.deg2rad(lat)) ** 2)[:, None] * np.ones(lon.size)[None, :]
    da = xr.DataArray(
        vals,
        dims=("latitude", "longitude"),
        coords={"latitude": lat, "longitude": lon},
    )
    rgr = _s2_regridder(da, _latlon_target(ny=12, nx=24, lat_max=85))
    out = rgr.regrid(da).values
    # Output values should stay in [0, 1] (the source range), not hemispheres.
    finite = np.isfinite(out)
    assert np.all(out[finite] >= -1e-12)
    assert np.all(out[finite] <= 1.0 + 1e-12)


@needs_spherely
def test_s2_cell_areas_are_not_complements():
    """Each s2 cell polygon must be the cell itself, not its hemisphere-sized
    complement. A descending-coordinate winding bug balloons every cell to
    ~4*pi sr; here a near-global grid's cell areas must each be < 2*pi and tile
    the sphere (~4*pi total) for BOTH ascending and descending latitude."""
    lon = np.linspace(-179, 179, 120)
    for lat in (np.linspace(-89, 89, 90), np.linspace(89, -89, 90)):
        grid = _build_s2_grid(lon, lat)
        cell_areas = np.asarray(spherely.area(grid.s2_polys, radius=1.0))
        assert np.all(cell_areas < 2 * np.pi), "a cell came out as its complement"
        np.testing.assert_allclose(cell_areas.sum(), 4 * np.pi, rtol=0.02)


@needs_spherely
def test_s2_descending_latitude_matches_ascending():
    """s2 regridding must be invariant to coordinate direction: a non-constant
    field on a descending-latitude grid (the CF 90 -> -90 convention) must
    regrid to the same result as the ascending grid. A clockwise/complement
    winding bug diverges wildly here while leaving the constant-field and
    mass-identity tests green."""
    asc = _latlon_da(ny=30, nx=40, lat_max=80, lon_max=170, seed=3)
    target = _latlon_target(ny=12, nx=20, lat_max=65)
    out_asc = _s2_regridder(asc, target).regrid(asc)
    desc = asc.isel(latitude=slice(None, None, -1))  # same field, descending lat
    out_desc = _s2_regridder(desc, target).regrid(desc)
    np.testing.assert_allclose(
        out_asc.transpose(*out_desc.dims).values,
        out_desc.values,
        rtol=1e-9,
        atol=1e-9,
        equal_nan=True,
    )


def _s2_all_pairs_target_areas(da, target):
    """Brute-force per-target-cell coverage: intersect EVERY (dst, src) s2 cell
    pair with spherely (no STRtree). Independent ground truth for the candidate
    search — if the STRtree drops a real overlap, the regridder's target_areas
    fall short of this reference."""
    sgrid = _build_s2_grid(
        np.asarray(da.longitude.values), np.asarray(da.latitude.values)
    )
    dgrid = _build_s2_grid(
        np.asarray(target.longitude.values), np.asarray(target.latitude.values)
    )
    ns, nd = len(sgrid.s2_polys), len(dgrid.s2_polys)
    dst_i, src_i = np.meshgrid(np.arange(nd), np.arange(ns), indexing="ij")
    inter = spherely.intersection(
        dgrid.s2_polys[dst_i.ravel()], sgrid.s2_polys[src_i.ravel()]
    )
    return np.asarray(spherely.area(inter, radius=1.0)).reshape(nd, ns).sum(axis=1)


@needs_spherely
def test_s2_high_latitude_coverage_is_complete():
    """Great-circle cell edges bulge poleward past their planar bbox, so a naive
    bbox candidate filter drops real high-latitude overlaps. The bulge-faithful
    shadow must recover them: target coverage matches the all-pairs reference."""
    rng = np.random.default_rng(0)
    da = xr.DataArray(
        rng.normal(size=(14, 18)),
        dims=("latitude", "longitude"),
        coords={
            "latitude": np.linspace(60, 86, 14),  # coarse + high latitude
            "longitude": np.linspace(-170, 170, 18),
        },
    )
    target = xr.Dataset(
        coords={
            "latitude": np.linspace(62, 84, 8),
            "longitude": np.linspace(-160, 160, 12),
        }
    )
    rgr = ConservativeRegridder(
        da, target, x_coord="longitude", y_coord="latitude", manifold="s2"
    )
    got = np.ravel(rgr.areas.sum(axis=1).todense())
    np.testing.assert_allclose(
        got, _s2_all_pairs_target_areas(da, target), rtol=1e-9, atol=1e-12
    )


@needs_spherely
def test_s2_antimeridian_coverage_is_complete():
    """Cells adjacent across the antimeridian genuinely overlap on the sphere
    but have far-apart planar bboxes. The ±360° candidate query must pair them:
    target coverage matches the all-pairs reference for a global grid."""
    rng = np.random.default_rng(1)
    da = xr.DataArray(
        rng.normal(size=(30, 60)),
        dims=("latitude", "longitude"),
        coords={
            "latitude": np.linspace(-87, 87, 30),
            "longitude": np.linspace(-177, 177, 60),
        },
    )
    target = xr.Dataset(
        coords={
            "latitude": np.linspace(-80, 80, 16),
            "longitude": np.linspace(-174, 180, 30),  # last cell straddles ±180
        }
    )
    rgr = ConservativeRegridder(
        da, target, x_coord="longitude", y_coord="latitude", manifold="s2"
    )
    got = np.ravel(rgr.areas.sum(axis=1).todense())
    np.testing.assert_allclose(
        got, _s2_all_pairs_target_areas(da, target), rtol=1e-9, atol=1e-12
    )
