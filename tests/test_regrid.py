from pathlib import Path

import numpy as np
import pytest
import xarray as xr
from numpy.testing import assert_array_equal

try:
    import xesmf
except ImportError:
    xesmf = None

import xarray_regrid

DATA_PATH = Path(__file__).parent.parent / "docs" / "notebooks" / "benchmarks" / "data"

CDO_FILES = {
    "linear": DATA_PATH / "cdo_bilinear_64b.nc",
    "nearest": DATA_PATH / "cdo_nearest_64b.nc",
    "conservative": DATA_PATH / "cdo_conservative_64b.nc",
}

CHUNK_SCHEMES = [{}, {"time": 1}, {"longitude": 100, "latitude": 100}]


@pytest.fixture(scope="session")
def sample_input_data() -> xr.Dataset:
    ds = xr.open_dataset(DATA_PATH / "era5_2m_dewpoint_temperature_2000_monthly.nc")
    # Convert to dask so regridding is lazy for attr-only tests
    return ds.compute().chunk()


@pytest.fixture(scope="session")
def conservative_input_data() -> xr.Dataset:
    ds = xr.open_dataset(DATA_PATH / "era5_total_precipitation_2020_monthly.nc")
    return ds.compute().chunk()


@pytest.fixture(scope="session")
def cdo_comparison_data() -> dict[str, xr.Dataset]:
    data = {}
    for method, path in CDO_FILES.items():
        data[method] = xr.open_dataset(path).compute()
    return data


@pytest.fixture(scope="session")
def sample_grid_ds():
    grid = xarray_regrid.Grid(
        north=90,
        east=180,
        south=45,
        west=90,
        resolution_lat=0.17,
        resolution_lon=0.17,
    )

    return xarray_regrid.create_regridding_dataset(grid)


@pytest.fixture(scope="session")
def conservative_sample_grid():
    grid = xarray_regrid.Grid(
        north=90,
        east=360,
        south=-90,
        west=0,
        resolution_lat=2.2,
        resolution_lon=2.2,
    )

    return xarray_regrid.create_regridding_dataset(grid)


@pytest.mark.parametrize("method", ["linear", "nearest"])
@pytest.mark.parametrize("chunks", CHUNK_SCHEMES)
def test_basic_regridders_ds(
    sample_input_data, sample_grid_ds, cdo_comparison_data, method, chunks
):
    """Test the dataset regridders (except conservative)."""
    regridder = getattr(sample_input_data.chunk(chunks).regrid, method)
    ds_regrid = regridder(sample_grid_ds)
    ds_cdo = cdo_comparison_data[method]
    xr.testing.assert_allclose(ds_regrid, ds_cdo, rtol=0.002, atol=2e-5)


@pytest.mark.parametrize("method", ["linear", "nearest"])
@pytest.mark.parametrize("chunks", CHUNK_SCHEMES)
def test_basic_regridders_da(
    sample_input_data, sample_grid_ds, cdo_comparison_data, method, chunks
):
    """Test the dataarray regridders (except conservative)."""
    regridder = getattr(sample_input_data["d2m"].chunk(chunks).regrid, method)
    da_regrid = regridder(sample_grid_ds)
    da_cdo = cdo_comparison_data[method]["d2m"]
    xr.testing.assert_allclose(da_regrid, da_cdo, rtol=0.002, atol=2e-5)


@pytest.mark.parametrize("chunks", CHUNK_SCHEMES)
def test_conservative_regridder(
    conservative_input_data,
    conservative_sample_grid,
    cdo_comparison_data,
    chunks,
):
    input_data = conservative_input_data.chunk(chunks)
    ds_regrid = input_data.regrid.conservative(
        conservative_sample_grid, latitude_coord="latitude"
    )
    ds_cdo = cdo_comparison_data["conservative"]

    xr.testing.assert_allclose(
        ds_regrid["tp"],
        ds_cdo["tp"],
        rtol=0.002,
        atol=4e-5,
    )


def test_conservative_nans(
    conservative_input_data, conservative_sample_grid, cdo_comparison_data
):
    ds = conservative_input_data
    ds["tp"] = ds["tp"].where(ds.latitude >= 0).where(ds.longitude < 180)
    ds_regrid = ds.regrid.conservative(
        conservative_sample_grid, latitude_coord="latitude"
    )
    ds_cdo = cdo_comparison_data["conservative"]

    no_nans = {"latitude": slice(1, 90), "longitude": slice(1, 179)}
    xr.testing.assert_allclose(
        ds_regrid["tp"].sel(no_nans),
        ds_cdo["tp"].sel(no_nans),
        rtol=0.002,
        atol=4e-5,
    )


@pytest.mark.parametrize("method", ["linear", "nearest", "cubic"])
def test_attrs_dataarray(sample_input_data, sample_grid_ds, method):
    regridder = getattr(sample_input_data["d2m"].regrid, method)
    da_regrid = regridder(sample_grid_ds)
    assert da_regrid.attrs == sample_input_data["d2m"].attrs
    assert da_regrid["longitude"].attrs == sample_input_data["longitude"].attrs


def test_attrs_dataarray_conservative(sample_input_data, sample_grid_ds):
    da_regrid = sample_input_data["d2m"].regrid.conservative(
        sample_grid_ds, latitude_coord="latitude"
    )
    assert da_regrid.attrs == sample_input_data["d2m"].attrs
    assert da_regrid["longitude"].attrs == sample_input_data["longitude"].attrs


@pytest.mark.parametrize("method", ["linear", "nearest", "cubic"])
def test_attrs_dataset(sample_input_data, sample_grid_ds, method):
    regridder = getattr(sample_input_data.regrid, method)
    ds_regrid = regridder(sample_grid_ds)
    assert ds_regrid.attrs == sample_input_data.attrs
    assert ds_regrid["longitude"].attrs == sample_input_data["longitude"].attrs


def test_attrs_dataset_conservative(sample_input_data, sample_grid_ds):
    ds_regrid = sample_input_data.regrid.conservative(
        sample_grid_ds, latitude_coord="latitude"
    )
    assert ds_regrid.attrs == sample_input_data.attrs
    assert ds_regrid["d2m"].attrs == sample_input_data["d2m"].attrs
    assert ds_regrid["longitude"].attrs == sample_input_data["longitude"].attrs


def test_conservative_nan_aggregation_over_dims():
    """Check the behavior of valid cell aggregation across multiple dimensions.
    If we correctly accumulate the NaN count across dims, this should output 1.666,
    vs 1.5 or 1.75 if we naively aggregate dimensions separately without accumulating
    the NaN count. Also checks the ability to handle a singleton target grid."""
    data = xr.DataArray([[1, np.nan], [2, 2]], coords={"x": [-1, 1], "y": [-1, 1]})
    # Add a time dim and mask off a single slice to make sure we keep all possible
    # valid points
    data = data.expand_dims(time=[0, 1, 2], axis=0)
    data = data.where(data.time < 2)
    target = xr.Dataset(coords={"x": [0], "y": [0]})

    result = data.regrid.conservative(target, skipna=True, nan_threshold=1)
    np.testing.assert_allclose(result[0].mean().item(), data[0].mean().item())


@pytest.mark.parametrize("nan_threshold", [0, 1])
def test_conservative_nan_thresholds_against_coarsen(nan_threshold):
    """Compare nan_threshold regridding to behavior of xarray coarsen, where coarsen
    with skipna=True should map to nan_threshold=1.0 and coarsen with skipna=False
    should map to nan_threshold=0.0 when the source grid evenly divides the target
    grid."""
    da = xr.DataArray(
        [
            [1, np.nan, np.nan, np.nan],
            [2, 2, np.nan, np.nan],
            [3, 3, 3, np.nan],
            [4, 4, 4, 4],
        ],
        coords={"x": [0, 1, 2, 3], "y": [0, 1, 2, 3]},
        name="foo",
    )
    da_coarsen = da.coarsen(x=2, y=2).mean(skipna=bool(nan_threshold))
    target = da_coarsen.to_dataset()[["x", "y"]]
    da_regrid = da.regrid.conservative(target, skipna=True, nan_threshold=nan_threshold)

    xr.testing.assert_allclose(da_coarsen, da_regrid)


def test_conservative_output_chunks():
    data = xr.DataArray(
        np.ones((8, 8)),
        coords={"x": np.linspace(0, 1, 8), "y": np.linspace(0, 1, 8)},
        dims=("x", "y"),
    )
    target = xr.Dataset(coords={"x": np.linspace(0, 1, 4), "y": np.linspace(0, 1, 4)})

    # Non-dask input should return non-dask output
    result = data.regrid.conservative(target)
    assert result.chunks is None

    # Dask input with unspecified output chunks should match the input chunks
    result = data.chunk({"x": 2, "y": 2}).regrid.conservative(target)
    assert result.chunks == ((2, 2), (2, 2))

    # Specified output chunks should be respected
    result = data.chunk({"x": 4, "y": 4}).regrid.conservative(
        target, output_chunks={"x": 2, "y": 2}
    )
    assert result.chunks == ((2, 2), (2, 2))


@pytest.mark.skipif(xesmf is None, reason="xesmf required")
def test_conservative_nan_thresholds_against_xesmf():
    ds = xr.tutorial.open_dataset("ersstv5").sst.isel(time=[0]).compute()
    ds = ds.rename(lon="longitude", lat="latitude")
    new_grid = xarray_regrid.Grid(
        north=90,
        east=360,
        south=-90,
        west=0,
        resolution_lat=2,
        resolution_lon=2,
    )
    target_dataset = xarray_regrid.create_regridding_dataset(new_grid)
    regridder = xesmf.Regridder(ds, target_dataset, "conservative")

    for nan_threshold in [0.0, 0.25, 0.5, 0.75, 1.0]:
        data_regrid = ds.regrid.conservative(
            target_dataset, skipna=True, nan_threshold=nan_threshold
        )
        data_esmf = regridder(ds, keep_attrs=True, na_thres=nan_threshold, skipna=True)
        xr.testing.assert_equal(data_regrid.isnull(), data_esmf.isnull())


def test_conservative_conserves_known_integral():
    """Gold-standard conservation for the axis-factored method.

    ``cos^2(lat) * (1.5 + sin(lon))`` integrates to ``4*pi`` over the unit
    sphere. Regridded with the spherical correction (``latitude_coord``) onto a
    co-extensive coarser global grid, the integral is conserved to the grid
    quadrature floor when measured with INDEPENDENT analytic spherical cell
    areas (sin-latitude bands times dlon) -- not the regridder's own weights --
    and the source integral approaches the known value.

    A spatially-varying field + independent areas + matched domains exercises
    the real weights, unlike a constant field (which any row-normalized
    regridder reproduces) or a self-area check (a row-sum identity).
    """

    def centers(n, lo, hi):  # global cell centers; edges land exactly on lo/hi
        edges = np.linspace(lo, hi, n + 1)
        return 0.5 * (edges[:-1] + edges[1:])

    def analytic_area(n_lon, n_lat):  # independent of the regridder
        lat_e = np.deg2rad(np.linspace(-90, 90, n_lat + 1))
        lon_e = np.deg2rad(np.linspace(-180, 180, n_lon + 1))
        return np.diff(np.sin(lat_e))[:, None] * np.diff(lon_e)[None, :]

    ns_lat, ns_lon, nt_lat, nt_lon = 120, 240, 40, 80
    lat_s, lon_s = centers(ns_lat, -90, 90), centers(ns_lon, -180, 180)
    lat_t, lon_t = centers(nt_lat, -90, 90), centers(nt_lon, -180, 180)
    grid_lat, grid_lon = np.meshgrid(lat_s, lon_s, indexing="ij")
    field = np.cos(np.deg2rad(grid_lat)) ** 2 * (1.5 + np.sin(np.deg2rad(grid_lon)))
    da = xr.DataArray(field, dims=("lat", "lon"), coords={"lat": lat_s, "lon": lon_s})
    target = xr.Dataset(coords={"lat": lat_t, "lon": lon_t})

    out = (
        da.regrid.conservative(target, latitude_coord="lat", skipna=False)
        .transpose("lat", "lon")
        .values
    )
    assert np.isfinite(out).all()  # co-extensive global grids -> full coverage

    i_src = float((da.values * analytic_area(ns_lon, ns_lat)).sum())
    i_tgt = float((out * analytic_area(nt_lon, nt_lat)).sum())
    np.testing.assert_allclose(i_tgt, i_src, rtol=1e-5)  # mass conserved
    np.testing.assert_allclose(i_src, 4 * np.pi, rtol=2e-3)  # ~ the known value


def test_conservative_returns_dense_output():
    """Regression guard: the regridded result must be a dense ndarray, not a
    ``sparse.COO`` array.

    Sparse weights are applied with a scipy CSR matmul (see
    ``methods.conservative.apply_weights``), which produces a dense result. A
    ``sparse.COO`` result here means the apply regressed to the sparse ``xr.dot``
    path -- 20-100x slower without ``opt_einsum``, and a wasteful sparse
    container for a dense field.
    """
    lat = np.linspace(-89, 89, 60)
    lon = np.linspace(-179, 179, 120)
    da = xr.DataArray(
        np.cos(np.deg2rad(lat))[:, None] * np.ones(lon.size),
        dims=("lat", "lon"),
        coords={"lat": lat, "lon": lon},
    )
    target = xr.Dataset(
        coords={"lat": np.linspace(-88, 88, 30), "lon": np.linspace(-178, 178, 60)}
    )
    out = da.regrid.conservative(target, latitude_coord="lat")
    assert isinstance(out.data, np.ndarray), (
        f"expected dense ndarray, got {type(out.data).__name__}"
    )


class TestCoordOrder:
    @pytest.mark.parametrize("method", ["linear", "nearest", "cubic"])
    @pytest.mark.parametrize("dataarray", [True, False])
    def test_original(self, sample_input_data, sample_grid_ds, method, dataarray):
        input_data = sample_input_data["d2m"] if dataarray else sample_input_data
        regridder = getattr(input_data.regrid, method)
        ds_regrid = regridder(sample_grid_ds)
        assert_array_equal(ds_regrid["latitude"], sample_grid_ds["latitude"])
        assert_array_equal(ds_regrid["longitude"], sample_grid_ds["longitude"])

    @pytest.mark.parametrize("coord", ["latitude", "longitude"])
    @pytest.mark.parametrize("method", ["linear", "nearest", "cubic"])
    @pytest.mark.parametrize("dataarray", [True, False])
    def test_reversed(
        self, sample_input_data, sample_grid_ds, method, coord, dataarray
    ):
        input_data = sample_input_data["d2m"] if dataarray else sample_input_data
        regridder = getattr(input_data.regrid, method)
        sample_grid_ds[coord] = list(reversed(sample_grid_ds[coord]))
        ds_regrid = regridder(sample_grid_ds)
        assert_array_equal(ds_regrid["latitude"], sample_grid_ds["latitude"])
        assert_array_equal(ds_regrid["longitude"], sample_grid_ds["longitude"])

    @pytest.mark.parametrize("dataarray", [True, False])
    def test_conservative_original(self, sample_input_data, sample_grid_ds, dataarray):
        input_data = sample_input_data["d2m"] if dataarray else sample_input_data
        ds_regrid = input_data.regrid.conservative(
            sample_grid_ds, latitude_coord="latitude"
        )
        assert_array_equal(ds_regrid["latitude"], sample_grid_ds["latitude"])
        assert_array_equal(ds_regrid["longitude"], sample_grid_ds["longitude"])

    @pytest.mark.parametrize("coord", ["latitude", "longitude"])
    @pytest.mark.parametrize("dataarray", [True, False])
    def test_conservative_reversed(
        self, sample_input_data, sample_grid_ds, coord, dataarray
    ):
        input_data = sample_input_data["d2m"] if dataarray else sample_input_data
        sample_grid_ds[coord] = list(reversed(sample_grid_ds[coord]))
        ds_regrid = input_data.regrid.conservative(
            sample_grid_ds, latitude_coord="latitude"
        )
        assert_array_equal(ds_regrid["latitude"], sample_grid_ds["latitude"])
        assert_array_equal(ds_regrid["longitude"], sample_grid_ds["longitude"])
