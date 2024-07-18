import numpy as np
import pytest
import xarray as xr
from numpy.testing import assert_array_equal

from xarray_regrid import Grid, create_regridding_dataset


@pytest.fixture
def dummy_lc_data():
    data = np.array(
        [
            [2, 2, 2, 2, 0, 0, 0, 0, 0, 0, 0],
            [2, 2, 0, 2, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [3, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
            [3, 3, 3, 3, 0, 0, 0, 0, 1, 1, 1],
            [3, 3, 0, 3, 0, 0, 0, 0, 1, 1, 1],
        ]
    )
    lat_coords = np.linspace(0, 40, num=11)
    lon_coords = np.linspace(0, 40, num=11)

    return xr.Dataset(
        data_vars={
            "lc": (["longitude", "latitude"], data),
        },
        coords={
            "longitude": (["longitude"], lon_coords),
            "latitude": (["latitude"], lat_coords),
        },
        attrs={"test": "not empty"},
    )


@pytest.fixture
def dummy_target_grid():
    new_grid = Grid(
        north=40,
        east=40,
        south=0,
        west=0,
        resolution_lat=8,
        resolution_lon=8,
    )
    return create_regridding_dataset(new_grid)


@pytest.fixture
def oversized_dummy_target_grid():
    new_grid = Grid(
        north=48,
        east=48,
        south=-8,
        west=-8,
        resolution_lat=8,
        resolution_lon=8,
    )
    return create_regridding_dataset(new_grid)


def test_most_common(dummy_lc_data, dummy_target_grid):
    expected_data = np.array(
        [
            [2, 2, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [3, 3, 0, 0, 0, 1],
        ]
    )

    lat_coords = np.linspace(0, 40, num=6)
    lon_coords = np.linspace(0, 40, num=6)

    expected = xr.Dataset(
        data_vars={
            "lc": (["longitude", "latitude"], expected_data),
        },
        coords={
            "longitude": (["longitude"], lon_coords),
            "latitude": (["latitude"], lat_coords),
        },
    )
    xr.testing.assert_equal(
        dummy_lc_data.regrid.most_common(dummy_target_grid)["lc"],
        expected["lc"],
    )


def test_oversized_most_common(dummy_lc_data, oversized_dummy_target_grid):
    expected_data = np.array(
        [
            [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
            [np.nan, 2, 2, 0, 0, 0, 0, np.nan],
            [np.nan, 0, 0, 0, 0, 0, 0, np.nan],
            [np.nan, 0, 0, 0, 0, 0, 0, np.nan],
            [np.nan, 0, 0, 0, 0, 0, 0, np.nan],
            [np.nan, 0, 0, 0, 0, 0, 0, np.nan],
            [np.nan, 3, 3, 0, 0, 0, 1, np.nan],
            [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
        ]
    )

    lat_coords = np.linspace(-8, 48, num=8)
    lon_coords = np.linspace(-8, 48, num=8)

    expected = xr.Dataset(
        data_vars={
            "lc": (["longitude", "latitude"], expected_data),
        },
        coords={
            "longitude": (["longitude"], lon_coords),
            "latitude": (["latitude"], lat_coords),
        },
    )
    xr.testing.assert_equal(
        dummy_lc_data.regrid.most_common(oversized_dummy_target_grid)["lc"],
        expected["lc"],
    )


def test_attrs_dataarray(dummy_lc_data, dummy_target_grid):
    dummy_lc_data["lc"].attrs = {"test": "testing"}
    da_regrid = dummy_lc_data["lc"].regrid.most_common(dummy_target_grid)
    assert da_regrid.attrs != {}
    assert da_regrid.attrs == dummy_lc_data["lc"].attrs
    assert da_regrid["longitude"].attrs == dummy_lc_data["longitude"].attrs


def test_attrs_dataset(dummy_lc_data, dummy_target_grid):
    ds_regrid = dummy_lc_data.regrid.most_common(
        dummy_target_grid,
    )
    assert ds_regrid.attrs != {}
    assert ds_regrid.attrs == dummy_lc_data.attrs
    assert ds_regrid["longitude"].attrs == dummy_lc_data["longitude"].attrs


@pytest.mark.parametrize("dataarray", [True, False])
def test_coord_order_original(dummy_lc_data, dummy_target_grid, dataarray):
    input_data = dummy_lc_data["lc"] if dataarray else dummy_lc_data
    ds_regrid = input_data.regrid.most_common(dummy_target_grid)
    assert_array_equal(ds_regrid["latitude"], dummy_target_grid["latitude"])
    assert_array_equal(ds_regrid["longitude"], dummy_target_grid["longitude"])


@pytest.mark.parametrize("coord", ["latitude", "longitude"])
@pytest.mark.parametrize("dataarray", [True, False])
def test_coord_order_reversed(dummy_lc_data, dummy_target_grid, coord, dataarray):
    input_data = dummy_lc_data["lc"] if dataarray else dummy_lc_data
    dummy_target_grid[coord] = list(reversed(dummy_target_grid[coord]))
    ds_regrid = input_data.regrid.most_common(dummy_target_grid)
    assert_array_equal(ds_regrid["latitude"], dummy_target_grid["latitude"])
    assert_array_equal(ds_regrid["longitude"], dummy_target_grid["longitude"])
