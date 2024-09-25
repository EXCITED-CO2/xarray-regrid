"""Utility functions shared between methods."""

from typing import Any, overload

import numpy as np
import pandas as pd
import xarray as xr


def construct_intervals(coord: np.ndarray) -> pd.IntervalIndex:
    """Create pandas.intervals with given coordinates."""
    step_size = np.median(np.diff(coord, n=1))
    breaks = np.append(coord, coord[-1] + step_size) - step_size / 2

    # Note: closed="both" triggers an `NotImplementedError`
    return pd.IntervalIndex.from_breaks(breaks, closed="left")


@overload
def restore_properties(
    result: xr.DataArray,
    original_data: xr.DataArray | xr.Dataset,
    target_ds: xr.Dataset,
    coords: list[str],
    fill_value: Any,
) -> xr.DataArray: ...


@overload
def restore_properties(
    result: xr.Dataset,
    original_data: xr.DataArray | xr.Dataset,
    target_ds: xr.Dataset,
    coords: list[str],
    fill_value: Any,
) -> xr.Dataset: ...


def restore_properties(
    result: xr.DataArray | xr.Dataset,
    original_data: xr.DataArray | xr.Dataset,
    target_ds: xr.Dataset,
    coords: list[str],
    fill_value: Any,
) -> xr.DataArray | xr.Dataset:
    """Restore coord names, copy values and attributes of target, & add NaN padding."""
    result.attrs = original_data.attrs

    result = result.rename({f"{coord}_bins": coord for coord in coords})
    for coord in coords:
        result[coord] = target_ds[coord]
        result[coord].attrs = target_ds[coord].attrs

        # Replace zeros outside of original data grid with NaNs
        uncovered_target_grid = (target_ds[coord] <= original_data[coord].max()) & (
            target_ds[coord] >= original_data[coord].min()
        )
        if fill_value is None:
            result = result.where(uncovered_target_grid)
        else:
            result = result.where(uncovered_target_grid, fill_value)

    return result.transpose(*original_data.dims)


@overload
def reduce_data_to_new_domain(
    data: xr.DataArray,
    target_ds: xr.Dataset,
    coords: list[str],
) -> xr.DataArray: ...


@overload
def reduce_data_to_new_domain(
    data: xr.Dataset,
    target_ds: xr.Dataset,
    coords: list[str],
) -> xr.Dataset: ...


def reduce_data_to_new_domain(
    data: xr.DataArray | xr.Dataset,
    target_ds: xr.Dataset,
    coords: list[str],
) -> xr.DataArray | xr.Dataset:
    """Slice the input data to bounds of the target dataset, to reduce computations."""
    data = data.sortby(list(coords))
    for coord in coords:
        coord_res = np.median(np.diff(target_ds[coord].to_numpy(), 1))
        data = data.sel(
            {
                coord: slice(
                    target_ds[coord].min().to_numpy() - coord_res,
                    target_ds[coord].max().to_numpy() + coord_res,
                )
            }
        )
    return data
