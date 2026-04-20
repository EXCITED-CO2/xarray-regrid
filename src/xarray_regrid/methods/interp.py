"""Methods based on xr.interp."""

from typing import Literal, cast, overload

import xarray as xr


@overload
def interp_regrid(
    data: xr.DataArray,
    target_ds: xr.Dataset,
    method: Literal["linear", "nearest", "cubic"],
) -> xr.DataArray: ...


@overload
def interp_regrid(
    data: xr.Dataset,
    target_ds: xr.Dataset,
    method: Literal["linear", "nearest", "cubic"],
) -> xr.Dataset: ...


def interp_regrid(
    data: xr.DataArray | xr.Dataset,
    target_ds: xr.Dataset,
    method: Literal["linear", "nearest", "cubic"],
) -> xr.DataArray | xr.Dataset:
    """Refine a dataset using xarray's interp method.

    Args:
        data: Input dataset.
        target_ds: Dataset which coordinates the input dataset should be regrid to.
        method: Which interpolation method to use (e.g. 'linear', 'nearest').

    Returns:
        Regridded input dataset
    """
    coord_names = set(target_ds.coords).intersection(set(data.coords))
    coords = {name: target_ds[name] for name in coord_names}
    coord_attrs = {coord: data[coord].attrs for coord in coord_names}

    interped = data.interp(
        coords=coords,
        method=method,
    )

    # xarray's interp drops some of the coordinate's attributes (e.g. long_name)
    for coord in coord_names:
        interped[coord].attrs = coord_attrs[coord]

    return cast("xr.DataArray | xr.Dataset", interped)
