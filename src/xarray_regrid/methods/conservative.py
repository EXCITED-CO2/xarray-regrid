"""Conservative regridding implementation."""
from collections.abc import Hashable
from typing import overload

import dask.array
import numpy as np
import xarray as xr

from xarray_regrid import utils


@overload
def conservative_regrid(
    data: xr.DataArray,
    target_ds: xr.Dataset,
    latitude_coord: str | None,
    skipna: bool = True,
    nan_threshold: float = 1.0,
) -> xr.DataArray:
    ...


@overload
def conservative_regrid(
    data: xr.Dataset,
    target_ds: xr.Dataset,
    latitude_coord: str | None,
    skipna: bool = True,
    nan_threshold: float = 1.0,
) -> xr.Dataset:
    ...


def conservative_regrid(
    data: xr.DataArray | xr.Dataset,
    target_ds: xr.Dataset,
    latitude_coord: str | None,
    skipna: bool = True,
    nan_threshold: float = 1.0,
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
            infer it from the first coordinate containing the string 'lat'.
        skipna: If True, enable handling for NaN values. This adds some overhead,
            so should be disabled for optimal performance on data without NaNs.
        nan_threshold: Threshold value that will retain any output points containing
            at least this many non-null input points. The default value is 1.0,
            which will keep output points containing any non-null inputs. The threshold
            is applied sequentially to each dimension, and may produce different results
            than a threshold applied concurrently to all regridding dimensions.

    Returns:
        Regridded input dataset
    """
    if latitude_coord is None:
        for coord in data.coords:
            if "lat" in coord.lower():
                latitude_coord = coord
                break
    if latitude_coord not in data.coords:
        msg = "Latitude coord not in input data!"
        raise ValueError(msg)

    dim_order = list(target_ds.dims)

    coord_names = set(target_ds.coords).intersection(set(data.coords))
    target_ds_sorted = target_ds.sortby(list(coord_names))
    coords = {name: target_ds_sorted[name] for name in coord_names}
    data = data.sortby(list(coord_names))

    if isinstance(data, xr.Dataset):
        regridded_data = conservative_regrid_dataset(
            data, coords, latitude_coord, skipna, nan_threshold
        ).transpose(*dim_order, ...)
    else:
        regridded_data = conservative_regrid_dataarray(  # type: ignore
            data, coords, latitude_coord, skipna, nan_threshold
        ).transpose(*dim_order, ...)

    regridded_data = regridded_data.reindex_like(target_ds, copy=False)

    return regridded_data


def conservative_regrid_dataset(
    data: xr.Dataset,
    coords: dict[Hashable, xr.DataArray],
    latitude_coord: str,
    skipna: bool,
    nan_threshold: float,
) -> xr.Dataset:
    """Dataset implementation of the conservative regridding method."""
    data_vars = list(data.data_vars)
    data_coords = list(data.coords)
    dataarrays = [data[var] for var in data_vars]

    attrs = data.attrs
    da_attrs = [da.attrs for da in dataarrays]
    coord_attrs = [data[coord].attrs for coord in data_coords]

    # track which target coordinate values are not covered by the source grid
    uncovered_target_grid = {}
    for coord in coords:
        uncovered_target_grid[coord] = (coords[coord] <= data[coord].max()) & (
            coords[coord] >= data[coord].min()
        )

        target_coords = coords[coord].to_numpy()
        source_coords = data[coord].to_numpy()
        weights = get_weights(source_coords, target_coords)

        # Modify weights to correct for latitude distortion
        if str(coord) == latitude_coord:
            dot_array = utils.create_dot_dataarray(
                weights, str(coord), target_coords, source_coords
            )
            dot_array = apply_spherical_correction(dot_array, latitude_coord)
            weights = dot_array.to_numpy()

        for i in range(len(dataarrays)):
            if coord in dataarrays[i].coords:
                da = dataarrays[i].transpose(coord, ...)
                dataarrays[i] = apply_weights(
                    da, weights, coord, target_coords, skipna, nan_threshold
                )

    for da, attr in zip(dataarrays, da_attrs, strict=True):
        da.attrs = attr
    regridded = xr.merge(dataarrays)

    # Replace zeros outside of original data grid with NaNs
    for coord in coords:
        regridded = regridded.where(uncovered_target_grid[coord])

    regridded.attrs = attrs

    new_coords = [regridded[coord] for coord in data_coords]
    for coord, attr in zip(new_coords, coord_attrs, strict=True):
        coord.attrs = attr

    return regridded  # TODO: add other coordinates/data variables back in.


def conservative_regrid_dataarray(
    data: xr.DataArray,
    coords: dict[Hashable, xr.DataArray],
    latitude_coord: str,
    skipna: bool,
    nan_threshold: float,
) -> xr.DataArray:
    """DataArray implementation of the conservative regridding method."""
    data_coords = list(data.coords)

    attrs = data.attrs
    coord_attrs = [data[coord].attrs for coord in data_coords]

    for coord in coords:
        uncovered_target_grid = (coords[coord] <= data[coord].max()) & (
            coords[coord] >= data[coord].min()
        )

        if coord in data.coords:
            target_coords = coords[coord].to_numpy()
            source_coords = data[coord].to_numpy()

            weights = get_weights(source_coords, target_coords)

            # Modify weights to correct for latitude distortion
            if str(coord) == latitude_coord:
                dot_array = utils.create_dot_dataarray(
                    weights, str(coord), target_coords, source_coords
                )
                dot_array = apply_spherical_correction(dot_array, latitude_coord)
                weights = dot_array.to_numpy()

            data = data.transpose(coord, ...)
            data = apply_weights(
                data, weights, coord, target_coords, skipna, nan_threshold
            )

            # Replace zeros outside of original data grid with NaNs
            data = data.where(uncovered_target_grid)

    new_coords = [data[coord] for coord in data_coords]
    for coord, attr in zip(new_coords, coord_attrs, strict=True):
        coord.attrs = attr
    data.attrs = attrs

    return data


def apply_weights(
    da: xr.DataArray,
    weights: np.ndarray,
    coord_name: Hashable,
    new_coords: np.ndarray,
    skipna: bool,
    nan_threshold: float,
) -> xr.DataArray:
    """Apply the weights to convert data to the new coordinates."""
    new_data: np.ndarray | dask.array.Array

    new_data = compute_einsum(da, weights, skipna, nan_threshold)

    coord_mapping = {coord_name: new_coords}
    coords = list(da.dims)
    coords.remove(coord_name)
    for coord in coords:
        coord_mapping[coord] = da[coord].to_numpy()

    return xr.DataArray(
        data=new_data,
        coords=coord_mapping,
        name=da.name,
    )


def compute_einsum(
    da: xr.DataArray, weights: np.ndarray, skipna: bool, nan_threshold: float
) -> np.ndarray | dask.array.Array:
    """Compute the einsum between the data and weights. If nan-skipping is
    enabled, renormalize the weights by the non-null input points, and keep
    output points containing up to the specified fraction of NaN values."""
    new_data: np.ndarray | dask.array.Array

    if skipna:
        data = da.fillna(0).data
        notnull = da.notnull().data
    else:
        data = da.data

    backend = np if da.chunks is None else dask.array
    new_data = backend.einsum(
        "i...,ij->j...", data, weights, optimize="greedy"
    )

    if skipna:
        valid_weight_sum = backend.einsum(
            "i...,ij->j...", notnull, weights, optimize="greedy"
        )
        new_data = new_data / backend.clip(valid_weight_sum, 1e-6, None)
        weight_threshold = get_weight_threshold(nan_threshold)
        new_data = backend.where(valid_weight_sum > weight_threshold, new_data, np.nan)

    return new_data


def get_weight_threshold(nan_threshold: float) -> float:
    """Make sure the specified nan_threshold is in [0,1], invert it,
    and coerce to just above zero and below one to handle numerical
    precision limitations in the weight sum."""
    if not 0.0 <= nan_threshold <= 1.0:
        msg = "nan_threshold must be between [0, 1]]"
        raise ValueError(msg)
    # This matches xesmf where na_thresh=0 keeps points with any valid data
    weight_threshold = 1 - np.clip(nan_threshold, 1e-6, 1.0 - 1e-6)
    return weight_threshold


def get_weights(source_coords: np.ndarray, target_coords: np.ndarray) -> np.ndarray:
    """Determine the weights to map from the old coordinates to the new coordinates.

    Args:
        source_coords: Source coordinates (center points)
        target_coords Target coordinates (center points)

    Returns:
        Weights, which can be used with a dot product to apply the conservative regrid.
    """
    # TODO: better resolution/IntervalIndex inference
    target_intervals = utils.to_intervalindex(
        target_coords, resolution=target_coords[1] - target_coords[0]
    )

    source_intervals = utils.to_intervalindex(
        source_coords, resolution=source_coords[1] - source_coords[0]
    )
    overlap = utils.overlap(source_intervals, target_intervals)
    return utils.normalize_overlap(overlap)


def apply_spherical_correction(
    dot_array: xr.DataArray, latitude_coord: str
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
