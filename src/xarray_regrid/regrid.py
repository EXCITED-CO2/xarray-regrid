import xarray as xr

from xarray_regrid.methods import conservative, interp, most_common


@xr.register_dataarray_accessor("regrid")
@xr.register_dataset_accessor("regrid")
class Regridder:
    """Regridding xarray dataarrays.

    Available methods:
        linear: linear, bilinear, or higher dimensional linear interpolation.
        nearest: nearest-neighbor regridding.
        cubic: cubic spline regridding.
        conservative: conservative regridding.
        most_common: most common value regridder
    """

    def __init__(self, xarray_obj: xr.DataArray | xr.Dataset):
        self._obj = xarray_obj

    def linear(
        self,
        ds_target_grid: xr.Dataset,
        time_dim: str = "time",
    ) -> xr.DataArray | xr.Dataset:
        """Regrid to the coords of the target dataset with linear interpolation.

        Args:
            ds_target_grid: Dataset containing the target coordinates.
            time_dim: The name of the time dimension/coordinate

        Returns:
            Data regridded to the target dataset coordinates.
        """
        ds_target_grid = validate_input(self._obj, ds_target_grid, time_dim)
        return interp.interp_regrid(self._obj, ds_target_grid, "linear")

    def nearest(
        self,
        ds_target_grid: xr.Dataset,
        time_dim: str = "time",
    ) -> xr.DataArray | xr.Dataset:
        """Regrid to the coords of the target with nearest-neighbor interpolation.

        Args:
            ds_target_grid: Dataset containing the target coordinates.
            time_dim: The name of the time dimension/coordinate

        Returns:
            Data regridded to the target dataset coordinates.
        """
        ds_target_grid = validate_input(self._obj, ds_target_grid, time_dim)
        return interp.interp_regrid(self._obj, ds_target_grid, "nearest")

    def cubic(
        self,
        ds_target_grid: xr.Dataset,
        time_dim: str = "time",
    ) -> xr.DataArray | xr.Dataset:
        ds_target_grid = validate_input(self._obj, ds_target_grid, time_dim)
        """Regrid to the coords of the target dataset with cubic interpolation.

        Args:
            ds_target_grid: Dataset containing the target coordinates.
            time_dim: The name of the time dimension/coordinate

        Returns:
            Data regridded to the target dataset coordinates.
        """
        return interp.interp_regrid(self._obj, ds_target_grid, "cubic")

    def conservative(
        self,
        ds_target_grid: xr.Dataset,
        latitude_coord: str | None = None,
        time_dim: str = "time",
        skipna: bool = True,
        nan_threshold: float = 0.0,
    ) -> xr.DataArray | xr.Dataset:
        """Regrid to the coords of the target dataset with a conservative scheme.

        Args:
            ds_target_grid: Dataset containing the target coordinates.
            latitude_coord: Name of the latitude coord, to be used for applying the
                spherical correction. By default, attempt to infer a latitude coordinate
                as anything starting with "lat".
            time_dim: The name of the time dimension/coordinate.
            skipna: If True, enable handling for NaN values. This adds some overhead,
                so can be disabled for optimal performance on data without any NaNs.
                Warning: with `skipna=False`, isolated NaNs will propagate throughout
                the dataset due to the sequential regridding scheme over each dimension.
            nan_threshold: Threshold value that will retain any output points
                containing at least this many non-null input points. The default value
                is 1.0, which will keep output points containing any non-null inputs,
                while a value of 0.0 will only keep output points where all inputs are
                non-null.

        Returns:
            Data regridded to the target dataset coordinates.
        """
        if not 0.0 <= nan_threshold <= 1.0:
            msg = "nan_threshold must be between [0, 1]]"
            raise ValueError(msg)

        ds_target_grid = validate_input(self._obj, ds_target_grid, time_dim)
        return conservative.conservative_regrid(
            self._obj, ds_target_grid, latitude_coord, skipna, nan_threshold
        )

    def most_common(
        self,
        ds_target_grid: xr.Dataset,
        time_dim: str = "time",
        max_mem: int = int(1e9),
    ) -> xr.DataArray | xr.Dataset:
        """Regrid by taking the most common value within the new grid cells.

        To be used for regridding data to a much coarser resolution, not for regridding
        when the source and target grids are of a similar resolution.

        Note that in the case of two unqiue values with the same count, the behaviour
        is not deterministic, and the resulting "most common" one will randomly be
        either of the two.

        Args:
            ds_target_grid: Target grid dataset
            time_dim: Name of the time dimension. Defaults to "time".
            max_mem: (Approximate) maximum memory in bytes that the regridding routine
                can use. Note that this is not the total memory consumption and does not
                include the size of the final dataset. Defaults to 1e9 (1 GB).

        Returns:
            Regridded data.
        """
        ds_target_grid = validate_input(self._obj, ds_target_grid, time_dim)
        return most_common.most_common_wrapper(
            self._obj, ds_target_grid, time_dim, max_mem
        )


def validate_input(
    data: xr.DataArray | xr.Dataset,
    ds_target_grid: xr.Dataset,
    time_dim: str,
) -> xr.Dataset:
    if time_dim in ds_target_grid.coords:
        ds_target_grid = ds_target_grid.isel(time=0).reset_coords()

    if len(set(data.dims).intersection(set(ds_target_grid.dims))) == 0:
        msg = (
            "None of the target dims are in the data:\n"
            " regridding is not possible.\n"
            f"Target dims: {list(ds_target_grid.dims)}\n"
            f"Source dims: {list(data.dims)}"
        )
        raise ValueError(msg)

    if len(set(data.coords).intersection(set(ds_target_grid.coords))) == 0:
        msg = (
            "None of the target coords are in the data:\n"
            " regridding is not possible.\n"
            f"Target coords: {ds_target_grid.coords}\n"
            f"Dataset coords: {data.coords}"
        )
        raise ValueError(msg)

    return ds_target_grid
