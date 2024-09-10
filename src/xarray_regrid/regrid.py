import numpy as np
import xarray as xr

from xarray_regrid.methods import conservative, flox_reduce, interp


@xr.register_dataarray_accessor("regrid")
@xr.register_dataset_accessor("regrid")
class Regridder:
    """Regridding xarray datasets and dataarrays.

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
        time_dim: str | None = "time",
    ) -> xr.DataArray | xr.Dataset:
        """Regrid to the coords of the target dataset with linear interpolation.

        Args:
            ds_target_grid: Dataset containing the target coordinates.
            time_dim: Name of the time dimension. Defaults to "time". Use `None` to
                force regridding over the time dimension.

        Returns:
            Data regridded to the target dataset coordinates.
        """
        ds_target_grid = validate_input(self._obj, ds_target_grid, time_dim)
        return interp.interp_regrid(self._obj, ds_target_grid, "linear")

    def nearest(
        self,
        ds_target_grid: xr.Dataset,
        time_dim: str | None = "time",
    ) -> xr.DataArray | xr.Dataset:
        """Regrid to the coords of the target with nearest-neighbor interpolation.

        Args:
            ds_target_grid: Dataset containing the target coordinates.
            time_dim: Name of the time dimension. Defaults to "time". Use `None` to
                force regridding over the time dimension.

        Returns:
            Data regridded to the target dataset coordinates.
        """
        ds_target_grid = validate_input(self._obj, ds_target_grid, time_dim)
        return interp.interp_regrid(self._obj, ds_target_grid, "nearest")

    def cubic(
        self,
        ds_target_grid: xr.Dataset,
        time_dim: str | None = "time",
    ) -> xr.DataArray | xr.Dataset:
        ds_target_grid = validate_input(self._obj, ds_target_grid, time_dim)
        """Regrid to the coords of the target dataset with cubic interpolation.

        Args:
            ds_target_grid: Dataset containing the target coordinates.
            time_dim: Name of the time dimension. Defaults to "time". Use `None` to
                force regridding over the time dimension.

        Returns:
            Data regridded to the target dataset coordinates.
        """
        return interp.interp_regrid(self._obj, ds_target_grid, "cubic")

    def conservative(
        self,
        ds_target_grid: xr.Dataset,
        latitude_coord: str | None = None,
        time_dim: str | None = "time",
        skipna: bool = True,
        nan_threshold: float = 0.0,
    ) -> xr.DataArray | xr.Dataset:
        """Regrid to the coords of the target dataset with a conservative scheme.

        Args:
            ds_target_grid: Dataset containing the target coordinates.
            latitude_coord: Name of the latitude coord, to be used for applying the
                spherical correction. By default, attempt to infer a latitude coordinate
                as anything starting with "lat".
            time_dim: Name of the time dimension. Defaults to "time". Use `None` to
                force regridding over the time dimension.
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
        expected_groups: np.ndarray,
        time_dim: str | None = "time",
        inverse: bool = False,
    ) -> xr.DataArray:
        """Regrid by taking the most common value within the new grid cells.

        To be used for regridding data to a much coarser resolution, not for regridding
        when the source and target grids are of a similar resolution.

        Note that in the case of two unqiue values with the same count, the behaviour
        is not deterministic, and the resulting "most common" one will randomly be
        either of the two.

        Args:
            ds_target_grid: Target grid dataset
            expected_groups: Numpy array containing all labels expected to be in the
                input data. For example, `np.array([0, 2, 4])`, if the data only
                contains the values 0, 2 and 4.
            time_dim: Name of the time dimension. Defaults to "time". Use `None` to
                force regridding over the time dimension.
            inverse: Find the least-common-value (anti-mode).

        Returns:
            Regridded data.
        """
        ds_target_grid = validate_input(self._obj, ds_target_grid, time_dim)

        if isinstance(self._obj, xr.Dataset):
            msg = (
                "The 'most common value' regridder is not implemented for\n",
                "xarray.Dataset, as it requires specifying the expected labels.\n"
                "Please select only a single variable (as DataArray),\n"
                " and regrid it separately.",
            )
            raise ValueError(msg)

        return flox_reduce.get_most_common_value(
            self._obj,
            ds_target_grid,
            expected_groups,
            time_dim,
            inverse,
        )

    def stat(
        self,
        ds_target_grid: xr.Dataset,
        method: str,
        time_dim: str | None = "time",
        skipna: bool = False,
    ) -> xr.DataArray | xr.Dataset:
        """Upsampling of data using statistical methods (e.g. the mean or variance).

        We use flox Aggregations to perform a "groupby" over multiple dimensions, which
        we reduce using the specified method.
        https://flox.readthedocs.io/en/latest/aggregations.html

        Args:
            ds_target_grid: Target grid dataset
            method: One of the following reduction methods: "sum", "mean", "var", "std",
                or "median.
            time_dim: Name of the time dimension. Defaults to "time". Use `None` to
                force regridding over the time dimension.
            skipna: If NaN values should be ignored.

        Returns:
            xarray.dataset with regridded land cover categorical data.
        """
        ds_target_grid = validate_input(self._obj, ds_target_grid, time_dim)

        return flox_reduce.statistic_reduce(
            self._obj, ds_target_grid, time_dim, method, skipna
        )


def validate_input(
    data: xr.DataArray | xr.Dataset,
    ds_target_grid: xr.Dataset,
    time_dim: str | None,
) -> xr.Dataset:
    if time_dim is not None and time_dim in ds_target_grid.coords:
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
