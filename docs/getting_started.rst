Quickstart
==========

``xarray-regrid`` allows you to regrid xarray Datasets or DataArrays to a new resolution.
You can install xarray-regrid with pip:

.. code:: shell

   pip install xarray-regrid


To use the package, import ``xarray_regrid``. This will register the ``.regrid`` 'accessor' so it can be used.
Next load in the data you want to regrid, and the data with a grid you want to regrid to:

.. code:: python

    import xarray_regrid
    import xarray

    ds = xr.open_dataset("input_data.nc")
    ds_grid = xr.open_dataset("target_grid.nc")

    ds = ds.regrid.linear(ds_grid)

    # or, for example:
    ds = ds.regrid.conservative(ds_grid, latitude_coord="lat")


Multiple regridding methods are available:

* `linear interpolation <autoapi/xarray_regrid/regrid/index.html#xarray_regrid.regrid.Regridder.linear>`_ (``.regrid.linear``)
* `nearest-neighbor <autoapi/xarray_regrid/regrid/index.html#xarray_regrid.regrid.Regridder.nearest>`_ (``.regrid.nearest``)
* `cubic interpolation  <autoapi/xarray_regrid/regrid/index.html#xarray_regrid.regrid.Regridder.cubic>`_ (``.regrid.cubic``)
* `conservative regridding <autoapi/xarray_regrid/regrid/index.html#xarray_regrid.regrid.Regridder.conservative>`_ (``.regrid.conservative``)
* `conservative 2D regridding <autoapi/xarray_regrid/regrid/index.html#xarray_regrid.regrid.Regridder.conservative_2d>`_ (``.regrid.conservative_2d``) for curvilinear grids, unstructured meshes, and grid-to-polygon aggregation
* `zonal statistics <autoapi/xarray_regrid/regrid/index.html#xarray_regrid.regrid.Regridder.stat>`_ (``.regrid.stat``) is available to compute statistics such as the maximum value, or variance. 

Additionally, there are separate methods available to compute the
`most common value <autoapi/xarray_regrid/regrid/index.html#xarray_regrid.regrid.Regridder.most_common>`_
(``.regrid.most_common``) and `least common value <autoapi/xarray_regrid/regrid/index.html#xarray_regrid.regrid.Regridder.least_common>`_
(``.regrid.least_common``). This can be used to upscale very fine categorical data to a more course resolution.
