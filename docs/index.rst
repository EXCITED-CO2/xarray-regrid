.. toctree::
  :maxdepth: 3
  :hidden:

  self


.. toctree::
  :maxdepth: 3
  :caption: User Guide
  :hidden:
  
  Quickstart <getting_started>
  Example Notebooks <notebooks/index>


.. toctree::
  :maxdepth: 3
  :caption: Technical information
  :hidden:

  Changelog <changelog_link>

xarray-regrid: Regridding utilities for xarray
**********************************************

|PyPI| |DOI|

Overview
========

``xarray-regrid`` extends xarray with regridding methods, making it possibly to easily and effiently regrid between two rectilinear grids. 

The following methods are supported:

* `Linear <autoapi/xarray_regrid/regrid/index.html#xarray_regrid.regrid.Regridder.linear>`_
* `Nearest-neighbor <autoapi/xarray_regrid/regrid/index.html#xarray_regrid.regrid.Regridder.nearest>`_
* `Conservative <autoapi/xarray_regrid/regrid/index.html#xarray_regrid.regrid.Regridder.conservative>`_
* `Conservative 2D (curvilinear / unstructured / polygon aggregation) <autoapi/xarray_regrid/regrid/index.html#xarray_regrid.regrid.Regridder.conservative_2d>`_
* `Cubic <autoapi/xarray_regrid/regrid/index.html#xarray_regrid.regrid.Regridder.cubic>`_
* `Zonal statistics <autoapi/xarray_regrid/regrid/index.html#xarray_regrid.regrid.Regridder.stat>`_
* `"Most common value" (zonal statistics) <autoapi/xarray_regrid/regrid/index.html#xarray_regrid.regrid.Regridder.most_common>`_
* `"Least common value" (zonal statistics) <autoapi/xarray_regrid/regrid/index.html#xarray_regrid.regrid.Regridder.least_common>`_

Note that "Most/least common value" is designed to regrid categorical data to a coarse resolution. For regridding categorical data to a finer resolution, please use "nearest-neighbor" regridder.

For usage examples, please refer to the `quickstart guide <getting_started>`_ and the `example notebooks <notebooks/index>`_.

Installing
==========

.. code:: shell

   pip install xarray-regrid


Acknowledgements
================

This package was developed under Netherlands eScience Center grant `NLESC.OEC.2022.017 <https://research-software-directory.org/projects/excited>`_.

Some methods were inspired by discussions in the `Pangeo <https://pangeo.io>`_ community.

.. |PyPI| image:: https://img.shields.io/pypi/v/xarray-regrid.svg?style=flat
   :target: https://pypi.python.org/pypi/xarray-regrid/

.. |DOI| image:: https://zenodo.org/badge/DOI/10.5281/zenodo.10203304.svg
   :target: https://doi.org/10.5281/zenodo.10203304
