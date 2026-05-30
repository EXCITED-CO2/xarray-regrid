# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## Unreleased

Added:
 - `conservative_2d` regridding for grids that aren't 1D-separable: curvilinear (2D `lat`/`lon`) grids, unstructured meshes (`ConservativeRegridder.from_polygons`), and grid-to-polygon aggregation (e.g. gridded data → region shapes). Built on shapely 2 polygon intersection with optional `sparse` weight storage, an analytic cylindrical-equal-area `manifold="cea"` option for spherical areas on lat/lon grids, antimeridian handling, and netCDF weight-matrix caching. Exposed via the `.regrid.conservative_2d` accessor and the reusable `ConservativeRegridder` class; install with the `conservative-2d` extra ([#70](https://github.com/xarray-contrib/xarray-regrid/pull/70)).

## 0.4.2 (2026-01-28)

New contributors:
 - [@tdickinson29](https://github.com/tdickinson29)

Fixed:
 - When validating input with a source dataset, use the passed `time_dim` argument ([#62](https://github.com/xarray-contrib/xarray-regrid/pull/62)).
 - Fixed issue with lon values being immutable, thanks @shenyulu ([#65](https://github.com/xarray-contrib/xarray-regrid/issues/65), [#66](https://github.com/xarray-contrib/xarray-regrid/pull/66)).
  
## 0.4.1 (2025-04-08)

Fixed:
 - Attributes are now properly preserved when updating coordinates during pre-formatting for regridding ([#54](https://github.com/xarray-contrib/xarray-regrid/pull/54)).
 - Handle datasets with inconsistent chunksizes during pre-formatting ([#57](https://github.com/xarray-contrib/xarray-regrid/pull/57)).
 - `regrid.stats` should properly regrid input data even when coordinates are not monotonic ([#58](https://github.com/xarray-contrib/xarray-regrid/pull/58)).

## 0.4.0 (2024-09-26)

Changed:
 - the "most common" routine has been overhauled, thanks to [@dcherian](https://github.com/dcherian). It is now much more efficient, and can operate fully lazily on dask arrays. Users do need to provide the expected groups (i.e., unique labels in the data), and the regridder is only available for `xr.DataArray` currently ([#46](https://github.com/xarray-contrib/xarray-regrid/pull/46)).
 - you can now use `None` as input to the `time_dim` kwarg in the regridding methods to force regridding over the time dimension (as long as it's numeric) ([#46](https://github.com/xarray-contrib/xarray-regrid/pull/46)).
 - Performance of the conservative method has been improved by simultaneously aggregating over all regridding dimensions. Conservative regridding now also produces outputs with the same grid chunks as the inputs, unless explicit chunksizes are passed via the `output_chunks` argument. ([#51](https://github.com/xarray-contrib/xarray-regrid/pull/51)).

Added:
 - `.regrid.stat` for reducing datasets using statistical methods such as the variance or median ([#46](https://github.com/xarray-contrib/xarray-regrid/pull/46)).
 - a "least common" routine (i.e. anti-mode), which is the inverse of the most common value ([#46](https://github.com/xarray-contrib/xarray-regrid/pull/46)).
 - If latitude/longitude coordinates are detected and the domain is global, apply automatic padding at the boundaries, which gives behavior more consistent with common tools like ESMF and CDO ([#45](https://github.com/xarray-contrib/xarray-regrid/pull/45)).
 - Conservative regridding weights are converted to sparse matrices if the optional [sparse](https://github.com/pydata/sparse) package is installed, which improves compute and memory performance in most cases ([#49](https://github.com/xarray-contrib/xarray-regrid/pull/49)).

## 0.3.0 (2024-09-05)

New contributors:
 - [@slevang](https://github.com/slevang)

Fixed:
 - conservative regridding now can be constructed fully lazily [#39](https://github.com/EXCITED-CO2/xarray-regrid/pull/39).

Added:
 - documentation for the package, including readthedocs integration ([#40](https://github.com/EXCITED-CO2/xarray-regrid/pull/40)).
 - better handling of NaNs to the conservative regridding routine, with a `nan_threshold` keyword argument. For more information see the notebooks on the documentation [#39](https://github.com/EXCITED-CO2/xarray-regrid/pull/39) \& [#41](https://github.com/EXCITED-CO2/xarray-regrid/pull/41).
 - `create_regridding_dataset` as a method of the `xarray_regrid.Grid` dataclass [#41](https://github.com/EXCITED-CO2/xarray-regrid/pull/41).

## v0.2.3 (2024-02-29)

New contributors:
 - [@kjdoore](https://github.com/kjdoore)

Fixed:
 - Ensure all attributes are kept upon regridding (dataset, variable and coordinate attrs) ([#27](https://github.com/EXCITED-CO2/xarray-regrid/pull/27)).
 - The target grid can now have coordinates sorted in decending order, instead of the regridding failing ([#29](https://github.com/EXCITED-CO2/xarray-regrid/pull/29)).
 - Regridding to larger grid now result in NaNs at locations outside of starting data grid ([#33](https://github.com/EXCITED-CO2/xarray-regrid/pull/33)).

Changed:
 - Moved to the Ruff formatter, instead of black ([#27](https://github.com/EXCITED-CO2/xarray-regrid/pull/27)).

## v0.2.2 (2023-11-24)

Added:
 - CITATION.cff file for Zenodo integration.

## v0.2.1 (2023-09-05)

Fixed:
 - Datasets containing NaN values can now be regridded using the conservative method. This previously produced only NaN values.

## v0.2.0 (2023-09-02)

Added:
 - xarray DataArrays are now supported
 - Conservative regridding method, along with a benchmark notebook.
 - A "most common value" regridding method, along with a demo notebook.

Changed:
 - The API has changed. Regridding is now done with `xr.Dataset.regrid.method()`. 
   - E.g. `xr.Dataset.regrid.linear()`


## v0.1.0 (2023-08-17)
First release of `xarray-regrid`, containing:
- the implementation of linear, nearest-neighbor and cubic regridding.
- benchmarks against CDO and xESMF for linear and nearest-neighbor.
