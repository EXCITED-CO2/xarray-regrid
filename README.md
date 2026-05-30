# xarray-regrid: Regridding utilities for xarray.

<img align="right" width="100" alt="Logo" src="./docs/assets/logo.png">


With xarray-regrid you can regrid between rectilinear grids, and conservatively onto curvilinear grids, unstructured meshes, or arbitrary polygons (e.g. country shapes). The following methods are supported:
 - Linear
 - Nearest-neighbor
 - Conservative
 - Conservative 2D: conservative regridding for grids that aren't 1D-separable — curvilinear coordinates, unstructured meshes, and grid-to-polygon aggregation
 - Cubic
 - "Most common value", as well as other zonal statistics (e.g., variance or median).

All regridding methods can operate lazily on [Dask arrays](https://docs.xarray.dev/en/latest/user-guide/dask.html).

Note that "Most common value" is designed to regrid categorical data to a coarse resolution. For regridding categorical data to a finer resolution, please use "nearest-neighbor" regridder.

[![PyPI](https://img.shields.io/pypi/v/xarray-regrid.svg?style=flat)](https://pypi.python.org/pypi/xarray-regrid/)
[![conda-forge](https://anaconda.org/conda-forge/xarray-regrid/badges/version.svg)](https://anaconda.org/conda-forge/xarray-regrid)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.10203304.svg)](https://doi.org/10.5281/zenodo.10203304)
[![Docs](https://readthedocs.org/projects/xarray-regrid/badge/?version=latest&style=flat)](https://xarray-regrid.readthedocs.org/)

## Why xarray-regrid?

Regridding is a common operation in earth science and other fields. While xarray does have some interpolation methods available, these are not always straightforward to use. Additionally, methods such as conservative regridding, or taking the most common value, are not available in xarray.

## Installation

For a minimal install:
```console
pip install xarray-regrid
```

*Note: xarray-regrid is also [available on conda-forge](https://anaconda.org/conda-forge/xarray-regrid).*

To improve performance in certain cases:
```console
pip install xarray-regrid[accel]
```

which includes optional extras such as:
 - `dask`: parallelization over chunked data
 - `sparse`: for performing conservative regridding using sparse weight matrices
 - `opt-einsum`: optimized einsum routines used in conservative regridding

Benchmarking varies across different hardware specifications, but the inclusion of these extras can often provide significant speedups.

For conservative regridding onto curvilinear grids, unstructured meshes, or polygons (the `conservative_2d` method), install the `conservative-2d` extra:
```console
pip install xarray-regrid[conservative-2d]
```
which adds `shapely`, `sparse`, and `h5netcdf` (the last for caching weight matrices to netCDF).

## Usage
The xarray-regrid routines are accessed using the "regrid" accessor on an xarray Dataset:
```py
import xarray_regrid

ds = xr.open_dataset("input_data.nc")
ds_grid = xr.open_dataset("target_grid.nc")

ds.regrid.linear(ds_grid)
```

For examples, see the benchmark notebooks and the demo notebooks.

## Benchmarks
The benchmark notebooks contain comparisons to more standard methods (CDO, xESMF).

To be able to run the notebooks, a conda environment is required (due to ESMF and CDO).
You can install this environment using the `environment.yml` file in this repository.
[Micromamba](https://mamba.readthedocs.io/en/latest/user_guide/micromamba.html) is a lightweight version of the much faster "mamba" conda alternative.

```sh
micromamba create -n environment_name -f environment.yml
```

## Acknowledgements

This package was developed under Netherlands eScience Center grant [NLESC.OEC.2022.017](https://research-software-directory.org/projects/excited).
