SYCL-BLAS samples
===

## How to compile the samples

At the moment any project using SYCL-BLAS requires:

* OpenCL
* [ComputeCpp](http://www.computecpp.com)
* The [ComputeCpp-SDK](https://github.com/codeplaysoftware/computecpp-sdk)
  headers (imported in this repo as a submodule, make sure to clone with the
  `--recursive` option).
* SYCL-BLAS, either:
  * as a library (install the library and include `sycl_blas.h` in an application)
  * as a header-only framework (include `sycl_blas.hpp` in an application)

### With CMake

This folder includes a basic CMake configuration file. Set the `ComputeCpp_DIR`
option to your ComputeCpp root path, and `SyclBLAS_DIR` to your SYCL-BLAS root
path.

```bash
mkdir build
cd build
cmake .. -GNinja -DComputeCpp_DIR=/path/to/computecpp \
                 -DSyclBLAS_DIR=~/path/to/syclblas
ninja
```
