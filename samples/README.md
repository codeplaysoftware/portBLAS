portBLAS samples
===

## How to compile the samples

At the moment any project using portBLAS requires:

* OpenCL
* [ComputeCpp](http://www.computecpp.com)
* portBLAS, either:
  * as a library (install the library and include `portblas.h` in an application)
  * as a header-only framework (include `portblas.hpp` in an application)

### With CMake

This folder contains a basic CMake configuration file and a module to find
portBLAS (which will be used as a header-only framework). It also uses a module
to find ComputeCpp that is located in the folder `cmake/Modules`.

Usage:

* set `ComputeCpp_DIR` to your ComputeCpp root path
* set `PORTBLAS_DIR` to your portBLAS root path

```bash
mkdir build
cd build
cmake .. -GNinja -DComputeCpp_DIR=/path/to/computecpp \
                 -DPORTBLAS_DIR=~/path/to/portblas
ninja
```
