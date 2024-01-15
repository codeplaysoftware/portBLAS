portBLAS samples
===

## How to compile the samples
A SYCL Compiler (DPCPP, hipSYCL or ComputeCpp) along with the target device's 
relevant compute drivers *(OpenCL, CUDA etc..)* are required to compile and 
run the samples.   
Any project that integrates portBLAS can either use it as :
  * a library *(install the library and include `portblas.h` in an application)*
  * a header-only framework *(include `portblas.hpp` in an application)*

### CMake Configuration

This folder contains a basic CMake configuration file and a module to find
portBLAS *(which will be used as a header-only framework)*. It also uses a module
to find the SYCL Compiler(DPCPP, hipSYCL or computeCpp *-deprecated-*) that is 
located in the folder `cmake/Modules`.

Sample usage with DPCPP Compiler: 

```bash
mkdir build && cd build
export CXX=/path/to/dpcpp/clang++
export CC=/path/to/dpcpp/clang
cmake .. -GNinja -DPORTBLAS_DIR=~/path/to/portblas
ninja
```
