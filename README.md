SYCL-BLAS Implementation
===

[![Build and Test](https://github.com/codeplaysoftware/sycl-blas/actions/workflows/build-and-test.yml/badge.svg?event=push)](https://github.com/codeplaysoftware/sycl-blas/actions/workflows/build-and-test.yml)

SYCL-BLAS implements BLAS - [Basic Linear Algebra Subroutines](https://en.wikipedia.org/wiki/Basic_Linear_Algebra_Subprograms) - using [SYCL 1.2](
https://www.khronos.org/registry/sycl/specs/sycl-1.2.pdf), the
[Khronos](http://www.khronos.org) abstraction layer for [OpenCL](https://www.khronos.org/opencl/).

SYCL-BLAS is a current work in progress research project from an ongoing
collaboration with the *High Performance Computing & Architectures (HPCA) group*
from the Universitat Jaume I [UJI](http://www.hpca.uji.es/).

SYCL-BLAS is written using modern C++. The current implementation uses C++11
features.
See [Roadmap](Roadmap.md) for details on the current status and plans for
the project.

## Table of Contents

  * [Motivation](#motivation)
  * [Basic Concepts](#basic-concepts)
    * [Views](#views)
    * [Operations](#operations)
    * [Executors](#executors)
    * [Interface](#interface)
  * [API description](#api-description)
    * [BLAS 1](#blas-1)
    * [BLAS 2](#blas-2)
    * [BLAS 3](#blas-3)
  * [Requirements](#requirements)
  * [Setup](#setup)
    * [Compile with ComputeCpp](#Compile-with-ComputeCpp)
    * [Compile with DPC++](#Compile-with-DPC++)
    * [Instaling SYCL-BLAS](#Instaling-SYCL-BLAS)
    * [POWER_VR support (ComputeCpp Only)](#POWER_VR-support-(ComputeCpp-Only))
    * [Doxygen](#Doxygen)
    * [How to compile](#how-to-compile)
    * [CMake options](#cmake-options)
    * [Cross-Compile](#cross-compile)
  * [Tests and benchmarks](#tests-and-benchmarks)
  * [Contributing to the project](#contributing-to-the-project)

## Motivation

The same numerical operations are computed to solve many scientific problems
and engineering applications, such as image and signal processing,
telecommunication, computational finance, materials science simulations,
structural biology, data mining, bio-informatics, fluid dynamics, and many other
areas. Thus, it was identified that around the 90% percent of the computational
cost is consumed on the 10% of the code, and therefore any improvement in this
10% of code would have a great impact in the performances of the applications.
Numerical Linear Algebra is the science area in charge of identifying the most
common operations and seeking their best implementation. To do this, the
researchers should consider the numerical stability of the selected algorithm,
and the platform on which the operation will be solved. The first analysis
studies the accuracy of the solution while the second one compares the
performances of the different implementations to select the best one.

Nowadays, all the numerical computations are based on a set of standard
libraries on which the most common operations are implemented. These libraries
are different for dense matrices (BLAS, LAPACK, ScaLAPACK, ...) and for sparse
matrices (SparseBLAS, ...). Moreover, there are  vendor implementations which
are adjusted to the platform features:
  - For multicores: ACML (AMD), ATLAS, Intel-MKL, OpenBLAS, ...
  - For GPUs: cuBLAS (Nvidia), clBLAS, CLBlast, MAGMA, ...

But, in any case, BLAS is always the lowest level in the hierarchy
of numerical libraries, such that
a good BLAS implementation improves the performances of all the other
libraries.  The development of numerical libraries on SYCL is one of the most
important objectives, because it will improve the performance of other SYCL
applications. Obviously, it makes sense SYCL-BLAS was the first step in this
task.

On GPUs, the data communication to/from the device and the grain of the kernels
play an important rule on the performances of the developments. On one
hand, to reduce the communication cost, the most of the data should be mapped
on the device, even the scalars. On the other hand, growing the size of the
kernels allows the CPU to complete other tasks while the GPU is computing or to
enter an energy-efficient C-state, reducing the energy consumption.

To enlarge the grain of the kernels is a complex task, in which many aspects
should be considered as the dependency between kernels, the grid topology, the
grid sizes, etc. This complexity justifies that, usually, the fused kernels are
manually written. An alternative to simplify this task could be to build a
expression tree on which all the single operation which are required to solve a
problem appears. This structure could be analysed by the compiler to decide how
to merge the different kernel and the best grid topology to execute the fused
kernel.  The use of expression trees is one of most important features of
SYCL-BLAS.

## Basic Concepts

SYCL-BLAS uses C++ Expression Tree templates to generate SYCL Kernels via
kernel composition.
Expression Tree templates are a widely used technique to implement expressions
on C++, that facilitate development and composition of operations.
In particular,
[Kernel composition in SYCL](http://dl.acm.org/citation.cfm?id=2791332) has
been used in various projects to create efficient domain-specific embedded
languages that enable users to easily fuse GPU kernels.

SYCL-BLAS can be used
- either as a header-only framework by including `sycl_blas.hpp` in
an application and passing the `src` folder in the list of include directories
- or as a library by including `sycl_blas.h` in an application.

All the relevant files can be found in
the `include` directory.

There are four components in SYCL-BLAS, the *View*, the *Operations*,
the *Executors* and the *Interface* itself.

### Views

The input data to all the operations in SYCL-BLAS is passed to the library
using *Views*.
A *View* represents data on top of a container, passed by reference.
Views *do not store data*, they only map a visualization of the data on top
of a container.
This enables the library to implement the different indexing modes of the
BLAS API, such as strides.
Note than a view can be of a different size than a container.

All views derive from the base view class or the base matrix view class, which
represents a view of a container as a vector or as a matrix.
The container does not need to be multi-dimensional to store a matrix.
The current restriction is that container must obey the 
[*LegacyRandomAccessIterator*](https://en.cppreference.com/w/cpp/named_req/RandomAccessIterator)
properties of the C++11 standard.

### Operations

Operations among elements of vectors (or matrices) are expressed in the
set of Operation Classes.
Operations are templated classes that take templated types as input.
Operations form the nodes of the SYCL-BLAS expression tree.
Refer to the documentation of each node type for details.

Composing these is how the compile-time Expression tree is created:
Given an operation node, the leaves of the node are other Operations.
The leaf nodes of an Expression Tree are Views or Scalar types (data).
The intermediate nodes of the Expression Tree are operations (e.g,
binary operations, unary operations, etc).

### Executors

An executor traverses the Expression Tree to evaluate the operations that it
defines.
Executors use different techniques to evaluate the expression tree.
The basic C++ executor performs a for loop on the size of the data and calls
the evaluation function on each item.

The SYCL evaluator transform the tree into a device tree (i.e, converting
buffer to accessors) and then evaluates the Expression Tree on the device.

### Interface

The different headers on the interface directory implement the traditional
BLAS interface.
Files are organised per BLAS level (1, 2, 3).

When the SYCL-BLAS BLAS interface is called, the Expression Tree for each
operation is constructed, and then executed.
Some API calls may execute several kernels (e.g, when a reduction is required).
The expression trees in the API allow to compile-time fuse operations.

Note that, although this library features a BLAS interface, users are allowed
to directly compose their own expression trees to compose multiple operations.
The CG example shows an implementation of the Conjugate Gradient that uses
various expression tree to demonstrate how to achieve compile-time kernel fusion
of multiple BLAS operations.

## API description

This section references all the supported operations and their interface.

All operations take as their first argument a reference to the executor, a
`blas::Executor` created with a `sycl::queue`. The return value is usually an
array of SYCL events (except for some operations that can return a scalar or
a tuple). The containers for the vectors and matrices (and scalars written by
the BLAS operations) are iterator buffers that can be created with
`make_sycl_iterator_buffer`.

We recommend checking the [samples](samples) to get started with SYCL-BLAS. It
is better to be familiar with BLAS:

- [Wikipedia](https://en.wikipedia.org/wiki/Basic_Linear_Algebra_Subprograms)
- [Netlib reference](http://www.netlib.org/lapack/explore-html/d1/df9/group__blas.html)

### BLAS 1

The following table sums up the interface that can be found in
[blas1_interface.h](include/interface/blas1_interface.h).

For all these operations:

* `vx` and `vy` are containers for vectors `x` and `y`.
* `incx` and `incy` are their increments (number of steps to jump to the next
   value, 1 for contiguous values).
* `N`, an integer, is the size of the vectors (less than or equal to the size of
  the containers).
* `alpha` is a scalar.
* `rs` is a container of size 1, containing either a scalar, an integer, or an
  index-value tuple.
* `c` and `s` for `_rot` are scalars (cosine and sine)

| operation | arguments | description |
|---|---|---|
| `_axpy` | `ex`, `N`, `alpha`, `vx`, `incx`, `vy`, `incy` | Vector multiply-add: `y = alpha * x + y` |
| `_copy`  | `ex`, `N`, `vx`, `incx`, `vy`, `incy` | Copies a vector to another: `y = x` |
| `_dot` | `ex`, `N`, `vx`, `incx`, `vy`, `incy` [, `rs`] | Dot product of two vectors `x` and `y`; written in `rs` if passed, else returned |
| `_asum` | `ex`, `N`, `vx`, `incx` [, `rs`] | Absolute sum of the vector `x`; written in `rs` if passed, else returned |
| `_iamax` | `ex`, `N`, `vx`, `incx` [, `rs`] | First index and value of the maximum element of `x`; written in `rs` if passed, else the index only is returned |
| `_iamin` | `ex`, `N`, `vx`, `incx` [, `rs`] | First index and value of the minimum element of `x`; written in `rs` if passed, else the index only is returned |
| `_swap`  | `ex`, `N`, `vx`, `incx`, `vy`, `incy` | Interchanges two vectors: `y = x` and `x = y` |
| `_scal` | `ex`, `N`, `alpha`, `vx`, `incx` | Scalar product of a vector: `x = alpha * x` |
| `_nrm2` | `ex`, `N`, `vx`, `incx` [, `rs`] | Euclidean norm of the vector `x`; written in `rs` if passed, else returned |
| `_rot` | `ex`, `N`, `vx`, `incx`, `vy`, `incy`, `c`, `s` | Applies a plane rotation to `x` and `y` with a cosine `c` and a sine `s`  |

### BLAS 2

The following table sums up the interface that can be found in
[blas2_interface.h](include/interface/blas2_interface.h).

For all these operations:

* `trans` is a `char` representing the transpose mode of the matrix: `'n'`,
  `'t'`, or `'c'`; respectively identity, transpose and Hermitian transpose
  (note: the latter is not relevant yet as complex numbers are not supported).
* `uplo` is a `char` that provides information about triangular matrices: `u` for
  upper triangular and `l` for lower triangular matrices.
* `diag` is a `char` that provides information about the diagonal elements of a
  triangular matrix: `u` if the matrix is unit triangular (all diagonal elements
  are 1), else `n`.
* `M` and `N` are the numbers of rows and columns of the matrix. They also
  determine the sizes of the vectors so that dimensions match, depending on the
  BLAS operation. For operations on square matrices, only `N` is given.
* `alpha` and `beta` are scalars.
* `mA` is a container for a column-major matrix `A`.
* `lda` is the leading dimension of `mA`, i.e the step between an element and
  its neighbor in the next column and same row. `lda` must be at least `M`.
* `vx` and `vy` are containers for vectors `x` and `y`.
* `incx` and `incy` are their increments (cf BLAS 1).

| operation | arguments | description |
|---|---|---|
| `_gemv` | `ex`, `trans`, `M`, `N`, `alpha`, `mA`, `lda`, `vx`, `incx`, `beta`, `vy`, `incy`  | Generalised matrix-vector product followed by a vector sum: `y = alpha * A * x + beta * y`. *Note: the dimensions of the vectors depend on the transpose mode (`x`: `N` and `y`: `M` for mode `'n'` ; `x`: `M` and `y`: `N` otherwise)* |
| `_trmv`  | `ex`, `uplo`, `trans`, `diag`, `N`, `alpha`, `mA`, `lda`, `vx`, `incx` | Matrix-vector product for a triangular matrix: `x = A * x` |
| `_symv` | `ex`, `uplo`, `N`, `alpha`, `mA`, `lda`, `vx`, `incx`, `beta`, `vy`, `incy` | Variant of GEMV for a symmetric matrix (`y = alpha * A * x + beta * y`). *Note: `uplo` specifies which side of the matrix will be read* |
| `_ger` | `ex`, `M`, `N`, `alpha`, `vx`, `incx`, `vy`, `incy`, `mA`, `lda` | Generalised vector-vector product followed by a matrix sum: `A = alpha * x * yT + A` |
| `_syr` | `ex`, `uplo`, `N`, `alpha`, `vx`, `incx`, `mA`, `lda` | Generalised vector squaring followed by a sum with a symmetric matrix: `A = alpha * x * xT + A` |
| `_syr2` | `ex`, `uplo`, `N`, `alpha`, `vx`, `incx`, `vy`, `incy`, `mA`, `lda` | Generalised vector products followed by a sum with a symmetric matrix: `A = alpha*x*yT + alpha*y*xT + A` |

### BLAS 3

The following table sums up the interface that can be found in
[blas3_interface.h](include/interface/blas3_interface.h).

For all these operations:

* `A`, `B` and `C` are containers for the column-major matrices A, B and C.
* `lda`, `ldb` and `ldc` are the leading dimensions of the matrices A, B and C
  (cf BLAS 2). The leading dimension of a matrix must be greater than or equal
  to its number of rows.
* `transa` and `transb` are the transpose modes of the matrices A and B
  (cf BLAS 2).
* `M`, `N` and `K` are the dimensions of the matrices. The dimensions
  **after transposition** are A: `M`x`K`, B: `K`x`N`, C: `M`x`N`.
* `alpha` and `beta` are scalars.
* `batch_size` is an integer.
* `side` is `l` for left or `r` for right.
* `uplo` is a `char` that provides information about triangular matrices: `u` for
  upper triangular and `l` for lower triangular matrices.
* `diag` is a `char` that provides information about the diagonal elements of a
  triangular matrix: `u` if the matrix is unit triangular (all diagonal elements
  are 1), else `n`.

| operation | arguments | description |
|---|---|---|
| `_gemm` | `ex`, `transa`, `transb`, `M`, `N`, `K`, `alpha`, `A`, `lda`, `B`, `ldb`, `beta`, `C`, `ldc` | Generalised matrix-matrix multiplication followed by matrix addition: `C = alpha * A * B + beta * C` |
| `_gemm_batched` | `ex`, `transa`, `transb`, `M`, `N`, `K`, `alpha`, `A`, `lda`, `B`, `ldb`, `beta`, `C`, `ldc`, `batch_size` | Same as `_gemm` but the containers contain `batch_size` end-to-end matrices. GEMM operations are performed independently with matching matrices. |
| `_trsm` | `ex`, `side`, `uplo`, `trans`, `diag`, `M`, `N`, `alpha`, `A`, `lda`, `B`, `ldb` | Triangular solve with Multiple Right-Hand Sides. |

## Requirements

SYCL-BLAS is designed to work with any SYCL 1.2.1 implementation.
We do not use any OpenCL interoperability, hence, the code is pure C++.
The project is developed using [ComputeCpp CE Edition](http://www.computecpp.com)
using Ubuntu 16.04 on Intel OpenCL CPU and Intel GPU.
In order to build the sources, GCC 5.4 or higher is required.
The build system is CMake version 3.4.2 or higher.
We rely on the `FindComputeCpp.cmake` imported from the 
[ComputeCpp SDK](https://github.com/codeplaysoftware/computecpp-sdk)
to build the project.

A BLAS library, such as [OpenBLAS](https://github.com/xianyi/OpenBLAS), is also
required to build and verify the test results. 
Instructions for building and installing
OpenBLAS can be found [on this page](https://github.com/xianyi/OpenBLAS/wiki/User-Manual). 
Please note that although some distributions may provide packages for OpenBLAS 
these versions are typically quite old and may have issues with the TRMV implementation 
which can cause random test failures. Any version of OpenBLAS `>= 0.3.0` will not suffer
from these issues.

When using OpenBLAS or any other BLAS library the installation directory must be
added to the `CMAKE_PREFIX_PATH` when building SYCL-BLAS (see
[below](###cmake-options)).

## Setup

1. Clone the SYCL-BLAS repository, making sure to pass the `--recursive` option, in order to clone submodule(s), such as the computecpp-sdk.
2. Create a build directory
3. Run `CMake` from the build directory (see options in the section below):

### Compile with ComputeCpp
```bash
cd build
cmake -GNinja ../ -DComputeCpp_DIR=/path/to/computecpp -DSYCL_COMPILER=computecpp
ninja
```

### Compile with DPC++
```bash
export CC=[path/to/intel/clang]
export CXX=[path/to/intel/clang++]
cd build
cmake -GNinja ../ -DSYCL_COMPILER=dpcpp
ninja
```
The target triplet can be set by adding `-DDPCPP_SYCL_TARGET=<triplet>`. If it
is not set, the default values is `spir64_x86_64-unknown-unknown`, which targets
the CPU. Other possible triplets targeting GPUs are: `spir64-unknown-unknown`
(Intel), `nvptx64-nvidia-cuda` (Nvidia), and `amdgcn-amd-amdhsa` (AMD).

If building for NVIDIA or AMD devices, it is advisable to specify the actual
device architecture by means of `-DDPCPP_SYCL_ARCH=<arch>`,
e.g., `<arch>` can be `sm_80` for NVIDIA or `gfx908` for AMD.
*Note:* specifying the architecture is required in case of AMD devices.

### Compile with hipSYCL
```bash
cd build
cmake -GNinja ../ -DhipSYCL_DIR=/path/to/hipSYCL/install/lib/cmake/hipSYCL -DSYCL_COMPILER=hipsycl
ninja
```
To build for other than the default devices (`omp`), set the `HIPSYCL_TARGETS` environment variable or specify `-DHIPSYCL_TARGETS` as [documented](https://github.com/illuhad/hipSYCL/blob/develop/doc/using-hipsycl.md).

### Instaling SYCL-BLAS
To install the SYCL-BLAS library (see `CMAKE_INSTALL_PREFIX` below)

```bash
ninja install
```
### POWER_VR support (ComputeCpp Only)

To enable the PowerVR backend, pass: `-DTARGET=POWER_VR`

To use the neural network library from Imagination, pass: `-DIMGDNN_DIR=path/to/library`

### Doxygen

Doxygen documentation can be generated by running:

```bash
doxygen doc/Doxyfile
```

### CMake options

CMake options are given using `-D` immediately followed by the option name, the
symbol `=` and a value (`ON` and `OFF` can be used for boolean options and are
equivalent to 1 and 0). Example: `-DBLAS_ENABLE_TESTING=OFF`

Some of the supported options are:

| name | value | description |
|---|---|---|
| `BLAS_ENABLE_TESTING` | `ON`/`OFF` | Set it to `OFF` to avoid building the tests (`ON` is the default value) |
| `BLAS_ENABLE_BENCHMARK` | `ON`/`OFF` | Set it to `OFF` to avoid building the benchmarks (`ON` is the default value) |
| `SYCL_COMPILER` | name | Used to determine which SYCL implementation to use. By default, the first implementation found is used. Supported values are: `computecpp`, `dpcpp` and `hipsycl`. |
| `TUNING_TARGET` | name | By default SYCL-BLAS library is tuned for CPU. Use that flag to tune it for a target (**highly recommended** for performance). The supported targets are: `INTEL_GPU`, `AMD_GPU`, `ARM_GPU`, `RCAR` |
| `CMAKE_PREFIX_PATH` | path | List of paths to check when searching for dependencies |
| `CMAKE_INSTALL_PREFIX` | path | Specify the install location, used when invoking `ninja install` |
| `BUILD_SHARED_LIBS` | `ON`/`OFF` | Build as shared library (`ON` by default) |
| `ENABLE_EXPRESSION_TESTS` | `ON`/`OFF` | Build additional tests that use the header-only framework (e.g to test expression trees); `OFF` by default |
| `BLAS_VERIFY_BENCHMARK` | `ON`/`OFF` | Verify the results of the benchmarks instead of only measuring the performance. See the documentation of the benchmarks for more details. `ON` by default |
| `BLAS_MODEL_OPTIMIZATION` | name | Pass a model name here to use optimized GEMM configurations for specific convolution models/sizes. Currently this only affects the `ARM_GPU` target. The supported models are: `RESNET_50`, `VGG_16` |
| `BLAS_ENABLE_CONST_INPUT` | `ON`/`OFF` | Determines whether to enable kernel instantiation with const input buffer (`ON` by default) |
| `BLAS_ENABLE_EXTENSIONS` | `ON`/`OFF` | Determines whether to enable sycl-blas extensions (`ON` by default) |
| `BLAS_DATA_TYPES` | `half;float;double` | Determines the floating-point types to instantiate BLAS operations for. Default is `float` |
| `BLAS_INDEX_TYPES` | `int32_t;int64_t` | Determines the type(s) to use for `index_t` and `increment_t`. Default is `int` |


### Cross-Compile (ComputeCpp Only)

To cross-compile SYCL-BLAS first the following environment variables must be
set:

```bash
export COMPUTECPP_TOOLCHAIN_DIR="PATH TO TOOLCHAIN_DIR"
export COMPUTECPP_TARGET_TRIPLE="PATH TO TARGET_TRIPLE"
export COMPUTECPP_SYSROOT_DIR="$PATH TO SYSROOT_DIR"
```
The following CMake command can be used to cross-compile SYCL-BLAS:

```bash
cmake  -GNinja                                                                                           \
    ${SOURCE_ROOT}                                                                                       \
   -DCMAKE_PREFIX_PATH="${OPENBLAS_PATH}"                                                                 \
   -DComputeCpp_DIR="${COMPUTECPP_DEVICE_PATH}"                                                          \
   -DComputeCpp_HOST_DIR="${COMPUTECPP_X86_PATH}"                                                        \
   -DCMAKE_TOOLCHAIN_FILE="${SYCL_BLAS_PATH}/external/computecpp-sdk/cmake/toolchains/gcc-generic.cmake" \
   -DCMAKE_BUILD_TYPE='Release'                                                                          \
   -DCMAKE_INSTALL_PREFIX=${CROSS_COMPILED_SYCLBLAS_INSTALL}                                             \
   -DOpenCL_INCLUDE_DIR="${OpenCL_Headers_PATH}"                                                         \
   -DOpenCL_LIBRARY="${OpenCL_LIBRARY}"                                                                  \
   -DCOMPUTECPP_BITCODE="${DEVICE_BITCODE}"                                                              \
   -DCMAKE_CXX_FLAGS='-O3'                                                                               \
   -DTARGET="${CHOSEN_TARGET}"
```


## Tests and benchmarks

The tests and benchmarks have their own documentation:

- [Documentation of the tests](test/README.md)
- [Documentation of the benchmarks](benchmark/README.md)


## Contributing to the project

SYCL-BLAS is an Open Source project maintained by the HPCA group and
Codeplay Software Ltd.
Feel free to create an issue on the Github tracker to request features or
report bugs.

### Guides and Other Documents

- [How to add a new operation](doc/AddingBlas3Op.md)
- [Autotuner Developer Guide](doc/Autotuner.md)
