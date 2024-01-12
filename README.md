# portBLAS Implementation
===

[![Build and Test](https://github.com/codeplaysoftware/portBLAS/actions/workflows/build-and-test.yml/badge.svg?event=push)](https://github.com/codeplaysoftware/portBLAS/actions/workflows/build-and-test.yml)

portBLAS implements BLAS - [Basic Linear Algebra Subroutines](https://en.wikipedia.org/wiki/Basic_Linear_Algebra_Subprograms) - using [SYCL](https://www.khronos.org/sycl/).

portBLAS is an ongoing collaboration with the *High Performance Computing 
& Architectures (HPCA) group* from the Universitat Jaume I [UJI](http://www.hpca.uji.es/).

portBLAS is written using modern C++. The current implementation uses C++11
features.
See [Roadmap](Roadmap.md) for details on the current status and plans for
the project.

## Table of Contents

- [portBLAS Implementation](#portBLAS-implementation)
  - [Table of Contents](#table-of-contents)
  - [Motivation](#motivation)
  - [Basic Concepts](#basic-concepts)
    - [Views](#views)
    - [Operations](#operations)
    - [SB\_Handle](#sb_handle)
    - [Interface](#interface)
  - [API description](#api-description)
    - [BLAS 1](#blas-1)
    - [BLAS 2](#blas-2)
    - [BLAS 3](#blas-3)
    - [Experimental Joint Matrix Support](#jm_support)
  - [Requirements](#requirements)
  - [Setup](#setup)
    - [Compile with DPC++](#compile-with-dpc)
    - [Compile with hipSYCL](#compile-with-hipsycl)
    - [Instaling portBLAS](#instaling-portBLAS)
    - [Doxygen](#doxygen)
    - [CMake options](#cmake-options)
    - [ComputeCpp Compilation (Deprecated)](#computecpp-deprecated)
      - [Compile with ComputeCpp](#compile-with-computecpp)
      - [POWER\_VR support](#power_vr-support-computecpp-only)
      - [Cross-Compile](#cross-compile-computecpp-only)
  - [Tests and benchmarks](#tests-and-benchmarks)
  - [Contributing to the project](#contributing-to-the-project)
    - [Guides and Other Documents](#guides-and-other-documents)

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
applications. Obviously, it makes sense portBLAS was the first step in this
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
portBLAS.

## Basic Concepts

portBLAS uses C++ Expression Tree templates to generate SYCL Kernels via
kernel composition.
Expression Tree templates are a widely used technique to implement expressions
on C++, that facilitate development and composition of operations.
In particular,
[Kernel composition in SYCL](http://dl.acm.org/citation.cfm?id=2791332) has
been used in various projects to create efficient domain-specific embedded
languages that enable users to easily fuse GPU kernels.

portBLAS can be used
- either as a header-only framework by including `portblas.hpp` in
an application and passing the `src` folder in the list of include directories
- or as a library by including `portblas.h` in an application.

All the relevant files can be found in
the `include` directory.

There are four components in portBLAS, the *View*, the *Operations*,
the *SB_Handle* and the *Interface* itself.

### Views

The input data to all the operations in portBLAS is passed to the library
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
Operations form the nodes of the portBLAS expression tree.
Refer to the documentation of each node type for details.

Composing these is how the compile-time Expression tree is created:
Given an operation node, the leaves of the node are other Operations.
The leaf nodes of an Expression Tree are Views or Scalar types (data).
The intermediate nodes of the Expression Tree are operations (e.g,
binary operations, unary operations, etc).

### SB_Handle

An SB_Handle traverses the Expression Tree to evaluate the operations that it
defines.
SB_Handle use different techniques to evaluate the expression tree.
The SYCL evaluator transform the tree into a device tree (i.e, converting
buffer to accessors) and then evaluates the Expression Tree on the device.

### Interface

The different headers on the interface directory implement the traditional
BLAS interface.
Files are organised per BLAS level (1, 2, 3).

When the portBLAS BLAS interface is called, the Expression Tree for each
operation is constructed, and then executed.
Some API calls may execute several kernels (e.g, when a reduction is required).
The expression trees in the API allow to compile-time fuse operations.

Note that, although this library features a BLAS interface, users are allowed
to directly compose their own expression trees to compose multiple operations.
The CG example shows an implementation of the Conjugate Gradient that uses
various expression tree to demonstrate how to achieve compile-time kernel fusion
of multiple BLAS operations.

## API description

This section references all the supported operations and their interface. The 
library follows the [oneAPI MKL BLAS specification](https://spec.oneapi.io/versions/latest/elements/oneMKL/source/domains/blas/blas.html) 
as reference for the api. We have support for both USM and Buffer api, however 
the group apis for USM are not supported. We don't support mixing USM and Buffer 
arguments together to compile the library, and instead stick to the aformentioned 
reference specification.

All operations take as their first argument a reference to the SB_Handle, a
`blas::SB_Handle` created with a `sycl::queue`. The last argument for all operators
is a vector of dependencies of type `cl::sycl::event` (empty by default). The return value 
is usually an array of SYCL events (except for some operations that can return a scalar or
a tuple). The containers for the vectors and matrices (and scalars written by
the BLAS operations) can either be `raw usm pointers` or `iterator buffers` that can be 
created with a call to `cl::sycl::malloc_device` or `make_sycl_iterator_buffer` respectively.

The USM support in portBLAS is limited to `device allocated` memory only and we don't support
`shared` or `host` allocations with USM. 

We recommend checking the [samples](samples) to get started with portBLAS. It
is better to be familiar with BLAS:

- [Wikipedia](https://en.wikipedia.org/wiki/Basic_Linear_Algebra_Subprograms)
- [Netlib reference](http://www.netlib.org/lapack/explore-html/d1/df9/group__blas.html)

### BLAS 1

The following table sums up the interface that can be found in
[blas1_interface.h](include/interface/blas1_interface.h).

For all these operations:

* `vx` and `vy` are containers for vectors `x` and `y`.
* `incx` and `incy` are their increments *(number of steps to jump to the next
   value, 1 for contiguous values)*.
* `N`, an integer, is the size of the vectors *(less than or equal to the size of
  the containers)*.
* `alpha` is a scalar.
* `rs` is a container of size 1, containing either a scalar, an integer, or an
  index-value tuple.
* `c` and `s` for `_rot` are scalars *(cosine and sine)*.
* `sb` for `_sdsdot` is a single precision scalar to be added to output.

| operation | arguments                                       | description                                                                                                                                                                  |
|-----------|-------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `_asum`   | `sb_handle`, `N`, `vx`, `incx` [, `rs`]                | Absolute sum of the vector `x`; written in `rs` if passed, else returned                                                                                                                               |
| `_axpy`   | `sb_handle`, `N`, `alpha`, `vx`, `incx`, `vy`, `incy`  | Vector multiply-add: `y = alpha * x + y`                                                                                                                                     |
| `_copy`   | `sb_handle`, `N`, `vx`, `incx`, `vy`, `incy`           | Copies a vector to another: `y = x`                                                                                                                                          |
| `_dot`    | `sb_handle`, `N`, `vx`, `incx`, `vy`, `incy`   [, `rs`] | Dot product of two vectors `x` and `y`; written to `rs` if passed, else returned                                                                                             |
| `_sdsdot`    | `sb_handle`, `N`, `sb`, `vx`, `incx`, `vy`, `incy`[, `rs`] | Compute sum of a constant `sb` with the double precision  dot product of two single precision vectors `x` and `y`; written in `rs` if passed, else returned                                                                                            |
| `_nrm2`   | `sb_handle`, `N`, `vx`, `incx` [, `rs`]                | Euclidean norm of the vector `x`; written in `rs` if passed, else returned                                                                                                   |
| `_rot`    | `sb_handle`, `N`, `vx`, `incx`, `vy`, `incy`, `c`, `s` | Applies a plane rotation to `x` and `y` with a cosine `c` and a sine `s`                                                                                                     |
| `_rotg`   | `sb_handle`, `a`, `b`, `c`, `s`                        | Given the Cartesian coordinates (`a`, `b`) of a point, return the parameters `c`, `s`, `r`, and `z` associated with the Givens rotation.                                     |
| `_rotm`   | `sb_handle`, `N`, `vx`, `incx`, `vy`, `incy`, `param`  | Applies a modified Givens rotation to `x` and `y`.                                                                                                                           |
| `_rotmg`  | `sb_handle`, `d1`, `d2`, `x1`, `y1` `param`            | Given the Cartesian coordinates (`x1`, `y1`) of a point, return the components of a modified Givens transformation matrix that zeros the y-component of the resulting point. |
| `_scal`   | `sb_handle`, `N`, `alpha`, `vx`, `incx`                | Scalar product of a vector: `x = alpha * x`                                                                                                                                  |
| `_swap`   | `sb_handle`, `N`, `vx`, `incx`, `vy`, `incy`           | Interchanges two vectors: `y = x` and `x = y`                                                                                                                                |
| `_iamax`  | `sb_handle`, `N`, `vx`, `incx` [, `rs`]                | Index of the first occurence of the maximum element in `x`; written to `rs` if passed, else returned.                                                              |
| `_iamin`  | `sb_handle`, `N`, `vx`, `incx` [, `rs`]                | Index of the first occurence of the minimum element in `x`; written to `rs` if passed, else returned.                                                            |

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
* `K` Number of sub/super-diagonals of the matrix.

| operation | arguments | description |
|---|---|---|
| `_gbmv` | `sb_handle`, `trans`, `M`, `N`, `KL`, `KU`, `alpha`, `mA`, `lda`, `vx`, `incx`, `beta`, `vy`, `incy`  | Generalised band matrix-vector product followed by a vector sum: `y = alpha * A * x + beta * y`. *Note: the dimensions of the vectors depend on the transpose mode (`x`: `N` and `y`: `M` for mode `'n'` ; `x`: `M` and `y`: `N` otherwise)* |
| `_gemv` | `sb_handle`, `trans`, `M`, `N`, `alpha`, `mA`, `lda`, `vx`, `incx`, `beta`, `vy`, `incy`  | Generalised matrix-vector product followed by a vector sum: `y = alpha * A * x + beta * y`. *Note: the dimensions of the vectors depend on the transpose mode (`x`: `N` and `y`: `M` for mode `'n'` ; `x`: `M` and `y`: `N` otherwise)* |
| `_ger` | `sb_handle`, `M`, `N`, `alpha`, `vx`, `incx`, `vy`, `incy`, `mA`, `lda` | Generalised vector-vector product followed by a matrix sum: `A = alpha * x * yT + A` |
| `_sbmv`| `sb_handle`, `uplo`, `alpha`, `mA`, `lda`, `vx`, `incx`, `beta`, `vy`, `incy` | Compute a scalar-matrix-vector product and add the result to a scalar-vector product, with a symmetric band matrix: `y = alpha * mA * x + beta * y` |
| `_spmv` | `sb_handle`, `uplo`, `N`, `alpha`, `mA`, `vx`, `incx`, `beta`, `vy`, `incy` |  Symmetric packed matrix-vector product: `y = alpha * A * x + beta * y` |
| `_spr` | `sb_handle`, `uplo`, `N`, `alpha`, `vx`, `incx`, `mPA` | Symmetric vector-vector product followed by a matrix sum: `mPA = alpha * x * xT + mPA` |
| `_spr2` | `sb_handle`, `uplo`, `N`, `alpha`, `vx`, `incx`, `vy`, `incy`, `mPA` | Compute two scalar-vector-vector products and add them to a symmetric packed matrix: `mPA = alpha * x * yT + alpha * y * xT + mPA` |
| `_symv` | `sb_handle`, `uplo`, `N`, `alpha`, `mA`, `lda`, `vx`, `incx`, `beta`, `vy`, `incy` | Variant of GEMV for a symmetric matrix (`y = alpha * A * x + beta * y`). *Note: `uplo` specifies which side of the matrix will be read* |
| `_syr` | `sb_handle`, `uplo`, `N`, `alpha`, `vx`, `incx`, `mA`, `lda` | Generalised vector squaring followed by a sum with a symmetric matrix: `A = alpha * x * xT + A` |
| `_syr2` | `sb_handle`, `uplo`, `N`, `alpha`, `vx`, `incx`, `vy`, `incy`, `mA`, `lda` | Generalised vector products followed by a sum with a symmetric matrix: `A = alpha*x*yT + alpha*y*xT + A` |
| `_tbmv` | `sb_handle`, `uplo`, `trans`, `diag`, `N`, `K`, `mA`, `lda`, `vx`, `incx` |  Compute a matrix-vector product with a triangular band matrix:  `A = A * x` |
| `_tbsv` | `sb_handle`, `uplo`, `trans`, `diag`, `N`, `K`, `mA`, `lda`, `vx`, `incx` | Solve a system of linear equations whose coefficients are in a triangular band matrix: `A * x = b` |
| `_tpmv`| `sb_handle`, `uplo`, `trans`, `diag`, `N`, `mA`, `vx`, `incx` | Triangular packed matrix-vector product: `x = A * x` |
| `_tpsv` | `sb_handle`, `uplo`, `trans`, `diag`, `N`, `mA`, `vx`, `incx` | Solve a system of linear equations whose coefficients are in a triangular packed matrix: `A * x = b` |
| `_trmv`  | `sb_handle`, `uplo`, `trans`, `diag`, `N`, `alpha`, `mA`, `lda`, `vx`, `incx` | Matrix-vector product for a triangular matrix: `x = A * x` |
| `_trsv` | `sb_handle`, `uplo`, `trans`, `diag`, `N`, `mA`, `lda`, `vx`, `incx` | Compute a matrix-vector product with a triangular band matrix: `A * x = b` |

### BLAS 3

The following table sums up the interface that can be found in
[blas3_interface.h](include/interface/blas3_interface.h).

For all these operations:

* `mA`, `mB` and `mC` are containers for the column-major matrices A, B and C.
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
  triangular matrix: `u` if the matrix is unit triangular *(all diagonal elements
  are 1)*, else `n`.
* `stride_a`, `stride_b` and `stride_c` are the striding size between consecutive 
matrices in a batched-strided entry for the inputs/outputs. 
* `batch_type` for `_gemm_batched` is either `strided` *(by default)* or `interleaved`*(More details about it here : [Gemm.md](doc/Gemm.md))*.

| operation | arguments | description |
|---|---|---|
| `_gemm` | `sb_handle`, `transa`, `transb`, `M`, `N`, `K`, `alpha`, `mA`, `lda`, `mB`, `ldb`, `beta`, `mC`, `ldc` | Generalised matrix-matrix multiplication followed by matrix addition: `C = alpha * A * B + beta * C` |
| `_gemm_batched` | `sb_handle`, `transa`, `transb`, `M`, `N`, `K`, `alpha`, `mA`, `lda`, `mB`, `ldb`, `beta`, `mC`, `ldc`, `batch_size`, `batch_type` | Same as `_gemm` but the containers contain `batch_size` end-to-end matrices. GEMM operations are performed independently with matching matrices. |
| `_gemm_strided_batched` | `sb_handle`, `transa`, `transb`, `M`, `N`, `K`, `alpha`, `mA`, `lda`, `stride_a`, `mB`, `ldb`, `stride_b`, `beta`, `mC`, `ldc`, `stride_c`, `batch_size` | Same as `_gemm` but the containers contain `batch_size` end-to-end matrices. GEMM operations are performed independently with matching matrices.
| `_symm` | `sb_handle`, `side` , `uplo` , `M`, `N`, `alpha`, `mA`, `lda`, `mB`, `ldb`, `beta`, `mC`, `ldc`| Compute a scalar-matrix-matrix product and add the result to a scalar-matrix product, where one of the matrices in the multiplication is symmetric. |
| `_trsm` | `sb_handle`, `side`, `uplo`, `trans`, `diag`, `M`, `N`, `alpha`, `mA`, `lda`, `mB`, `ldb` | Triangular solve with Multiple Right-Hand Sides. |

### EXTENSION

The following table sums up the interface that can be found in
[extension_interface.h](include/interface/extension_interface.h).

For all these operations:

* `A`, `B` and `C` are containers for the column-major matrices A, B and C.
* `lda`, `ldb` and `ldc` are the leading dimensions of the matrices A, B and C
  (cf BLAS 2). The leading dimension of a matrix must be greater than or equal
  to its number of rows. In the case of in-place copy/transpose, the same matrix `A`
  is used with two different leading dimensions for input & output.
* `stride_a`, `stride_b` and `stride_c` are the striding size between consecutive 
matrices in a batched entry for the inputs/outputs. 
* `inc_a` and `inc_b` are the jump-count between consecutive elements in A & B matrices.
* `transa` and `transb` are the transpose modes of the matrices A and B
  (cf BLAS 2).
* `M` and `N` are the dimensions of the matrices (Rows and Columns respectively).
* `alpha` and `beta` are scaling scalars.
* `batch_size` is the number of batch matrices.

| operation | arguments | description |
|---|---|---|
| `_axpy_batch` | `sb_handle`, `N`, `alpha`, `vx`, `incx`, `stride_x`, `vy`, `incy`, `stride_y`, `batch_size` | Perform multiple axpy operators in batch |
| `_omatcopy` | `sb_handle`, `transa`, `M`, `N`, `alpha`, `A`, `lda`, `B`, `ldb`  | Perform an out-of-place scaled matrix transpose or copy operation using a general dense matrix. |
| `_omatcopy2`| `sb_handle`, `transa`, `M`, `N`, `alpha`, `A`, `lda`, `inc_a`, `B`, `ldb`, `inc_b`  | Computes two-strided scaling and out-of-place transposition or copying of general dense matrices. |
| `_omatadd`| `sb_handle`, `transa`, `transb`, `M`, `N`, `alpha`, `A`, `lda`, `beta`, `B`, `ldb`, `C`,`ldc`  | Computes scaled general dense matrix addition with possibly transposed arguments. |
| `_omatcopy_batch` | `sb_handle`, `transa`, `M`, `N`, `alpha`, `A`, `lda`, `stride_a`, `B`, `ldb`, `stride_b`, `batch_size` | Perform an out-of-place scaled batched-strided matrix transpose or copy operation using a general dense matrix. |
| `_imatcopy_batch` | `sb_handle`, `transa`, `M`, `N`, `alpha`, `A`, `lda`, `ldb`, `stride`, `batch_size` | Perform an in-place scaled batched-strided matrix transpose* or copy operation using a general dense matrix. (*: Currently the transpose case is not supported). |
| `_omatadd_batch`| `sb_handle`, `transa`, `transb`, `M`, `N`, `alpha`, `A`, `lda`, `stride_a`, `beta`, `B`, `ldb`, `stride_b`, `C`,`ldc`, `stride_c`, `batch_size`  | Computes a batch of scaled general dense matrix addition with optionally transposed arguments. |

Other non-official extension operators : 
| operation | arguments | description |
|---|---|---|
| `_transpose` | `sb_handle`, `M`, `N`, `A`, `lda`, `B`, `ldb`  | Computes an out-of-place matrix transpose operation using a general dense matrix. |
| `_transpose*` | `sb_handle`, `M`, `N`, `A`, `lda`, `ldb`  | Computes an in-place matrix transpose operation using a general dense matrix, lda & ldb being input and output leading dimensions of A respectively _(*Not implemented)_. |
### Experimental Joint Matrix Support

portBLAS now supports sub-group based collective GEMM operation using the experimental 
[`joint_matrix`](https://github.com/intel/llvm/blob/sycl/sycl/doc/extensions/experimental/sycl_ext_matrix/sycl_ext_oneapi_matrix.asciidoc) extension provided by DPC++. This support is only accessible for the latest 
NVIDIA Ampere GPUs and beyond. The requirements for using this experimental support 
are: 
```bash
DPCPP_SYCL_TARGET = "nvptx64-nvidia-cuda"
DPCPP_SYCL_ARCH = "sm_80" | "sm_90"
```
To invoke the `joint_matrix` based GEMM, you need to set the following environment variable:
```bash
export SB_ENABLE_JOINT_MATRIX=1
```
The user should expect erroneous behaviour from the code if both of these requirements are not met.

## Requirements

portBLAS is designed to work with any SYCL implementation.
We do not use any OpenCL interoperability, hence, the code is pure C++.
The project is developed using [DPCPP open source](https://github.com/intel/llvm)
or [oneapi release](https://www.intel.com/content/www/us/en/developer/tools/oneapi/base-toolkit.html#gs.2iaved),
using Ubuntu 22.04 on Intel OpenCL CPU, Intel GPU, NVIDIA GPU and AMD GPU.
The build system is CMake version 3.4.3 or higher.

A BLAS library, such as [OpenBLAS](https://github.com/xianyi/OpenBLAS), is also
required to build and verify the test results. 
Instructions for building and installing
OpenBLAS can be found [on this page](https://github.com/xianyi/OpenBLAS/wiki/User-Manual). 
Please note that although some distributions may provide packages for OpenBLAS 
these versions are typically quite old and may have issues with the TRMV implementation 
which can cause random test failures. Any version of OpenBLAS `>= 0.3.0` will not suffer
from these issues.

When using OpenBLAS or any other BLAS library the installation directory must be
added to the `CMAKE_PREFIX_PATH` when building portBLAS (see
[below](###cmake-options)).

## Setup

**IMPORTANT NOTE:** The `TARGET` CMake variable is no longer supported. It has
been replaced by `TUNING_TARGET`, which accepts the same options.
`TUNING_TARGET` affects only the tuning configuration, applicable for some operators such
as GEMM, and has no effect on the target triplet for DPC++ or the hipSYCL target. Please 
refer to the sections below for setting them.

1. Clone the portBLAS repository, making sure to pass the `--recursive` option, in order 
to clone submodule(s).
2. Create a build directory
3. Run `CMake` from the build directory *(see options in the section below)*:

### Compile with DPC++
```bash
export CC=[path/to/intel/clang]
export CXX=[path/to/intel/clang++]
cd build
cmake -GNinja ../ -DSYCL_COMPILER=dpcpp
ninja
```
The target triplet can be set by adding `-DDPCPP_SYCL_TARGET=<triplet>`. If it
is not set, the default values is `spir64`, which compiles for generic SPIR-V
targets.

Other possible triplets are `nvptx64-nvidia-cuda`, and
`amdgcn-amd-amdhsa` for compiling for NVIDIA and AMD GPUs. In this case, it is
advisable for NVIDIA and **mandatory for AMD** to provide the specific device
architecture through `-DDPCPP_SYCL_ARCH=<arch>`, e.g., `<arch>` can be `sm_80`
for NVIDIA or `gfx908` for AMD.

### Compile with hipSYCL
```bash
cd build
cmake -GNinja ../ -DhipSYCL_DIR=/path/to/hipSYCL/install/lib/cmake/hipSYCL -DSYCL_COMPILER=hipsycl
ninja
```
To build for other than the default devices (`omp`), set the `HIPSYCL_TARGETS` environment variable or specify `-DHIPSYCL_TARGETS` as [documented](https://github.com/illuhad/hipSYCL/blob/develop/doc/using-hipsycl.md).

### Installing portBLAS
To install the portBLAS library (see `CMAKE_INSTALL_PREFIX` below)

```bash
ninja install
```

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
| `SYCL_COMPILER` | name | Used to determine which SYCL implementation to use. By default, the first implementation found is used. Supported values are: `dpcpp`, `hipsycl` and `computecpp`*(deprecated)*. |
| `TUNING_TARGET` | name | By default, this flag is set to `DEFAULT_CPU` to restrict any device specific compiler optimizations. Use this flag to tune the code for a target (**highly recommended** for performance). The supported targets are: `INTEL_GPU`, `NVIDIA_GPU`, `AMD_GPU` |
| `CMAKE_PREFIX_PATH` | path | List of paths to check when searching for dependencies |
| `CMAKE_INSTALL_PREFIX` | path | Specify the install location, used when invoking `ninja install` |
| `BUILD_SHARED_LIBS` | `ON`/`OFF` | Build as shared library (`ON` by default) |
| `ENABLE_EXPRESSION_TESTS` | `ON`/`OFF` | Build additional tests that use the header-only framework (e.g to test expression trees); `OFF` by default |
| `BLAS_VERIFY_BENCHMARK` | `ON`/`OFF` | Verify the results of the benchmarks instead of only measuring the performance. See the documentation of the benchmarks for more details. `ON` by default |
| `BLAS_ENABLE_CONST_INPUT` | `ON`/`OFF` | Determines whether to enable kernel instantiation with const input buffer (`ON` by default) |
| `BLAS_ENABLE_EXTENSIONS` | `ON`/`OFF` | Determines whether to enable portBLAS extensions (`ON` by default) |
| `BLAS_DATA_TYPES` | `half;float;double` | Determines the floating-point types to instantiate BLAS operations for. Default is `float` |
| `BLAS_INDEX_TYPES` | `int32_t;int64_t` | Determines the type(s) to use for `index_t` and `increment_t`. Default is `int` |
| `BLAS_ENABLE_COMPLEX` | `ON`/`OFF` | Determines whether to enable Complex data type support *(GEMM Operators only)* (`ON` by default) |

## ComputeCpp Compilation *(Deprecated)*

portBLAS ComputeCpp compilation is deprecated since ComputeCpp releasing has been
discontinued. More information about this are found in this [announcement](https://codeplay.com/portal/news/2023/07/07/the-future-of-computecpp). 

### Compile with ComputeCpp

```bash
cd build
cmake -GNinja ../ -DComputeCpp_DIR=/path/to/computecpp -DSYCL_COMPILER=computecpp
ninja
```

### Cross-Compile *(ComputeCpp Only)*

To cross-compile portBLAS first the following environment variables must be
set:

```bash
export COMPUTECPP_TOOLCHAIN_DIR="PATH TO TOOLCHAIN_DIR"
export COMPUTECPP_TARGET_TRIPLE="PATH TO TARGET_TRIPLE"
export COMPUTECPP_SYSROOT_DIR="$PATH TO SYSROOT_DIR"
```

Clone the [ComputeCpp-SDK](https://github.com/codeplaysoftware/computecpp-sdk) to retrieve the toolchain file.
The following CMake command can be used to cross-compile portBLAS:

```bash
cmake  -GNinja                                                                         \
    ${SOURCE_ROOT}                                                                     \
   -DCMAKE_PREFIX_PATH="${OPENBLAS_PATH}"                                              \
   -DComputeCpp_DIR="${COMPUTECPP_DEVICE_PATH}"                                        \
   -DComputeCpp_HOST_DIR="${COMPUTECPP_X86_PATH}"                                      \
   -DCMAKE_TOOLCHAIN_FILE="/path/to/computecpp-sdk/cmake/toolchains/gcc-generic.cmake" \
   -DCMAKE_BUILD_TYPE='Release'                                                        \
   -DCMAKE_INSTALL_PREFIX=${CROSS_COMPILED_PORTBLAS_INSTALL}                           \
   -DOpenCL_INCLUDE_DIR="${OpenCL_Headers_PATH}"                                       \
   -DOpenCL_LIBRARY="${OpenCL_LIBRARY}"                                                \
   -DCOMPUTECPP_BITCODE="${DEVICE_BITCODE}"                                            \
   -DCMAKE_CXX_FLAGS='-O3'                                                             \
   -DTUNING_TARGET="${CHOSEN_TARGET}"
```

### POWER_VR support *(ComputeCpp Only)*

To enable the PowerVR target tuning, pass: `-DTUNING_TARGET=POWER_VR`

To use the neural network library from Imagination, pass: `-DIMGDNN_DIR=path/to/library`


## Tests and benchmarks

The tests and benchmarks have their own documentation:

- [Documentation of the tests](test/README.md)
- [Documentation of the benchmarks](benchmark/README.md)


## Contributing to the project

portBLAS is an Open Source project maintained by the HPCA group and
Codeplay Software Ltd.
Feel free to create an issue on the Github tracker to request features or
report bugs.

### Guides and Other Documents

- [How to add a new operation](doc/AddingBlas3Op.md)
- [Autotuner Developer Guide](doc/Autotuner.md)
