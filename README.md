SYCL-BLAS Implementation
=========================================


SYCL-BLAS implements BLAS - [Basic Linear Algebra Subroutines](https://en.wikipedia.org/wiki/Basic_Linear_Algebra_Subprogram) - using [SYCL 1.2](
https://www.khronos.org/registry/sycl/specs/sycl-1.2.pdf), the
[Khronos](http://www.khronos.org) abstraction layer for [OpenCL](https://www.khronos.org/opencl/). 

SYCL-BLAS is a current work in progress research project from an ongoing 
collaboration with the *High Performance Computing & Architectures (HPCA) group*
from the Universitat Jaume I [UJI](http://www.hpca.uji.es/).

SYCL-BLAS is written using modern C++. The current implementation uses C++11 
features but we aim to move to C++14 in the short term.
See [Roadmap](Roadmap.md) for details on the current status and plans for
the project.

Table of Contents
------------------

  * [SYCL-BLAS Implementation](#sycl-blas-implementation)
    * [Table of Contents](#table-of-contents)
    * [Motivation](#motivation)
    * [Basic Concepts](#basic-concepts)
      * [Views](#views)
      * [Operations](#operations)
      * [Executors](#executors)
      * [Interface](#interface)
    * [Requirements](#requirements)
    * [Setup](#setup)
      * [Cross-Compile](#cross-compile)
    * [Tests](#tests)
    * [Contributing to the project](#contributing-to-the-project)



Motivation
------------

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
and the platform on the operation will be solved. The first analysis studies
the accuracy of the solution while the second one compares the performances of
the different implementations to select the best one.

Nowadays, all the numerical computations are based on a set of standard
libraries on which the most common operations are implemented. These libraries
are different for dense matrices (BLAS, LAPACK, ScaLAPACK, ...) and for sparse
matrices (SparseBLAS, ...). Moreover, there are  vendor implementations which
are adjusted to the platform features:
  - For multicores: ACML (AMD), ATLAS, Intel-MKL, OpenBLAS, ...
  - For GPUs: cuBLAS(nVidia), clBLAS, MAGMA, ...  

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

Basic Concepts
-----------------

SYCL-BLAS uses C++ Expression Tree templates to generate SYCL Kernels via
kernel composition.
Expression Tree templates are a widely used technique to implement expressions
on C++, that facilitate development and composition of operations.
In particular,
[Kernel composition in SYCL](http://dl.acm.org/citation.cfm?id=2791332) has 
been used in various projects to create efficient domain specific embedded
languages that enable users to easily fuse GPU kernels.

SYCL-BLAS can be used 
- either as a header-only framework by including sycl_blas.hpp in 
an application and passing src as pass to the application include directory.
- or as a library by including sycl_blas.h in an application. 

All the relevant files can be found in 
the include directory. 
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
The current restriction is that container must obey the *RandomAccessIterator*
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

The different headers on the interface directory implements the traditional
BLAS interface. 
Files are organised per BLAS level (1,2,3).

When the SYCL-BLAS BLAS interface is called, the Expression Tree for each
operation is constructed, and then executed.
Some API calls may execute several kernels (e.g, when a reduction is required).
The expression trees in the API allow to compile-time fuse operations.

Note that, although this library features a BLAS interface, users are allowed
to directly compose their own expression trees to compose multiple operations.
The CG example shows an implementation of the Conjugate Gradient that uses 
various expression tree to demonstrate how to achieve compile-time kernel fusion
of multiple BLAS operations.

Requirements
----------------

SYCL-BLAS is designed to work with any SYCL 1.2.1 implementation. 
We do not use any OpenCL interoperability, hence, the code is pure C++.
The project is developed using (ComputeCpp CE Edition 1.0.2)[http://www.computecpp.com]
 using Ubuntu 16.04 on Intel OpenCL CPU and AMD GPU.
In order to build the sources, GCC 5.4 or higher is required. 
The build system is CMake version 3.2.2 or higher.
We rely on the `FindComputeCpp.cmake` imported from the Computecpp SDK to
build the project.

A BLAS library, such as OpenBLAS, is also required to verify the test results. This can be installed on
Ubuntu from the `libopenblas-dev` package.

Setup
-----------------

1. Clone the SYCL-BLAS repository, making sure to pass the `--recursive` option, in order to clone submodule(s), such as the computecpp-sdk. 
2. Create a build directory
3. Run `CMake` from the build directory:

```
$ cd build; cmake -GNinja ../ -DComputeCpp_DIR=/path/to/computecpp
```

```
$ ninja
```
CMake options:

- To build a SYCL-BLAS Library only without the testing and benchmarking set 
```
-DBLAS_ENABLE_TESTING=OFF -DBLAS_ENABLE_BENCHMARK=OFF
```
Doxygen documentation can be generated by running

- By default SYCL-BLAS library is built for CPU. To compile it for a specific 
backend the TARGET flag should be passed to the CMake. Currently the following 
TARGETS are supported:
  - INTEL_GPU
  - AMD_GPU
  - ARM_GPU
  - RCAR
  - ARM_GPU
  
- SYCL-BLAS requires a System BLAS for verifying the test result. 
If BLAS_ENABLE_TESTING is enabled a system blas is required to be installed in 
a machine. If it is installed in a custom place 
```
-DSYSTEM_BLAS_ROOT=/path/to/root
```
can be used to set blas_path.

```
$ doxygen doc/Doxyfile
```

To install SYCL-BLAS library 
```
ninja install
```
The -DCMAKE_INSTALL_PREFIX can be used to set the install path.

### Cross-Compile

- To cross-compile SYCL-BLAS first the following Environment variable must be set

```
export COMPUTECPP_TOOLCHAIN_DIR="PATH TO TOOLCHAIN_DIR"
export COMPUTECPP_TARGET_TRIPLE="PATH TO TARGET_TRIPLE"
export COMPUTECPP_SYSROOT_DIR="$PATH TO SYSROOT_DIR"
```
The following cmake command can be used to cross-compile SYCL-BLAS

```
cmake  -GNinja                                                                                           \
    ${SOURCE_ROOT}                                                                                       \
   -DSYSTEM_BLAS_ROOT="${OPENBLAS_PATH}"                                                                 \
   -DComputeCpp_DIR="${COMPUTECPP_DEVICE_PATH}"                                                          \
   -DComputeCpp_HOST_DIR="${COMPUTECPP_X86_PATH}"                                                        \
   -DCMAKE_TOOLCHAIN_FILE="${SYCL_BLAS_PATH}/external/computecpp-sdk/cmake/toolchains/gcc-generic.cmake" \
   -DCMAKE_BUILD_TYPE=Release                                                                            \
   -DCMAKE_INSTALL_PREFIX=${CROSS_COMPILED_SYCLBLAS_INSTALL}                                             \
   -DOpenCL_INCLUDE_DIR="${OpenCL_Headers_PATH}"                                                         \
   -DOpenCL_LIBRARY="${OpenCL_LIBRARY}"                                                                  \
   -DCOMPUTECPP_BITCODE="${DEVICE BITECODE}"                                                             \
   -DCMAKE_CXX_FLAGS='-O3'                                                                               \
   -DTARGET="CHOSEN TARGET}"
```


Tests
-----------------

The sample code of the project is designed also as a test.
The project uses `Ctest` to run the different test.
The default platform reported by ComputeCpp is used as the execution platform.


Contributing to the project
-----------------------------

SYCL-BLAS is an Open Source project maintained by the HPCA group and
Codeplay Software Ltd.
Feel free to create an issue on the github tracker to request features or 
report bugs. 


