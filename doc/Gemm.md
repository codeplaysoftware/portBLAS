# SYCL-BLAS GEMM Developer Documentation

The following is documentation for the `GEMM` kernels and associated areas of code within `SYCL-BLAS`.
This should give you a good understanding of how everything works and where/how to do things such as:

- Work on or create a new `GEMM` kernel
- Work with the `GEMM` dispatch system.
- Add or modify `GEMM` configurations for different backends.

Please note that while this document primarily refers to `GEMM` and `Blas3` operations many of the same concepts around source code generation apply to other operations and blas levels.

# Contents

- [SYCL-BLAS GEMM Developer Documentation](#sycl-blas-gemm-developer-documentation)
- [Contents](#contents)
- [GEMM](#gemm)
  - [What is GEMM?](#what-is-gemm)
  - [SYCL-BLAS GEMM Kernels](#sycl-blas-gemm-kernels)
  - [Relevant CMake Variables](#relevant-cmake-variables)
  - [Kernel Structure](#kernel-structure)
  - [Vectorized Loading/Storing](#vectorized-loadingstoring)
  - [Batched Gemm](#batched-gemm)
- [GEMM Dispatch](#gemm-dispatch)
  - [GEMM Backends](#gemm-backends)
  - [GEMM Launcher](#gemm-launcher)
  - [Source Code Generation](#source-code-generation)
- [GEMM Configurations](#gemm-configurations)
  - [Backend Configurations](#backend-configurations)
  - [CMake Configurations](#cmake-configurations)
- [Common Tasks](#common-tasks)
  - [Adding a new configuration](#adding-a-new-configuration)
  - [Adding a new kernel](#adding-a-new-kernel)

# GEMM

## What is GEMM?

`GEMM` stands for General Matrix Multiplication and solves the equation of the form:

```
C = alpha * A * B + beta * C
where A, B and C are matrices and alpha and beta are scalars.
```

## SYCL-BLAS GEMM Kernels

`SYCL-BLAS` currently contains the following `GEMM` kernels in <src/operations/blas3/>:

- `gemm_ref.hpp` - A naive, reference implementation of `GEMM` with no optimizations.

- `gemm_partial_local.hpp` - Used for tall, skinny `GEMM` optimizations.

- `gemm_local.hpp` - Uses local memory for increased performance on platforms that have it. Supports full vectorization.

- `gemm_no_local_partial_vec.hpp` - Doesn't use local memory. Only supports partial vectorization (can only vectorize in some specific cases).

- `gemm_no_local_full_vec.hpp` - Doesn't use local memory. Supports full vectorization.

- `gemm_interleaved.hpp` - An alternative approach to batched `GEMM` calculations where the inputs are interleaved in contiguous memory. 
Uses no local memory and corresponds to HWN data layout (NWH in column major, which is what `SYCL-BLAS` uses).

## Relevant CMake Variables

There are several CMake variables which are specific to `GEMM` :

- `NAIVE_GEMM` (Default: `OFF`) - Forces the use of the naive, reference `GEMM` implementation.
- `GEMM_VECTORIZATION_SUPPORT` (Default: `OFF`) - Enables vectorization within the `GEMM` kernels. 
If `OFF` it is equivalent to passing `1` for the vector size to the `Gemm` launcher.
- `GEMM_TALL_SKINNY_SUPPORT` (Default: `ON`) - Enables optimizations for tall, skinny matrices. Not used on all targets.
- `BLAS_MODEL_OPTIMIZATION` - Passing a machine learning model name here (`VGG_16` or `RESNET_50`) enables optimizations for the `GEMM` sizes used in these models. 
Only applies to the `ARM_GPU` target.

## Kernel Structure

Kernels are created as partial template specializations of the `Gemm` class to optimize for more specific cases or use different features (such as using local memory). 
The `Gemm` class has a number of template parameters, some of which are typically used for partial specialization. 
The class definition (along with the naive, reference implementation) is located in `gemm_ref.hpp`.

There are some small member functions which do things such as determine the number of work groups required to execute each `GEMM`. 
However, the actual work of each kernel is done in `Gemm::eval()`. 

The general goals for programming a `GEMM` kernel can be summarized as follows:

- Balance register pressure with loop unrolling to achieve an optimal balance of performance.
- Using SFINAE and templates where applicable to minimize branching and keep as many things const and compile-time as possible for increased performance.

Outside of the naive, reference `GEMM` all the kernels are tile based, where each instance of the kernel calculates a small portion of the overall matrix. 
Much of the work in the `::eval()` functions tends to be precalculating indices, offsets and other values for use in the actual computation. 
Because they are tile based one of the main considerations is whether the current tile is internal or external. 
If it's internal that means that boundary checks can be avoided which is a significant time save and performance increase.

The core of the `GEMM` computation is as follows:

1. Loop over the K dimension.

    1. Load a block of A and B matrices.
    2. Multiply together and store in some intermediate memory (local or not).

2. Store the final results in the appropriate part of the output matrix.

## Vectorized Loading/Storing

Many of the `GEMM` kernels support vectorized loads/stores using functions located in `gemm_load_store.hpp` in `src/operations/blas3/` . 
These functions are pretty simple but there are some special considerations for how they are used, particularly around whether the matrices are transposed or not. 
If a matrix is transposed this changes the data layout such that elements are no longer contiguous in memory.

You can see examples of how to handle these issues by looking at the `gemm_local.hpp` and `gemm_no_local_full_vec.hpp` kernels.

## Batched Gemm

Batched `GEMM` is not officially part of the BLAS specification but is a common use case, particularly when you have a series of smaller matrices to multiply it makes more sense to perform them as a batched operation. 
All `GEMM` kernels support batched operations but the interleaved `GEMM` can only be used for batched operations as it is designed specifically for it.

Batched `GEMM` is called with a separate `_gemm_batched` function, however beyond the user facing functions all `GEMM` calls take the same path, with `batch_size` and `batch_type` parameters controlling if and how a batched operation takes place.

# GEMM Dispatch

As previously mentioned, the `Gemm` class has a lot of template parameters, and many of these are based on values passed at runtime by the user when they call `_gemm` . 
So there is a series of calls to enable translating some of these runtime values to template parameters when calling subsequent parts of the `GEMM` dispatch. Typically this happens with `enum` or `bool` values and looks like:

```c++
template <bool templateParam>
void foo(){
  //do something here
}

void bar(bool runtimeValue)
{
  if(runtimeValue){
    foo<true>();
  }
  else{
    foo<false>();
  }
}
```

You can also see this technique at work inside the `GEMM` kernels themselves.

The notable calls in the stack are (all located in `src/interface/gemm_interface.hpp`):

- `blas::internal::_gemm`

  - calls `_gemm_backend()` always passing `strided` for the `gemm` batch type (as the interleaved kernel is intended only for batch operations). 
  The `_batch_gemm` call instead passes the batch type through.

- `blas::internal::_gemm_backend()`

  - calls `_gemm_is_beta_zero` with different transpose template parameters depending on the runtime values passed.

- `blas::internal::_gemm_is_beta_zero()`

  - calls `_gemm_platform_specific` depending on whether beta == 0 or not.

- `blas::internal::_gemm_platform_specific`

  - calls `blas::gemm::backend::_gemm` which is the backend target specific GEMM.

## GEMM Backends

GEMM backends are a mechanism to provide different compile-time configurations for different hardware platforms/backends.
Backend selection is controlled by passing the cmake variable `TUNING_TARGET` during CMake configuration, for example passing `-DTUNING_TARGET=INTEL_GPU` would select the appropriate configurations for Intel GPUs.
This cmake variable causes a corresponding define for the selected platform to be included in the source which then controls backend selection through `#ifdef`s in `src/interface/blas3/backend/backend.hpp` like so:

```c++
#if defined(RCAR)
#include "interface/blas3/backend/rcar.hpp"
#elif defined INTEL_GPU
#include "interface/blas3/backend/intel_gpu.hpp"
#elif defined AMD_GPU
#include "interface/blas3/backend/amd_gpu.hpp"
#elif defined ARM_GPU
#include "interface/blas3/backend/arm_gpu.hpp"
#elif defined POWER_VR
#include "interface/blas3/backend/power_vr.hpp"
#else
#include "interface/blas3/backend/default_cpu.hpp"
#endif
```

These backend headers call `Gemm_Launcher::_select_gemm()` with various parameters depending on the inputs given. 
For example, they commonly call different configurations depending on input size to obtain optimal performance for a given size or range of sizes. 
Backend configurations are covered in further detail in [this section](#backend-configurations).

## GEMM Launcher

The `Gemm_Launcher` class wraps the creation of the actual `Gemm` class as well as the creation of the matrix views (which are what is actually passed to the `Gemm` class for use in the kernel). 
This happens in the `::select_gemm()` member function where it also executes the created `GEMM` through the passed in sb_handle and returns the associated event.

```c++
namespace blas {

/*!
 * @brief Wrapper around Gemm. Creates the views, then makes and launches Gemm
 */
template <int WgSize, bool DoubleBuffer, bool ConflictA, bool ConflictB, 
          int ClSize, typename TileT, bool TransA, bool TransB,
          int GemmMemoryType, int GemmAlgorithm, int GemmVectorization,
          bool is_beta_zero, int VectorSize, int BatchType>

template <typename SB_Handle, typename container_t0, typename container_t1, 
          typename container_t2, typename element_t, typename index_t>

typename SB_Handle::event_t Gemm_Launcher<
    WgSize, DoubleBuffer, ConflictA, ConflictB, ClSize, TileT, TransA, TransB,
    GemmMemoryType, GemmAlgorithm, GemmVectorization, is_beta_zero, VectorSize,
    BatchType>::_select_gemm(SB_Handle& sb_handle, index_t _M, index_t _N, index_t _K,
                             element_t _alpha, container_t0 a_, index_t _lda,
                             container_t1 b_, index_t _ldb, element_t _beta,
                             container_t2 _C, index_t _ldc,
                             index_t batch_size) {

  //Helper functions used to make matrix views
  auto buffer_a = make_matrix_view<col_major>(a_, _M, _K, _lda); 
  auto buffer_b = make_matrix_view<col_major>(b_, _K, _N, _ldb); 
  auto buffer_c = make_matrix_view<col_major>(_C, _M, _N, _ldc); 

  //Helper function to construct the Gemm object
  auto gemm = make_gemm<DoubleBuffer, ConflictA, ConflictB, ClSize, TileT, 
                        TransA, TransB, GemmMemoryType, GemmAlgorithm,
                        GemmVectorization, is_beta_zero, VectorSize, BatchType>(
      buffer_a, buffer_b, buffer_c, element_t(_alpha), element_t(_beta),
      batch_size);

  //Execute the gemm and return the associated event
  return sb_handle.execute(gemm); 
}
}  // namespace blas
```

## Source Code Generation

In order to correctly link a user's application to the SYCL-BLAS library the configurations for both `Gemm_Launcher` and `Gemm` must be instantiated explicitly in `.cpp` files to prevent linking errors. 
These instantiations are generated using a template file and several python scripts which replace variables in the template file with the appropriate types for different configurations. 
This is driven by CMake and covered more extensively in [this section](#cmake-configurations).

The templates are located in `src/interface/blas3/` while the Python scripts are located in `python_generator`.

The template for `Gemm` looks like this:

```c++
#include "container/sycl_iterator.hpp"
#include "sb_handle/sycl_blas_handle.hpp"
#include "interface/gemm_interface.hpp"
#include "operations/blas_constants.hpp"
#include "sycl_blas_helper.h"
#include "views/view_sycl.hpp"

namespace blas {
namespace internal {
// gemm
template typename SB_Handle::event_t _gemm(
    SB_Handle& sb_handle, char _TransA, char _TransB, ${INDEX_TYPE} _M,
    ${INDEX_TYPE} _N, ${INDEX_TYPE} _K, ${DATA_TYPE} _alpha, ${container_t0} a_,
    ${INDEX_TYPE} _lda, ${container_t1} b_, ${INDEX_TYPE} _ldb,
    ${DATA_TYPE} _beta, ${container_t2} _C, ${INDEX_TYPE} _ldc);
// batched gemm
template typename SB_Handle::event_t _gemm_batched(
    SB_Handle& sb_handle, char _TransA, char _TransB, ${INDEX_TYPE} _M,
    ${INDEX_TYPE} _N, ${INDEX_TYPE} _K, ${DATA_TYPE} _alpha, ${container_t0} a_,
    ${INDEX_TYPE} _lda, ${container_t1} b_, ${INDEX_TYPE} _ldb,
    ${DATA_TYPE} _beta, ${container_t2} _C, ${INDEX_TYPE} _ldc,
    ${INDEX_TYPE} batch_size, gemm_batch_type_t batch_type);
}  // namespace internal
}  // namespace blas
```

It includes instantiations for both `_gemm` and `_gemm_batched` . 
The placeholders like `${INDEX_TYPE}` are replaced with the correct types to instantiate the various configurations.

# GEMM Configurations

As previously touched on, tailored configurations for `GEMM` are provided on a per-target basis (along with the default CPU target configurations). 
Typically these are based on things like input size to provide optimal configurations for different use cases.

## Backend Configurations

Each backend header calls `Gemm_Launcher` with various configurations of template parameters to select different `GEMM` kernels and achieve the best performance within those kernels. 
The relevant parameters are:

- Tile size by passing a `Tile<>` (found in `include/operations/blas3_trees.h`), has parameters for batch sizes and for rows and columns of tile sizes at several levels:

  - Item level, the size of the block of elements processed by each work item running the `GEMM` kernel.
  - Work group level, made up of a number of item level tiles.
  - Sub group level, the size of any sub groups within a work group.
  - Tile level, the topmost level made up of a number of workgroup level tiles.

- Cache line size (in bytes) which influences the data layout and access within the kernel to try and optimize for the cache size of the hardware.
- Double buffering, whether to double buffer the loads and stores of the kernel, can increase performance.
- Bank conflicts, whether to modify storage in the kernel to avoid bank conflicts.
- Memory type, whether to use local memory or not.
- Gemm Algorithm, whether to use naive, tall skinny or standard (everything else) `GEMM` kernels.
- Vectorization, whether to enable partial or full vectorization.
- Vector size, the number of elements to use in vectorized loads/stores.
- Batch type, whether to use strided (most `GEMM` kernels) or the interleaved `GEMM` for batched calls.

For an example of a backend target header and some of the ways that configurations are selected let's look at `src/interface/blas3/backend/default_cpu.hpp` :

```c++
template <bool _t_a, bool _t_b, bool is_beta_zero, typename sb_handle_t, 
          typename container_0_t, typename container_1_t,
          typename container_2_t, typename element_t, typename index_t>

typename sb_handle_t::event_t _gemm(
    sb_handle_t& sb_handle, index_t _M, index_t _N, index_t _K, element_t _alpha,
    container_0_t _a, index_t _lda, container_1_t _b, index_t _ldb,
    element_t _beta, container_2_t _c, index_t _ldc, index_t batch_size,
    gemm_batch_type_t batch_type) {
  if (batch_type == gemm_batch_type_t::interleaved) {
    return blas::Gemm_Launcher<
        64, false, false, false, 64, Tile<2, 2, 4, 4, 1, 1, 1, 1, 4, 4>, _t_a, _t_b,
        static_cast<int>(gemm_memory_t::no_local),
        static_cast<int>(gemm_algorithm_t::standard),
        static_cast<int>(gemm_vectorization_t::full), is_beta_zero, 4,
        static_cast<int>(
            gemm_batch_type_t::interleaved)>::template _select_gemm(sb_handle, _M, _N,
                                                                    _K, _alpha,
                                                                    _a, _lda,
                                                                    _b, _ldb,
                                                                    _beta, _c,
                                                                    _ldc,
                                                                    batch_size);
  }
```

The first configuration is only used if `interleaved` is specified for the `GEMM` batch type.

```c++
#if defined(NAIVE_GEMM)
  return blas::Gemm_Launcher<
      64, false, false, false, 64, Tile<8, 8, 8, 8>, _t_a, _t_b,
      static_cast<int>(gemm_memory_t::no_local),
      static_cast<int>(gemm_algorithm_t::naive),
      static_cast<int>(gemm_vectorization_t::partial), is_beta_zero, 1,
      static_cast<int>(
          gemm_batch_type_t::strided)>::template _select_gemm(sb_handle, _M, _N, _K,
                                                              _alpha, _a, _lda,
                                                              _b, _ldb, _beta,
                                                              _c, _ldc,
                                                              batch_size);
#else
```

Next we have an `#if` directive for when we want to force naive `GEMM` configurations. 
This is triggered by a cmake variable. 
You can see other examples like this in `arm_gpu.hpp` which does similar things for different values of the cmake variable `BLAS_MODEL_OPTIMIZATION` .

```c++
if (_M <= 128 && _N <= 128 && _K <= 128) {
    return blas::Gemm_Launcher<
        64, false, false, false, 64, Tile<2, 2, 8, 8>, _t_a, _t_b,
        static_cast<int>(gemm_memory_t::no_local),
        static_cast<int>(gemm_algorithm_t::standard),
        static_cast<int>(gemm_vectorization_t::full), is_beta_zero, 2,
        static_cast<int>(
            gemm_batch_type_t::strided)>::template _select_gemm(sb_handle, _M, _N, _K,
                                                                _alpha, _a,
                                                                _lda, _b, _ldb,
                                                                _beta, _c, _ldc,
                                                                batch_size);
  } else {
    return blas::Gemm_Launcher<
        64, false, false, false, 64, Tile<8, 8, 8, 8>, _t_a, _t_b,
        static_cast<int>(gemm_memory_t::no_local),
        static_cast<int>(gemm_algorithm_t::standard),
        static_cast<int>(gemm_vectorization_t::partial), is_beta_zero, 1,
        static_cast<int>(
            gemm_batch_type_t::strided)>::template _select_gemm(sb_handle, _M, _N, _K,
                                                                _alpha, _a,
                                                                _lda, _b, _ldb,
                                                                _beta, _c, _ldc,
                                                                batch_size);
  }
#endif
}
```

Finally we provide a targeted configuration for small sizes (if all dimensions are less than or equal to 128) and a sensible default case for all other sizes.

## CMake Configurations

The generation of the `Gemm`, `Gemm_Launcher` and other operation's instantiations are driven through CMake and make use of the template files and python scripts previously touched on in [the section on source code generation](#source-code-generation).

The configurations to be generated, along with associated functions, are located in `cmake/CmakeFunctionHelper.cmake` and these functions are called from `src/interface/<blas_level>/CMakeLists.txt`. 
Configurations are provided per backend target and will be generated for each data type set during CMake configuration with the variable `BLAS_DATA_TYPES`.

As an example let's look at the configurations in `CmakeFunctionHelper.cmake` for the `RCAR` target backend, inside the function `generate_blas_gemm_objects`:

```cmake
if(${TUNING_TARGET} STREQUAL "RCAR")
  set(supported_types
    "float"
  )
  foreach(data ${supported_types})
    add_gemm_configuration(
      "${data}" 32 "false" "false" "false"
      128 4 8 8 4 1 1 1 1 1 1 "local" "standard" "full" 4 "strided")
    add_gemm_configuration(
      "${data}" 32 "false" "false" "false"
      128 8 4 4 8 1 1 1 1 1 1 "local" "standard" "full" 4 "strided")
    add_gemm_configuration(
      "${data}" 64 "false" "false" "false"
      64 4 4 4 4 1 1 1 1 4 4 "no_local" "standard" "full" 4 "interleaved")
  endforeach()
```

First we are setting the data types supported by the target. 
In this case RCAR only supports float but other platforms might also include `half` or `double` . 
Then we iterate over these supported data types calling `add_gemm_configuration()` for each configuration that we want to add. 
If a data type is passed which the user has not explicitly enabled with `BLAS_DATA_TYPES` then that configuration will be silently skipped. 
The configurations listed here must mirror those in the header for the backend, in this case `interface/blas3/backend/rcar.hpp` .

If you encounter errors after adding a new configuration this is the first place to check for inconsistencies. 
Having configurations in CMake which are _not_ present in the backend target header will not cause errors.

# Common Tasks

## Adding a new configuration

The following is a checklist of steps to add a new `GEMM` configuration to a chosen backend. 
The steps are the same for modifying an existing configuration, just modify in each relevant step instead of adding a new config.

1. Locate your target backend's header in `src/interface/blas3/backends/`.
2. Add your configuration to the ones already in the file. 
See the section on [backend configurations](#backend-configurations) for an example along with an explanation of the relevant template parameters of `Gemm_Launcher`.
3. Mirror the configuration you've added in the chosen target's section of `CmakeFunctionHelper.cmake`, see [the section on cmake configurations](#cmake-configurations) for more detail.

## Adding a new kernel

The following is a checklist of steps to add a new `GEMM` kernel to `SYCL-BLAS` .

1. Create your kernel header file in `src/operations/blas3/` and give it a sensible name that follows the convention of the others. 
For example, if your new kernel is very fast call it `gemm_very_fast.hpp`.
2. In this header create your partial specialization of the `Gemm` class. See `gemm_local.hpp` for an example.
3. Include your new very fast header in `src/operations/blas3_trees.hpp`
4. Modify backend configurations as necessary to enable the usage of your new specialization. See [Gemm Configurations](#gemm-configurations) for more information.
