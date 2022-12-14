# Adding a new BLAS level 3 operation

This document is meant to be a guide on how to add a new BLAS Level 3
operation to SYCL-BLAS. It is mostly based on the work done when adding the
TRSM algorithm to SYCL-BLAS.

The steps described here are just a guideline and may vary when doing the
actual implementation.

## Add the operation interface

The folder `include/interface` contains the public interface of SYCL-BLAS.
The files in this folder contain the functions that users can call to 
run the available blas operations.

This folder has one file for each level of blas. For example, the `trsm`
function will be located in `include/interface/blas3_interface.h`.

When defining a new level 3 operation, the first step is to define the
user-facing function, in this case `blas::_trsm`, and declare the
internal function that will be implemented inside SYCL-BLAS, 
called `blas::internal::_trsm`, similar to the following:

```c++
namespace blas {
namespace internal {

// Internal function that will be implemented in the sycl-blas library
template <typename sb_handle_t, typename container_0_t, typename container_1_t,
          typename element_t, typename index_t>
typename sb_handle_t::event_t _trsm(sb_handle_t& sb_handle, char Side,
                                             char Triangle, char Transpose,
                                             char Diagonal, index_t M,
                                             index_t N, element_t alpha,
                                             container_0_t A, index_t lda,
                                             container_1_t B, index_t ldb);
    
}

// User-facing function to call the TRSM operation
template <typename sb_handle_t, typename container_0_t, typename container_1_t,
          typename element_t, typename index_t>
typename sb_handle_t::event_t inline _trsm(
    sb_handle_t& sb_handle, char Side, char Triangle, char Transpose, char Diagonal,
    index_t M, index_t N, element_t alpha, container_0_t A, index_t lda,
    container_1_t B, index_t ldb) {
  return internal::_trsm(sb_handle, Side, Triangle, Transpose, Diagonal, M, N, alpha, A, lda,
                         B, ldb);
}
} // namespace internal
} // namespace blas
```

## Define a test for the function

> Defining a test for the new function early on is the best way to proceed since
it will allow adding new pieces of code in small batches. This can prevent
several complex compilation issues when adding the actual implementation for 
the new operation.

Templated functions don't get instantiated until they are used somewhere,
so the next step in adding a new operation is to define a test for it.

This can be done by adding a new file, corresponding to the new operation
in the folder `test/blas3`. Here we create the file `test/blas3/blas3_trsm_test.cpp`
with the following contents:

```c++
#include "blas_test.hpp"

// A list of parameters to be passed to the test. This type will be used to
// generate several combinations of input values for the function to be tested.
template <typename scalar_t>
using combination_t = std::tuple<int, int, char, char, char, char, scalar_t,
                                 scalar_t, scalar_t, scalar_t>;

// Register the function that will run the test
template <typename scalar_t>
void run_test(const combination_t<scalar_t> combi) {
  int m;
  int n;
  char transA;
  char side;
  char diag;
  char triangle;
  scalar_t alpha;
  scalar_t ldaMul;
  scalar_t ldbMul;
  scalar_t unusedValue;
  std::tie(m, n, transA, side, diag, triangle, alpha, ldaMul, ldbMul,
           unusedValue) = combi;

  //
  // Perform any initialization here
  //

  // Create the SB_Handle for the test
  auto q = make_queue();
  SB_Handle sb_handle(q);

  // Invoke the newly added operation
  _trsm(sb_handle, side, triangle, transA, diag, m, n, alpha, a_gpu, lda, b_gpu, ldb);

  // Verify the results
}

// Create the combinations of parameters to invoke the test
const auto combi = ::testing::Combine(::testing::Values(7, 513, 1027),  // m
                                      ::testing::Values(7, 513, 1027),  // n
                                      ::testing::Values('n', 't'),  // transA
                                      ::testing::Values('l', 'r'),  // side
                                      ::testing::Values('u', 'n'),  // diag
                                      ::testing::Values('l', 'u'),  // triangle
                                      ::testing::Values(1.0, 2.0),  // alpha
                                      ::testing::Values(1.0, 2.0),  // lda_mul
                                      ::testing::Values(1.0, 2.0),  // ldb_mul
                                      ::testing::Values(0.0, NaN)   // unused
);

// Finaly register the new test
BLAS_REGISTER_TEST_ALL(Trsm, combination_t, combi);
```

By this point, you are expected to get a linker error saying the symbol for the internal
version of `blas::internal::_trsm` is undefined, and that is expected since we didn't 
provide the internal implementation for the newly added `_trsm` function.

## Implement the internal version of `_trsm`

The next step is to provide the implementation of `blas::internal::_trsm` and instantiate
it somewhere.

We start by creating the internal file `src/interface/trsm_interface.hpp`. This file
will contain the actual implementation of `_trsm`, but note that the function is not
yet being instantiated so the linker error will persist at this point:

```c++
namespace blas {
namespace internal {

template <typename sb_handle_t, typename container_0_t, typename container_1_t,
          typename element_t, typename index_t>
typename sb_handle_t::event_t _trsm(
    sb_handle_t& sb_handle, char Side, char Triangle, char Transpose, char Diagonal,
    index_t M, index_t N, element_t alpha, container_0_t A, index_t lda,
    container_1_t B, index_t ldb) {
  // Implementation of the new operation
  // This will probably invoke a kernel which we don't yet have defined.
    
  // This function can access any other operation defined in this
  // or lower levels of blas. For example, one may want to invoke a gemm
  // operation like the following
  auto gemmEvent = internal::_gemm(
            sb_handle, 'n', isTranspose ? 't' : 'n', M, currentBlockSize,
            currentBlockSize, (i == 0) ? alpha : element_t{1}, B + i * ldb, ldb,
            invA + i * blockSize, blockSize, element_t{0}, X + i * ldx, ldx);
  trsmEvents = concatenate_vectors(trsmEvents, gemmEvent);  

  // Ultimately, a list of all events created in this function is returned
  // to the user that can be used to wait on all or part of the internal
  // operations required to perform the full computation
  return trsmEvents;    
}

} // namespace internal
} // namespace blas
```

## Instantiate the new operation

The next step is to define the `_trsm` function in a `.cpp` file. In this way, when the client
binary is linked against SYCL-BLAS, the linker will find the definition of the missing symbol.

To do this, we create the source file that will contain instantiations of the new `_trsm` operation.
The file is located at `src/interface/blas3/trsm.cpp.in`. This is not the file that will be 
compiled, but a template file that the python script `python_generator/py_gen_blas_binary.py`
will use to generate the actual source file where the instantiation of `_trsm` will happen.

The file `src/interface/blas3/trsm.cpp.in` must include all files that are necessary to successfully
compile `blas::internal::_trsm`, for this particular example, this file looks like the following:

```c++
#include "container/sycl_iterator.hpp"
#include "sb_handle/sycl_blas_handle.hpp"
#include "sb_handle/kernel_constructor.hpp"
#include "operations/blas_constants.hpp"
#include "views/view_sycl.hpp"
#include "sycl_blas_helper.h"
#include "interface/blas1_interface.hpp"
#include "interface/trsm_interface.hpp"
#include "operations/blas3/trsm.hpp"

namespace blas {
namespace internal {


template typename SB_Handle::event_t _trsm(
  SB_Handle sb_handle, char Side, char Triangle, char Transpose, char Diagonal,
  ${INDEX_TYPE} M, ${INDEX_TYPE} N, ${DATA_TYPE} alpha,
  ${container_t0} A, ${INDEX_TYPE} lda,
  ${container_t1} B, ${INDEX_TYPE} ldb);


} // namespace internal
} // namespace blas
```

Where `${INDEX_TYPE}, ${DATA_TYPE}, ${container_t0}` and `${container_t1}` are going
to be replaced by the appropriate types required to explicitly instantiate the new function
Finally, the file `src/interface/blas3/CMakeLists.txt` must be changed
in order to generate instantiations of `_trsm`.
The following entry must be added:

```cmake
generate_blas_binary_objects(blas3 trsm)
```

There are predefined functions to be used depending on the number of inputs the function expects.
`_trsm` is a case of *binary* function, since only two buffers are required as input. `_gemm` is
an example of ternary function, since it requires three buffers as input (in this case `${container_t2}`
can be used in the `.cpp` file).

After this, the new object file created must be added to the library and is done by adding a new
entry to `cmake/CmakeFunctionHelper.cmake`. At the end of this file there is a list of all object
files that are archived to form the SYCL-BLAS library. A new entry must be added in the function
`build_library`:

```cmake
function (build_library LIB_NAME)
add_library(${LIB_NAME}
                #
                # Other objects here
                #
                $<TARGET_OBJECTS:trsm> # Add the new object here
            )
endfunction(build_library)
```

After this step the test defined for the new function will compile and link successfully and 
`blas::internal::_trsm` will be called from the test.

## Adding a kernel

So far we have looked at how to add a new public facing function and how to instantiate it internally so that
the linker can find it when compiling user code. If the new operation can be implemented in terms of pre-existing
operations, the job would be done, however, it is often the case that a new operation requires a new kernel
to be implemented in order to perform some computation.

In the case of `_trsm`, we need to implement the diagonal blocks inversion algorithm using SYCL.
To add a new kernel we need to add the corresponding expression tree in the file `include/operations/blas3_trees.h`.
In the case of `_trsm`, we add a new struct called `DiagonalBlocksInverter` with the following interface:

```c++
namespace blas {
template <bool UnitDiag, bool Upper, int BlockSize, typename matrix_t>
struct DiagonalBlocksInverter {
  using index_t = typename std::make_signed<typename matrix_t::index_t>::type;
  using value_t = typename std::remove_cv<typename matrix_t::value_t>::type;
  static constexpr index_t internalBlockSize = BlockSize;
  static constexpr index_t outterBlockSize = BlockSize;
  matrix_t A_;
  matrix_t invA_;
  index_t lda_;
  index_t N_;

  DiagonalBlocksInverter(matrix_t& A, matrix_t& invA);
  bool valid_thread(cl::sycl::nd_item<1> id) const;
  void bind(cl::sycl::handler& cgh);
  void adjust_access_displacement();

  template <typename local_memory_t>
  void eval(local_memory_t localMem, cl::sycl::nd_item<1> id) noexcept;
};

template <bool UnitDiag, bool Upper, int BlockSize, typename matrix_t>
DiagonalBlocksInverter<UnitDiag, Upper, BlockSize, matrix_t>
make_diag_blocks_inverter(matrix_t& A, matrix_t& invA) {
  return DiagonalBlocksInverter<UnitDiag, Upper, BlockSize, matrix_t>(A, invA);
}
}
```

The `make_diag_blocks_inverter` is a utility function that will be used in `blas::internal::_trsm`
to instantiate the kernel. With this new interface defined, we can implement it internally by creating
a new file called `src/operations/blas3/trsm.hpp`. This file will contain the implementation
of `DiagonalBlocksInverter` and `make_diag_blocks_inverter`. The actual kernel is implemented
in `DiagonalBlocksInverter::eval` in this file.

The last part is to add this file in the include list in `src/operations/blas3/blas3_trees.hpp`.

## Adding a call to the system blas version of the new operation

SYCL-BLAS implements the netlib-blas interface, so there will be a system blas version of the operation
being added. In the case of `_trsm`, the
[netlib-blas version of the function](http://www.netlib.org/lapack/explore-html/d2/d8b/strsm_8f.html)
can be called to verify that our implementation produces the correct results for a certain combination of parameters.

SYCL-BLAS provides a utility header that is used to invoke different system-blas functions for testing and benchmarking purposes.
To make the new operation available, add it to the file `include/utils/system_reference_blas.hpp`, like the following:

```c++
namespace reference_blas {
template <typename scalar_t>
void trsm(const char* side, const char* triangle, const char* transA,
          const char *diag, int m, int n, scalar_t alpha, scalar_t A[], int lda,
          scalar_t B[], int ldb) {
  auto func = blas_system_function<scalar_t>(&cblas_strsm, &cblas_dtrsm);
  func(CblasColMajor, c_side(*side), c_uplo(*triangle), c_trans(*transA),
       c_diag(*diag), m, n, alpha, A, lda, B, ldb);
}
} // namespace reference_blas
```

In this way, this function can be used in tests or benchmarks:

```c++
reference_blas::trsm(&side, &triangle, &transA, &diag, m, n,
                       static_cast<data_t>(alpha), A.data(), lda, cpu_B.data(),
                       ldb);
```
So results from the reference implementation can be used to check if the operation implemented is correct.

## Conclusion

By following the steps described in this document, you can add a new operation in SYCL-BLAS alongside
the tests required to validate the implementation.
