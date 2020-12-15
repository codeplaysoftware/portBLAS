
#ifndef SYCL_BLAS_BLAS3_REF_TRSM_HPP
#define SYCL_BLAS_BLAS3_REF_TRSM_HPP

#include "operations/blas3_trees.h"
#include "views/view.h"

#include <CL/sycl.hpp>

namespace blas {

template <bool UnitDiag, bool Upper, typename matrix_t>
SYCL_BLAS_INLINE
DiagonalBlocksInverter<UnitDiag, Upper, matrix_t>::DiagonalBlocksInverter(
    matrix_t& A, matrix_t& invA)
    : A_(A), invA_(invA), N_(A_.get_size_col()), lda_(A_.getSizeL()) {}

template <bool UnitDiag, bool Upper, typename matrix_t>
SYCL_BLAS_INLINE bool
DiagonalBlocksInverter<UnitDiag, Upper, matrix_t>::valid_thread(
    cl::sycl::nd_item<1> id) const {
  return true;
}

template <bool UnitDiag, bool Upper, typename matrix_t>
SYCL_BLAS_INLINE void DiagonalBlocksInverter<UnitDiag, Upper, matrix_t>::bind(
    cl::sycl::handler& cgh) {
  A_.bind(cgh);
  invA_.bind(cgh);
}

template <bool UnitDiag, bool Upper, typename matrix_t>
SYCL_BLAS_INLINE void DiagonalBlocksInverter<
    UnitDiag, Upper, matrix_t>::adjust_access_displacement() {
  A_.adjust_access_displacement();
  invA_.adjust_access_displacement();
}

template <bool UnitDiag, bool Upper, typename matrix_t>
template <typename local_memory_t>
SYCL_BLAS_INLINE void DiagonalBlocksInverter<UnitDiag, Upper, matrix_t>::eval(
    local_memory_t localMem, cl::sycl::nd_item<1> item) noexcept {
  auto A = A_.get_data().get_pointer() + A_.get_access_displacement();
  auto invA = invA_.get_data().get_pointer() + invA_.get_access_displacement();
  value_t* local = localMem.localAcc.get_pointer();

  const size_t threadIndex = item.get_local_id(0);
  const size_t i = threadIndex;
  const size_t blockIndex = item.get_group(0);

  // Sets the offset for this particular block in the source and destination
  // matrices
  const size_t blockIndexPerBlock = blockIndex * internalBlockSize;
  const size_t srcBlockOffset =
      blockIndex * (internalBlockSize + lda_ * internalBlockSize);
  const size_t numInnerBlocks = outterBlockSize / internalBlockSize;
  const size_t blockIndexDiv = blockIndex / numInnerBlocks;
  const size_t blockIndexMod = blockIndex % numInnerBlocks;
  // go to the blockIndexDiv outer outterBlockSize*outterBlockSize block
  const size_t offsetPart1 = blockIndexDiv * outterBlockSize * outterBlockSize;
  // then to the blockIndexMod inner internalBlockSize*internalBlockSize block
  // inside that
  const size_t offsetPart2 =
      blockIndexMod * (outterBlockSize * internalBlockSize + internalBlockSize);
  const size_t destBlockOffset = offsetPart1 + offsetPart2;

  // Loads the source lower triangle into local memory. Any values in the upper
  // triangle or outside of the matrix are set to zero
  for (size_t j = 0; j < internalBlockSize; ++j) {
    bool isInRange = false;
    if (upper) {
      isInRange = (i <= j) && (blockIndexPerBlock + j < N_);
    } else {
      isInRange = (i >= j) && ((blockIndexPerBlock + i) < N_);
    }
    if (isInRange) {
      const size_t srcIndex = j * lda_ + i + srcBlockOffset;
      local[j + i * internalBlockSize] = A[srcIndex];
    } else {
      local[j + i * internalBlockSize] = value_t{0};
    }
  }
  item.barrier(cl::sycl::access::fence_space::local_space);

  // Inverts the diagonal elements
  if (!unitDiag_) {
    local[i + i * internalBlockSize] =
        value_t{1} / local[i + i * internalBlockSize];
    item.barrier(cl::sycl::access::fence_space::local_space);
  }

  if (upper) {
    for (cl::sycl::cl_int j = 1; j < internalBlockSize; ++j) {
      value_t sum = value_t{0};
      if (i < j) {
        for (cl::sycl::cl_int k = 0; k < j; ++k) {
          sum = cl::sycl::mad(local[k + j * internalBlockSize],
                          local[j + k * internalBlockSize], sum);
        }
      }
      item.barrier(cl::sycl::access::fence_space::local_space);
      if (i < j) {
        const value_t negDiagValue = -local[j + j * internalBlockSize];
        local[j + i * internalBlockSize] = sum * negDiagValue;
      }
      item.barrier(cl::sycl::access::fence_space::local_space);
    }
  } else {
    // Computes the elements j+1:internalBlock-1 of the j-th column
    for (cl::sycl::cl_int j = internalBlockSize - 2; j >= 0; --j) {
      value_t sum = value_t{0};
      if (i > j) {
        for (cl::sycl::cl_int k = j + 1; k < internalBlockSize; ++k) {
          sum = cl::sycl::mad(local[k + i * internalBlockSize],
                          local[j + k * internalBlockSize], sum);
        }
      }
      item.barrier(cl::sycl::access::fence_space::local_space);
      if (i > j) {
        const value_t negDiagValue = -local[j + j * internalBlockSize];
        local[j + i * internalBlockSize] = sum * negDiagValue;
      }
      item.barrier(cl::sycl::access::fence_space::local_space);
    }
  }

  // Writes the result to global memory
  for (cl::sycl::cl_int j = 0; j < internalBlockSize; ++j) {
    invA[j * outterBlockSize + i + destBlockOffset] =
        local[j + i * internalBlockSize];
  }
}

}  // namespace blas

#endif  // SYCL_BLAS_BLAS3_REF_TRSM_HPP