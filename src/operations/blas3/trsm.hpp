/***************************************************************************
 *  @license
 *  Copyright (C) Codeplay Software Limited
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  For your convenience, a copy of the License has been included in this
 *  repository.
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 *
 *  SYCL-BLAS: BLAS implementation using SYCL
 *
 **************************************************************************/

#ifndef SYCL_BLAS_BLAS3_TRSM_HPP
#define SYCL_BLAS_BLAS3_TRSM_HPP

#include "operations/blas3_trees.h"
#include "views/view.h"

#include <CL/sycl.hpp>

namespace blas {

template <bool UnitDiag, bool Upper, int BlockSize, typename matrix_t>
SYCL_BLAS_INLINE DiagonalBlocksInverter<UnitDiag, Upper, BlockSize, matrix_t>::
    DiagonalBlocksInverter(matrix_t& A, matrix_t& invA)
    : A_(A), invA_(invA), N_(A_.get_size_col()), lda_(A_.getSizeL()) {}

template <bool UnitDiag, bool Upper, int BlockSize, typename matrix_t>
SYCL_BLAS_INLINE bool
DiagonalBlocksInverter<UnitDiag, Upper, BlockSize, matrix_t>::valid_thread(
    cl::sycl::nd_item<1> id) const {
  return true;
}

template <bool UnitDiag, bool Upper, int BlockSize, typename matrix_t>
SYCL_BLAS_INLINE void
DiagonalBlocksInverter<UnitDiag, Upper, BlockSize, matrix_t>::bind(
    cl::sycl::handler& cgh) {
  A_.bind(cgh);
  invA_.bind(cgh);
}

template <bool UnitDiag, bool Upper, int BlockSize, typename matrix_t>
SYCL_BLAS_INLINE void DiagonalBlocksInverter<
    UnitDiag, Upper, BlockSize, matrix_t>::adjust_access_displacement() {
  A_.adjust_access_displacement();
  invA_.adjust_access_displacement();
}

template <bool UnitDiag, bool Upper, int BlockSize, typename matrix_t>
template <typename local_memory_t>
SYCL_BLAS_INLINE void
DiagonalBlocksInverter<UnitDiag, Upper, BlockSize, matrix_t>::eval(
    local_memory_t localMem, cl::sycl::nd_item<1> item) noexcept {
  auto A = A_.get_data().get_pointer() + A_.get_access_displacement();
  auto invA = invA_.get_data().get_pointer() + invA_.get_access_displacement();
  value_t* local = localMem.localAcc.get_pointer();

  const index_t i = item.get_local_id(0);
  const index_t blockIndex = item.get_group(0);

  // Sets the offset for this particular block in the source and destination
  // matrices
  const index_t blockIndexPerBlock = blockIndex * internalBlockSize;
  const index_t srcBlockOffset =
      blockIndex * (internalBlockSize + lda_ * internalBlockSize);
  const index_t numInnerBlocks = outterBlockSize / internalBlockSize;
  const index_t blockIndexDiv = blockIndex / numInnerBlocks;
  const index_t blockIndexMod = blockIndex % numInnerBlocks;
  // go to the blockIndexDiv outer outterBlockSize*outterBlockSize block
  const index_t offsetPart1 = blockIndexDiv * outterBlockSize * outterBlockSize;
  // then to the blockIndexMod inner internalBlockSize*internalBlockSize block
  // inside that
  const index_t offsetPart2 =
      blockIndexMod * (outterBlockSize * internalBlockSize + internalBlockSize);
  const index_t destBlockOffset = offsetPart1 + offsetPart2;

  // Loads the source lower triangle into local memory. Any values in the upper
  // triangle or outside of the matrix are set to zero
  for (index_t j = 0; j < internalBlockSize; ++j) {
    bool isInRange = false;
    isInRange = (Upper) ? (i <= j) && ((blockIndexPerBlock + j) < N_)
                        : (i >= j) && ((blockIndexPerBlock + i) < N_);
    local[j + i * internalBlockSize] =
        (isInRange) ? A[j * lda_ + i + srcBlockOffset] : value_t{0};
  }
  item.barrier(cl::sycl::access::fence_space::local_space);

  // Inverts the diagonal elements
  if (!UnitDiag) {
    local[i + i * internalBlockSize] =
        value_t{1} / local[i + i * internalBlockSize];
    item.barrier(cl::sycl::access::fence_space::local_space);
  }

  if (Upper) {
    for (index_t j = 1; j < internalBlockSize; ++j) {
      value_t sum = value_t{0};
      if (i < j) {
        for (index_t k = 0; k < j; ++k) {
          sum = cl::sycl::mad(local[k + i * internalBlockSize],
                              local[j + k * internalBlockSize], sum);
        }
      }
      item.barrier(cl::sycl::access::fence_space::local_space);
      if (i < j) {
        local[j + i * internalBlockSize] =
            sum * -local[j + j * internalBlockSize];
      }
      item.barrier(cl::sycl::access::fence_space::local_space);
    }
  } else {
    // Computes the elements j+1:internalBlock-1 of the j-th column
    for (index_t j = internalBlockSize - 2; j >= 0; --j) {
      value_t sum = value_t{0};
      if (i > j) {
        for (index_t k = j + 1; k < internalBlockSize; ++k) {
          sum = cl::sycl::mad(local[k + i * internalBlockSize],
                              local[j + k * internalBlockSize], sum);
        }
      }
      item.barrier(cl::sycl::access::fence_space::local_space);
      if (i > j) {
        local[j + i * internalBlockSize] =
            sum * -local[j + j * internalBlockSize];
      }
      item.barrier(cl::sycl::access::fence_space::local_space);
    }
  }

  // Writes the result to global memory
  for (index_t j = 0; j < internalBlockSize; ++j) {
    invA[j * outterBlockSize + i + destBlockOffset] =
        local[j + i * internalBlockSize];
  }
}

}  // namespace blas

#endif  // SYCL_BLAS_BLAS3_TRSM_HPP
