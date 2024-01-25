/***************************************************************************
 *
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
 *  portBLAS: BLAS implementation using SYCL
 *
 **************************************************************************/

#ifndef PORTBLAS_BLAS3_TRSM_INTERFACE_HPP
#define PORTBLAS_BLAS3_TRSM_INTERFACE_HPP

#include "blas_meta.h"
#include "interface/gemm_interface.hpp"
#include "operations/blas3_trees.h"
#include "sb_handle/portblas_handle.h"
#include "portblas_helper.h"
#include "views/view.h"

namespace blas {
namespace internal {

/**
 * @brief Implementation of Triangle Solve with Multiple Right Hand Sides
 * (TRSM).
 * @param side Indicates if A is on the left or right of X
 * @param uplo Indicates if A is lower or upper triangular
 * @param trans Indicates the form that the matrix A will take in the
 * multiplication
 * @param diag Indicates if A has a non-unit diagonal or is assumed to be
 *             unit diagonal.
 * @param M The number of rows of matrix B, must be at least 1
 * @param N The number of columns of B, must be at least 1
 * @param alpha The scalar alpha that is applied to B
 * @param A Memory object that holds the input matrix A
 * @param lda Leading dimension of matrix A
 * @param B Memory object that holds the input/output matrix B
 * @param ldb Leading dimension of matrix B
 *
 * @note both matrices A and B are expected to be stored in column major order
 *
 * Documentation from LAPACK's reference implementation can be found here:
 * http://www.netlib.org/lapack/explore-html/d1/d54/group__double__blas__level3_ga6a0a7704f4a747562c1bd9487e89795c.html#ga6a0a7704f4a747562c1bd9487e89795c.
 *
 * TRSM solves one of the following matrix equations
 *
 * op(A)*X = alpha*B      or     X*op(A) = alpha*B
 *
 * where alpha is a scalar, X and B are m by n matrices, A is a unit or
 * non-unit, upper or lower triangular matrix and op(A) is
 *
 * op(A) = A    or     op(A) = A^{T}
 *
 * The matrix X, which contains the result, is copied to B at the end.
 *
 * This is the parallel version of TRSM, that works by solving the equation
 * AX = B as X = A^{-1}B. Inverting the matrix A is usually not the recommended
 * way of solving this system, however, being a triangular matrix, we can use
 * a block decomposition like the following:
 *
 *        A            X       alpha *  B
 *   [ A00   0  ] * [ X0 ]  =         [ B0 ]
 *   [ A10  A11 ]   [ X1 ]            [ B1 ]
 *
 * This is an example where A is on the left side and is a lower triangular
 * matrix. The matrices can be divided in as many blocks as necessary. This
 * decomposition yields:
 *
 * A00*X0          = alpha*B0    ==>   X0 = alpha*A00^{-1}*B0
 * A01*X0 + A11*X1 = alpha*B1    ==>   X1 = A11^{-1}*(alpha*B1 - A10*X0)
 *
 * Which implies that we only need to invert A00 and A11 (or the diagonal blocks
 * of matrix A). The function @ref make_diagonal_blocks_inverter can be used to
 * perform this operation. The process of obtaining X0 and X1 is now mapped into
 * 3 GEMM calls, one to solve X0, and 2 two solve X1.
 *
 * GEMM evaluates the expression C = alpha*A*B + beta*C, so solving for X0
 * becomes a GEMM call in the format:
 *
 *  X0 = alpha * A00^{-1}*B0 + 0*X0
 *
 * With X0 calculated we can solve X1 with two more GEMM calls
 *
 *  B1 = -1 * A01*X0      + alpha*B1
 *  X1 =  1 * A11^{-1}*B1 +     0*X1
 *
 * This step can be repeated as many times as necessary for larger matrices.
 * Despite having to invert blocks of the matrix A, this TRSM implementation
 * takes advantage of GEMM calls that are heavily optimized for the target
 * hardware, thus running with maximum performance.
 */
template <typename sb_handle_t, typename container_0_t, typename container_1_t,
          typename element_t, typename index_t>
typename sb_handle_t::event_t _trsm(
    sb_handle_t& sb_handle, char side, char uplo, char trans, char diag,
    index_t M, index_t N, element_t alpha, container_0_t A, index_t lda,
    container_1_t B, index_t ldb,
    const typename sb_handle_t::event_t& _dependencies) {
  // Makes sure all dimensions are larger than zero
  if ((M == 0) || (N == 0) || (lda == 0) || (ldb == 0)) {
    throw std::invalid_argument("invalid matrix size argument");
  }

  side = tolower(side);
  uplo = tolower(uplo);
  trans = tolower(trans);
  diag = tolower(diag);

  if (side != 'l' && side != 'r') {
    throw std::invalid_argument("invalid Side argument");
  } else if (uplo != 'u' && uplo != 'l') {
    throw std::invalid_argument("invalid Triangle argument");
  } else if (trans != 'n' && trans != 't') {
    throw std::invalid_argument("invalid Transpose argument");
  } else if (diag != 'u' && diag != 'n') {
    throw std::invalid_argument("invalid Diagonal argument");
  }

  // Computes the k dimension. This is based on whether or not matrix is A (on
  // the left) or B (on the right) in the gemm routine.
  const index_t K = (side == 'l') ? M : N;

  const bool isUnitDiag = diag == 'u';
  const bool isUpper = uplo == 'u';
  const bool isLeft = side == 'l';
  const bool isTranspose = trans == 't';

  constexpr index_t blockSize = 16;

  typename sb_handle_t::event_t trsmEvents;

  // Temporary buffer for the inverse of the diagonal blocks of the matrix A
  // filled with zeroes
  const index_t invASize = roundUp<index_t>(K, blockSize) * blockSize;
  constexpr bool is_usm = std::is_pointer<container_0_t>::value;
  auto invA = sb_handle.template acquire_temp_mem < is_usm
                  ? helper::AllocType::usm
                  : helper::AllocType::buffer,
       element_t > (invASize);
  typename sb_handle_t::event_t event = {blas::helper::fill(
      sb_handle.get_queue(), invA, element_t{0}, invASize, _dependencies)};
  trsmEvents = concatenate_vectors(trsmEvents, event);

  // Create the matrix views from the input buffers
  typename MatrixViewType<container_0_t, index_t, col_major>::type bufferA =
      make_matrix_view<col_major>(A, K, K, lda);
  auto bufferInvA =
      make_matrix_view<col_major>(invA, blockSize, blockSize, lda);

  // Calculate the parameters for the diagonal blocks inversion
  const index_t numBlocks = roundUp<index_t>(K, blockSize) / blockSize;
  const index_t numInternalBlocks = roundUp<index_t>(K, blockSize) / blockSize;
  const index_t globalSize = numInternalBlocks * blockSize;
  const index_t localSize = blockSize;
  const index_t localMemSize = blockSize * blockSize;

  // Instantiate the appropriate diagonal blocks inversion based on the matrix
  // type
  typename sb_handle_t::event_t invertBlocksEvent;
  if (isUnitDiag && isUpper) {
    auto diagInverter =
        make_diag_blocks_inverter<true, true, blockSize>(bufferA, bufferInvA);
    invertBlocksEvent = sb_handle.execute(diagInverter, localSize, globalSize,
                                          localMemSize, event);
  } else if (!isUnitDiag && isUpper) {
    auto diagInverter =
        make_diag_blocks_inverter<false, true, blockSize>(bufferA, bufferInvA);
    invertBlocksEvent = sb_handle.execute(diagInverter, localSize, globalSize,
                                          localMemSize, event);
  } else if (isUnitDiag && !isUpper) {
    auto diagInverter =
        make_diag_blocks_inverter<true, false, blockSize>(bufferA, bufferInvA);
    invertBlocksEvent = sb_handle.execute(diagInverter, localSize, globalSize,
                                          localMemSize, event);
  } else if (!isUnitDiag && !isUpper) {
    auto diagInverter =
        make_diag_blocks_inverter<false, false, blockSize>(bufferA, bufferInvA);
    invertBlocksEvent = sb_handle.execute(diagInverter, localSize, globalSize,
                                          localMemSize, event);
  }
  trsmEvents = concatenate_vectors(trsmEvents, invertBlocksEvent);

  // Creates a copy of B to avoid overwriting the input in GEMM. While computing
  // output X will hold the TRSM result and will be copied to B at the end
  const index_t BSize = ldb * (N - 1) + M;
  const index_t ldx = ldb;
  auto X = sb_handle.template acquire_temp_mem < is_usm
               ? helper::AllocType::usm
               : helper::AllocType::buffer,
       element_t > (BSize);
  trsmEvents = concatenate_vectors(
      trsmEvents,
      internal::_copy<sb_handle_t, index_t, decltype(B), decltype(X), index_t>(
          sb_handle, BSize, B, 1, X, 1, trsmEvents));

  if (isLeft) {
    if ((isUpper && isTranspose) || (!isUpper && !isTranspose)) {
      // Solves the system AX = alpha*B, as described in the documentation of
      // the function when X is lower triangular.
      //
      //         A            X                 B
      //    [ A00   0  ] * [ X0 ]  =  alpha * [ B0 ]
      //    [ A10  A11 ]   [ X1 ]             [ B1 ]
      //
      // yields:
      //
      //  X0 = alpha*A00{-1}*B0
      //  B1 = -1 * A10*X0 + alpha*B1
      //  X1 = A11{-1}*B1  + 0*X1
      //

      // True when (lower triangular) or (upper triangular and transposed)
      for (index_t i = 0; i < M; i += blockSize) {
        const index_t currentBlockSize = std::min(M - i, blockSize);
        auto gemmEvent = internal::_gemm(
            sb_handle, isTranspose ? 't' : 'n', 'n', currentBlockSize, N,
            currentBlockSize, (i == 0) ? alpha : element_t{1},
            invA + i * blockSize, blockSize, B + i, ldb, element_t{0}, X + i,
            ldx, trsmEvents);
        trsmEvents = concatenate_vectors(trsmEvents, gemmEvent);

        if ((i + blockSize) >= M) {
          break;
        }

        const std::ptrdiff_t offsetA = !isTranspose
                                           ? ((i + blockSize) + (i * lda))
                                           : (i + (blockSize + i) * lda);

        helper::add_const<container_0_t> a_ = A + offsetA;
        helper::add_const<container_1_t> b_ = X + i;
        gemmEvent = internal::_gemm(
            sb_handle, isTranspose ? 't' : 'n', 'n', M - i - blockSize, N,
            blockSize, element_t{-1}, a_, lda, b_, ldx,
            (i == 0) ? alpha : element_t{1}, B + i + blockSize, ldb, gemmEvent);
        trsmEvents = concatenate_vectors(trsmEvents, gemmEvent);
      }
    } else {
      // Solves the system AX = alpha*B when X is upper triangular
      //
      //         A            X                 B
      //    [ A00  A01  ] * [ X0 ]  =  alpha * [ B0 ]
      //    [  0   A11  ]   [ X1 ]             [ B1 ]
      //
      // yields:
      //
      //  X1 = alpha*A11{-1}*B1
      //  B0 = -1 * A01*X1 + alpha*B0
      //  X0 = A00{-1}*B0  + 0*X0
      //

      // True when (upper triangular) or (lower triangular and transposed)
      const index_t specialBlockSize =
          (M % blockSize == 0) ? blockSize : (M % blockSize);
      const index_t iStart = M - specialBlockSize;
      for (index_t i = iStart; i >= 0; i -= blockSize) {
        const index_t currentBlockSize =
            (i == iStart) ? specialBlockSize : blockSize;
        auto gemmEvent = internal::_gemm(
            sb_handle, isTranspose ? 't' : 'n', 'n', currentBlockSize, N,
            currentBlockSize, (i == iStart) ? alpha : element_t{1},
            invA + i * blockSize, blockSize, B + i, ldb, element_t{0}, X + i,
            ldx, trsmEvents);
        trsmEvents = concatenate_vectors(trsmEvents, gemmEvent);

        if ((i - blockSize) < 0) {
          break;
        }

        helper::add_const<container_0_t> a_ =
            A + (!isTranspose ? (i * lda) : i);
        helper::add_const<container_1_t> b_ = X + i;
        gemmEvent = internal::_gemm(
            sb_handle, isTranspose ? 't' : 'n', 'n', i, N, currentBlockSize,
            element_t{-1}, a_, lda, b_, ldx,
            (i == iStart) ? alpha : element_t{1}, B, ldb, gemmEvent);
        trsmEvents = concatenate_vectors(trsmEvents, gemmEvent);
      }
    }
  } else {
    // Right side

    if ((isUpper && isTranspose) || (!isUpper && !isTranspose)) {
      // Solves the system XA = alpha*B when A is lower triangular

      //         X     *      A                        B
      //    [ X0  X1 ]   [ A00   0   ]  =  alpha * [ B0  B1 ]
      //                 [ A10  A11  ]
      //
      // yields:
      //
      //  X1 = alpha*B1*A11{-1}
      //  B0 = -1 * X1*A10 + alpha*B0
      //  X0 = B0*A00{-1}  + 0*X0
      //

      // True when (lower triangular) or (upper triangular and transposed)
      const index_t specialBlockSize =
          (N % blockSize == 0) ? blockSize : (N % blockSize);
      const index_t iStart = N - specialBlockSize;
      for (index_t i = iStart; i >= 0; i -= blockSize) {
        const index_t currentBlockSize =
            (i == iStart) ? specialBlockSize : blockSize;
        auto gemmEvent = internal::_gemm(
            sb_handle, 'n', isTranspose ? 't' : 'n', M, currentBlockSize,
            currentBlockSize, (i == iStart) ? alpha : element_t{1}, B + i * ldb,
            ldb, invA + i * blockSize, blockSize, element_t{0}, X + i * ldx,
            ldx, trsmEvents);
        trsmEvents = concatenate_vectors(trsmEvents, gemmEvent);

        if ((i - blockSize) < 0) {
          break;
        }

        helper::add_const<container_1_t> a_ = X + i * ldx;
        helper::add_const<container_0_t> b_ =
            A + (!isTranspose ? i : (i * lda));
        gemmEvent = internal::_gemm(
            sb_handle, 'n', isTranspose ? 't' : 'n', M, i, currentBlockSize,
            element_t{-1}, a_, ldx, b_, lda,
            (i == iStart) ? alpha : element_t{1}, B, ldb, gemmEvent);
        trsmEvents = concatenate_vectors(trsmEvents, gemmEvent);
      }

    } else {
      // Solves the system XA = alpha*B when A is upper triangular

      //      X        *      A                         B
      //    [ X0  X1 ]   [ A00  A01  ]  =  alpha * [ B0  B1 ]
      //                 [  0   A11  ]
      //
      // yields:
      //
      //  X0 = alpha*B0*A00^{-1}
      //  B1 = -1 * X0*A01 + alpha*B1
      //  X1 = B1*A11{-1}  + 0*X1
      //

      // True when (upper triangular) or (lower triangular and transposed)
      for (index_t i = 0; i < N; i += blockSize) {
        const index_t currentBlockSize = std::min(N - i, blockSize);

        auto gemmEvent = internal::_gemm(
            sb_handle, 'n', isTranspose ? 't' : 'n', M, currentBlockSize,
            currentBlockSize, (i == 0) ? alpha : element_t{1}, B + i * ldb, ldb,
            invA + i * blockSize, blockSize, element_t{0}, X + i * ldx, ldx,
            trsmEvents);
        trsmEvents = concatenate_vectors(trsmEvents, gemmEvent);

        if ((i + blockSize) > N) {
          break;
        }

        const std::ptrdiff_t offset = !isTranspose
                                          ? (i + (blockSize + i) * lda)
                                          : (i + blockSize) + (i * lda);

        helper::add_const<container_1_t> a_ = X + i * ldx;
        helper::add_const<container_0_t> b_ = A + offset;
        gemmEvent =
            internal::_gemm(sb_handle, 'n', isTranspose ? 't' : 'n', M,
                            N - i - blockSize, blockSize, element_t{-1}, a_,
                            ldx, b_, lda, (i == 0) ? alpha : element_t{1},
                            B + (i + blockSize) * ldb, ldb, gemmEvent);
        trsmEvents = concatenate_vectors(trsmEvents, gemmEvent);
      }
    }
  }

  // Copy bufferX to bufferB as the TRSM result
  typename sb_handle_t::event_t lastEvent;
  trsmEvents = concatenate_vectors(
      trsmEvents, lastEvent = internal::_copy<sb_handle_t, index_t, decltype(X),
                                              decltype(B), index_t>(
                      sb_handle, BSize, X, 1, B, 1, trsmEvents));

  sb_handle.template release_temp_mem(lastEvent, invA);

  sb_handle.template release_temp_mem(lastEvent, X);

  return trsmEvents;
}

}  // namespace internal
}  // namespace blas

#endif  // PORTBLAS_BLAS3_TRSM_INTERFACE_HPP
