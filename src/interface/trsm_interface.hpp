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
 *  SYCL-BLAS: BLAS implementation using SYCL
 *
 **************************************************************************/

#ifndef SYCL_BLAS_BLAS3_TRSM_INTERFACE_HPP
#define SYCL_BLAS_BLAS3_TRSM_INTERFACE_HPP

#include "blas_meta.h"
#include "executors/executor.h"
#include "interface/gemm_interface.hpp"
#include "operations/blas3_trees.h"
#include "policy/sycl_policy_handler.h"

namespace blas {
namespace internal {

/**
 * @brief Implementation of Triangle Solve with Multiple Right Hand Sides
 * (TRSM).
 * @param Side Indicates if A is on the left or right of X
 * @param Triangle Indicates if A is lower or upper triangular
 * @param Transpose Indicates the form that the matrix A will take in the
 * multiplication
 * @param Diagonal Indicates if A has a unit or non-unit diagonal
 * @param M The number of rows of matrix B, must be at least 1
 * @param N The number of columns of B, must be at least 1
 * @param alpha The scalar alpha that is applied to B
 * @param A Buffer that holds the input matrix A
 * @param lda Leading dimension of matrix A
 * @param B Buffer that holds the input/output matrix B
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
template <typename executor_t, typename container_0_t, typename container_1_t,
          typename element_t, typename index_t>
typename executor_t::policy_t::event_t _trsm_impl(
    executor_t& ex, char Side, char Triangle, char Transpose, char Diagonal,
    index_t M, index_t N, element_t alpha, container_0_t A, index_t lda,
    container_1_t B, index_t ldb) {
  // Makes sure all dimensions are larger than zero
  if ((M == 0) || (N == 0) || (lda == 0) || (ldb == 0)) {
    throw std::invalid_argument("invalid matrix size argument");
  }

  Side = tolower(Side);
  Triangle = tolower(Triangle);
  Transpose = tolower(Transpose);
  Diagonal = tolower(Diagonal);

  if (Side != 'l' && Side != 'r') {
    throw std::invalid_argument("invalid Side argument");
  } else if (Triangle != 'u' && Triangle != 'l') {
    throw std::invalid_argument("invalid Triangle argument");
  } else if (Transpose != 'n' && Transpose != 't') {
    throw std::invalid_argument("invalid Transpose argument");
  } else if (Diagonal != 'u' && Diagonal != 'n') {
    throw std::invalid_argument("invalid Diagonal argument");
  }

  // Computes the k dimension. This is based on whether or not matrix is A (on
  // the left) or B (on the right) in the gemm routine.
  const index_t K = (Side == 'l') ? M : N;

  const bool isUnitDiag = Diagonal == 'u';
  const bool isUpper = Triangle == 'u';
  const bool isLeft = Side == 'l';
  const bool isTranspose = Transpose == 't';

  constexpr index_t blockSize = 16;

  typename executor_t::policy_t::event_t trsmEvents;

  // Temporary buffer for the inverse of the diagonal blocks of the matrix A
  // filled with zeroes
  const index_t invASize = roundUp<index_t>(K, blockSize) * blockSize;
  auto invA = make_sycl_iterator_buffer<element_t>(invASize);
  trsmEvents = concatenate_vectors(
      trsmEvents, ex.get_policy_handler().fill(invA, element_t{0}, invASize));

  // Create the matrix views from the input buffers
  auto bufferA = make_matrix_view<col_major>(ex, A, K, K, lda);
  auto bufferInvA =
      make_matrix_view<col_major>(ex, invA, blockSize, blockSize, lda);
  auto bufferB = make_matrix_view<col_major>(ex, B, M, N, ldb);

  // Calculate the parameters for the diagonal blocks inversion
  const index_t numBlocks = roundUp<index_t>(K, blockSize) / blockSize;
  const index_t numInternalBlocks = roundUp<index_t>(K, blockSize) / blockSize;
  const index_t globalSize = numInternalBlocks * blockSize;
  const index_t localSize = blockSize;
  const index_t localMemSize = blockSize * blockSize;

  // Instantiate the appropriate diagonal blocks inversion based on the matrix
  // type
  typename executor_t::policy_t::event_t invertBlocksEvent;
  if (isUnitDiag && isUpper) {
    auto diagInverter =
        make_diag_blocks_inverter<true, true, blockSize>(bufferA, bufferInvA);
    invertBlocksEvent =
        ex.execute(diagInverter, localSize, globalSize, localMemSize);
  } else if (!isUnitDiag && isUpper) {
    auto diagInverter =
        make_diag_blocks_inverter<false, true, blockSize>(bufferA, bufferInvA);
    invertBlocksEvent =
        ex.execute(diagInverter, localSize, globalSize, localMemSize);
  } else if (isUnitDiag && !isUpper) {
    auto diagInverter =
        make_diag_blocks_inverter<true, false, blockSize>(bufferA, bufferInvA);
    invertBlocksEvent =
        ex.execute(diagInverter, localSize, globalSize, localMemSize);
  } else if (!isUnitDiag && !isUpper) {
    auto diagInverter =
        make_diag_blocks_inverter<false, false, blockSize>(bufferA, bufferInvA);
    invertBlocksEvent =
        ex.execute(diagInverter, localSize, globalSize, localMemSize);
  }

  // Creates a copy of B to avoid overwriting the input in GEMM. While computing
  // output X will hold the TRSM result and will be copied to B at the end
  const index_t BSize = ldb * (N - 1) + M;
  const index_t ldx = ldb;
  auto X = make_sycl_iterator_buffer<element_t>(BSize);
  trsmEvents =
      concatenate_vectors(trsmEvents, internal::_copy(ex, BSize, B, 1, X, 1));

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
        auto gemmEvent = internal::_gemm(ex, isTranspose ? 't' : 'n', 'n',
                                         currentBlockSize, N, currentBlockSize,
                                         (i == 0) ? alpha : element_t{1},
                                         invA + i * blockSize, blockSize, B + i,
                                         ldb, element_t{0}, X + i, ldx);
        trsmEvents = concatenate_vectors(trsmEvents, gemmEvent);

        if ((i + blockSize) >= M) {
          break;
        }

        const std::ptrdiff_t offsetA = !isTranspose
                                           ? ((i + blockSize) + (i * lda))
                                           : (i + (blockSize + i) * lda);
        internal::_gemm(ex, isTranspose ? 't' : 'n', 'n', M - i - blockSize, N,
                        blockSize, element_t{-1}, A + offsetA, lda, X + i, ldx,
                        (i == 0) ? alpha : element_t{1}, B + i + blockSize,
                        ldb);
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
      for (int i = iStart; i >= 0; i -= blockSize) {
        const index_t currentBlockSize =
            (i == iStart) ? specialBlockSize : blockSize;
        auto gemmEvent = internal::_gemm(ex, isTranspose ? 't' : 'n', 'n',
                                         currentBlockSize, N, currentBlockSize,
                                         (i == iStart) ? alpha : element_t{1},
                                         invA + i * blockSize, blockSize, B + i,
                                         ldb, element_t{0}, X + i, ldx);
        trsmEvents = concatenate_vectors(trsmEvents, gemmEvent);

        if ((i - blockSize) < 0) {
          break;
        }

        gemmEvent = internal::_gemm(
            ex, isTranspose ? 't' : 'n', 'n', i, N, currentBlockSize,
            element_t{-1}, A + (!isTranspose ? (i * lda) : i), lda, X + i, ldx,
            (i == iStart) ? alpha : element_t{1}, B, ldb);
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
      for (int i = iStart; i >= 0; i -= blockSize) {
        const index_t currentBlockSize =
            (i == iStart) ? specialBlockSize : blockSize;
        auto gemmEvent = internal::_gemm(
            ex, 'n', isTranspose ? 't' : 'n', M, currentBlockSize,
            currentBlockSize, (i == iStart) ? alpha : element_t{1}, B + i * ldb,
            ldb, invA + i * blockSize, blockSize, element_t{0}, X + i * ldx,
            ldx);
        trsmEvents = concatenate_vectors(trsmEvents, gemmEvent);

        if ((i - blockSize) < 0) {
          break;
        }

        gemmEvent = internal::_gemm(
            ex, 'n', isTranspose ? 't' : 'n', M, i, currentBlockSize,
            element_t{-1}, X + i * ldx, ldx, A + (!isTranspose ? i : (i * lda)),
            lda, (i == iStart) ? alpha : element_t{1}, B, ldb);
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
            ex, 'n', isTranspose ? 't' : 'n', M, currentBlockSize,
            currentBlockSize, (i == 0) ? alpha : element_t{1}, B + i * ldb, ldb,
            invA + i * blockSize, blockSize, element_t{0}, X + i * ldx, ldx);
        trsmEvents = concatenate_vectors(trsmEvents, gemmEvent);

        if ((i + blockSize) > N) {
          break;
        }

        const std::ptrdiff_t offset = !isTranspose
                                          ? (i + (blockSize + i) * lda)
                                          : (i + blockSize) + (i * lda);
        gemmEvent = internal::_gemm(
            ex, 'n', isTranspose ? 't' : 'n', M, N - i - blockSize, blockSize,
            element_t{-1}, X + i * ldx, ldx, A + offset, lda,
            (i == 0) ? alpha : element_t{1}, B + (i + blockSize) * ldb, ldb);
        trsmEvents = concatenate_vectors(trsmEvents, gemmEvent);
      }
    }
  }

  // Copy bufferX to bufferB as the TRSM result
  trsmEvents =
      concatenate_vectors(trsmEvents, internal::_copy(ex, BSize, X, 1, B, 1));

  return trsmEvents;
}

template <typename executor_t, typename container_0_t, typename container_1_t,
          typename element_t, typename index_t>
typename executor_t::policy_t::event_t inline _trsm(
    executor_t& ex, char Side, char Triangle, char Transpose, char Diagonal,
    index_t M, index_t N, element_t alpha, container_0_t A, index_t lda,
    container_1_t B, index_t ldb) {
  return _trsm_impl(ex, Side, Triangle, Transpose, Diagonal, M, N, alpha, A,
                    lda, B, ldb);
}

}  // namespace internal
}  // namespace blas

#endif  // SYCL_BLAS_BLAS3_TRSM_INTERFACE_HPP
