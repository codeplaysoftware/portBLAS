
#ifndef SYCL_BLAS_BLAS3_TRSM_HPP
#define SYCL_BLAS_BLAS3_TRSM_HPP

#include "blas_meta.h"
#include "executors/executor.h"
#include "interface/gemm.hpp"
#include "operations/blas3_trees.h"
#include "policy/sycl_policy_handler.h"

namespace blas {
namespace internal {

/**
 * @brief Implementation of Triangle Solve with Multiple Right Hand Sides
 * (TRSM).
 * @param Side Indicates if A is on the left or right of X
 * @param Triangle If A is lower or upper triangular
 * @param Transpose indicates the form that the matrix A will take in the
 * multiplication
 * @param Diagonal If A has a unit or non-unit diagonal
 * @param M The number of rows in of matrix B, must be at least 0
 * @param N The number of columns of B, must be at least 0
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
 *        A            X         B
 *   [ A00   0  ] * [ X0 ]  =  [ B0 ]
 *   [ A10  A11 ]   [ X1 ]     [ B1 ]
 *
 * This is an example where A is on the left side and is a lower triangular
 * matrix and alpha is 1. The matrices can be divided in as many blocks as
 * necessary. This decomposition yields:
 *
 * A00*X0          = B0    ==>   X0 = A00^{-1}*B0
 * A01*X0 + A11*X1 = B1    ==>   X1 = A11^{-1}*(B1 - A01*X0)
 *
 * Which implies that we only need to invert A00 and A11 (or the diagonal blocks
 * of matrix A). The function @ref invert_diagonal_blocks can be used to perform
 * this operation. The process of obtaining X0 and X1 is now mapped into 3 GEMM
 * calls, one to solve X0, and 2 two solve X1.
 *
 * GEMM evaluates the expression C = alpha*A*B + beta*C, so solving for X0
 * becomes a GEMM call in the format:
 *
 *  X0 = 1 * A00^{-1}*B0 + 0*X0
 *
 * With X0 calculated we can solve X1 with two more GEMM calls
 *
 *  B1 = -1 * A01*X0      + 1*B1
 *  X1 =  1 * A11^{-1}*B1 + 0*X1
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
  const index_t invASize = roundUp<index_t>(N, blockSize) * blockSize;
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
        make_diag_blocks_inverter<true, true>(bufferA, bufferInvA);
    invertBlocksEvent =
        ex.execute(diagInverter, localSize, globalSize, localMemSize);
  } else if (!isUnitDiag && isUpper) {
    auto diagInverter =
        make_diag_blocks_inverter<false, true>(bufferA, bufferInvA);
    invertBlocksEvent =
        ex.execute(diagInverter, localSize, globalSize, localMemSize);
  } else if (isUnitDiag && !isUpper) {
    auto diagInverter =
        make_diag_blocks_inverter<true, false>(bufferA, bufferInvA);
    invertBlocksEvent =
        ex.execute(diagInverter, localSize, globalSize, localMemSize);
  } else if (!isUnitDiag && !isUpper) {
    auto diagInverter =
        make_diag_blocks_inverter<false, false>(bufferA, bufferInvA);
    invertBlocksEvent =
        ex.execute(diagInverter, localSize, globalSize, localMemSize);
  }

  // Creates a copy of B to avoid overwriting the input in GEMM. While computing
  // output X will hold the TRSM result and will be copied to B at the end
  const index_t BSize = ldb * (N - 1) + M;
  const index_t ldx = ldb;
  auto X = make_sycl_iterator_buffer<element_t>(BSize);
  auto bufferX = make_matrix_view<col_major>(ex, X, M, N, ldx);
  trsmEvents = concatenate_vectors(
      trsmEvents, ex.execute(make_op<Assign>(bufferX, bufferB), BSize));

  if (isLeft) {
    if ((isUpper && isTranspose) || (!isUpper && !isTranspose)) {
      // Solves the system AX = alpha*B, as described in the documentation of
      // the function when X is lower triangular

      // True when (lower triangular) or (upper triangular and transposed)
      for (index_t i = 0; i < M; i += blockSize) {
        const index_t currentBlockSize = std::min(M - i, blockSize);
        index_t gemmM = currentBlockSize;
        index_t gemmN = N;
        index_t gemmK = currentBlockSize;
        char gemmTransA = isTranspose ? 't' : 'n';
        char gemmTransB = 'n';
        element_t gemmAlpha = (i == 0) ? alpha : element_t{1};
        element_t gemmBeta = element_t{0};
        index_t gemmLda = blockSize;
        index_t gemmLdb = ldb;
        index_t gemmLdc = ldx;
        std::ptrdiff_t offsetA = i * blockSize;
        std::ptrdiff_t offsetB = i;
        std::ptrdiff_t offsetC = i;
        auto gemmBufferA = invA + offsetA;
        auto gemmBufferB = B + offsetB;
        auto gemmBufferC = X + offsetC;
        auto gemmEvent =
            internal::_gemm(ex, gemmTransA, gemmTransB, gemmM, gemmN, gemmK,
                            gemmAlpha, gemmBufferA, gemmLda, gemmBufferB,
                            gemmLdb, gemmBeta, gemmBufferC, gemmLdc);
        trsmEvents = concatenate_vectors(trsmEvents, gemmEvent);

        if ((i + blockSize) >= M) {
          break;
        }

        gemmM = M - i - blockSize;
        gemmN = N;
        gemmK = blockSize;
        gemmTransA = isTranspose ? 't' : 'n';
        gemmTransB = 'n';
        gemmAlpha = element_t{-1};
        gemmBeta = (i == 0) ? alpha : element_t{1};
        gemmLda = lda;
        gemmLdb = ldx;
        gemmLdc = ldb;
        offsetA = !isTranspose ? ((i + blockSize) + (i * lda))
                               : (i + (blockSize + i) * lda);
        offsetB = i;
        offsetC = i + blockSize;
        gemmBufferA = A + offsetA;
        gemmBufferB = X + offsetB;
        gemmBufferC = B + offsetC;
        gemmEvent =
            internal::_gemm(ex, gemmTransA, gemmTransB, gemmM, gemmN, gemmK,
                            gemmAlpha, gemmBufferA, gemmLda, gemmBufferB,
                            gemmLdb, gemmBeta, gemmBufferC, gemmLdc);
        trsmEvents = concatenate_vectors(trsmEvents, gemmEvent);
      }
    } else {
      // Solves the system AX = alpha*B when X is upper triangular

      // True when (upper triangular) or (lower triangular and transposed)
      const index_t specialBlockSize =
          (M % blockSize == 0) ? blockSize : (M % blockSize);
      const index_t iStart = M - specialBlockSize;
      for (int i = iStart; i >= 0; i -= blockSize) {
        const index_t currentBlockSize =
            (i == iStart) ? specialBlockSize : blockSize;
        index_t gemmM = currentBlockSize;
        index_t gemmN = N;
        index_t gemmK = currentBlockSize;
        char gemmTransA = isTranspose ? 't' : 'n';
        char gemmTransB = 'n';
        element_t gemmAlpha = (i == iStart) ? alpha : element_t{1};
        element_t gemmBeta = element_t{0};
        index_t gemmLda = static_cast<int>(blockSize);
        index_t gemmLdb = static_cast<int>(ldb);
        index_t gemmLdc = static_cast<int>(ldx);
        std::ptrdiff_t gemmOffsetA = i * blockSize;
        std::ptrdiff_t gemmOffsetB = i;
        std::ptrdiff_t gemmOffsetC = i;
        auto gemmBufferA = invA + gemmOffsetA;
        auto gemmBufferB = B + gemmOffsetB;
        auto gemmBufferC = X + gemmOffsetC;
        auto gemmEvent =
            internal::_gemm(ex, gemmTransA, gemmTransB, gemmM, gemmN, gemmK,
                            gemmAlpha, gemmBufferA, gemmLda, gemmBufferB,
                            gemmLdb, gemmBeta, gemmBufferC, gemmLdc);
        trsmEvents = concatenate_vectors(trsmEvents, gemmEvent);

        if ((i - blockSize) < 0) {
          break;
        }

        gemmM = i;
        gemmN = N;
        gemmK = currentBlockSize;
        gemmTransA = isTranspose ? 't' : 'n';
        gemmTransB = 'n';
        gemmAlpha = element_t{-1};
        gemmBeta = (i == iStart) ? alpha : element_t{1};
        gemmLda = lda;
        gemmLdb = ldx;
        gemmLdc = ldb;
        gemmOffsetA = !isTranspose ? (i * lda) : i;
        gemmOffsetB = i;
        gemmOffsetC = 0;
        gemmBufferA = A + gemmOffsetA;
        gemmBufferB = X + gemmOffsetB;
        gemmBufferC = B + gemmOffsetC;
        gemmEvent =
            internal::_gemm(ex, gemmTransA, gemmTransB, gemmM, gemmN, gemmK,
                            gemmAlpha, gemmBufferA, gemmLda, gemmBufferB,
                            gemmLdb, gemmBeta, gemmBufferC, gemmLdc);
        trsmEvents = concatenate_vectors(trsmEvents, gemmEvent);
      }
    }
  } else {
    // Right side

    if ((isUpper && isTranspose) || (!isUpper && !isTranspose)) {
      // Solves the system XA = alpha*B when A is lower triangular

      // True when (lower triangular) or (upper triangular and transposed)
      const index_t specialBlockSize =
          (N % blockSize == 0) ? blockSize : (N % blockSize);
      const index_t iStart = N - specialBlockSize;
      for (int i = iStart; i >= 0; i -= blockSize) {
        const index_t currentBlockSize =
            (i == iStart) ? specialBlockSize : blockSize;

        index_t gemmM = M;
        index_t gemmN = currentBlockSize;
        index_t gemmK = currentBlockSize;
        char gemmTransA = 'n';
        char gemmTransB = isTranspose ? 't' : 'n';
        element_t gemmAlpha = (i == iStart) ? alpha : element_t{1};
        element_t gemmBeta = element_t{0};
        index_t gemmLda = ldb;
        index_t gemmLdb = blockSize;
        index_t gemmLdc = ldx;
        std::ptrdiff_t gemmOffsetA = i * ldb;
        std::ptrdiff_t gemmOffsetB = i * blockSize;
        std::ptrdiff_t gemmOffsetC = i * ldx;
        auto gemmBufferA = B + gemmOffsetA;
        auto gemmBufferB = invA + gemmOffsetB;
        auto gemmBufferC = X + gemmOffsetC;
        auto gemmEvent =
            internal::_gemm(ex, gemmTransA, gemmTransB, gemmM, gemmN, gemmK,
                            gemmAlpha, gemmBufferA, gemmLda, gemmBufferB,
                            gemmLdb, gemmBeta, gemmBufferC, gemmLdc);
        trsmEvents = concatenate_vectors(trsmEvents, gemmEvent);

        if ((i - blockSize) < 0) {
          break;
        }

        gemmM = M;
        gemmN = i;
        gemmK = currentBlockSize;
        gemmTransA = 'n';
        gemmTransB = isTranspose ? 't' : 'n';
        gemmAlpha = element_t{-1};
        gemmBeta = (i == iStart) ? alpha : element_t{1};
        gemmLda = ldx;
        gemmLdb = lda;
        gemmLdc = ldb;
        gemmOffsetA = i * ldx;
        gemmOffsetB = !isTranspose ? i : (i * lda);
        gemmOffsetC = 0;
        gemmBufferA = X + gemmOffsetA;
        gemmBufferB = A + gemmOffsetB;
        gemmBufferC = B + gemmOffsetC;
        gemmEvent =
            internal::_gemm(ex, gemmTransA, gemmTransB, gemmM, gemmN, gemmK,
                            gemmAlpha, gemmBufferA, gemmLda, gemmBufferB,
                            gemmLdb, gemmBeta, gemmBufferC, gemmLdc);
        trsmEvents = concatenate_vectors(trsmEvents, gemmEvent);
      }

    } else {
      // Solves the system XA = alpha*B when A is upper triangular

      // True when (upper triangular) or (lower triangular and transposed)
      for (index_t i = 0; i < N; i += blockSize) {
        const index_t currentBlockSize = std::min(N - i, blockSize);

        index_t gemmM = M;
        index_t gemmN = currentBlockSize;
        index_t gemmK = currentBlockSize;
        char gemmTransA = 'n';
        char gemmTransB = isTranspose ? 't' : 'n';
        element_t gemmAlpha = (i == 0) ? alpha : element_t{1};
        element_t gemmBeta = element_t{0};
        index_t gemmLda = ldb;
        index_t gemmLdb = blockSize;
        index_t gemmLdc = ldx;
        std::ptrdiff_t gemmOffsetA = i * ldb;
        std::ptrdiff_t gemmOffsetB = i * blockSize;
        std::ptrdiff_t gemmOffsetC = i * ldx;
        auto gemmBufferA = B + gemmOffsetA;
        auto gemmBufferB = invA + gemmOffsetB;
        auto gemmBufferC = X + gemmOffsetC;
        auto gemmEvent =
            internal::_gemm(ex, gemmTransA, gemmTransB, gemmM, gemmN, gemmK,
                            gemmAlpha, gemmBufferA, gemmLda, gemmBufferB,
                            gemmLdb, gemmBeta, gemmBufferC, gemmLdc);
        trsmEvents = concatenate_vectors(trsmEvents, gemmEvent);

        if ((i + blockSize) > N) {
          break;
        }

        gemmM = M;
        gemmN = N - i - blockSize;
        gemmK = blockSize;
        gemmTransA = 'n';
        gemmTransB = isTranspose ? 't' : 'n';
        gemmAlpha = element_t{-1};
        gemmBeta = (i == 0) ? alpha : element_t{1};
        gemmLda = ldx;
        gemmLdb = lda;
        gemmLdc = ldb;
        gemmOffsetA = i * ldx;
        gemmOffsetB = !isTranspose ? (i + (blockSize + i) * lda)
                                   : (i + blockSize) + (i * lda);
        gemmOffsetC = (i + blockSize) * ldb;
        gemmBufferA = X + gemmOffsetA;
        gemmBufferB = A + gemmOffsetB;
        gemmBufferC = B + gemmOffsetC;
        gemmEvent =
            internal::_gemm(ex, gemmTransA, gemmTransB, gemmM, gemmN, gemmK,
                            gemmAlpha, gemmBufferA, gemmLda, gemmBufferB,
                            gemmLdb, gemmBeta, gemmBufferC, gemmLdc);
        trsmEvents = concatenate_vectors(trsmEvents, gemmEvent);
      }
    }
  }

  // Copy bufferX to bufferB as the TRSM result
  trsmEvents = concatenate_vectors(
      trsmEvents, ex.execute(make_op<Assign>(bufferB, bufferX), BSize));

  return trsmEvents;
}

template <typename executor_t, typename container_0_t, typename container_1_t,
          typename element_t, typename index_t>
typename executor_t::policy_t::event_t inline _trsm(
    executor_t& ex, char Side, char Triangle, char Transpose, char Diagonal,
    index_t M, index_t N, element_t alpha, container_0_t A, index_t lda,
    container_1_t B, index_t ldb) {
  return _trsm_impl(ex, Side, Triangle, Transpose, Diagonal, M, N, alpha, A, lda, B, ldb);
}

}  // namespace internal
}  // namespace blas

#endif  // SYCL_BLAS_BLAS3_TRSM_HPP
