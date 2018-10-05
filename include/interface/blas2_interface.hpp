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
 *  @filename blas2_interface.hpp
 *
 **************************************************************************/

#ifndef BLAS2_INTERFACE_HPP
#define BLAS2_INTERFACE_HPP

#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>

#include <interface/blas_interface_sycl.hpp>

#include <executors/executor_sycl.hpp>
#include <operations/blas2_trees.hpp>

namespace blas {
namespace internal {

/*! _gemv.
 * @brief Implementation of the General Matrix Vector product.
 * 
 */
template <typename Executor, typename IndexType, typename T,
          typename ContainerT0, typename ContainerT1, typename IncrementType,
          typename ContainerT2>
typename Executor::Return_Type _gemv_impl(
    Executor& ex, char _Trans, IndexType _M, IndexType _N, T _alpha,
    ContainerT0 _mA, IndexType _lda, ContainerT1 _vx, IncrementType _incx,
    T _beta, ContainerT2 _vy, IncrementType _incy,
    IndexType OptimisedLocalSize = 0, IndexType OptimisedScratchPadSize = 0,
    IndexType Optimised_n_rows_WG = 0, IndexType Optimised_n_cols_WG = 0) {
  typename Executor::Return_Type ret;
  _Trans = tolower(_Trans);

  if ((_Trans != 'n') && (_Trans != 't') && (_Trans != 'c')) {
    throw std::invalid_argument("Erroneous parameter");
  }
  int accessOpr = (_Trans == 'n');

  IndexType M = (_Trans == 'n') ? _M : _N;
  IndexType N = (_Trans == 'n') ? _N : _M;

  auto mA = make_matrix_view(ex, _mA, M, N, _lda, accessOpr);
  auto vx = make_vector_view(ex, _vx, _incx, N);
  auto vy = make_vector_view(ex, _vy, _incy, M);

  const IndexType interLoop = 1;
  const IndexType localSize =
      (OptimisedLocalSize == 0) ? ex.get_policy_handler().get_work_group_size()
                                : OptimisedLocalSize;
  const IndexType n_rows_WG = (Optimised_n_rows_WG == 0)
                                  ? ((mA.is_row_access()) ? 1 : localSize)
                                  : std::min(M, Optimised_n_rows_WG);
  const IndexType n_cols_WG = (Optimised_n_cols_WG == 0)
                                  ? ((mA.is_row_access()) ? N : localSize)
                                  : std::min(N, Optimised_n_cols_WG);
  const IndexType scratchPadSize =
      (OptimisedLocalSize == 0) ? localSize : OptimisedScratchPadSize;

  const IndexType nWG_col = (N - 1) / n_cols_WG + 1;
  const IndexType nWG_row = (M - 1) / n_rows_WG + 1;
  const IndexType globalSize = localSize * nWG_row * nWG_col;

  const IndexType scratchSize =
      (mA.is_row_access())
          ? (((scratchPadSize == 0) ? std::min(N, localSize) : 1) * nWG_col)
          : nWG_col;

  auto valT1 = blas::helper::make_sycl_iterator_buffer<T>(M * scratchSize);
  auto mat1 = make_matrix_view(ex, valT1, M, scratchSize, scratchSize, 0);

  if (mA.is_row_access()) {
    auto gemvR = make_Gemv_Row<interLoop>(mat1, mA, vx, nWG_row, nWG_col,
                                          scratchPadSize);
    ret = ex.execute(gemvR, localSize, globalSize, scratchPadSize);
  } else {
    auto gemvC = make_Gemv_Col(mat1, mA, vx, nWG_row, nWG_col, scratchPadSize);
    ret = ex.execute(gemvC, localSize, globalSize, scratchPadSize);
  }

  auto scalOp1 = make_op<ScalarOp, prdOp2_struct>(_beta, vy);
  auto addMOp = make_addSetColumns(mat1);
  auto scalOp2 = make_op<ScalarOp, prdOp2_struct>(_alpha, addMOp);
  auto addOp = make_op<BinaryOp, addOp2_struct>(scalOp1, scalOp2);
  auto assignOp = make_op<Assign>(vy, addOp);
  ret = ex.execute(assignOp, localSize);
  return ret;
}

/*! _TRMV.
 * @brief Implementation of the Triangular Matrix Vector product.
 */

template <typename Executor, typename IndexType, typename ContainerT0,
          typename ContainerT1, typename IncrementType>
typename Executor::Return_Type _trmv_impl(
    Executor& ex, char _Uplo, char _Trans, char _Diag, IndexType _N,
    ContainerT0 _mA, IndexType _lda, ContainerT1 _vx, IncrementType _incx,
    IndexType OptimisedLocalSize = 0, IndexType OptimisedScratchPadSize = 0,
    IndexType Optimised_n_rows_WG = 0, IndexType Optimised_n_cols_WG = 0) {
  _Trans = tolower(_Trans);
  _Uplo = tolower(_Uplo);
  _Diag = tolower(_Diag);

  if ((_Trans != 'n') && (_Trans != 't') && (_Trans != 'c') && (_Uplo != 'u') &&
      (_Uplo != 'l') && (_Diag != 'u') && (_Diag != 'n')) {
    throw std::invalid_argument("Erroneous parameter");
  }

  int accessOpr = (_Trans == 'n');
  int triangOpr = (accessOpr) ? (_Uplo == 'u') : (_Uplo == 'l');
  int unitDiag = (_Diag == 'u');
  IndexType N = _N;
  auto mA = make_matrix_view(ex, _mA, N, N, _lda, accessOpr);
  auto vx = make_vector_view(ex, _vx, _incx, N);

  const IndexType interLoop = 1;
  const IndexType localSize =
      (OptimisedLocalSize == 0) ? ex.get_policy_handler().get_work_group_size()
                                : OptimisedLocalSize;
  const IndexType n_rows_WG = (Optimised_n_rows_WG == 0)
                                  ? ((mA.is_row_access()) ? 1 : localSize)
                                  : std::min(N, Optimised_n_rows_WG);
  const IndexType n_cols_WG = (Optimised_n_cols_WG == 0)
                                  ? ((mA.is_row_access()) ? N : localSize)
                                  : std::min(N, Optimised_n_cols_WG);
  const IndexType scratchPadSize =
      (OptimisedLocalSize == 0) ? localSize : OptimisedScratchPadSize;

  const IndexType nWG_col = (N - 1) / n_cols_WG + 1;
  const IndexType nWG_row = (N - 1) / n_rows_WG + 1;
  const IndexType scratchSize =
      (mA.is_row_access())
          ? (((scratchPadSize == 0) ? std::min(N, localSize) : 1) * nWG_col)
          : nWG_col;
  const IndexType globalSize = localSize * nWG_row * nWG_col;

  using T = typename scalar_type<ContainerT0>::ScalarT;
  auto valT1 = blas::helper::make_sycl_iterator_buffer<T>(N * scratchSize);
  auto mat1 = make_matrix_view(ex, valT1, N, scratchSize, scratchSize, 0);

  typename Executor::Return_Type ret;

  if (mA.is_row_access()) {  // ROWS ACCESS
    if (triangOpr == 1) {
      if (unitDiag == 1) {
        auto gemvR = make_Gemv_Row<interLoop, false, true, true, true>(
            mat1, mA, vx, nWG_row, nWG_col, scratchPadSize);
        ret = ex.execute(gemvR, localSize, globalSize, scratchPadSize);
      } else {
        auto gemvR = make_Gemv_Row<interLoop, false, true, true>(
            mat1, mA, vx, nWG_row, nWG_col, scratchPadSize);
        ret = ex.execute(gemvR, localSize, globalSize, scratchPadSize);
      }
    } else {
      if (unitDiag == 1) {
        auto gemvR = make_Gemv_Row<interLoop, true, true, false, true>(
            mat1, mA, vx, nWG_row, nWG_col, scratchPadSize);
        ret = ex.execute(gemvR, localSize, globalSize, scratchPadSize);
      } else {
        auto gemvR = make_Gemv_Row<interLoop, true, true, false>(
            mat1, mA, vx, nWG_row, nWG_col, scratchPadSize);
        ret = ex.execute(gemvR, localSize, globalSize, scratchPadSize);
      }
    }
  } else {
    if (triangOpr == 1) {
      if (unitDiag == 1) {
        auto gemvC = make_Gemv_Col<false, true, true, true>(
            mat1, mA, vx, nWG_row, nWG_col, scratchPadSize);
        ret = ex.execute(gemvC, localSize, globalSize, scratchPadSize);
      } else {
        auto gemvC = make_Gemv_Col<false, true, true>(mat1, mA, vx, nWG_row,
                                                      nWG_col, scratchPadSize);
        ret = ex.execute(gemvC, localSize, globalSize, scratchPadSize);
      }
    } else {
      if (unitDiag == 1) {
        auto gemvC = make_Gemv_Col<true, true, false, true>(
            mat1, mA, vx, nWG_row, nWG_col, scratchPadSize);
        ret = ex.execute(gemvC, localSize, globalSize, scratchPadSize);
      } else {
        auto gemvC = make_Gemv_Col<true, true, false>(mat1, mA, vx, nWG_row,
                                                      nWG_col, scratchPadSize);
        ret = ex.execute(gemvC, localSize, globalSize, scratchPadSize);
      }
    }
  }
  auto addMOp = make_addSetColumns(mat1);
  auto assignOp = make_op<Assign>(vx, addMOp);
  ret = ex.execute(assignOp, localSize);
  return ret;
}

/*! _SYMV.
 * @brief Implementation of the Symmetric Matrix Vector product.
 */
/*
ssymv 	( 	character  	UPLO,
   integer  	N,
   real  	ALPHA,
   real, dimension(lda,*)  	A,
   integer  	LDA,
   real, dimension(*)  	X,
   integer  	INCX,
   real  	BETA,
   real, dimension(*)  	Y,
   integer  	INCY
 ) 	*/
template <typename Executor, typename IndexType, typename T,
          typename ContainerT0, typename ContainerT1, typename IncrementType,
          typename ContainerT2>
typename Executor::Return_Type _symv_impl(
    Executor& ex, char _Uplo, IndexType _N, T _alpha, ContainerT0 _mA,
    IndexType _lda, ContainerT1 _vx, IncrementType _incx, T _beta,
    ContainerT2 _vy, IncrementType _incy, IndexType OptimisedLocalSize = 0,
    IndexType OptimisedScratchPadSize = 0, IndexType Optimised_n_rows_WG = 0,
    IndexType Optimised_n_cols_WG = 0) {
  _Uplo = tolower(_Uplo);

  if ((_Uplo != 'u') && (_Uplo != 'l')) {
    throw std::invalid_argument("Erroneous parameter");
  }
  int accessOpr = 1;
  int triangOpr = (_Uplo == 'u');
  IndexType N = _N;
  auto mA = make_matrix_view(ex, _mA, N, N, _lda, accessOpr);
  auto vx = make_vector_view(ex, _vx, _incx, N);
  auto vy = make_vector_view(ex, _vy, _incy, N);
  auto mAT = make_matrix_view(ex, _mA, N, N, _lda, int(0));

  const IndexType interLoop = 1;

  const IndexType localSize =
      (OptimisedLocalSize == 0) ? ex.get_policy_handler().get_work_group_size()
                                : OptimisedLocalSize;
  const IndexType scratchPadSize =
      (OptimisedLocalSize == 0) ? localSize : OptimisedScratchPadSize;

  const IndexType n_rows_WG_R =
      (Optimised_n_rows_WG == 0) ? 1 : std::min(N, Optimised_n_rows_WG);
  const IndexType n_cols_WG_R =
      (Optimised_n_cols_WG == 0) ? N : std::min(N, Optimised_n_cols_WG);

  const IndexType nWG_row_R = (N - 1) / n_rows_WG_R + 1;
  const IndexType nWG_col_R = (N - 1) / n_cols_WG_R + 1;
  const IndexType globalSize_R = localSize * nWG_row_R * nWG_col_R;

  const IndexType n_rows_WG_C =
      (Optimised_n_rows_WG == 0) ? localSize : Optimised_n_rows_WG;
  const IndexType n_cols_WG_C =
      (Optimised_n_cols_WG == 0) ? localSize : Optimised_n_cols_WG;

  const IndexType nWG_row_C = (N - 1) / n_rows_WG_C + 1;
  const IndexType nWG_col_C = (N - 1) / n_cols_WG_C + 1;
  const IndexType globalSize_C = localSize * nWG_row_C * nWG_col_C;

  const IndexType scratchSize_R =
      ((scratchPadSize == 0) ? std::min(N, localSize) : 1) * nWG_col_R;

  auto valTR = blas::helper::make_sycl_iterator_buffer<T>(N * scratchSize_R);
  auto matR = make_matrix_view(ex, valTR, N, scratchSize_R, scratchSize_R, 0);

  const IndexType scratchSize_C = nWG_col_C;

  auto valTC = blas::helper::make_sycl_iterator_buffer<T>(N * scratchSize_C);
  auto matC = make_matrix_view(ex, valTC, N, scratchSize_C, scratchSize_C, 0);

  if (mA.is_row_access()) {  // ROWS ACCESS
    if (triangOpr == 1) {
      auto gemvR = make_Gemv_Row<interLoop, false, true, true>(
          matR, mA, vx, nWG_row_R, nWG_col_R, scratchPadSize);
      auto gemvC = make_Gemv_Col<true, false, false>(matC, mAT, vx, nWG_row_C,
                                                     nWG_col_C, scratchPadSize);
      ex.execute(gemvR, localSize, globalSize_R, scratchPadSize);
      ex.execute(gemvC, localSize, globalSize_C, scratchPadSize);
    } else {
      auto gemvR = make_Gemv_Row<interLoop, true, true, false>(
          matR, mA, vx, nWG_row_R, nWG_col_R, scratchPadSize);
      auto gemvC = make_Gemv_Col<false, false, true>(matC, mAT, vx, nWG_row_C,
                                                     nWG_col_C, scratchPadSize);
      ex.execute(gemvR, localSize, globalSize_R, scratchPadSize);
      ex.execute(gemvC, localSize, globalSize_C, scratchPadSize);
    }

  } else {  // col major

    if (triangOpr == 1) {
      auto gemvC = make_Gemv_Col<false, true, true>(matC, mA, vx, nWG_row_C,
                                                    nWG_col_C, scratchPadSize);
      auto gemvR = make_Gemv_Row<interLoop, true, false, false>(
          matR, mAT, vx, nWG_row_R, nWG_col_R, scratchPadSize);
      ex.execute(gemvC, localSize, globalSize_C, scratchPadSize);
      ex.execute(gemvR, localSize, globalSize_R, scratchPadSize);
    } else {
      auto gemvC = make_Gemv_Col<true, true, false>(matC, mA, vx, nWG_row_C,
                                                    nWG_col_C, scratchPadSize);
      auto gemvR = make_Gemv_Row<interLoop, false, false, true>(
          matR, mAT, vx, nWG_row_R, nWG_col_R, scratchPadSize);
      ex.execute(gemvC, localSize, globalSize_C, scratchPadSize);
      ex.execute(gemvR, localSize, globalSize_R, scratchPadSize);
    }
  }
  auto scalOp1 = make_op<ScalarOp, prdOp2_struct>(_beta, vy);
  auto addMOpR = make_addSetColumns(matR);
  auto addMOpC = make_addSetColumns(matC);
  auto addMOp = make_op<BinaryOp, addOp2_struct>(addMOpR, addMOpC);
  auto scalOp2 = make_op<ScalarOp, prdOp2_struct>(_alpha, addMOp);
  auto addOp = make_op<BinaryOp, addOp2_struct>(scalOp1, scalOp2);
  auto assignOp = make_op<Assign>(vy, addOp);
  return ex.execute(assignOp, localSize);
}

/**** RANK 1 MODIFICATION ****/

template <typename Executor, typename IndexType, typename T,
          typename ContainerT0, typename IncrementType, typename ContainerT1,
          typename ContainerT2>
typename Executor::Return_Type _ger_impl(
    Executor& ex, IndexType _M, IndexType _N, T _alpha, ContainerT0 _vx,
    IncrementType _incx, ContainerT1 _vy, IncrementType _incy, ContainerT2 _mA,
    IndexType _lda, IndexType OptimisedLocalSize = 0,
    IndexType OptimisedScratchPadSize = 0, IndexType Optimised_n_rows_WG = 0,
    IndexType Optimised_n_cols_WG = 0) {
  IndexType M = _M;
  IndexType N = _N;
  int accessOpr = 1;
  auto mA = make_matrix_view(ex, _mA, M, N, _lda, accessOpr);
  auto vx = make_vector_view(ex, _vx, _incx, M);
  auto vy = make_vector_view(ex, _vy, _incy, N);

  const IndexType localSize =
      (OptimisedLocalSize == 0) ? ex.get_policy_handler().get_work_group_size()
                                : OptimisedLocalSize;
  const IndexType n_rows_WG = (Optimised_n_rows_WG == 0)
                                  ? ((mA.is_row_access()) ? 1 : localSize)
                                  : std::min(M, Optimised_n_rows_WG);
  ;
  const IndexType n_cols_WG = (Optimised_n_cols_WG == 0)
                                  ? ((mA.is_row_access()) ? N : localSize)
                                  : std::min(N, Optimised_n_cols_WG);
  ;
  const IndexType scratchPadSize =
      (OptimisedLocalSize == 0) ? localSize : OptimisedScratchPadSize;

  const IndexType nWG_col = (N - 1) / n_cols_WG + 1;
  const IndexType nWG_row = (M - 1) / n_rows_WG + 1;
  const IndexType globalSize = localSize * nWG_row * nWG_col;

  typename Executor::Return_Type ret;

  if (mA.is_row_access()) {  // rowmajor
    auto assignOp =
        make_Ger_Row(mA, _alpha, vx, vy, nWG_row, nWG_col, scratchPadSize);
    ret = ex.execute(assignOp, localSize, globalSize, scratchPadSize);
  } else {  // colmajor
    auto assignOp =
        make_Ger_Col(mA, _alpha, vx, vy, nWG_row, nWG_col, scratchPadSize);
    ret = ex.execute(assignOp, localSize, globalSize, scratchPadSize);
  }
  return ret;
}

/*! _SYR.
 * @brief Implementation of the rank 1 operation
 */
/*
ssyr 	( 	character  	UPLO,
   integer  	N,
   real  	ALPHA,
   real, dimension(*)  	X,
   integer  	INCX,
   real, dimension(lda,*)  	A,
   integer  	LDA
 )
*/
template <typename Executor, typename IndexType, typename T,
          typename ContainerT0, typename IncrementType, typename ContainerT1>
typename Executor::Return_Type _syr_impl(
    Executor& ex, char _Uplo, IndexType _N, T _alpha, ContainerT0 _vx,
    IncrementType _incx, ContainerT1 _mA, IndexType _lda,
    IndexType OptimisedLocalSize = 0, IndexType OptimisedScratchPadSize = 0,
    IndexType Optimised_n_rows_WG = 0, IndexType Optimised_n_cols_WG = 0) {
  _Uplo = tolower(_Uplo);
  int accessOpr = true;
  int triangOpr = (_Uplo == 'u');
  IndexType N = _N;
  auto mA = make_matrix_view(ex, _mA, N, N, _lda, accessOpr);
  auto vx = make_vector_view(ex, _vx, _incx, N);

  const IndexType localSize =
      (OptimisedLocalSize == 0) ? ex.get_policy_handler().get_work_group_size()
                                : OptimisedLocalSize;
  const IndexType n_rows_WG = (Optimised_n_rows_WG == 0)
                                  ? ((mA.is_row_access()) ? 1 : localSize)
                                  : std::min(N, Optimised_n_rows_WG);
  const IndexType n_cols_WG = (Optimised_n_cols_WG == 0)
                                  ? ((mA.is_row_access()) ? N : localSize)
                                  : std::min(N, Optimised_n_cols_WG);
  const IndexType scratchPadSize =
      (OptimisedLocalSize == 0) ? localSize : OptimisedScratchPadSize;

  const IndexType nWG_row = (N - 1) / n_rows_WG + 1;
  const IndexType nWG_col = (N - 1) / n_cols_WG + 1;
  const IndexType globalSize = localSize * nWG_row * nWG_col;

  if (mA.is_row_access()) {  // ROWS ACCESS
    if (triangOpr) {
      auto assignOp = make_Ger_Row<true, false, true, true>(
          mA, _alpha, vx, vx, nWG_row, nWG_col, scratchPadSize);
      ex.execute(assignOp, localSize, globalSize, scratchPadSize);

    } else {
      auto assignOp = make_Ger_Row<true, true, true, false>(
          mA, _alpha, vx, vx, nWG_row, nWG_col, scratchPadSize);
      ex.execute(assignOp, localSize, globalSize, scratchPadSize);
    }

  } else {  // COLUMN ACCESS
    if (triangOpr) {
      auto assignOp = make_Ger_Col<true, false, true, true>(
          mA, _alpha, vx, vx, nWG_row, nWG_col, scratchPadSize);
      ex.execute(assignOp, localSize, globalSize, scratchPadSize);
    } else {
      auto assignOp = make_Ger_Col<true, true, true, false>(
          mA, _alpha, vx, vx, nWG_row, nWG_col, scratchPadSize);
      ex.execute(assignOp, localSize, globalSize, scratchPadSize);
    }
  }
}

/*
    ssyr2 	( 	character  	UPLO,
                integer  	N,
                real  	ALPHA,
                real, dimension(*)  	X,
                integer  	INCX,
                real, dimension(*)  	Y,
                integer  	INCY,
                real, dimension(lda,*)  	A,
                integer  	LDA
        )
*/
template <typename Executor, typename IndexType, typename T,
          typename ContainerT0, typename IncrementType, typename ContainerT1,
          typename ContainerT2>
typename Executor::Return_Type _syr2_impl(
    Executor& ex, char _Uplo, IndexType _N, T _alpha, ContainerT0 _vx,
    IncrementType _incx, ContainerT1 _vy, IncrementType _incy, ContainerT2 _mA,
    IndexType _lda, IndexType OptimisedLocalSize = 0,
    IndexType OptimisedScratchPadSize = 0, IndexType Optimised_n_rows_WG = 0,
    IndexType Optimised_n_cols_WG = 0) {
  _Uplo = tolower(_Uplo);

  int accessOpr = true;
  int triangOpr = (_Uplo == 'u');
  IndexType N = _N;

  auto mA = make_matrix_view(ex, _mA, _N, _N, _lda, accessOpr);
  auto vx = make_vector_view(ex, _vx, _incx, _N);
  auto vy = make_vector_view(ex, _vy, _incy, _N);

  const IndexType localSize =
      (OptimisedLocalSize == 0) ? ex.get_policy_handler().get_work_group_size()
                                : OptimisedLocalSize;
  const IndexType n_rows_WG = (Optimised_n_rows_WG == 0)
                                  ? ((mA.is_row_access()) ? 1 : localSize)
                                  : std::min(N, Optimised_n_rows_WG);
  const IndexType n_cols_WG = (Optimised_n_cols_WG == 0)
                                  ? ((mA.is_row_access()) ? N : localSize)
                                  : std::min(N, Optimised_n_cols_WG);
  const IndexType scratchPadSize =
      (OptimisedLocalSize == 0) ? 2 * localSize : OptimisedScratchPadSize;

  const IndexType nWG_row = (N - 1) / n_rows_WG + 1;
  const IndexType nWG_col = (N - 1) / n_cols_WG + 1;
  const IndexType globalSize = localSize * nWG_row * nWG_col;

  if (mA.is_row_access()) {  // ROWS ACCESS
    if (triangOpr) {
      auto assignOp = make_Ger_Row<false, false, true, true>(
          mA, _alpha, vx, vy, nWG_row, nWG_col, scratchPadSize);
      ex.execute(assignOp, localSize, globalSize, scratchPadSize);
    } else {
      auto assignOp = make_Ger_Row<false, true, true, false>(
          mA, _alpha, vx, vy, nWG_row, nWG_col, scratchPadSize);
      ex.execute(assignOp, localSize, globalSize, scratchPadSize);
    }
  } else {  // COLUMN ACCESS
    if (triangOpr) {
      auto assignOp = make_Ger_Col<false, false, true, true>(
          mA, _alpha, vx, vy, nWG_row, nWG_col, scratchPadSize);
      ex.execute(assignOp, localSize, globalSize, scratchPadSize);
    } else {
      auto assignOp = make_Ger_Col<false, true, true, false>(
          mA, _alpha, vx, vy, nWG_row, nWG_col, scratchPadSize);
      ex.execute(assignOp, localSize, globalSize, scratchPadSize);
    }
  }
}
}  // namespace internal

/*!
 @brief Generalised matrix vector product with rectangular non-symmetric matrices. 
 
 Generalised matrix vector product with rectangular non-symmetric matrices, i.e. computing 
 the mathematical operation: 

 y = alpha*A*x + beta*y

 See the netlib blas interface documentation for more details of the high level interface: 
     http://www.netlib.org/lapack/explore-html/db/d58/sgemv_8f.html

 _gemv(
    Executor& ex,        -- Executor (sycl, parallel, serial, etc)
    char _Trans,         -- The transposition of the matrix parameter ("n","t","c")
    IndexType _M,        -- The size of dimension M of the matrix (rows)
    IndexType _N,        -- The size of dimension N of the matrix (columns)
    T _alpha,            -- Scalar parameter Alpha
    ContainerT0 _mA,     -- An array (LDA,N), with the first m*n elements containing the matrix
    IndexType _lda,      -- Specifies the first dimension of a, max(1, m)
    ContainerT1 _vx,     -- An array of dimension at least: (1+(n-1)*abs(incx)) when trans = 'n'
                                                            (1+(m-1)*abs(incx)) otherwise
                            containing the vector "x"
    IncrementType _incx, -- The increment for elements in x, which must be nonzero
    T _beta,             -- Scalar parameter Beta
    ContainerT2 _vy,     -- An array of dimension at least: (1+(m-1)*abs(incy)) when trans = "n"
                                                            (1+(n-1)*abs(incy)) otherwise
                            containing the vector "y" (if beta is nonzero). When finished, 
                            y is overwritten with the updated vector. 
    IncrementType _incy  -- The increment for elements in y, which must be nonzero.
  ) 
 */
template <typename Executor, typename IndexType, typename T,
          typename ContainerT0, typename ContainerT1, typename IncrementType,
          typename ContainerT2>
typename Executor::Return_Type inline _gemv(
    Executor& ex, char _Trans, IndexType _M, IndexType _N, T _alpha,
    ContainerT0 _mA, IndexType _lda, ContainerT1 _vx, IncrementType _incx,
    T _beta, ContainerT2 _vy, IncrementType _incy) {
  // TODO: Here we can use some heuristics to select localn global, local, and
  // scratch size per device
  return internal::_gemv_impl(ex, _Trans, _M, _N, _alpha, _mA, _lda, _vx, _incx,
                              _beta, _vy, _incy);
}
template <typename Executor, typename IndexType, typename ContainerT0,
          typename ContainerT1, typename IncrementType>
typename Executor::Return_Type inline _trmv(Executor& ex, char _Uplo,
                                            char _Trans, char _Diag,
                                            IndexType _N, ContainerT0 _mA,
                                            IndexType _lda, ContainerT1 _vx,
                                            IncrementType _incx) {
  // TODO: Here we can use some heuristics to select localn global, local, and
  // scratch size per device
  return internal::_trmv_impl(ex, _Uplo, _Trans, _Diag, _N, _mA, _lda, _vx,
                              _incx);
}
template <typename Executor, typename IndexType, typename T,
          typename ContainerT0, typename ContainerT1, typename IncrementType,
          typename ContainerT2>
typename Executor::Return_Type inline _symv(
    Executor& ex, char _Uplo, IndexType _N, T _alpha, ContainerT0 _mA,
    IndexType _lda, ContainerT1 _vx, IncrementType _incx, T _beta,
    ContainerT2 _vy, IncrementType _incy) {
  // TODO: Here we can use some heuristics to select localn global, local, and
  // scratch size per device
  return internal::_symv_impl(ex, _Uplo, _N, _alpha, _mA, _lda, _vx, _incx,
                              _beta, _vy, _incy);
}
template <typename Executor, typename IndexType, typename T,
          typename ContainerT0, typename IncrementType, typename ContainerT1,
          typename ContainerT2>
typename Executor::Return_Type inline _ger(Executor& ex, IndexType _M,
                                           IndexType _N, T _alpha,
                                           ContainerT0 _vx, IncrementType _incx,
                                           ContainerT1 _vy, IncrementType _incy,
                                           ContainerT2 _mA, IndexType _lda) {
  // TODO: Here we can use some heuristics to select localn global, local, and
  // scratch size per device
  return internal::_ger_impl(ex, _M, _N, _alpha, _vx, _incx, _vy, _incy, _mA,
                             _lda);
}
template <typename Executor, typename IndexType, typename T,
          typename ContainerT0, typename IncrementType, typename ContainerT1>
typename Executor::Return_Type inline _syr(Executor& ex, char _Uplo,
                                           IndexType _N, T _alpha,
                                           ContainerT0 _vx, IncrementType _incx,
                                           ContainerT1 _mA, IndexType _lda) {
  // TODO: Here we can use some heuristics to select localn global, local, and
  // scratch size per device
  return internal::_syr_impl(ex, _Uplo, _N, _alpha, _vx, _incx, _mA, _lda);
}
template <typename Executor, typename IndexType, typename T,
          typename ContainerT0, typename IncrementType, typename ContainerT1,
          typename ContainerT2>
typename Executor::Return_Type inline _syr2(
    Executor& ex, char _Uplo, IndexType _N, T _alpha, ContainerT0 _vx,
    IncrementType _incx, ContainerT1 _vy, IncrementType _incy, ContainerT2 _mA,
    IndexType _lda) {
  // TODO: Here we can use some heuristics to select localn global, local, and
  // scratch size per device
  return internal::_syr2_impl(ex, _Uplo, _N, _alpha, _vx, _incx, _vy, _incy,
                              _mA, _lda);
}

}  // namespace blas

#endif  // BLAS2_INTERFACE_HPP
