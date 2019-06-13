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

#ifndef SYCL_BLAS_BLAS2_INTERFACE_HPP
#define SYCL_BLAS_BLAS2_INTERFACE_HPP

#include "blas_meta.h"
#include "container/sycl_iterator.h"
#include "executors/executor.h"
#include "interface/blas2_interface.h"
#include "operations/blas2_trees.h"
#include "operations/blas_constants.h"
#include "operations/blas_operators.hpp"
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace blas {
namespace internal {

/*! _gemv.
 * @brief Implementation of the General Matrix Vector product.
 *
 */
template <transpose_type trn, typename Executor, typename index_t,
          typename element_t, typename container_t0, typename container_t1,
          typename increment_t, typename container_t2>
typename Executor::policy_t::event_t _gemv_impl(
    Executor& ex, index_t _M, index_t _N, element_t _alpha, container_t0 _mA,
    index_t _lda, container_t1 _vx, increment_t _incx, element_t _beta,
    container_t2 _vy, increment_t _incy, index_t _localSize = 0,
    index_t _scratchPadSize = 0, index_t _nRowsWG = 0, index_t _nColsWG = 0) {
  typename Executor::policy_t::event_t ret{};

  index_t M = (trn == transpose_type::Normal) ? _M : _N;
  index_t N = (trn == transpose_type::Normal) ? _N : _M;

  static constexpr auto data_layout_access =
      Choose<trn == transpose_type::Normal, access_layout,
             access_layout::col_major, access_layout::row_major>::type;
  using data_layout_t = typename Layout<data_layout_access>::type;
  auto mA = make_matrix_view<data_layout_t>(ex, _mA, M, N, _lda);
  auto vx = make_vector_view(ex, _vx, _incx, N);
  auto vy = make_vector_view(ex, _vy, _incy, M);

  const index_t interLoop = 1;
  const index_t localSize = (_localSize == 0)
                                ? ex.get_policy_handler().get_work_group_size()
                                : _localSize;
  const index_t nRowsWG =
      (_nRowsWG == 0) ? ((data_layout_t::is_col_major()) ? localSize : 1)
                      : std::min(M, _nRowsWG);
  const index_t nColsWG =
      (_nColsWG == 0) ? ((data_layout_t::is_col_major()) ? localSize : N)
                      : std::min(N, _nColsWG);
  const index_t scratchPadSize =
      (_localSize == 0) ? localSize : _scratchPadSize;

  const index_t nWGPerCol = (N - 1) / nColsWG + 1;
  const index_t nWGPerRow = (M - 1) / nRowsWG + 1;
  const index_t globalSize = localSize * nWGPerRow * nWGPerCol;

  const index_t scratchSize =
      (data_layout_t::is_col_major())
          ? nWGPerCol
          : (((scratchPadSize == 0) ? std::min(N, localSize) : 1) * nWGPerCol);

  auto valT1 = blas::make_sycl_iterator_buffer<element_t>(M * scratchSize);
  // this is column major
  auto mat1 =
      make_matrix_view<row_major>(ex, valT1, M, scratchSize, scratchSize);

  if (data_layout_t::is_col_major()) {
    auto gemvC =
        make_Gemv_Col(mat1, mA, vx, nWGPerRow, nWGPerCol, scratchPadSize);
    ret = concatenate_vectors(
        ret, ex.execute(gemvC, localSize, globalSize, scratchPadSize));
  } else {
    auto gemvR = make_Gemv_Row<interLoop>(mat1, mA, vx, nWGPerRow, nWGPerCol,
                                          scratchPadSize);
    ret = concatenate_vectors(
        ret, ex.execute(gemvR, localSize, globalSize, scratchPadSize));
  }

  // beta * y
  auto scalOp1 = make_op<ScalarOp, ProductOperator>(_beta, vy);
  // Finish the mv?
  auto addMOp = make_addSetColumns(mat1);
  // (..) * alpha
  auto scalOp2 = make_op<ScalarOp, ProductOperator>(_alpha, addMOp);
  // add up
  auto addOp = make_op<BinaryOp, AddOperator>(scalOp1, scalOp2);
  // assign the result to
  auto assignOp = make_op<Assign>(vy, addOp);
  ret = concatenate_vectors(ret, ex.execute(assignOp, localSize));
  return ret;
}

/*! _TRMV.
 * @brief Implementation of the Triangular Matrix Vector product.
 */

template <transpose_type trn, typename Executor, typename index_t,
          typename container_t0, typename container_t1, typename increment_t>
typename Executor::policy_t::event_t _trmv_impl(
    Executor& ex, char _Uplo, char _Diag, index_t _N, container_t0 _mA,
    index_t _lda, container_t1 _vx, increment_t _incx, index_t _localSize = 0,
    index_t _scratchPadSize = 0, index_t _nRowsWG = 0, index_t _nColsWG = 0) {
  typename Executor::policy_t::event_t ret{};
  _Uplo = tolower(_Uplo);
  _Diag = tolower(_Diag);

  if ((_Uplo != 'u') && (_Uplo != 'l') && (_Diag != 'u') && (_Diag != 'n')) {
    throw std::invalid_argument("Erroneous parameter");
  }

  static constexpr auto data_layout_access =
      Choose<trn == transpose_type::Normal, access_layout,
             access_layout::col_major, access_layout::row_major>::type;
  using data_layout_t = typename Layout<data_layout_access>::type;
  int triangOpr =
      (data_layout_t::is_col_major()) ? (_Uplo == 'u') : (_Uplo == 'l');
  int unitDiag = (_Diag == 'u');
  index_t N = _N;
  auto mA = make_matrix_view<data_layout_t>(ex, _mA, N, N, _lda);
  auto vx = make_vector_view(ex, _vx, _incx, N);
  const index_t interLoop = 1;
  const index_t localSize = (_localSize == 0)
                                ? ex.get_policy_handler().get_work_group_size()
                                : _localSize;
  const index_t nRowsWG =
      (_nRowsWG == 0) ? ((data_layout_t::is_col_major()) ? localSize : 1)
                      : std::min(N, _nRowsWG);
  const index_t nColsWG =
      (_nColsWG == 0) ? ((data_layout_t::is_col_major()) ? localSize : N)
                      : std::min(N, _nColsWG);
  const index_t scratchPadSize =
      (_localSize == 0) ? localSize : _scratchPadSize;

  const index_t nWGPerCol = (N - 1) / nColsWG + 1;
  const index_t nWGPerRow = (N - 1) / nRowsWG + 1;
  const index_t scratchSize =
      (data_layout_t::is_col_major())
          ? nWGPerCol
          : (((scratchPadSize == 0) ? std::min(N, localSize) : 1) * nWGPerCol);
  const index_t globalSize = localSize * nWGPerRow * nWGPerCol;

  using element_t = typename ValueType<container_t0>::type;
  auto valT1 = blas::make_sycl_iterator_buffer<element_t>(N * scratchSize);
  auto mat1 =
      make_matrix_view<row_major>(ex, valT1, N, scratchSize, scratchSize);

  if (data_layout_t::is_col_major()) {
    if (triangOpr == 1) {
      if (unitDiag == 1) {
        auto gemvC = make_Gemv_Col<false, true, true, true>(
            mat1, mA, vx, nWGPerRow, nWGPerCol, scratchPadSize);
        ret = concatenate_vectors(
            ret, ex.execute(gemvC, localSize, globalSize, scratchPadSize));
      } else {
        auto gemvC = make_Gemv_Col<false, true, true>(
            mat1, mA, vx, nWGPerRow, nWGPerCol, scratchPadSize);
        ret = concatenate_vectors(
            ret, ex.execute(gemvC, localSize, globalSize, scratchPadSize));
      }
    } else {
      if (unitDiag == 1) {
        auto gemvC = make_Gemv_Col<true, true, false, true>(
            mat1, mA, vx, nWGPerRow, nWGPerCol, scratchPadSize);
        ret = concatenate_vectors(
            ret, ex.execute(gemvC, localSize, globalSize, scratchPadSize));
      } else {
        auto gemvC = make_Gemv_Col<true, true, false>(
            mat1, mA, vx, nWGPerRow, nWGPerCol, scratchPadSize);
        ret = concatenate_vectors(
            ret, ex.execute(gemvC, localSize, globalSize, scratchPadSize));
      }
    }
  } else {  // row_major
    if (triangOpr == 1) {
      if (unitDiag == 1) {
        auto gemvR = make_Gemv_Row<interLoop, false, true, true, true>(
            mat1, mA, vx, nWGPerRow, nWGPerCol, scratchPadSize);
        ret = concatenate_vectors(
            ret, ex.execute(gemvR, localSize, globalSize, scratchPadSize));
      } else {
        auto gemvR = make_Gemv_Row<interLoop, false, true, true>(
            mat1, mA, vx, nWGPerRow, nWGPerCol, scratchPadSize);
        ret = concatenate_vectors(
            ret, ex.execute(gemvR, localSize, globalSize, scratchPadSize));
      }
    } else {
      if (unitDiag == 1) {
        auto gemvR = make_Gemv_Row<interLoop, true, true, false, true>(
            mat1, mA, vx, nWGPerRow, nWGPerCol, scratchPadSize);
        ret = concatenate_vectors(
            ret, ex.execute(gemvR, localSize, globalSize, scratchPadSize));
      } else {
        auto gemvR = make_Gemv_Row<interLoop, true, true, false>(
            mat1, mA, vx, nWGPerRow, nWGPerCol, scratchPadSize);
        ret = concatenate_vectors(
            ret, ex.execute(gemvR, localSize, globalSize, scratchPadSize));
      }
    }
  }
  auto addMOp = make_addSetColumns(mat1);
  auto assignOp = make_op<Assign>(vx, addMOp);
  ret = concatenate_vectors(ret, ex.execute(assignOp, localSize));
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
template <typename Executor, typename index_t, typename element_t,
          typename container_t0, typename container_t1, typename increment_t,
          typename container_t2>
typename Executor::policy_t::event_t _symv_impl(
    Executor& ex, char _Uplo, index_t _N, element_t _alpha, container_t0 _mA,
    index_t _lda, container_t1 _vx, increment_t _incx, element_t _beta,
    container_t2 _vy, increment_t _incy, index_t _localSize = 0,
    index_t _scratchPadSize = 0, index_t _nRowsWG = 0, index_t _nColsWG = 0) {
  _Uplo = tolower(_Uplo);
  typename Executor::policy_t::event_t ret;
  if ((_Uplo != 'u') && (_Uplo != 'l')) {
    throw std::invalid_argument("Erroneous parameter");
  }
  int triangOpr = (_Uplo == 'u');
  index_t N = _N;
  auto mA = make_matrix_view<col_major>(ex, _mA, N, N, _lda);
  auto vx = make_vector_view(ex, _vx, _incx, N);
  auto vy = make_vector_view(ex, _vy, _incy, N);
  auto mAT = make_matrix_view<row_major>(ex, _mA, N, N, _lda);

  const index_t interLoop = 1;

  const index_t localSize = (_localSize == 0)
                                ? ex.get_policy_handler().get_work_group_size()
                                : _localSize;
  const index_t scratchPadSize =
      (_localSize == 0) ? localSize : _scratchPadSize;

  const index_t nRowsWG_R = (_nRowsWG == 0) ? 1 : std::min(N, _nRowsWG);
  const index_t nColsWG_R = (_nColsWG == 0) ? N : std::min(N, _nColsWG);

  const index_t nWGPerRow_R = (N - 1) / nRowsWG_R + 1;
  const index_t nWGPerCol_R = (N - 1) / nColsWG_R + 1;
  const index_t globalSize_R = localSize * nWGPerRow_R * nWGPerCol_R;

  const index_t nRowsWG_C = (_nRowsWG == 0) ? localSize : _nRowsWG;
  const index_t nColsWG_C = (_nColsWG == 0) ? localSize : _nColsWG;

  const index_t nWGPerRow_C = (N - 1) / nRowsWG_C + 1;
  const index_t nWGPerCol_C = (N - 1) / nColsWG_C + 1;
  const index_t globalSize_C = localSize * nWGPerRow_C * nWGPerCol_C;

  const index_t scratchSize_R =
      ((scratchPadSize == 0) ? std::min(N, localSize) : 1) * nWGPerCol_R;

  auto valTR = blas::make_sycl_iterator_buffer<element_t>(N * scratchSize_R);
  auto matR =
      make_matrix_view<row_major>(ex, valTR, N, scratchSize_R, scratchSize_R);

  const index_t scratchSize_C = nWGPerCol_C;

  auto valTC = blas::make_sycl_iterator_buffer<element_t>(N * scratchSize_C);
  auto matC =
      make_matrix_view<row_major>(ex, valTC, N, scratchSize_C, scratchSize_C);

  if (triangOpr == 1) {
    auto gemvC = make_Gemv_Col<false, true, true>(matC, mA, vx, nWGPerRow_C,
                                                  nWGPerCol_C, scratchPadSize);
    auto gemvR = make_Gemv_Row<interLoop, true, false, false>(
        matR, mAT, vx, nWGPerRow_R, nWGPerCol_R, scratchPadSize);
    ret = concatenate_vectors(
        ret, ex.execute(gemvC, localSize, globalSize_C, scratchPadSize));
    ret = concatenate_vectors(
        ret, ex.execute(gemvR, localSize, globalSize_R, scratchPadSize));
  } else {
    auto gemvC = make_Gemv_Col<true, true, false>(matC, mA, vx, nWGPerRow_C,
                                                  nWGPerCol_C, scratchPadSize);
    auto gemvR = make_Gemv_Row<interLoop, false, false, true>(
        matR, mAT, vx, nWGPerRow_R, nWGPerCol_R, scratchPadSize);
    ret = concatenate_vectors(
        ret, ex.execute(gemvC, localSize, globalSize_C, scratchPadSize));
    ret = concatenate_vectors(
        ret, ex.execute(gemvR, localSize, globalSize_R, scratchPadSize));
  }

  auto scalOp1 = make_op<ScalarOp, ProductOperator>(_beta, vy);
  auto addMOpR = make_addSetColumns(matR);
  auto addMOpC = make_addSetColumns(matC);
  auto addMOp = make_op<BinaryOp, AddOperator>(addMOpR, addMOpC);
  auto scalOp2 = make_op<ScalarOp, ProductOperator>(_alpha, addMOp);
  auto addOp = make_op<BinaryOp, AddOperator>(scalOp1, scalOp2);
  auto assignOp = make_op<Assign>(vy, addOp);
  ret = concatenate_vectors(ret, ex.execute(assignOp, localSize));
  return ret;
}

/**** RANK 1 MODIFICATION ****/

template <typename Executor, typename index_t, typename element_t,
          typename container_t0, typename increment_t, typename container_t1,
          typename container_t2>
typename Executor::policy_t::event_t _ger_impl(
    Executor& ex, index_t _M, index_t _N, element_t _alpha, container_t0 _vx,
    increment_t _incx, container_t1 _vy, increment_t _incy, container_t2 _mA,
    index_t _lda, index_t _localSize = 0, index_t _scratchPadSize = 0,
    index_t _nRowsWG = 0, index_t _nColsWG = 0) {
  index_t M = _M;
  index_t N = _N;
  auto mA = make_matrix_view<col_major>(ex, _mA, M, N, _lda);
  auto vx = make_vector_view(ex, _vx, _incx, M);
  auto vy = make_vector_view(ex, _vy, _incy, N);

  const index_t localSize = (_localSize == 0)
                                ? ex.get_policy_handler().get_work_group_size()
                                : _localSize;
  const index_t nRowsWG = (_nRowsWG == 0) ? localSize : std::min(M, _nRowsWG);

  const index_t nColsWG = (_nColsWG == 0) ? localSize : std::min(N, _nColsWG);

  const index_t scratchPadSize =
      (_localSize == 0) ? localSize : _scratchPadSize;

  const index_t nWGPerCol = (N - 1) / nColsWG + 1;
  const index_t nWGPerRow = (M - 1) / nRowsWG + 1;
  const index_t globalSize = localSize * nWGPerRow * nWGPerCol;

  typename Executor::policy_t::event_t ret;
  auto assignOp =
      make_Ger_Col(mA, _alpha, vx, vy, nWGPerRow, nWGPerCol, scratchPadSize);
  return ex.execute(assignOp, localSize, globalSize, scratchPadSize);
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
template <typename Executor, typename index_t, typename element_t,
          typename container_t0, typename increment_t, typename container_t1>
typename Executor::policy_t::event_t _syr_impl(
    Executor& ex, char _Uplo, index_t _N, element_t _alpha, container_t0 _vx,
    increment_t _incx, container_t1 _mA, index_t _lda, index_t _localSize = 0,
    index_t _scratchPadSize = 0, index_t _nRowsWG = 0, index_t _nColsWG = 0) {
  typename Executor::policy_t::event_t ret;
  _Uplo = tolower(_Uplo);
  int triangOpr = (_Uplo == 'u');
  index_t N = _N;
  auto mA = make_matrix_view<col_major>(ex, _mA, N, N, _lda);
  auto vx = make_vector_view(ex, _vx, _incx, N);

  const index_t localSize = (_localSize == 0)
                                ? ex.get_policy_handler().get_work_group_size()
                                : _localSize;
  const index_t nRowsWG = (_nRowsWG == 0) ? localSize : std::min(N, _nRowsWG);
  const index_t nColsWG = (_nColsWG == 0) ? localSize : std::min(N, _nColsWG);
  const index_t scratchPadSize =
      (_localSize == 0) ? localSize : _scratchPadSize;

  const index_t nWGPerRow = (N - 1) / nRowsWG + 1;
  const index_t nWGPerCol = (N - 1) / nColsWG + 1;
  const index_t globalSize = localSize * nWGPerRow * nWGPerCol;

  if (triangOpr) {
    auto assignOp = make_Ger_Col<true, false, true, true>(
        mA, _alpha, vx, vx, nWGPerRow, nWGPerCol, scratchPadSize);
    return ret = concatenate_vectors(
               ret,
               ex.execute(assignOp, localSize, globalSize, scratchPadSize));
  } else {
    auto assignOp = make_Ger_Col<true, true, true, false>(
        mA, _alpha, vx, vx, nWGPerRow, nWGPerCol, scratchPadSize);
    return ret = concatenate_vectors(
               ret,
               ex.execute(assignOp, localSize, globalSize, scratchPadSize));
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
template <typename Executor, typename index_t, typename element_t,
          typename container_t0, typename increment_t, typename container_t1,
          typename container_t2>
typename Executor::policy_t::event_t _syr2_impl(
    Executor& ex, char _Uplo, index_t _N, element_t _alpha, container_t0 _vx,
    increment_t _incx, container_t1 _vy, increment_t _incy, container_t2 _mA,
    index_t _lda, index_t _localSize = 0, index_t _scratchPadSize = 0,
    index_t _nRowsWG = 0, index_t _nColsWG = 0) {
  _Uplo = tolower(_Uplo);
  int triangOpr = (_Uplo == 'u');
  index_t N = _N;

  auto mA = make_matrix_view<col_major>(ex, _mA, _N, _N, _lda);
  auto vx = make_vector_view(ex, _vx, _incx, _N);
  auto vy = make_vector_view(ex, _vy, _incy, _N);

  const index_t localSize = (_localSize == 0)
                                ? ex.get_policy_handler().get_work_group_size()
                                : _localSize;
  const index_t nRowsWG = (_nRowsWG == 0) ? localSize : std::min(N, _nRowsWG);
  const index_t nColsWG = (_nColsWG == 0) ? localSize : std::min(N, _nColsWG);
  const index_t scratchPadSize =
      (_localSize == 0) ? 2 * localSize : _scratchPadSize;

  const index_t nWGPerRow = (N - 1) / nRowsWG + 1;
  const index_t nWGPerCol = (N - 1) / nColsWG + 1;
  const index_t globalSize = localSize * nWGPerRow * nWGPerCol;

  if (triangOpr) {
    auto assignOp = make_Ger_Col<false, false, true, true>(
        mA, _alpha, vx, vy, nWGPerRow, nWGPerCol, scratchPadSize);
    return ex.execute(assignOp, localSize, globalSize, scratchPadSize);
  } else {
    auto assignOp = make_Ger_Col<false, true, true, false>(
        mA, _alpha, vx, vy, nWGPerRow, nWGPerCol, scratchPadSize);
    return ex.execute(assignOp, localSize, globalSize, scratchPadSize);
  }
}

/*!
 @brief Generalised matrix vector product with rectangular non-symmetric
 matrices.

 Generalised matrix vector product with rectangular non-symmetric matrices, i.e.
 computing the mathematical operation:

 y = alpha*A*x + beta*y

 See the netlib blas interface documentation for more details of the high level
 interface: http://www.netlib.org/lapack/explore-html/db/d58/sgemv_8f.html

 */
template <typename Executor, typename index_t, typename element_t,
          typename container_t0, typename container_t1, typename increment_t,
          typename container_t2>
typename Executor::policy_t::event_t inline _gemv(
    Executor& ex,       // Executor (sycl, parallel, serial, etc)
    char _trans,        // The transposition of the matrix ('n', 't', 'c')
    index_t _M,         // The size of dimension M of the matrix (rows)
    index_t _N,         // The size of dimension N of the matrix (columns)
    element_t _alpha,   // Scalar parameter Alpha
    container_t0 _mA,   // An array (LDA,N), with the first m*n elements
    index_t _lda,       // Specifies the first dimension of a, max(1, m)
    container_t1 _vx,   // An array of dimension at least: (1+(n-1)*abs(incx))
                        // when trans = 'n' and (1+(m-1)*abs(incx) otherwise,
                        // containing the vector "x"
    increment_t _incx,  // The increment for elements in x (nonzero).
    element_t _beta,    // Scalar parameter Beta
    container_t2 _vy,   // An array of dimension at least: (1+(m-1)*abs(incy))
                        // when trans = "n" and (1+(n-1)*abs(incy) otherwise,
    // containing the vector "y" (if beta is nonzero). When
    // finished, y is overwritten with the updated vector.
    increment_t _incy  // The increment for elements in y (nonzero).
) {
  return tolower(_trans) == 'n'
             ? _gemv_impl<transpose_type::Normal>(ex, _M, _N, _alpha, _mA, _lda,
                                                  _vx, _incx, _beta, _vy, _incy)
             : _gemv_impl<transpose_type::Transposed>(ex, _M, _N, _alpha, _mA,
                                                      _lda, _vx, _incx, _beta,
                                                      _vy, _incy);
}

template <typename Executor, typename index_t, typename container_t0,
          typename container_t1, typename increment_t>
typename Executor::policy_t::event_t inline _trmv(
    Executor& ex, char _Uplo, char _trans, char _Diag, index_t _N,
    container_t0 _mA, index_t _lda, container_t1 _vx, increment_t _incx) {
  // TODO: Here we can use some heuristics to select localn global, local, and
  // scratch size per device
  return tolower(_trans) == 'n'
             ? _trmv_impl<transpose_type::Normal>(ex, _Uplo, _Diag, _N, _mA,
                                                  _lda, _vx, _incx)
             : _trmv_impl<transpose_type::Transposed>(ex, _Uplo, _Diag, _N, _mA,
                                                      _lda, _vx, _incx);
}
template <typename Executor, typename index_t, typename element_t,
          typename container_t0, typename container_t1, typename increment_t,
          typename container_t2>
typename Executor::policy_t::event_t inline _symv(
    Executor& ex, char _Uplo, index_t _N, element_t _alpha, container_t0 _mA,
    index_t _lda, container_t1 _vx, increment_t _incx, element_t _beta,
    container_t2 _vy, increment_t _incy) {
  // TODO: Here we can use some heuristics to select localn global, local, and
  // scratch size per device
  return _symv_impl(ex, _Uplo, _N, _alpha, _mA, _lda, _vx, _incx, _beta, _vy,
                    _incy);
}
template <typename Executor, typename index_t, typename element_t,
          typename container_t0, typename increment_t, typename container_t1,
          typename container_t2>
typename Executor::policy_t::event_t inline _ger(
    Executor& ex, index_t _M, index_t _N, element_t _alpha, container_t0 _vx,
    increment_t _incx, container_t1 _vy, increment_t _incy, container_t2 _mA,
    index_t _lda) {
  // TODO: Here we can use some heuristics to select localn global, local, and
  // scratch size per device
  return _ger_impl(ex, _M, _N, _alpha, _vx, _incx, _vy, _incy, _mA, _lda);
}
template <typename Executor, typename index_t, typename element_t,
          typename container_t0, typename increment_t, typename container_t1>
typename Executor::policy_t::event_t inline _syr(
    Executor& ex, char _Uplo, index_t _N, element_t _alpha, container_t0 _vx,
    increment_t _incx, container_t1 _mA, index_t _lda) {
  // TODO: Here we can use some heuristics to select localn global, local, and
  // scratch size per device
  return _syr_impl(ex, _Uplo, _N, _alpha, _vx, _incx, _mA, _lda);
}
template <typename Executor, typename index_t, typename element_t,
          typename container_t0, typename increment_t, typename container_t1,
          typename container_t2>
typename Executor::policy_t::event_t inline _syr2(
    Executor& ex, char _Uplo, index_t _N, element_t _alpha, container_t0 _vx,
    increment_t _incx, container_t1 _vy, increment_t _incy, container_t2 _mA,
    index_t _lda) {
  // TODO: Here we can use some heuristics to select localn global, local, and
  // scratch size per device
  return _syr2_impl(ex, _Uplo, _N, _alpha, _vx, _incx, _vy, _incy, _mA, _lda);
}

}  // namespace internal
}  // namespace blas

#endif  // BLAS2_INTERFACE_HPP
