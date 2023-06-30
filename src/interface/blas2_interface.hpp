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
#include "interface/blas2/backend/backend.hpp"
#include "interface/blas2_interface.h"
#include "operations/blas2_trees.h"
#include "operations/blas_constants.h"
#include "operations/blas_operators.hpp"
#include "sb_handle/sycl_blas_handle.h"
#include "views/view.h"
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace blas {
namespace internal {

/*! _gemv_impl.
 * @brief Internal implementation of the General Matrix Vector product.
 *
 * This function contains the code that sets up and executes the kernels
 * required to perform the gemv operation.
 *
 * This function is called by blas::internal::backend::gemv which, dependant on
 * the platform being compiled for and other parameters, provides different
 * template parameters to ensure the most optimal kernel is constructed
 *
 * @tparam local_range  specifies the number of threads per work group used by
 *                      the kernel
 * @tparam cache_line_size  specifies the size in bytes of the cache line. This
 *                          value will determine the dimensions of tiles loaded
 *                          into local memory in the transposed local memory
 *                          version of the kernel
 * @tparam memory_type  specifies whether the kernel should use local shared
 *                      memory or not
 * @tparam trn  specifies whether the input matrix should be transposed
 *
 */
template <uint32_t local_range, uint32_t cache_line_size,
          gemv_memory_t memory_type, transpose_type trn, typename sb_handle_t,
          typename index_t, typename element_t, typename container_t0,
          typename container_t1, typename increment_t, typename container_t2>
typename sb_handle_t::event_t _gemv_impl(sb_handle_t& sb_handle, index_t _M,
                                         index_t _N, element_t _alpha,
                                         container_t0 _mA, index_t _lda,
                                         container_t1 _vx, increment_t _incx,
                                         element_t _beta, container_t2 _vy,
                                         increment_t _incy) {
  constexpr int cl_elems = cache_line_size / sizeof(element_t);
  constexpr bool is_transposed = trn != transpose_type::Normal;

  const auto x_vector_size = is_transposed ? _M : _N;
  const auto y_vector_size = is_transposed ? _N : _M;

  auto mA = make_matrix_view<col_major>(_mA, _M, _N, _lda);
  auto vx = make_vector_view(_vx, _incx, x_vector_size);
  auto vy = make_vector_view(_vy, _incy, y_vector_size);

  // Non-local memory kernel
  if (memory_type != gemv_memory_t::local) {
    // Leading dimension for dot products matrix
    const auto ld = is_transposed ? _N : _M;
    constexpr index_t one = 1;

    auto dot_products_buffer = blas::make_sycl_iterator_buffer<element_t>(ld);
    auto dot_products_matrix =
        make_matrix_view<col_major>(dot_products_buffer, ld, one, ld);

    const index_t global_size = roundUp<index_t>(ld, local_range);

    auto gemv = make_gemv<local_range, is_transposed, cache_line_size, 1>(
        dot_products_matrix, mA, vx, one, one);

    // Execute the GEMV kernel that calculate the partial dot products of rows
    // auto gemvEvent = sb_handle.execute(gemv, local_range, global_size);
    auto gemvEvent =
        sb_handle.execute(gemv, static_cast<index_t>(local_range), global_size);

    if (_beta != static_cast<element_t>(0)) {
      // vec_y * b
      auto betaMulYOp = make_op<ScalarOp, ProductOperator>(_beta, vy);

      // alpha * vec_dot_products
      auto alphaMulDotsOp =
          make_op<ScalarOp, ProductOperator>(_alpha, dot_products_matrix);

      // add up
      auto addOp = make_op<BinaryOp, AddOperator>(betaMulYOp, alphaMulDotsOp);

      // assign the result back to vec_y
      auto assignOp = make_op<Assign>(vy, addOp);

      // exectutes the above expression tree to yield the final GEMV result
      return concatenate_vectors(gemvEvent,
                                 sb_handle.execute(assignOp, local_range));
    } else {
      auto alphaMulDotsOp =
          make_op<ScalarOp, ProductOperator>(_alpha, dot_products_matrix);
      auto assignOp = make_op<Assign>(vy, alphaMulDotsOp);
      return concatenate_vectors(gemvEvent,
                                 sb_handle.execute(assignOp, local_range));
    }

  } else  // Local memory kernel
  {
    // Calculate number of work groups per each dimension based on the local
    // range
    const index_t WGs_per_NC =
        is_transposed ? (_N - 1) / local_range + 1 : (_M - 1) / local_range + 1;
    const index_t WGs_per_C =
        is_transposed ? (_M - 1) / local_range + 1 : (_N - 1) / local_range + 1;

    // Calculate the scratch size the kernel requires
    // When input matrix should be transposed then we add more memory to
    // transpose it on the fly
    // We add one to cl_elems to eliminate bank conflicts
    const index_t kernel_scratch_size =
        local_range + (is_transposed ? (cl_elems + 1) * local_range : 0);

    // Leading dimension for partial dot products matrix
    const auto ld = is_transposed ? _N : _M;
    const auto dot_products_buffer_size = ld * WGs_per_C;

    // Create the dot products buffer and matrix view
    auto dot_products_buffer =
        blas::make_sycl_iterator_buffer<element_t>(dot_products_buffer_size);
    auto dot_products_matrix =
        make_matrix_view<col_major>(dot_products_buffer, ld, WGs_per_C, ld);

    const index_t global_size = local_range * WGs_per_C * WGs_per_NC;

    // Create the gemv kernel
    auto gemv = make_gemv<local_range, is_transposed, cache_line_size, 1>(
        dot_products_matrix, mA, vx, WGs_per_NC, WGs_per_C);

    // Execute the GEMV kernel that calculate the partial dot products of rows
    auto gemvEvent = sb_handle.execute(gemv, static_cast<index_t>(local_range),
                                       global_size, kernel_scratch_size);

    // Sum the partial dot products results from the GEMV kernel
    auto sumColsOp = make_sum_matrix_columns(dot_products_matrix);

    if (_beta != static_cast<element_t>(0)) {
      // vec_y * b
      auto betaMulYOp = make_op<ScalarOp, ProductOperator>(_beta, vy);

      // alpha * vec_dot_products
      auto alphaMulDotsOp =
          make_op<ScalarOp, ProductOperator>(_alpha, sumColsOp);

      // add up
      auto addOp = make_op<BinaryOp, AddOperator>(betaMulYOp, alphaMulDotsOp);

      // assign the result back to vec_y
      auto assignOp = make_op<Assign>(vy, addOp);

      // exectutes the above expression tree to yield the final GEMV result
      return concatenate_vectors(gemvEvent,
                                 sb_handle.execute(assignOp, local_range));
    } else {
      auto alphaMulDotsOp =
          make_op<ScalarOp, ProductOperator>(_alpha, sumColsOp);
      auto assignOp = make_op<Assign>(vy, alphaMulDotsOp);
      return concatenate_vectors(gemvEvent,
                                 sb_handle.execute(assignOp, local_range));
    }
  }
}

/*! _TRMV.
 * @brief Implementation of the Triangular Matrix Vector product.
 */

template <transpose_type trn, typename sb_handle_t, typename index_t,
          typename container_t0, typename container_t1, typename increment_t>
typename sb_handle_t::event_t _trmv_impl(
    sb_handle_t& sb_handle, char _Uplo, char _Diag, index_t _N,
    container_t0 _mA, index_t _lda, container_t1 _vx, increment_t _incx,
    index_t _localSize = 0, index_t _scratchPadSize = 0, index_t _nRowsWG = 0,
    index_t _nColsWG = 0) {
  typename sb_handle_t::event_t ret{};
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
  auto mA = make_matrix_view<data_layout_t>(_mA, N, N, _lda);
  auto vx = make_vector_view(_vx, _incx, N);
  const index_t interLoop = 1;
  const index_t localSize =
      (_localSize == 0) ? sb_handle.get_work_group_size() : _localSize;
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
  auto mat1 = make_matrix_view<row_major>(valT1, N, scratchSize, scratchSize);

  if (data_layout_t::is_col_major()) {
    if (triangOpr == 1) {
      if (unitDiag == 1) {
        auto gemvC = make_gemv_col<false, true, true, true>(
            mat1, mA, vx, nWGPerRow, nWGPerCol, scratchPadSize);
        ret = concatenate_vectors(
            ret,
            sb_handle.execute(gemvC, localSize, globalSize, scratchPadSize));
      } else {
        auto gemvC = make_gemv_col<false, true, true>(
            mat1, mA, vx, nWGPerRow, nWGPerCol, scratchPadSize);
        ret = concatenate_vectors(
            ret,
            sb_handle.execute(gemvC, localSize, globalSize, scratchPadSize));
      }
    } else {
      if (unitDiag == 1) {
        auto gemvC = make_gemv_col<true, true, false, true>(
            mat1, mA, vx, nWGPerRow, nWGPerCol, scratchPadSize);
        ret = concatenate_vectors(
            ret,
            sb_handle.execute(gemvC, localSize, globalSize, scratchPadSize));
      } else {
        auto gemvC = make_gemv_col<true, true, false>(
            mat1, mA, vx, nWGPerRow, nWGPerCol, scratchPadSize);
        ret = concatenate_vectors(
            ret,
            sb_handle.execute(gemvC, localSize, globalSize, scratchPadSize));
      }
    }
  } else {  // row_major
    if (triangOpr == 1) {
      if (unitDiag == 1) {
        auto gemvR = make_gemv_row<interLoop, false, true, true, true>(
            mat1, mA, vx, nWGPerRow, nWGPerCol, scratchPadSize);
        ret = concatenate_vectors(
            ret,
            sb_handle.execute(gemvR, localSize, globalSize, scratchPadSize));
      } else {
        auto gemvR = make_gemv_row<interLoop, false, true, true>(
            mat1, mA, vx, nWGPerRow, nWGPerCol, scratchPadSize);
        ret = concatenate_vectors(
            ret,
            sb_handle.execute(gemvR, localSize, globalSize, scratchPadSize));
      }
    } else {
      if (unitDiag == 1) {
        auto gemvR = make_gemv_row<interLoop, true, true, false, true>(
            mat1, mA, vx, nWGPerRow, nWGPerCol, scratchPadSize);
        ret = concatenate_vectors(
            ret,
            sb_handle.execute(gemvR, localSize, globalSize, scratchPadSize));
      } else {
        auto gemvR = make_gemv_row<interLoop, true, true, false>(
            mat1, mA, vx, nWGPerRow, nWGPerCol, scratchPadSize);
        ret = concatenate_vectors(
            ret,
            sb_handle.execute(gemvR, localSize, globalSize, scratchPadSize));
      }
    }
  }
  auto addMOp = make_sum_matrix_columns(mat1);
  auto assignOp = make_op<Assign>(vx, addMOp);
  ret = concatenate_vectors(ret, sb_handle.execute(assignOp, localSize));
  return ret;
}

/*! _TRSV.
 * @brief Implementation of the Triangular Matrix Vector product.
 */
template <uint32_t subgroup_size, uint32_t subgroups, uplo_type uplo,
          transpose_type trn, diag_type diag, typename sb_handle_t,
          typename index_t, typename container_t0, typename container_t1,
          typename increment_t>
typename sb_handle_t::event_t _trsv_impl(sb_handle_t& sb_handle, index_t _N,
                                         container_t0 _mA, index_t _lda,
                                         container_t1 _vx, increment_t _incx) {
#if SYCL_LANGUAGE_VERSION < 202000
  throw std::runtime_error("trsv requires SYCL 2020");
#else
  static_assert(subgroup_size % subgroups == 0,
                "`subgroups` needs to be a multiple of `subgroup_size`.");

  using one = constant<increment_t, const_val::one>;
  constexpr bool is_upper = (uplo == uplo_type::Upper);
  constexpr bool is_transposed = (trn != transpose_type::Normal);
  constexpr bool is_unit = (diag == diag_type::Unit);

  constexpr bool is_forward =
      (is_upper && is_transposed) || (!is_upper && !is_transposed);

  auto mA = make_matrix_view<col_major>(_mA, _N, _N, _lda);
  auto vx = make_vector_view(_vx, _incx, _N);

  std::vector<int32_t> sync_vec(2);
  sync_vec[0] =
      is_forward ? 0
                 : ((roundUp<index_t>(_N, subgroup_size) / subgroup_size) - 1);
  sync_vec[1] = sync_vec[0];

  auto sync_buffer =
      blas::make_sycl_iterator_buffer<int32_t>(sync_vec, sync_vec.size());
  auto sync = make_vector_view(sync_buffer, one::value(), sync_vec.size());

  index_t unused;
  auto trsv = make_txsv<matrix_format_t::full, subgroup_size, subgroups,
                        is_upper, is_transposed, is_unit>(vx, mA, unused, sync);

  const index_t sub_num = subgroups;
  return sb_handle.execute(
      trsv, static_cast<index_t>(sub_num * subgroup_size),
      roundUp<index_t>(sub_num * _N, sub_num * subgroup_size),
      static_cast<index_t>(subgroup_size * (subgroup_size + 2 + sub_num)));
#endif
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
template <typename sb_handle_t, typename index_t, typename element_t,
          typename container_t0, typename container_t1, typename increment_t,
          typename container_t2>
typename sb_handle_t::event_t _symv_impl(
    sb_handle_t& sb_handle, char _Uplo, index_t _N, element_t _alpha,
    container_t0 _mA, index_t _lda, container_t1 _vx, increment_t _incx,
    element_t _beta, container_t2 _vy, increment_t _incy,
    index_t _localSize = 0, index_t _scratchPadSize = 0, index_t _nRowsWG = 0,
    index_t _nColsWG = 0) {
  _Uplo = tolower(_Uplo);
  typename sb_handle_t::event_t ret;
  if ((_Uplo != 'u') && (_Uplo != 'l')) {
    throw std::invalid_argument("Erroneous parameter");
  }
  int triangOpr = (_Uplo == 'u');
  index_t N = _N;
  auto mA = make_matrix_view<col_major>(_mA, N, N, _lda);
  auto vx = make_vector_view(_vx, _incx, N);
  auto vy = make_vector_view(_vy, _incy, N);
  auto mAT = make_matrix_view<row_major>(_mA, N, N, _lda);

  const index_t interLoop = 1;

  const index_t localSize =
      (_localSize == 0) ? sb_handle.get_work_group_size() : _localSize;
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
      make_matrix_view<row_major>(valTR, N, scratchSize_R, scratchSize_R);

  const index_t scratchSize_C = nWGPerCol_C;

  auto valTC = blas::make_sycl_iterator_buffer<element_t>(N * scratchSize_C);
  auto matC =
      make_matrix_view<row_major>(valTC, N, scratchSize_C, scratchSize_C);

  if (triangOpr == 1) {
    auto gemvC = make_gemv_col<false, true, true>(matC, mA, vx, nWGPerRow_C,
                                                  nWGPerCol_C, scratchPadSize);
    auto gemvR = make_gemv_row<interLoop, true, false, false>(
        matR, mAT, vx, nWGPerRow_R, nWGPerCol_R, scratchPadSize);
    ret = concatenate_vectors(
        ret, sb_handle.execute(gemvC, localSize, globalSize_C, scratchPadSize));
    ret = concatenate_vectors(
        ret, sb_handle.execute(gemvR, localSize, globalSize_R, scratchPadSize));
  } else {
    auto gemvC = make_gemv_col<true, true, false>(matC, mA, vx, nWGPerRow_C,
                                                  nWGPerCol_C, scratchPadSize);
    auto gemvR = make_gemv_row<interLoop, false, false, true>(
        matR, mAT, vx, nWGPerRow_R, nWGPerCol_R, scratchPadSize);
    ret = concatenate_vectors(
        ret, sb_handle.execute(gemvC, localSize, globalSize_C, scratchPadSize));
    ret = concatenate_vectors(
        ret, sb_handle.execute(gemvR, localSize, globalSize_R, scratchPadSize));
  }

  auto scalOp1 = make_op<ScalarOp, ProductOperator>(_beta, vy);
  auto addMOpR = make_sum_matrix_columns(matR);
  auto addMOpC = make_sum_matrix_columns(matC);
  auto addMOp = make_op<BinaryOp, AddOperator>(addMOpR, addMOpC);
  auto scalOp2 = make_op<ScalarOp, ProductOperator>(_alpha, addMOp);
  auto addOp = make_op<BinaryOp, AddOperator>(scalOp1, scalOp2);
  auto assignOp = make_op<Assign>(vy, addOp);
  ret = concatenate_vectors(ret, sb_handle.execute(assignOp, localSize));
  return ret;
}

/*! _gbmv_impl.
 * @brief Implementation of the Generic Band Matrix Vector product.
 *
 */
template <uint32_t local_range, transpose_type trn, typename sb_handle_t,
          typename index_t, typename element_t, typename container_t0,
          typename container_t1, typename increment_t, typename container_t2>
typename sb_handle_t::event_t _gbmv_impl(sb_handle_t& sb_handle, index_t _M,
                                         index_t _N, index_t _KL, index_t _KU,
                                         element_t _alpha, container_t0 _mA,
                                         index_t _lda, container_t1 _vx,
                                         increment_t _incx, element_t _beta,
                                         container_t2 _vy, increment_t _incy) {
  if ((_KL >= _M) || (_KU >= _N)) {
    throw std::invalid_argument("Erroneous parameter: _KL >= _M || _KU >= _N");
  }

  constexpr bool is_transposed = (trn != transpose_type::Normal);

  auto x_vector_size = is_transposed ? _M : _N;
  auto y_vector_size = is_transposed ? _N : _M;

  auto mA =
      make_matrix_view<col_major>(_mA, _KL + _KU + 1, x_vector_size, _lda);
  auto vx = make_vector_view(_vx, _incx, x_vector_size);
  auto vy = make_vector_view(_vy, _incy, y_vector_size);

  auto gbmv = make_gbmv<local_range, is_transposed>(_KL, _KU, _alpha, mA, vx,
                                                    _beta, vy);

  return sb_handle.execute(gbmv, static_cast<index_t>(local_range),
                           roundUp<index_t>(y_vector_size, local_range));
}

/*! _sbmv_impl.
 * @brief Implementation of the Symmetric Band Matrix Vector product.
 *
 */
template <uint32_t local_range, uplo_type uplo, typename sb_handle_t,
          typename index_t, typename element_t, typename container_t0,
          typename container_t1, typename increment_t, typename container_t2>
typename sb_handle_t::event_t _sbmv_impl(sb_handle_t& sb_handle, index_t _N,
                                         index_t _K, element_t _alpha,
                                         container_t0 _mA, index_t _lda,
                                         container_t1 _vx, increment_t _incx,
                                         element_t _beta, container_t2 _vy,
                                         increment_t _incy) {
  if (_K >= _N) {
    throw std::invalid_argument("Erroneous parameter: _K >= _N");
  }

  auto vector_size = _N;

  auto mA = make_matrix_view<col_major>(_mA, _K + 1, _N, _lda);
  auto vx = make_vector_view(_vx, _incx, vector_size);
  auto vy = make_vector_view(_vy, _incy, vector_size);

  auto sbmv = make_sbmv<local_range, uplo == uplo_type::Upper>(_K, _alpha, mA,
                                                               vx, _beta, vy);

  return sb_handle.execute(sbmv, static_cast<index_t>(local_range),
                           roundUp<index_t>(vector_size, local_range));
}

/*! _spmv_impl.
 * @brief Implementation of the Symmetric Packed Matrix Vector product.
 *
 */
template <uint32_t local_range_x, uint32_t local_range_y, uplo_type uplo,
          typename sb_handle_t, typename index_t, typename element_t,
          typename container_t0, typename container_t1, typename increment_t,
          typename container_t2>
typename sb_handle_t::event_t _spmv_impl(sb_handle_t& sb_handle, index_t _N,
                                         element_t _alpha, container_t0 _mA,
                                         container_t1 _vx, increment_t _incx,
                                         element_t _beta, container_t2 _vy,
                                         increment_t _incy) {
  static_assert(local_range_x % local_range_y == 0,
                "Local y range needs to be a multiple of local x range.");

  constexpr bool is_upper = (uplo == uplo_type::Upper);

  constexpr index_t one = 1;

  index_t vector_size = _N;
  index_t matrix_size = ((_N + 1) * _N) / 2;

  auto mA = make_matrix_view<col_major>(_mA, one, matrix_size, matrix_size);
  auto vx = make_vector_view(_vx, _incx, vector_size);
  auto vy = make_vector_view(_vy, _incy, vector_size);

  auto spmv =
      make_xpmv<local_range_x, local_range_y, true, is_upper, false, false>(
          _alpha, mA, vx, _beta, vy);

  const index_t loc_mem_leading_dim = local_range_x + 1;

  return sb_handle.execute(
      spmv, static_cast<index_t>(local_range_y * local_range_x),
      roundUp<index_t>(local_range_y * vector_size,
                       local_range_y * local_range_x),
      static_cast<index_t>(local_range_x * (loc_mem_leading_dim + 2)));
}

template <uint32_t local_range, uplo_type uplo, transpose_type trn,
          diag_type diag, typename sb_handle_t, typename index_t,
          typename container_t0, typename container_t1, typename increment_t>
typename sb_handle_t::event_t _tbmv_impl(sb_handle_t& sb_handle, index_t _N,
                                         index_t _K, container_t0 _mA,
                                         index_t _lda, container_t1 _vx,
                                         increment_t _incx) {
  constexpr bool is_upper = (uplo == uplo_type::Upper);
  constexpr bool is_transposed = (trn != transpose_type::Normal);
  constexpr bool is_unit = (diag == diag_type::Unit);

  if (_K >= _N) {
    throw std::invalid_argument("Erroneous parameter: _K >= _N");
  }

  using one = constant<index_t, const_val::one>;
  auto x_vector_size = _N;
  auto res_buffer =
      blas::make_sycl_iterator_buffer<typename container_t0::scalar_t>(
          x_vector_size);

  auto mA = make_matrix_view<col_major>(_mA, _K + 1, _N, _lda);
  auto vx = make_vector_view(_vx, _incx, x_vector_size);
  auto vres = make_vector_view(res_buffer, one::value(), x_vector_size);

  const index_t global_size = roundUp<index_t>(x_vector_size, local_range);
  auto tbmv = make_tbmv<local_range, is_upper, is_transposed, is_unit>(vres, mA,
                                                                       _K, vx);

  auto tbmvEvent =
      sb_handle.execute(tbmv, static_cast<index_t>(local_range), global_size);

  auto assignOp = make_op<Assign>(vx, vres);
  return concatenate_vectors(tbmvEvent,
                             sb_handle.execute(assignOp, local_range));
}

template <uint32_t local_range_x, uint32_t local_range_y, uplo_type uplo,
          transpose_type trn, diag_type diag, typename sb_handle_t,
          typename index_t, typename container_t0, typename container_t1,
          typename increment_t>
typename sb_handle_t::event_t _tpmv_impl(sb_handle_t& sb_handle, index_t _N,
                                         container_t0 _mA, container_t1 _vx,
                                         increment_t _incx) {
  static_assert(local_range_x % local_range_y == 0,
                "Local y range needs to be a multiple of local x range.");

  constexpr bool is_upper = (uplo == uplo_type::Upper);
  constexpr bool is_transposed = (trn != transpose_type::Normal);
  constexpr bool is_unit = (diag == diag_type::Unit);

  constexpr index_t one = 1;

  index_t vector_size = _N;
  index_t matrix_size = ((_N + 1) * _N) / 2;

  auto res_buffer =
      blas::make_sycl_iterator_buffer<typename container_t0::scalar_t>(
          vector_size);

  auto mA = make_matrix_view<col_major>(_mA, one, matrix_size, matrix_size);
  auto vx = make_vector_view(_vx, _incx, vector_size);
  auto vres = make_vector_view(res_buffer, one, vector_size);

  typename container_t0::scalar_t unused;

  auto tpmv = make_xpmv<local_range_x, local_range_y, false, is_upper,
                        is_transposed, is_unit>(unused, mA, vx, unused, vres);

  const index_t loc_mem_leading_dim = local_range_x + 1;

  auto tpmvEvent = sb_handle.execute(
      tpmv, static_cast<index_t>(local_range_y * local_range_x),
      roundUp<index_t>(local_range_y * vector_size,
                       local_range_y * local_range_x),
      static_cast<index_t>(local_range_x * (loc_mem_leading_dim + 2)));

  auto assignOp = make_op<Assign>(vx, vres);
  return concatenate_vectors(tpmvEvent, sb_handle.execute(assignOp));
}

template <uint32_t subgroup_size, uint32_t subgroups, uplo_type uplo,
          transpose_type trn, diag_type diag, typename sb_handle_t,
          typename index_t, typename container_t0, typename container_t1,
          typename increment_t>
typename sb_handle_t::event_t _tbsv_impl(sb_handle_t& sb_handle, index_t _N,
                                         index_t _K, container_t0 _mA,
                                         index_t _lda, container_t1 _vx,
                                         increment_t _incx) {
#if SYCL_LANGUAGE_VERSION < 202000
  throw std::runtime_error("tbsv requires SYCL 2020");
#else
  static_assert(subgroup_size % subgroups == 0,
                "`subgroups` needs to be a multiple of `subgroup_size`.");

  if (_K >= _N) throw std::invalid_argument("Erroneous parameter: _K >= _N");

  using one = constant<increment_t, const_val::one>;
  constexpr bool is_upper = (uplo == uplo_type::Upper);
  constexpr bool is_transposed = (trn != transpose_type::Normal);
  constexpr bool is_unit = (diag == diag_type::Unit);

  constexpr bool is_forward =
      (is_upper && is_transposed) || (!is_upper && !is_transposed);

  auto mA = make_matrix_view<col_major>(_mA, _K + 1, _N, _lda);
  auto vx = make_vector_view(_vx, _incx, _N);

  std::vector<int32_t> sync_vec(2);
  sync_vec[0] =
      is_forward ? 0
                 : ((roundUp<index_t>(_N, subgroup_size) / subgroup_size) - 1);
  sync_vec[1] = sync_vec[0];

  auto sync_buffer =
      blas::make_sycl_iterator_buffer<int32_t>(sync_vec, sync_vec.size());
  auto sync = make_vector_view(sync_buffer, one::value(), sync_vec.size());

  auto tbsv = make_txsv<matrix_format_t::banded, subgroup_size, subgroups,
                        is_upper, is_transposed, is_unit>(vx, mA, _K, sync);

  const index_t sub_num = subgroups;
  return sb_handle.execute(
      tbsv, static_cast<index_t>(sub_num * subgroup_size),
      roundUp<index_t>(sub_num * _N, sub_num * subgroup_size),
      static_cast<index_t>(subgroup_size * (subgroup_size + 2 + sub_num)));
#endif
}

template <uint32_t subgroup_size, uint32_t subgroups, uplo_type uplo,
          transpose_type trn, diag_type diag, typename sb_handle_t,
          typename index_t, typename container_t0, typename container_t1,
          typename increment_t>
typename sb_handle_t::event_t _tpsv_impl(sb_handle_t& sb_handle, index_t _N,
                                         container_t0 _mA, container_t1 _vx,
                                         increment_t _incx) {
#if SYCL_LANGUAGE_VERSION < 202000
  throw std::runtime_error("tpsv requires SYCL 2020");
#else
  static_assert(subgroup_size % subgroups == 0,
                "`subgroups` needs to be a multiple of `subgroup_size`.");

  using one = constant<increment_t, const_val::one>;
  constexpr bool is_upper = (uplo == uplo_type::Upper);
  constexpr bool is_transposed = (trn != transpose_type::Normal);
  constexpr bool is_unit = (diag == diag_type::Unit);

  constexpr bool is_forward =
      (is_upper && is_transposed) || (!is_upper && !is_transposed);

  index_t matrix_size = ((_N + 1) * _N) / 2;

  auto mA =
      make_matrix_view<col_major>(_mA, one::value(), matrix_size, matrix_size);
  auto vx = make_vector_view(_vx, _incx, _N);

  std::vector<int32_t> sync_vec(2);
  sync_vec[0] =
      is_forward ? 0
                 : ((roundUp<index_t>(_N, subgroup_size) / subgroup_size) - 1);
  sync_vec[1] = sync_vec[0];

  auto sync_buffer =
      blas::make_sycl_iterator_buffer<int32_t>(sync_vec, sync_vec.size());
  auto sync = make_vector_view(sync_buffer, one::value(), sync_vec.size());

  index_t unused;
  auto tpsv = make_txsv<matrix_format_t::packed, subgroup_size, subgroups,
                        is_upper, is_transposed, is_unit>(vx, mA, unused, sync);

  const index_t sub_num = subgroups;
  return sb_handle.execute(
      tpsv, static_cast<index_t>(sub_num * subgroup_size),
      roundUp<index_t>(sub_num * _N, sub_num * subgroup_size),
      static_cast<index_t>(subgroup_size * (subgroup_size + 2 + sub_num)));
#endif
}

/**** RANK 1 MODIFICATION ****/

template <typename sb_handle_t, typename index_t, typename element_t,
          typename container_t0, typename increment_t, typename container_t1,
          typename container_t2>
typename sb_handle_t::event_t _ger_impl(
    sb_handle_t& sb_handle, index_t _M, index_t _N, element_t _alpha,
    container_t0 _vx, increment_t _incx, container_t1 _vy, increment_t _incy,
    container_t2 _mA, index_t _lda, index_t _localSize = 0,
    index_t _scratchPadSize = 0, index_t _nRowsWG = 0, index_t _nColsWG = 0) {
  index_t M = _M;
  index_t N = _N;
  auto mA = make_matrix_view<col_major>(_mA, M, N, _lda);
  auto vx = make_vector_view(_vx, _incx, M);
  auto vy = make_vector_view(_vy, _incy, N);

  const index_t localSize =
      (_localSize == 0) ? sb_handle.get_work_group_size() : _localSize;
  const index_t nRowsWG = (_nRowsWG == 0) ? localSize : std::min(M, _nRowsWG);

  const index_t nColsWG = (_nColsWG == 0) ? localSize : std::min(N, _nColsWG);

  const index_t scratchPadSize =
      (_localSize == 0) ? localSize : _scratchPadSize;

  const index_t nWGPerCol = (N - 1) / nColsWG + 1;
  const index_t nWGPerRow = (M - 1) / nRowsWG + 1;
  const index_t globalSize = localSize * nWGPerRow * nWGPerCol;

  typename sb_handle_t::event_t ret;
  auto assignOp =
      make_ger_col(mA, _alpha, vx, vy, nWGPerRow, nWGPerCol, scratchPadSize);
  return sb_handle.execute(assignOp, localSize, globalSize, scratchPadSize);
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
template <typename sb_handle_t, typename index_t, typename element_t,
          typename container_t0, typename increment_t, typename container_t1>
typename sb_handle_t::event_t _syr_impl(
    sb_handle_t& sb_handle, char _Uplo, index_t _N, element_t _alpha,
    container_t0 _vx, increment_t _incx, container_t1 _mA, index_t _lda,
    index_t _localSize = 0, index_t _scratchPadSize = 0, index_t _nRowsWG = 0,
    index_t _nColsWG = 0) {
  typename sb_handle_t::event_t ret;
  _Uplo = tolower(_Uplo);
  int triangOpr = (_Uplo == 'u');
  index_t N = _N;
  auto mA = make_matrix_view<col_major>(_mA, N, N, _lda);
  auto vx = make_vector_view(_vx, _incx, N);

  const index_t localSize =
      (_localSize == 0) ? sb_handle.get_work_group_size() : _localSize;
  const index_t nRowsWG = (_nRowsWG == 0) ? localSize : std::min(N, _nRowsWG);
  const index_t nColsWG = (_nColsWG == 0) ? localSize : std::min(N, _nColsWG);
  const index_t scratchPadSize =
      (_localSize == 0) ? localSize : _scratchPadSize;

  const index_t nWGPerRow = (N - 1) / nRowsWG + 1;
  const index_t nWGPerCol = (N - 1) / nColsWG + 1;
  const index_t globalSize = localSize * nWGPerRow * nWGPerCol;

  if (triangOpr) {
    auto assignOp = make_ger_col<true, false, true, true>(
        mA, _alpha, vx, vx, nWGPerRow, nWGPerCol, scratchPadSize);
    return ret = concatenate_vectors(
               ret, sb_handle.execute(assignOp, localSize, globalSize,
                                      scratchPadSize));
  } else {
    auto assignOp = make_ger_col<true, true, true, false>(
        mA, _alpha, vx, vx, nWGPerRow, nWGPerCol, scratchPadSize);
    return ret = concatenate_vectors(
               ret, sb_handle.execute(assignOp, localSize, globalSize,
                                      scratchPadSize));
  }
}

/*! _SPR.
 * @brief Implementation of the rank 1 operation
 */
/*
sspr 	( 	character  	UPLO,
   integer  	N,
   real  	ALPHA,
   real, dimension(N)  	X,
   integer  	INCX,
   real, dimension(N, N + 1 / 2)  	AP
 )
*/
template <typename sb_handle_t, typename index_t, typename element_t,
          typename container_t0, typename increment_t, typename container_t1>
typename sb_handle_t::event_t _spr_impl(sb_handle_t& sb_handle, char _Uplo,
                                        index_t _N, element_t _alpha,
                                        container_t0 _vx, increment_t _incx,
                                        container_t1 _mPA) {
  // throw exception if invalid arguments
  if (_N <= 0) {
    throw std::invalid_argument("Invalid vector size");
  }

  // bail out early if alpha == 0
  if (_alpha == (element_t)0) {
    typename sb_handle_t::event_t event;
    return event;
  }

  typename sb_handle_t::event_t ret;
  _Uplo = tolower(_Uplo);
  const int Upper = _Uplo == 'u';
  auto mA = make_matrix_view<col_major>(_mPA, _N, (_N + 1) / 2, _N);
  auto vx = make_vector_view(_vx, _incx, _N);

  const index_t localSize = sb_handle.get_work_group_size();
  const index_t nColsWG = localSize;

  const index_t nWGPerCol = (_N * (_N + 1) / 2 - 1) / nColsWG + 1;
  const index_t globalSize = localSize * nWGPerCol;

  if (Upper) {
    auto spr = make_spr<true, true>(mA, _N, _alpha, vx, _incx, vx, _incx);
    return ret = concatenate_vectors(
               ret, sb_handle.execute(spr, localSize, globalSize));
  } else {
    auto spr = make_spr<true, false>(mA, _N, _alpha, vx, _incx, vx, _incx);
    return ret = concatenate_vectors(
               ret, sb_handle.execute(spr, localSize, globalSize));
  }
}

/*! _SPR2.
 * @brief Implementation of the rank 2 operation
 */
/*
sspr2 	( 	character  	UPLO,
   integer  	N,
   real  	ALPHA,
   real, dimension(N)  	X,
   integer  	INCX,
   real, dimension(N)  	Y,
   integer  	INCY,
   real, dimension(N, N + 1 / 2)  	AP
 )
*/
template <typename sb_handle_t, typename index_t, typename element_t,
          typename container_t0, typename increment_t, typename container_t1,
          typename container_t2>
typename sb_handle_t::event_t _spr2_impl(sb_handle_t& sb_handle, char _Uplo,
                                         index_t _N, element_t _alpha,
                                         container_t0 _vx, increment_t _incx,
                                         container_t1 _vy, increment_t _incy,
                                         container_t2 _mPA) {
  // throw exception if invalid arguments
  if (_N <= 0) {
    throw std::invalid_argument("Invalid vector size");
  }

  // bail out early if alpha == 0
  if (_alpha == (element_t)0) {
    typename sb_handle_t::event_t event;
    return event;
  }

  typename sb_handle_t::event_t ret;
  _Uplo = tolower(_Uplo);
  const int Upper = _Uplo == 'u';
  auto mA = make_matrix_view<col_major>(_mPA, _N, (_N + 1) / 2, _N);
  auto vx = make_vector_view(_vx, _incx, _N);
  auto vy = make_vector_view(_vy, _incy, _N);

  const index_t localSize = sb_handle.get_work_group_size();
  const index_t nColsWG = localSize;

  const index_t nWGPerCol = (_N * (_N + 1) / 2 - 1) / nColsWG + 1;
  const index_t globalSize = localSize * nWGPerCol;

  if (Upper) {
    auto spr2 = make_spr<false, true>(mA, _N, _alpha, vx, _incx, vy, _incy);
    return ret = concatenate_vectors(
               ret, sb_handle.execute(spr2, localSize, globalSize));
  } else {
    auto spr2 = make_spr<false, false>(mA, _N, _alpha, vx, _incx, vy, _incy);
    return ret = concatenate_vectors(
               ret, sb_handle.execute(spr2, localSize, globalSize));
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
template <typename sb_handle_t, typename index_t, typename element_t,
          typename container_t0, typename increment_t, typename container_t1,
          typename container_t2>
typename sb_handle_t::event_t _syr2_impl(
    sb_handle_t& sb_handle, char _Uplo, index_t _N, element_t _alpha,
    container_t0 _vx, increment_t _incx, container_t1 _vy, increment_t _incy,
    container_t2 _mA, index_t _lda, index_t _localSize = 0,
    index_t _scratchPadSize = 0, index_t _nRowsWG = 0, index_t _nColsWG = 0) {
  _Uplo = tolower(_Uplo);
  int triangOpr = (_Uplo == 'u');
  index_t N = _N;

  auto mA = make_matrix_view<col_major>(_mA, _N, _N, _lda);
  auto vx = make_vector_view(_vx, _incx, _N);
  auto vy = make_vector_view(_vy, _incy, _N);

  const index_t localSize =
      (_localSize == 0) ? sb_handle.get_work_group_size() : _localSize;
  const index_t nRowsWG = (_nRowsWG == 0) ? localSize : std::min(N, _nRowsWG);
  const index_t nColsWG = (_nColsWG == 0) ? localSize : std::min(N, _nColsWG);
  const index_t scratchPadSize =
      (_localSize == 0) ? 2 * localSize : _scratchPadSize;

  const index_t nWGPerRow = (N - 1) / nRowsWG + 1;
  const index_t nWGPerCol = (N - 1) / nColsWG + 1;
  const index_t globalSize = localSize * nWGPerRow * nWGPerCol;

  if (triangOpr) {
    auto assignOp = make_ger_col<false, false, true, true>(
        mA, _alpha, vx, vy, nWGPerRow, nWGPerCol, scratchPadSize);
    return sb_handle.execute(assignOp, localSize, globalSize, scratchPadSize);
  } else {
    auto assignOp = make_ger_col<false, true, true, false>(
        mA, _alpha, vx, vy, nWGPerRow, nWGPerCol, scratchPadSize);
    return sb_handle.execute(assignOp, localSize, globalSize, scratchPadSize);
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
template <typename sb_handle_t, typename index_t, typename element_t,
          typename container_t0, typename container_t1, typename increment_t,
          typename container_t2>
typename sb_handle_t::event_t inline _gemv(
    sb_handle_t& sb_handle,  // instance of SB_Handle
    char _trans,             // The transposition of the matrix ('n', 't', 'c')
    index_t _M,              // The size of dimension M of the matrix (rows)
    index_t _N,              // The size of dimension N of the matrix (columns)
    element_t _alpha,        // Scalar parameter Alpha
    container_t0 _mA,        // An array (LDA,N), with the first m*n elements
    index_t _lda,            // Specifies the first dimension of a, max(1, m)
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
             ? blas::gemv::backend::_gemv<transpose_type::Normal>(
                   sb_handle, _M, _N, _alpha, _mA, _lda, _vx, _incx, _beta, _vy,
                   _incy)
             : blas::gemv::backend::_gemv<transpose_type::Transposed>(
                   sb_handle, _M, _N, _alpha, _mA, _lda, _vx, _incx, _beta, _vy,
                   _incy);
}

template <typename sb_handle_t, typename index_t, typename container_t0,
          typename container_t1, typename increment_t>
typename sb_handle_t::event_t inline _trmv(sb_handle_t& sb_handle, char _Uplo,
                                           char _trans, char _Diag, index_t _N,
                                           container_t0 _mA, index_t _lda,
                                           container_t1 _vx,
                                           increment_t _incx) {
  // TODO: Here we can use some heuristics to select localn global, local, and
  // scratch size per device
  return tolower(_trans) == 'n'
             ? _trmv_impl<transpose_type::Normal>(sb_handle, _Uplo, _Diag, _N,
                                                  _mA, _lda, _vx, _incx)
             : _trmv_impl<transpose_type::Transposed>(
                   sb_handle, _Uplo, _Diag, _N, _mA, _lda, _vx, _incx);
}

#define INST_UPLO_TRANS_DIAG(func, ...)                           \
  if (tolower(_Uplo) == 'u') {                                    \
    if (tolower(_trans) == 'n') {                                 \
      if (tolower(_Diag) == 'n') {                                \
        return func<uplo_type::Upper, transpose_type::Normal,     \
                    diag_type::Nonunit>(__VA_ARGS__);             \
      } else {                                                    \
        return func<uplo_type::Upper, transpose_type::Normal,     \
                    diag_type::Unit>(__VA_ARGS__);                \
      }                                                           \
    } else {                                                      \
      if (tolower(_Diag) == 'n') {                                \
        return func<uplo_type::Upper, transpose_type::Transposed, \
                    diag_type::Nonunit>(__VA_ARGS__);             \
      } else {                                                    \
        return func<uplo_type::Upper, transpose_type::Transposed, \
                    diag_type::Unit>(__VA_ARGS__);                \
      }                                                           \
    }                                                             \
  } else {                                                        \
    if (tolower(_trans) == 'n') {                                 \
      if (tolower(_Diag) == 'n') {                                \
        return func<uplo_type::Lower, transpose_type::Normal,     \
                    diag_type::Nonunit>(__VA_ARGS__);             \
      } else {                                                    \
        return func<uplo_type::Lower, transpose_type::Normal,     \
                    diag_type::Unit>(__VA_ARGS__);                \
      }                                                           \
    } else {                                                      \
      if (tolower(_Diag) == 'n') {                                \
        return func<uplo_type::Lower, transpose_type::Transposed, \
                    diag_type::Nonunit>(__VA_ARGS__);             \
      } else {                                                    \
        return func<uplo_type::Lower, transpose_type::Transposed, \
                    diag_type::Unit>(__VA_ARGS__);                \
      }                                                           \
    }                                                             \
  }

template <typename sb_handle_t, typename index_t, typename container_t0,
          typename container_t1, typename increment_t>
typename sb_handle_t::event_t inline _trsv(sb_handle_t& sb_handle, char _Uplo,
                                           char _trans, char _Diag, index_t _N,
                                           container_t0 _mA, index_t _lda,
                                           container_t1 _vx,
                                           increment_t _incx) {
  INST_UPLO_TRANS_DIAG(blas::trsv::backend::_trsv, sb_handle, _N, _mA, _lda,
                       _vx, _incx)
}

template <typename sb_handle_t, typename index_t, typename element_t,
          typename container_t0, typename container_t1, typename increment_t,
          typename container_t2>
typename sb_handle_t::event_t inline _symv(sb_handle_t& sb_handle, char _Uplo,
                                           index_t _N, element_t _alpha,
                                           container_t0 _mA, index_t _lda,
                                           container_t1 _vx, increment_t _incx,
                                           element_t _beta, container_t2 _vy,
                                           increment_t _incy) {
  // TODO: Here we can use some heuristics to select localn global, local, and
  // scratch size per device
  return _symv_impl(sb_handle, _Uplo, _N, _alpha, _mA, _lda, _vx, _incx, _beta,
                    _vy, _incy);
}

template <typename sb_handle_t, typename index_t, typename element_t,
          typename container_t0, typename container_t1, typename increment_t,
          typename container_t2>
typename sb_handle_t::event_t inline _gbmv(sb_handle_t& sb_handle, char _trans,
                                           index_t _M, index_t _N, index_t _KL,
                                           index_t _KU, element_t _alpha,
                                           container_t0 _mA, index_t _lda,
                                           container_t1 _vx, increment_t _incx,
                                           element_t _beta, container_t2 _vy,
                                           increment_t _incy) {
  return tolower(_trans) == 'n'
             ? blas::gbmv::backend::_gbmv<transpose_type::Normal>(
                   sb_handle, _M, _N, _KL, _KU, _alpha, _mA, _lda, _vx, _incx,
                   _beta, _vy, _incy)
             : blas::gbmv::backend::_gbmv<transpose_type::Transposed>(
                   sb_handle, _M, _N, _KL, _KU, _alpha, _mA, _lda, _vx, _incx,
                   _beta, _vy, _incy);
}

template <typename sb_handle_t, typename index_t, typename element_t,
          typename container_t0, typename increment_t, typename container_t1,
          typename container_t2>
typename sb_handle_t::event_t inline _ger(sb_handle_t& sb_handle, index_t _M,
                                          index_t _N, element_t _alpha,
                                          container_t0 _vx, increment_t _incx,
                                          container_t1 _vy, increment_t _incy,
                                          container_t2 _mA, index_t _lda) {
  // TODO: Here we can use some heuristics to select localn global, local, and
  // scratch size per device
  return _ger_impl(sb_handle, _M, _N, _alpha, _vx, _incx, _vy, _incy, _mA,
                   _lda);
}

template <typename sb_handle_t, typename index_t, typename element_t,
          typename container_t0, typename container_t1, typename increment_t,
          typename container_t2>
typename sb_handle_t::event_t inline _sbmv(
    sb_handle_t& sb_handle, char _Uplo, index_t _N, index_t _K,
    element_t _alpha, container_t0 _mA, index_t _lda, container_t1 _vx,
    increment_t _incx, element_t _beta, container_t2 _vy, increment_t _incy) {
  return tolower(_Uplo) == 'u' ? blas::sbmv::backend::_sbmv<uplo_type::Upper>(
                                     sb_handle, _N, _K, _alpha, _mA, _lda, _vx,
                                     _incx, _beta, _vy, _incy)
                               : blas::sbmv::backend::_sbmv<uplo_type::Lower>(
                                     sb_handle, _N, _K, _alpha, _mA, _lda, _vx,
                                     _incx, _beta, _vy, _incy);
}

template <typename sb_handle_t, typename index_t, typename element_t,
          typename container_t0, typename container_t1, typename increment_t,
          typename container_t2>
typename sb_handle_t::event_t inline _spmv(sb_handle_t& sb_handle, char _Uplo,
                                           index_t _N, element_t _alpha,
                                           container_t0 _mA, container_t1 _vx,
                                           increment_t _incx, element_t _beta,
                                           container_t2 _vy,
                                           increment_t _incy) {
  return tolower(_Uplo) == 'u'
             ? blas::spmv::backend::_spmv<uplo_type::Upper>(
                   sb_handle, _N, _alpha, _mA, _vx, _incx, _beta, _vy, _incy)
             : blas::spmv::backend::_spmv<uplo_type::Lower>(
                   sb_handle, _N, _alpha, _mA, _vx, _incx, _beta, _vy, _incy);
}

template <typename sb_handle_t, typename index_t, typename element_t,
          typename container_t0, typename increment_t, typename container_t1>
typename sb_handle_t::event_t inline _syr(sb_handle_t& sb_handle, char _Uplo,
                                          index_t _N, element_t _alpha,
                                          container_t0 _vx, increment_t _incx,
                                          container_t1 _mA, index_t _lda) {
  // TODO: Here we can use some heuristics to select localn global, local, and
  // scratch size per device
  return _syr_impl(sb_handle, _Uplo, _N, _alpha, _vx, _incx, _mA, _lda);
}

template <typename sb_handle_t, typename index_t, typename element_t,
          typename container_t0, typename increment_t, typename container_t1>
typename sb_handle_t::event_t inline _spr(sb_handle_t& sb_handle, char _Uplo,
                                          index_t _N, element_t _alpha,
                                          container_t0 _vx, increment_t _incx,
                                          container_t1 _mPA) {
  return _spr_impl<sb_handle_t, index_t, element_t, container_t0, increment_t,
                   container_t1>(sb_handle, _Uplo, _N, _alpha, _vx, _incx,
                                 _mPA);
}

template <typename sb_handle_t, typename index_t, typename element_t,
          typename container_t0, typename increment_t, typename container_t1,
          typename container_t2>
typename sb_handle_t::event_t inline _spr2(sb_handle_t& sb_handle, char _Uplo,
                                           index_t _N, element_t _alpha,
                                           container_t0 _vx, increment_t _incx,
                                           container_t1 _vy, increment_t _incy,
                                           container_t2 _mPA) {
  return _spr2_impl<sb_handle_t, index_t, element_t, container_t0, increment_t,
                    container_t1, container_t2>(sb_handle, _Uplo, _N, _alpha,
                                                _vx, _incx, _vy, _incy, _mPA);
}

template <typename sb_handle_t, typename index_t, typename element_t,
          typename container_t0, typename increment_t, typename container_t1,
          typename container_t2>
typename sb_handle_t::event_t inline _syr2(sb_handle_t& sb_handle, char _Uplo,
                                           index_t _N, element_t _alpha,
                                           container_t0 _vx, increment_t _incx,
                                           container_t1 _vy, increment_t _incy,
                                           container_t2 _mA, index_t _lda) {
  // TODO: Here we can use some heuristics to select localn global, local, and
  // scratch size per device
  return _syr2_impl(sb_handle, _Uplo, _N, _alpha, _vx, _incx, _vy, _incy, _mA,
                    _lda);
}
template <typename sb_handle_t, typename index_t, typename container_t0,
          typename container_t1, typename increment_t>
typename sb_handle_t::event_t _tbmv(sb_handle_t& sb_handle, char _Uplo,
                                    char _trans, char _Diag, index_t _N,
                                    index_t _K, container_t0 _mA, index_t _lda,
                                    container_t1 _vx, increment_t _incx) {
  INST_UPLO_TRANS_DIAG(blas::tbmv::backend::_tbmv, sb_handle, _N, _K, _mA, _lda,
                       _vx, _incx)
}
template <typename sb_handle_t, typename index_t, typename container_t0,
          typename container_t1, typename increment_t>
typename sb_handle_t::event_t _tpmv(sb_handle_t& sb_handle, char _Uplo,
                                    char _trans, char _Diag, index_t _N,
                                    container_t0 _mA, container_t1 _vx,
                                    increment_t _incx) {
  INST_UPLO_TRANS_DIAG(blas::tpmv::backend::_tpmv, sb_handle, _N, _mA, _vx,
                       _incx)
}
template <typename sb_handle_t, typename index_t, typename container_t0,
          typename container_t1, typename increment_t>
typename sb_handle_t::event_t _tbsv(sb_handle_t& sb_handle, char _Uplo,
                                    char _trans, char _Diag, index_t _N,
                                    index_t _K, container_t0 _mA, index_t _lda,
                                    container_t1 _vx, increment_t _incx) {
  INST_UPLO_TRANS_DIAG(blas::tbsv::backend::_tbsv, sb_handle, _N, _K, _mA, _lda,
                       _vx, _incx)
}
template <typename sb_handle_t, typename index_t, typename container_t0,
          typename container_t1, typename increment_t>
typename sb_handle_t::event_t _tpsv(sb_handle_t& sb_handle, char _Uplo,
                                    char _trans, char _Diag, index_t _N,
                                    container_t0 _mA, container_t1 _vx,
                                    increment_t _incx) {
  INST_UPLO_TRANS_DIAG(blas::tpsv::backend::_tpsv, sb_handle, _N, _mA, _vx,
                       _incx)
}
}  // namespace internal
}  // namespace blas

#endif  // BLAS2_INTERFACE_HPP
