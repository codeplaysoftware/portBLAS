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
 *  @filename reduction_interface.hpp
 *
 **************************************************************************/

#ifndef PORTBLAS_EXTENSION_INTERFACE_HPP
#define PORTBLAS_EXTENSION_INTERFACE_HPP

#include "blas_meta.h"
#include "interface/extension/backend/backend.hpp"
#include "interface/extension_interface.h"
#include "operations/blas1_trees.h"
#include "operations/blas_operators.hpp"
#include "operations/extension/axpy_batch.h"
#include "operations/extension/matcopy_batch.h"
#include "operations/extension/reduction.h"
#include "operations/extension/transpose.h"
#include "portblas_helper.h"
#include "sb_handle/portblas_handle.h"
#include "views/view.h"

namespace blas {
namespace internal {

template <typename operator_t>
struct get_second_step_op {
  using type = operator_t;
};

template <>
struct get_second_step_op<MeanOperator> {
  using type = AddOperator;
};

/**
 * @brief Wrapping implementation of outplace transpose kernel.
 */
template <int Tile_size, int wg_size, int cl_size, bool local_memory,
          typename sb_handle_t, typename container_0_t, typename container_1_t,
          typename element_t, typename index_t>
typename sb_handle_t::event_t _transpose_outplace_impl(
    sb_handle_t& sb_handle, index_t _M, index_t _N, element_t _alpha,
    container_0_t in_, index_t _ld_in, index_t _inc_in, index_t _stride_in,
    container_1_t out_, index_t _ld_out, index_t _inc_out, index_t _stride_out,
    index_t _batch_size, const typename sb_handle_t::event_t& _dependencies) {
  constexpr const index_t num_line_elems =
      std::max(Tile_size, static_cast<int>(cl_size / sizeof(element_t)));
  constexpr const index_t num_tiles_per_line = num_line_elems / Tile_size;

  // Matrix Views
  auto in_view = make_matrix_view<col_major>(in_, _M, _N, _ld_in);
  auto out_view = make_matrix_view<col_major>(out_, _M, _N, _ld_out);

  // Work items & groups sizes
  index_t n_wg = ((_M - 1) / Tile_size + 1) * ((_N - 1) / Tile_size + 1);
  index_t global_size = n_wg * wg_size * _batch_size;

  // Transpose expression Tree
  auto trans_scale_tree =
      make_transpose<false, Tile_size, wg_size, cl_size, local_memory>(
          in_view, _inc_in, _stride_in, out_view, _inc_out, _stride_out, _alpha,
          _batch_size);

  if constexpr (local_memory) {
    index_t local_mem = static_cast<index_t>((num_line_elems + 1) * Tile_size /
                                             num_tiles_per_line);
    return sb_handle.execute(trans_scale_tree, static_cast<index_t>(wg_size),
                             global_size, local_mem, _dependencies);
  } else {
    return sb_handle.execute(trans_scale_tree, static_cast<index_t>(wg_size),
                             global_size, _dependencies);
  }
}

/**
 * @brief Implementation of matrix copy operators for transpose cases.
 */
template <bool in_place, bool trans, typename sb_handle_t, typename element_t,
          typename index_t, typename in_t, typename out_t>
typename std::enable_if<trans, typename sb_handle_t::event_t>::type
_matcopy_impl(sb_handle_t& sb_handle, index_t m, index_t n, element_t alpha,
              in_t in_memory, index_t ld_in, index_t inc_in, index_t stride_in,
              out_t out_memory, index_t ld_out, index_t inc_out,
              index_t stride_out, index_t batch_size,
              const typename sb_handle_t::event_t& _dependencies) {
  if constexpr (!in_place) {
    return blas::transpose::backend::_transpose_outplace<
        sb_handle_t, in_t, out_t, element_t, index_t>(
        sb_handle, m, n, alpha, in_memory, ld_in, inc_in, stride_in, out_memory,
        ld_out, inc_out, stride_out, batch_size, _dependencies);

  } else {
    // TODO
    // In-place transpose not implemented.
    throw std::runtime_error("In-place transpose not implemented.");
  }
}

/**
 * @brief Implementation of matrix copy operators for non transpose cases.
 */
template <bool in_place, bool trans, bool in_has_inc, bool out_has_inc,
          typename sb_handle_t, typename element_t, typename index_t,
          typename in_t, typename out_t>
typename std::enable_if<!trans, typename sb_handle_t::event_t>::type
_matcopy_impl(sb_handle_t& sb_handle, index_t m, index_t n, element_t alpha,
              in_t in_memory, index_t ld_in, index_t inc_in, out_t out_memory,
              index_t ld_out, index_t inc_out,
              const typename sb_handle_t::event_t& _dependencies) {
  typename sb_handle_t::event_t ret;
  typename MatrixViewType<in_t, index_t, col_major, in_has_inc>::type in_view =
      make_matrix_view<col_major, element_t, index_t, in_has_inc>(
          in_memory, m, n, ld_in, inc_in);
  typename MatrixViewType<out_t, index_t, col_major, out_has_inc>::type
      out_view = make_matrix_view<col_major, element_t, index_t, out_has_inc>(
          out_memory, m, n, ld_out, inc_out);
  // if alpha=1 no need to multiply
  if (alpha == 1) {
    auto copy_op = make_op<Assign>(out_view, in_view);
    ret = sb_handle.execute(copy_op, _dependencies);
  } else {
    auto scal_op = make_op<ScalarOp, ProductOperator>(alpha, in_view);
    auto copy_op = make_op<Assign>(out_view, scal_op);
    ret = sb_handle.execute(copy_op, _dependencies);
  }
  return ret;
}

template <bool in_place, bool trans, typename sb_handle_t, typename element_t,
          typename index_t, typename in_t, typename out_t>
typename std::enable_if<!trans, typename sb_handle_t::event_t>::type
_matcopy_impl(sb_handle_t& sb_handle, index_t m, index_t n, element_t alpha,
              in_t in_memory, index_t ld_in, index_t inc_in, index_t stride_in,
              out_t out_memory, index_t ld_out, index_t inc_out,
              index_t stride_out, index_t batch_size,
              const typename sb_handle_t::event_t& _dependencies) {
  // if alpha=1 no need to multiply
  if (inc_in == 1 && inc_out == 1) {
    return _matcopy_impl<in_place, trans, false, false>(
        sb_handle, m, n, alpha, in_memory, ld_in, inc_in, out_memory, ld_out,
        inc_out, _dependencies);
  } else if (inc_in == 1) {
    return _matcopy_impl<in_place, trans, false, true>(
        sb_handle, m, n, alpha, in_memory, ld_in, inc_in, out_memory, ld_out,
        inc_out, _dependencies);
  } else if (inc_out == 1) {
    return _matcopy_impl<in_place, trans, true, false>(
        sb_handle, m, n, alpha, in_memory, ld_in, inc_in, out_memory, ld_out,
        inc_out, _dependencies);
  } else {
    return _matcopy_impl<in_place, trans, true, true>(
        sb_handle, m, n, alpha, in_memory, ld_in, inc_in, out_memory, ld_out,
        inc_out, _dependencies);
  }
}

/**
 * @brief Implementation of matrix copy batch operators for non transpose cases.
 */
template <uint32_t TileSize, int TilePerWG, typename sb_handle_t,
          typename element_t, typename index_t, typename in_t, typename out_t>
typename sb_handle_t::event_t _matcopy_batch_impl(
    sb_handle_t& sb_handle, index_t m, index_t n, element_t alpha,
    in_t in_memory, index_t ld_in, index_t in_stride, out_t out_memory,
    index_t ld_out, index_t out_stride, index_t batch_size,
    const typename sb_handle_t::event_t& _dependencies) {
  typename MatrixViewType<in_t, index_t, col_major>::type in_view =
      make_matrix_view<col_major>(in_memory, m, n, ld_in);
  auto out_view = make_matrix_view<col_major>(out_memory, m, n, ld_out);
  const element_t beta = 0;
  const index_t ld_b = 0;
  const index_t stride_b = 0;
  auto copy_batch_tree = make_matcopy_batch<false, TileSize, TilePerWG>(
      out_view, in_view, in_view, alpha, beta, m, n, ld_out, ld_in, ld_b,
      out_stride, in_stride, stride_b, batch_size);
  constexpr index_t local_size = TileSize * TilePerWG;
  const index_t tile_per_matrix =
      (((m - 1) / TileSize) + 1) * (((n - 1) / TileSize) + 1);
  const index_t wg_size = (tile_per_matrix - 1) / TilePerWG + 1;
  const index_t global_size = (wg_size)*local_size * batch_size;
  return sb_handle.execute(copy_batch_tree, local_size, global_size,
                           _dependencies);
}

/*!
 * @brief Wrapper around Transpose-Add. Creates the views, then makes and
 * launches Transpose Add kernel
 */
template <bool both_trans, int Tile_size, int wg_size, int cl_size,
          bool local_memory, typename sb_handle_t, typename container_0_t,
          typename container_1_t, typename container_2_t, typename element_t,
          typename index_t>
typename sb_handle_t::event_t _transpose_add_impl(
    sb_handle_t& sb_handle, index_t _M, index_t _N, element_t _alpha,
    container_0_t a_, index_t _lda, index_t _nrows_a, index_t _ncols_a,
    index_t _stride_a, element_t _beta, container_1_t b_, index_t _ldb,
    index_t _nrows_b, index_t _ncols_b, index_t _stride_b, container_2_t c_,
    index_t _ldc, index_t _stride_c, index_t _batch_size,
    const typename sb_handle_t::event_t& _dependencies) {
  constexpr const index_t num_line_elems =
      std::max(Tile_size, static_cast<int>(cl_size / sizeof(element_t)));
  constexpr const index_t num_tiles_per_line = num_line_elems / Tile_size;
  // Matrix Views
  typename MatrixViewType<container_0_t, index_t, col_major>::type A_view =
      make_matrix_view<col_major>(a_, _nrows_a, _ncols_a, _lda);
  typename MatrixViewType<container_1_t, index_t, col_major>::type B_view =
      make_matrix_view<col_major>(b_, _nrows_b, _ncols_b, _ldb);

  auto C_view = make_matrix_view<col_major>(c_, _M, _N, _ldc);

  // Work items & groups sizes
  index_t n_wg = ((_M - 1) / Tile_size + 1) * ((_N - 1) / Tile_size + 1);
  index_t global_size = n_wg * wg_size * _batch_size;

  // Transpose Add expression Tree
  auto trans_scale_tree =
      make_transpose_add<both_trans, Tile_size, wg_size, cl_size, local_memory>(
          A_view, _stride_a, B_view, _stride_b, C_view, _stride_c, _alpha,
          _beta, _batch_size);

  if constexpr (local_memory) {
    index_t local_mem = static_cast<index_t>((num_line_elems + 1) * Tile_size /
                                             num_tiles_per_line);
    return sb_handle.execute(trans_scale_tree, static_cast<index_t>(wg_size),
                             global_size, local_mem, _dependencies);
  } else {
    return sb_handle.execute(trans_scale_tree, static_cast<index_t>(wg_size),
                             global_size, _dependencies);
  }
}

/*!
 * @brief _omatadd_impl in the (trans_a || trans_b) case : This specialization
 * covers the following 3 cases :
 *  - A transposed & B transposed
 *  - A transposed & B not transposed
 *  - A not transposed & B transposed
 *
 * For convenience purposes, these 3 cases can be brought down to 2 cases, where
 * 1. either both matrices are transposed OR 2. only the 'first' matrix is
 * transposed. Thus, this function assumes that if only one matrix is
 * transposed, it should be the matrix a (trans_a == true).
 *
 */
template <bool trans_a, bool trans_b, typename sb_handle_t, typename element_t,
          typename index_t, typename container_0_t, typename container_1_t,
          typename container_2_t>
typename std::enable_if<trans_a, typename sb_handle_t::event_t>::type
_omatadd_impl(sb_handle_t& sb_handle, index_t m, index_t n, element_t alpha,
              container_0_t a, index_t lda, index_t stride_a, element_t beta,
              container_1_t b, index_t ldb, index_t stride_b, container_2_t c,
              index_t ldc, index_t stride_c, index_t batch_size,
              const typename sb_handle_t::event_t& _dependencies) {
  const index_t a_rows = trans_a ? n : m;
  const index_t a_cols = trans_a ? m : n;
  const index_t b_rows = trans_b ? n : m;
  const index_t b_cols = trans_b ? n : m;

  constexpr const bool both_trans = trans_a && trans_b;

  return blas::transpose::backend::_transpose_add<both_trans>(
      sb_handle, m, n, alpha, a, lda, a_rows, a_cols, stride_a, beta, b, ldb,
      b_rows, b_cols, stride_b, c, ldc, stride_c, batch_size, _dependencies);
}

/*!
 * @brief _omatadd_impl in case of non-transpose matrix
 */
template <bool trans_a, bool trans_b, typename sb_handle_t, typename element_t,
          typename index_t, typename container_0_t, typename container_1_t,
          typename container_2_t>
typename std::enable_if<!trans_a && !trans_b,
                        typename sb_handle_t::event_t>::type
_omatadd_impl(sb_handle_t& sb_handle, index_t m, index_t n, element_t alpha,
              container_0_t a, index_t lda, index_t stride_a, element_t beta,
              container_1_t b, index_t ldb, index_t stride_b, container_2_t c,
              index_t ldc, index_t stride_c, index_t batch_size,
              const typename sb_handle_t::event_t& _dependencies) {
  // This implementation of omatadd must be used only for non batched version
  // of the operator. For this reason is not needed to check for batch size.
  // The batched implementation is completely different.
  typename sb_handle_t::event_t ret;
  typename MatrixViewType<container_0_t, index_t, col_major>::type m_a_view =
      make_matrix_view<col_major>(a, m, n, lda);
  typename MatrixViewType<container_1_t, index_t, col_major>::type m_b_view =
      make_matrix_view<col_major>(b, m, n, ldb);
  auto m_c_view = make_matrix_view<col_major>(c, m, n, ldc);
  auto scal_a = make_op<ScalarOp, ProductOperator>(alpha, m_a_view);
  auto scal_b = make_op<ScalarOp, ProductOperator>(beta, m_b_view);
  auto sum_op = make_op<BinaryOp, AddOperator>(scal_a, scal_b);
  auto copy_op = make_op<Assign>(m_c_view, sum_op);
  ret = sb_handle.execute(copy_op, _dependencies);
  return ret;
}

template <uint32_t TileSize, int TilePerWG, typename sb_handle_t,
          typename element_t, typename index_t, typename container_0_t,
          typename container_1_t, typename container_2_t>
typename sb_handle_t::event_t _omatadd_batch_impl(
    sb_handle_t& sb_handle, index_t m, index_t n, element_t alpha,
    container_0_t a, index_t lda, index_t stride_a, element_t beta,
    container_1_t b, index_t ldb, index_t stride_b, container_2_t c,
    index_t ldc, index_t stride_c, index_t batch_size,
    const typename sb_handle_t::event_t& _dependencies) {
  auto m_a_view = make_matrix_view<col_major>(a, m, n, lda);
  auto m_b_view = make_matrix_view<col_major>(b, m, n, ldb);
  auto m_c_view = make_matrix_view<col_major>(c, m, n, ldc);
  auto copy_batch_tree = make_matcopy_batch<true, TileSize, TilePerWG>(
      m_c_view, m_a_view, m_b_view, alpha, beta, m, n, ldc, lda, ldb, stride_c,
      stride_a, stride_b, batch_size);
  constexpr index_t local_size = TileSize * TilePerWG;
  const index_t tile_per_matrix =
      (((m - 1) / TileSize) + 1) * (((n - 1) / TileSize) + 1);
  const index_t wg_size = (tile_per_matrix - 1) / TilePerWG + 1;
  const index_t global_size = (wg_size)*local_size * batch_size;
  return sb_handle.execute(copy_batch_tree, local_size, global_size,
                           _dependencies);
}

/*!
 * @brief Wrapper around Reduction. Creates the views, then makes and launches
 * the Reduction kernel
 */
template <typename operator_t, reduction_dim_t reduction_dim,
          typename element_t, typename sb_handle_t, typename input_t,
          typename output_t, typename index_t>
typename sb_handle_t::event_t launch_type_based_reduction(
    sb_handle_t& sb_handle, input_t buffer_in, index_t ld, output_t buffer_out,
    index_t rows, index_t cols,
    const typename SB_Handle::event_t& dependencies) {
#ifdef POWER_VR
  constexpr int ClSize = 32;
  constexpr int WgSize = 64;
#else
  constexpr int ClSize = 64;
  constexpr int WgSize = 256;
#endif
  constexpr index_t reductions_per_thread = 64;

  using params_t = blas::ReductionParams<index_t, element_t, ClSize, WgSize,
                                         reductions_per_thread,
                                         static_cast<int>(reduction_dim)>;

  const auto reduced_group_count =
      params_t::calculate_reduced_group_count(rows, cols);

  /* Create an empty event vector */
  typename sb_handle_t::event_t reduction_event;

  auto matrix_buffer_in =
      make_matrix_view<col_major>(buffer_in, rows, cols, ld);
  const index_t out_rows =
      reduction_dim == reduction_dim_t::outer ? rows : index_t(1);
  const index_t out_cols =
      reduction_dim == reduction_dim_t::outer ? index_t(1) : cols;
  auto matrix_buffer_out =
      make_matrix_view<col_major>(buffer_out, out_rows, out_cols, out_rows);

  const bool two_step_reduction = reduced_group_count > 1;
  /* 2-step reduction */
  if (two_step_reduction) {
    /* Create a temporary buffer */
    auto temp_buffer = make_sycl_iterator_buffer<element_t>(
        (reduction_dim == reduction_dim_t::outer ? rows : cols) *
        reduced_group_count);

    const index_t temp_rows =
        reduction_dim == reduction_dim_t::outer ? rows : reduced_group_count;
    const index_t temp_cols =
        reduction_dim == reduction_dim_t::outer ? reduced_group_count : cols;
    auto temp_ = make_matrix_view<col_major>(temp_buffer, temp_rows, temp_cols,
                                             temp_rows);

    /* 1st step */
    auto reduction =
        blas::make_reduction<operator_t, params_t>(matrix_buffer_in, temp_);
    reduction_event = concatenate_vectors(
        reduction_event, sb_handle.execute(reduction, dependencies));

    /* 2nd step */
    auto reduction_step_2 =
        blas::make_reduction<typename get_second_step_op<operator_t>::type,
                             params_t>(temp_, matrix_buffer_out);
    reduction_event = concatenate_vectors(
        reduction_event, sb_handle.execute(reduction_step_2, reduction_event));
  } else {
    /* 1-step reduction */
    auto reduction = blas::make_reduction<operator_t, params_t>(
        matrix_buffer_in, matrix_buffer_out);
    reduction_event = concatenate_vectors(
        reduction_event, sb_handle.execute(reduction, dependencies));
  }

  return reduction_event;
}

template <bool in_place, typename sb_handle_t, typename element_t,
          typename index_t, typename in_t, typename out_t>
typename sb_handle_t::event_t _matcopy(
    sb_handle_t& sb_handle, char trans, index_t m, index_t n, element_t alpha,
    in_t in_memory, index_t ld_in, index_t inc_in, out_t out_memory,
    index_t ld_out, index_t inc_out,
    const typename sb_handle_t::event_t& _dependencies) {
  // bail out early if the leading dimensions are not correct
  if (ld_in < (inc_in * (m - 1) + 1) ||
      (ld_out - 1) < (trans == 't' ? inc_out * (n - 1) : inc_out * (m - 1))) {
    throw std::invalid_argument("invalid ld_in and/or ld_out, inc_out, inc_in");
  }

  const index_t stride = 1;
  const index_t batch_size = 1;

  if (trans == 't') {
    return _matcopy_impl<in_place, true>(
        sb_handle, m, n, alpha, in_memory, ld_in, inc_in, stride, out_memory,
        ld_out, inc_out, stride, index_t(1), _dependencies);
  } else {
    return _matcopy_impl<in_place, false>(
        sb_handle, m, n, alpha, in_memory, ld_in, inc_in, stride, out_memory,
        ld_out, inc_out, stride, batch_size, _dependencies);
  }
}

template <bool in_place, typename sb_handle_t, typename element_t,
          typename index_t, typename in_t, typename out_t>
typename sb_handle_t::event_t _matcopy_batch(
    sb_handle_t& sb_handle, char trans, index_t m, index_t n, element_t alpha,
    in_t in_memory, index_t ld_in, index_t stride_in, out_t out_memory,
    index_t ld_out, index_t stride_out, index_t batch_size,
    const typename sb_handle_t::event_t& _dependencies) {
  // bail out early if the leading dimensions / strides are not correct
  if (ld_in < m || (ld_out < (trans == 't' ? n : m))) {
    throw std::invalid_argument("invalid ld_in and/or ld_out");
  }
  if ((stride_in < ld_in * n) ||
      (stride_out < (ld_out * (trans == 't' ? m : n)))) {
    throw std::invalid_argument("invalid stride_in and/or stride_out");
  }

  const index_t increment = 1;

  if (trans == 't') {
    return _matcopy_impl<in_place, true>(
        sb_handle, m, n, alpha, in_memory, ld_in, increment, stride_in,
        out_memory, ld_out, increment, stride_out, batch_size, _dependencies);
  } else {
    return blas::matcopy_batch::backend::_matcopy_batch<false>(
        sb_handle, m, n, alpha, in_memory, ld_in, stride_in, out_memory, ld_out,
        stride_out, batch_size, _dependencies);
  }
}

template <typename sb_handle_t, typename element_t, typename index_t,
          typename container_0_t, typename container_1_t,
          typename container_2_t>
typename sb_handle_t::event_t _omatadd(
    sb_handle_t& sb_handle, char trans_a, char trans_b, index_t m, index_t n,
    element_t alpha, container_0_t a, index_t lda, element_t beta,
    container_1_t b, index_t ldb, container_2_t c, index_t ldc,
    const typename sb_handle_t::event_t& _dependencies) {
  // Bail out early if the leading dimensions are not correct
  if (ldc < m) {
    throw std::invalid_argument("Invalid ldc");
  } else if (lda < (trans_a == 't' ? n : m)) {
    throw std::invalid_argument("Invalid lda");
  } else if (ldb < (trans_b == 't' ? n : m)) {
    throw std::invalid_argument("Invalid ldb");
  }

  // Stride = 0 as a dummy value as it is not used when batch_size == 1
  const index_t stride_a = 0;
  const index_t stride_b = 0;
  const index_t stride_c = 0;
  const index_t batch_size = 1;

  if (trans_a == 't') {
    if (trans_b == 't') {
      return _omatadd_impl<true, true>(sb_handle, m, n, alpha, a, lda, stride_a,
                                       beta, b, ldb, stride_b, c, ldc, stride_c,
                                       batch_size, _dependencies);
    } else {
      return _omatadd_impl<true, false>(
          sb_handle, m, n, alpha, a, lda, stride_a, beta, b, ldb, stride_b, c,
          ldc, stride_c, batch_size, _dependencies);
    }
  } else if (trans_b == 't') {
    // In this case, (alpha,a) & (beta,b) parameters positions are swapped as
    // the kernel implementation assumes the first input matrix is the
    // transposed one for simplicity purposes.
    return _omatadd_impl<true, false>(sb_handle, m, n, beta, b, ldb, stride_b,
                                      alpha, a, lda, stride_a, c, ldc, stride_c,
                                      batch_size, _dependencies);
  } else {
    return _omatadd_impl<false, false>(sb_handle, m, n, alpha, a, lda, stride_a,
                                       beta, b, ldb, stride_b, c, ldc, stride_c,
                                       static_cast<index_t>(1), _dependencies);
  }
}

template <typename sb_handle_t, typename element_t, typename index_t,
          typename container_0_t, typename container_1_t,
          typename container_2_t>
typename sb_handle_t::event_t _omatadd_batch(
    sb_handle_t& sb_handle, char trans_a, char trans_b, index_t m, index_t n,
    element_t alpha, container_0_t a, index_t lda, index_t stride_a,
    element_t beta, container_1_t b, index_t ldb, index_t stride_b,
    container_2_t c, index_t ldc, index_t stride_c, index_t batch_size,
    const typename sb_handle_t::event_t& _dependencies) {
  // Bail out early if the leading dimensions are not correct
  if (ldc < m) {
    throw std::invalid_argument("Invalid ldc");
  } else if (lda < (trans_a == 't' ? n : m)) {
    throw std::invalid_argument("Invalid lda");
  } else if (ldb < (trans_b == 't' ? n : m)) {
    throw std::invalid_argument("Invalid ldb");
  }

  if (trans_a == 't') {
    if (trans_b == 't') {
      return _omatadd_impl<true, true>(sb_handle, m, n, alpha, a, lda, stride_a,
                                       beta, b, ldb, stride_b, c, ldc, stride_c,
                                       batch_size, _dependencies);
    } else {
      return _omatadd_impl<true, false>(
          sb_handle, m, n, alpha, a, lda, stride_a, beta, b, ldb, stride_b, c,
          ldc, stride_c, batch_size, _dependencies);
    }
  } else if (trans_b == 't') {
    // In this case, (alpha,a) & (beta,b) parameters positions are swapped as
    // the kernel implementation assumes the first input matrix is the
    // transposed one for simplicity purposes.
    return _omatadd_impl<true, false>(sb_handle, m, n, beta, b, ldb, stride_b,
                                      alpha, a, lda, stride_a, c, ldc, stride_c,
                                      batch_size, _dependencies);
  } else {
    return blas::omatadd_batch::backend::_omatadd_batch(
        sb_handle, m, n, alpha, a, lda, stride_a, beta, b, ldb, stride_b, c,
        ldc, stride_c, batch_size, _dependencies);
  }
}

template <bool in_place, typename element_t, typename sb_handle_t,
          typename index_t, typename in_t, typename out_t>
typename sb_handle_t::event_t _transpose(
    sb_handle_t& sb_handle, index_t m, index_t n, in_t A, index_t ld_a, out_t B,
    index_t ld_b, const typename sb_handle_t::event_t& _dependencies) {
  // bail out early if the leading dimensions are not correct
  if (ld_a < m) {
    throw std::invalid_argument("Invalid lda");
  } else if (ld_b < n) {
    throw std::invalid_argument("Invalid ldb");
  }

  const element_t alpha = 1;
  const index_t inc = 1;
  const index_t stride = 1;
  const index_t batch_size = 1;

  return _matcopy_impl<in_place, true>(sb_handle, m, n, alpha, A, ld_a, inc,
                                       stride, B, ld_b, inc, stride, batch_size,
                                       _dependencies);
}

template <typename operator_t, typename element_t, typename sb_handle_t,
          typename input_t, typename output_t, typename index_t>
typename sb_handle_t::event_t _reduction(
    sb_handle_t& sb_handle, input_t buffer_in, index_t ld, output_t buffer_out,
    index_t rows, index_t cols, reduction_dim_t reduction_dim,
    const typename sb_handle_t::event_t& dependencies) {
  if (reduction_dim == reduction_dim_t::inner) {
    return launch_type_based_reduction<operator_t, reduction_dim_t::inner,
                                       element_t>(
        sb_handle, buffer_in, ld, buffer_out, rows, cols, dependencies);
  } else {  // reduction_dim_t::outer
    return launch_type_based_reduction<operator_t, reduction_dim_t::outer,
                                       element_t>(
        sb_handle, buffer_in, ld, buffer_out, rows, cols, dependencies);
  }
}

template <typename sb_handle_t, typename container_0_t, typename container_1_t,
          typename element_t, typename index_t>
typename sb_handle_t::event_t _axpy_batch(
    sb_handle_t& sb_handle, index_t _N, element_t _alpha, container_0_t _vx,
    index_t _incx, index_t _stride_x, container_1_t _vy, index_t _incy,
    index_t _stride_y, index_t _batch_size,
    const typename sb_handle_t::event_t& _dependencies) {
  // if inc are of opposite sign the values are exchanged. It doesn't matter
  // which one is positive or negative, so to simplify index computation in
  // kernel we always set incx to be negative and incy to be positive.
  if (_incx > 0 && _incy < 0) {
    _incx = -_incx;
    _incy = -_incy;
  }
  typename VectorViewType<container_0_t, index_t, index_t>::type vx =
      make_vector_view(_vx, static_cast<index_t>(_incx),
                       static_cast<index_t>(_N * _batch_size));
  auto vy = make_vector_view(_vy, _incy, _N * _batch_size);
  const auto local_size = sb_handle.get_work_group_size();
  const auto nWG = (_N + local_size - 1) / local_size;
  const auto global_size = local_size * nWG * _batch_size;
  // If both vectors are read from the same side it doesn't matter the sign of
  // the increment
  if (_incx * _incy > 0) {
    auto op =
        make_axpy_batch<true>(vy, vx, _alpha, _N, std::abs(_incy), _stride_y,
                              std::abs(_incx), _stride_x, _batch_size);
    typename sb_handle_t::event_t ret =
        sb_handle.execute(op, local_size, global_size, _dependencies);
    return ret;
  } else {
    auto op = make_axpy_batch<false>(vy, vx, _alpha, _N, _incy, _stride_y,
                                     _incx, _stride_x, _batch_size);
    typename sb_handle_t::event_t ret =
        sb_handle.execute(op, local_size, global_size, _dependencies);
    return ret;
  }
}

}  // namespace internal
}  // namespace blas

#endif  // PORTBLAS_EXTENSION_INTERFACE_HPP
