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
 *  @filename reduction_interface.hpp
 *
 **************************************************************************/

#ifndef SYCL_BLAS_EXTENSION_INTERFACE_HPP
#define SYCL_BLAS_EXTENSION_INTERFACE_HPP

#include "blas_meta.h"
#include "interface/extension/backend/backend.hpp"
#include "interface/transpose_launcher.h"
#include "operations/blas1_trees.h"
#include "operations/blas_operators.hpp"
#include "operations/extension/reduction.h"
#include "sb_handle/sycl_blas_handle.h"
#include "sycl_blas_helper.h"
#include "views/view.h"

namespace blas {
namespace extension {
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
 * @brief Implementation of matrix copy operators for transpose cases with out
 * of place memory copy.
 */
template <bool in_place, bool trans, typename sb_handle_t, typename element_t,
          typename index_t, typename in_t, typename out_t>
typename std::enable_if<trans && !in_place, typename sb_handle_t::event_t>::type
_matcopy_impl(sb_handle_t& sb_handle, index_t m, index_t n, element_t alpha,
              in_t in_memory, index_t ld_in, index_t inc_in, out_t out_memory,
              index_t ld_out, index_t inc_out) {
  return blas::extension::backend::_transpose_outplace<sb_handle_t, in_t, out_t,
                                                       element_t, index_t>(
      sb_handle, m, n, alpha, in_memory, ld_in, inc_in, out_memory, ld_out,
      inc_out);
}

template <bool in_place, bool trans, typename sb_handle_t, typename element_t,
          typename index_t, typename in_t, typename out_t>
typename std::enable_if<trans && in_place, typename sb_handle_t::event_t>::type
_matcopy_impl(sb_handle_t& sb_handle, index_t m, index_t n, element_t alpha,
              in_t in_memory, index_t ld_in, index_t inc_in, out_t out_memory,
              index_t ld_out, index_t inc_out) {
  // TODO
  typename sb_handle_t::event_t ret;
  return ret;
}

/**
 * @brief Implementation of matrix copy operators for non transpose cases.
 */
template <bool in_place, bool trans, typename sb_handle_t, typename element_t,
          typename index_t, typename in_t, typename out_t>
typename std::enable_if<!trans, typename sb_handle_t::event_t>::type
_matcopy_impl(sb_handle_t& sb_handle, index_t m, index_t n, element_t alpha,
              in_t in_memory, index_t ld_in, index_t inc_in, out_t out_memory,
              index_t ld_out, index_t inc_out) {
  typename sb_handle_t::event_t ret;
  // if alpha=1 no need to multiply
  if (alpha == 1) {
    auto in_view = make_matrix_view<col_major>(in_memory, m, n, ld_in, inc_in);
    auto out_view =
        make_matrix_view<col_major>(out_memory, m, n, ld_out, inc_out);
    auto copy_op = make_op<Assign>(out_view, in_view);
    ret = sb_handle.execute(copy_op);
  } else {
    auto in_view = make_matrix_view<col_major>(in_memory, m, n, ld_in, inc_in);
    auto out_view =
        make_matrix_view<col_major>(out_memory, m, n, ld_out, inc_out);
    auto scal_op = make_op<ScalarOp, ProductOperator>(alpha, in_view);
    auto copy_op = make_op<Assign>(out_view, scal_op);
    ret = sb_handle.execute(copy_op);
  }
  return ret;
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
          typename index_t, typename container_t>
typename std::enable_if<trans_a, typename sb_handle_t::event_t>::type
_omatadd_impl(sb_handle_t& sb_handle, index_t m, index_t n, element_t alpha,
              container_t a, index_t lda, element_t beta, container_t b,
              index_t ldb, container_t c, index_t ldc) {
  typename sb_handle_t::event_t ret;

  const index_t a_rows = trans_a ? n : m;
  const index_t a_cols = trans_a ? m : n;
  const index_t b_rows = trans_b ? n : m;
  const index_t b_cols = trans_b ? n : m;

  constexpr const bool both_trans = trans_a && trans_b;

  return blas::extension::backend::_transpose_add<both_trans>(
      sb_handle, m, n, alpha, a, lda, a_rows, a_cols, beta, b, ldb, b_rows,
      b_cols, c, ldc);
}

template <bool trans_a, bool trans_b, typename sb_handle_t, typename element_t,
          typename index_t, typename container_t>
typename std::enable_if<!trans_a && !trans_b,
                        typename sb_handle_t::event_t>::type
_omatadd_impl(sb_handle_t& sb_handle, index_t m, index_t n, element_t alpha,
              container_t a, index_t lda, element_t beta, container_t b,
              index_t ldb, container_t c, index_t ldc) {
  typename sb_handle_t::event_t ret;
  auto m_a_view = make_matrix_view<col_major>(a, m, n, lda);
  auto m_b_view = make_matrix_view<col_major>(b, m, n, ldb);
  auto m_c_view = make_matrix_view<col_major>(c, m, n, ldc);
  auto scal_a = make_op<ScalarOp, ProductOperator>(alpha, m_a_view);
  auto scal_b = make_op<ScalarOp, ProductOperator>(beta, m_b_view);
  auto sum_op = make_op<BinaryOp, AddOperator>(scal_a, scal_b);
  auto copy_op = make_op<Assign>(m_c_view, sum_op);
  ret = sb_handle.execute(copy_op);
  return ret;
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
    index_t rows, index_t cols) {
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
    reduction_event =
        concatenate_vectors(reduction_event, sb_handle.execute(reduction));

    /* 2nd step */
    auto reduction_step_2 =
        blas::make_reduction<typename get_second_step_op<operator_t>::type,
                             params_t>(temp_, matrix_buffer_out);
    reduction_event = concatenate_vectors(reduction_event,
                                          sb_handle.execute(reduction_step_2));
  } else {
    /* 1-step reduction */
    auto reduction = blas::make_reduction<operator_t, params_t>(
        matrix_buffer_in, matrix_buffer_out);
    reduction_event =
        concatenate_vectors(reduction_event, sb_handle.execute(reduction));
  }

  return reduction_event;
}

template <bool in_place, typename sb_handle_t, typename element_t,
          typename index_t, typename in_t, typename out_t>
typename sb_handle_t::event_t _matcopy(sb_handle_t& sb_handle, char trans,
                                       index_t m, index_t n, element_t alpha,
                                       in_t in_memory, index_t ld_in,
                                       index_t inc_in, out_t out_memory,
                                       index_t ld_out, index_t inc_out) {
  // bail out early if the leading dimensions are not correct
  if (ld_in < (inc_in * (m - 1) + 1) ||
      (ld_out - 1) < (trans == 't' ? inc_out * (n - 1) : inc_out * (m - 1))) {
    typename sb_handle_t::event_t ret;
    return ret;
  }

  if (trans == 't') {
    return _matcopy_impl<in_place, true>(sb_handle, m, n, alpha, in_memory,
                                         ld_in, inc_in, out_memory, ld_out,
                                         inc_out);
  } else {
    return _matcopy_impl<in_place, false>(sb_handle, m, n, alpha, in_memory,
                                          ld_in, inc_in, out_memory, ld_out,
                                          inc_out);
  }
}

template <typename sb_handle_t, typename element_t, typename index_t,
          typename container_t>
typename sb_handle_t::event_t _omatadd(sb_handle_t& sb_handle, char trans_a,
                                       char trans_b, index_t m, index_t n,
                                       element_t alpha, container_t a,
                                       index_t lda, element_t beta,
                                       container_t b, index_t ldb,
                                       container_t c, index_t ldc) {
  if (trans_a == 't') {
    if (trans_b == 't') {
      return _omatadd_impl<true, true>(sb_handle, m, n, alpha, a, lda, beta, b,
                                       ldb, c, ldc);
    } else {
      return _omatadd_impl<true, false>(sb_handle, m, n, alpha, a, lda, beta, b,
                                        ldb, c, ldc);
    }
  } else if (trans_b == 't') {
    // In this case, (alpha,a) & (beta,b) parameters positions are swapped as
    // the kernel implementation assumes the first input matrix is the
    // transposed one for simplicity purposes.
    return _omatadd_impl<true, false>(sb_handle, m, n, beta, b, ldb, alpha, a,
                                      lda, c, ldc);
  } else {
    return _omatadd_impl<false, false>(sb_handle, m, n, alpha, a, lda, beta, b,
                                       ldb, c, ldc);
  }
}

template <bool in_place, typename element_t, typename sb_handle_t,
          typename index_t, typename in_t, typename out_t>
typename sb_handle_t::event_t _transpose(sb_handle_t& sb_handle, index_t m,
                                         index_t n, in_t A, index_t ld_a,
                                         out_t B, index_t ld_b) {
  // bail out early if the leading dimensions are not correct
  if (ld_a < m || ld_b < n) {
    typename sb_handle_t::event_t ret;
    return ret;
  }

  const index_t inc = 1;
  const element_t alpha = element_t(1);

  return _matcopy_impl<in_place, true>(sb_handle, m, n, alpha, A, ld_a, inc, B,
                                       ld_b, inc);
}

template <typename operator_t, typename element_t, typename sb_handle_t,
          typename input_t, typename output_t, typename index_t>
typename sb_handle_t::event_t _reduction(sb_handle_t& sb_handle,
                                         input_t buffer_in, index_t ld,
                                         output_t buffer_out, index_t rows,
                                         index_t cols,
                                         reduction_dim_t reduction_dim) {
  if (reduction_dim == reduction_dim_t::inner) {
    return launch_type_based_reduction<operator_t, reduction_dim_t::inner,
                                       element_t>(sb_handle, buffer_in, ld,
                                                  buffer_out, rows, cols);
  } else {  // reduction_dim_t::outer
    return launch_type_based_reduction<operator_t, reduction_dim_t::outer,
                                       element_t>(sb_handle, buffer_in, ld,
                                                  buffer_out, rows, cols);
  }
}

}  // namespace internal
}  // namespace extension
}  // namespace blas

#endif  // SYCL_BLAS_EXTENSION_INTERFACE_HPP
