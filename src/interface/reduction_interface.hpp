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

#ifndef SYCL_BLAS_REDUCTION_INTERFACE_HPP
#define SYCL_BLAS_REDUCTION_INTERFACE_HPP

#include "blas_meta.h"
#include "executors/executor.h"
#include "interface/gemm_interface.hpp"
#include "operations/blas3_trees.h"
#include "policy/policy_handler.h"

namespace blas {
namespace extension {
namespace internal {

template <typename operator_t, typename element_t,
          typename executor_t, typename input_t, typename output_t,
          typename index_t>
typename executor_t::policy_t::event_t _reduction(executor_t& ex,
                                                  input_t buffer_in, index_t ld,
                                                  output_t buffer_out,
                                                  index_t rows, index_t cols) {
  constexpr int ClSize = 64;
  constexpr int WgSize = 256;

  using params_t =
      blas::ReductionRows_Params<index_t, element_t, ClSize, WgSize>;
  auto policy_handler = ex.get_policy_handler();

  const index_t num_compute_units = policy_handler.get_num_compute_units();

  /* Create an empty event vector */
  typename executor_t::policy_t::event_t reduction_event;

  const index_t max_group_count_col =
      (cols - 1) / params_t::work_group_cols + 1;

  const index_t group_count_cols =
      params_t::work_group_cols < max_group_count_col
          ? params_t::work_group_cols
          : max_group_count_col;

  /* Choose at run-time whether to do a one-step or two-step reduction.
   * Two-step reduction is needed when we have more than 1 valid work groups
   * along the columns */
  const bool two_step_reduction = group_count_cols > 1;

  auto matrix_buffer_in = make_matrix_view<col_major>(ex, buffer_in, rows, cols, ld);
  auto matrix_buffer_out = make_matrix_view<col_major>(ex, buffer_out, rows, 1, rows);
  /* 2-step reduction */
  if (two_step_reduction) {
    /* Create a temporary buffer */
    auto temp_buffer =
        make_sycl_iterator_buffer<element_t>(rows * group_count_cols);
    auto temp_ = make_matrix_view<col_major>(ex, temp_buffer, rows,
                                             group_count_cols, rows);

    /* 1st step */
    reduction_event.push_back(
        launch_row_reduction_step<operator_t, ClSize, WgSize, element_t>(
            policy_handler.get_queue(), matrix_buffer_in, temp_, group_count_cols,
            params_t::local_memory_size, num_compute_units));
    policy_handler.wait(reduction_event);

    /* 2nd step */
    reduction_event.push_back(
        launch_row_reduction_step<operator_t, ClSize, WgSize, element_t>(
            policy_handler.get_queue(), temp_, matrix_buffer_out, index_t(1),
            params_t::local_memory_size, num_compute_units));
    policy_handler.wait(reduction_event);
  }
  /* 1-step reduction */
  else {
    reduction_event.push_back(
        launch_row_reduction_step<operator_t, ClSize, WgSize, element_t>(
            policy_handler.get_queue(), matrix_buffer_in, matrix_buffer_out, index_t(1),
            params_t::local_memory_size, num_compute_units));
    policy_handler.wait(reduction_event);
  }

  return reduction_event;
}

}  // namespace internal
}  // namespace extension
}  // namespace blas

#endif  // SYCL_BLAS_REDUCTION_INTERFACE_HPP
