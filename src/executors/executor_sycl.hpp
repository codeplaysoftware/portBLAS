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
 *  @filename executor_sycl.hpp
 *
 **************************************************************************/

#ifndef SYCL_BLAS_EXECUTOR_SYCL_HPP
#define SYCL_BLAS_EXECUTOR_SYCL_HPP

#include <algorithm>

#include "blas_meta.h"
#include "executors/executor.h"
#include "executors/kernel_constructor.h"
#include "operations/blas1_trees.hpp"
#include "operations/blas2_trees.hpp"
#include "operations/blas_operators.hpp"
#include "policy/sycl_policy_handler.h"
#include "views/view.h"

namespace blas {
/*! Executor<PolicyHandler<codeplay_policy>>.
 * @brief Executes an Expression expression_tree_t using SYCL.
 */
template class Executor<PolicyHandler<codeplay_policy>>;

/*!
 * @brief Executes the tree without defining required shared memory.
 */
template <>
template <typename expression_tree_t>
inline typename codeplay_policy::event_t
Executor<PolicyHandler<codeplay_policy>>::execute(expression_tree_t t) {
  const auto localSize = policy_handler_.get_work_group_size();
  auto _N = t.get_size();
  auto nWG = (_N + localSize - 1) / localSize;
  auto globalSize = nWG * localSize;

  return {execute_tree<using_local_memory::disabled>(
      policy_handler_.get_queue(), t, localSize, globalSize, 0)};
};

/*!
 * @brief Executes the tree fixing the localSize but without defining
 * required shared memory.
 */
template <>
template <typename expression_tree_t, typename index_t>
inline typename codeplay_policy::event_t
Executor<PolicyHandler<codeplay_policy>>::execute(expression_tree_t t,
                                                  index_t localSize) {
  auto _N = t.get_size();
  auto nWG = (_N + localSize - 1) / localSize;
  auto globalSize = nWG * localSize;
  return {execute_tree<using_local_memory::disabled>(
      policy_handler_.get_queue(), t, localSize, globalSize, 0)};
};

/*!
 * @brief Executes the tree fixing the localSize but without defining
 * required shared memory.
 */
template <>
template <typename expression_tree_t, typename index_t>
inline typename codeplay_policy::event_t
Executor<PolicyHandler<codeplay_policy>>::execute(expression_tree_t t,
                                                  index_t localSize,
                                                  index_t globalSize) {
  return {execute_tree<using_local_memory::disabled>(
      policy_handler_.get_queue(), t, localSize, globalSize, 0)};
}

/*!
 * @brief Executes the tree with specific local, global and shared
 * memory values.
 */
template <>
template <typename expression_tree_t, typename index_t>
inline typename codeplay_policy::event_t
Executor<PolicyHandler<codeplay_policy>>::execute(expression_tree_t t,
                                                  index_t localSize,
                                                  index_t globalSize,
                                                  index_t shMem) {
  return {execute_tree<using_local_memory::enabled>(
      policy_handler_.get_queue(), t, localSize, globalSize, shMem)};
}

/*!
 * @brief Applies a reduction to a tree.
 */
template <>
template <typename operator_t, typename lhs_t, typename rhs_t>
inline typename codeplay_policy::event_t
Executor<PolicyHandler<codeplay_policy>>::execute(
    AssignReduction<operator_t, lhs_t, rhs_t> t) {
  using expression_tree_t = AssignReduction<operator_t, lhs_t, rhs_t>;
  auto _N = t.get_size();
  auto localSize = t.local_num_thread_;
  // IF THERE ARE ENOUGH ELEMENTS, EACH BLOCK PROCESS TWO BLOCKS OF
  // ELEMENTS THEREFORE, 2*GLOBALSIZE ELEMENTS ARE PROCESSED IN A STEP
  // MOREOVER, A LOOP ALLOWS TO REPEAT THE PROCESS UNTIL
  // ALL THE ELEMENTS ARE PROCESSED
  auto nWG = (t.global_num_thread_ + (2 * localSize) - 1) / (2 * localSize);
  auto lhs = t.lhs_;
  auto rhs = t.rhs_;

  // Two accessors to local memory
  auto sharedSize = ((nWG < localSize) ? localSize : nWG);
  auto shMem1 = make_sycl_iterator_buffer<typename lhs_t::value_t>(sharedSize);
  auto shMem2 = make_sycl_iterator_buffer<typename lhs_t::value_t>(sharedSize);
  auto opShMem1 = lhs_t(shMem1, 1, sharedSize);
  auto opShMem2 = lhs_t(shMem2, 1, sharedSize);
  typename codeplay_policy::event_t event;
  bool frst = true;
  bool even = false;
  do {
    auto globalSize = nWG * localSize;
    if (frst) {
      // THE FIRST CASE USES THE ORIGINAL BINARY/TERNARY FUNCTION
      auto localTree = expression_tree_t(((nWG == 1) ? lhs : opShMem1), rhs,
                                         localSize, globalSize);
      event.push_back(execute_tree<using_local_memory::enabled>(
          policy_handler_.get_queue(), localTree, localSize, globalSize,
          sharedSize));
    } else {
      // THE OTHER CASES ALWAYS USE THE BINARY FUNCTION
      auto localTree = AssignReduction<operator_t, lhs_t, lhs_t>(
          ((nWG == 1) ? lhs : (even ? opShMem2 : opShMem1)),
          (even ? opShMem1 : opShMem2), localSize, globalSize);
      event.push_back(execute_tree<using_local_memory::enabled>(
          policy_handler_.get_queue(), localTree, localSize, globalSize,
          sharedSize));
    }
    _N = nWG;
    nWG = (_N + (2 * localSize) - 1) / (2 * localSize);
    frst = false;
    even = !even;
  } while (_N > 1);
  return event;
}

/*!
 * @brief Applies a reduction to a tree, receiving a scratch
 * BufferIterator.
 */
template <>
template <typename operator_t, typename lhs_t, typename rhs_t,
          typename local_memory_t>
inline typename codeplay_policy::event_t
Executor<PolicyHandler<codeplay_policy>>::execute(
    AssignReduction<operator_t, lhs_t, rhs_t> t, local_memory_t scr) {
  using expression_tree_t = AssignReduction<operator_t, lhs_t, rhs_t>;
  auto _N = t.get_size();
  auto localSize = t.local_num_thread_;
  // IF THERE ARE ENOUGH ELEMENTS, EACH BLOCK PROCESS TWO BLOCKS OF
  // ELEMENTS THEREFORE, 2*GLOBALSIZE ELEMENTS ARE PROCESSED IN A STEP
  // MOREOVER, A LOOP ALLOWS TO REPEAT THE PROCESS UNTIL
  // ALL THE ELEMENTS ARE PROCESSED
  auto nWG = (t.global_num_thread_ + (2 * localSize) - 1) / (2 * localSize);
  auto lhs = t.lhs_;
  auto rhs = t.rhs_;
  typename codeplay_policy::event_t event;
  // Two accessors to local memory
  auto sharedSize = ((nWG < localSize) ? localSize : nWG);
  auto opShMem1 = lhs_t(scr, 1, sharedSize);
  auto opShMem2 = lhs_t(scr + sharedSize, 1, sharedSize);

  bool frst = true;
  bool even = false;
  do {
    auto globalSize = nWG * localSize;
    if (frst) {
      // THE FIRST CASE USES THE ORIGINAL BINARY/TERNARY FUNCTION
      auto localTree = expression_tree_t(((nWG == 1) ? lhs : opShMem1), rhs,
                                         localSize, globalSize);
      event.push_back(execute_tree<using_local_memory::enabled>(
          policy_handler_.get_queue(), localTree, localSize, globalSize,
          sharedSize));
    } else {
      // THE OTHER CASES ALWAYS USE THE BINARY FUNCTION
      auto localTree = AssignReduction<operator_t, lhs_t, lhs_t>(
          ((nWG == 1) ? lhs : (even ? opShMem2 : opShMem1)),
          (even ? opShMem1 : opShMem2), localSize, globalSize);
      event.push_back(execute_tree<using_local_memory::enabled>(
          policy_handler_.get_queue(), localTree, localSize, globalSize,
          sharedSize));
    }
    _N = nWG;
    nWG = (_N + (2 * localSize) - 1) / (2 * localSize);
    frst = false;
    even = !even;
  } while (_N > 1);
  return event;
}

template <>
template <typename input_t, typename output_t, bool DoubleBuffer, bool NbcA,
          bool NbcB, int ClSize, typename tile_type, bool TransA, bool TransB,
          typename element_t, bool is_beta_zero, int GemmMemoryType,
          int GemmAlgorithm, int VectorSize>
inline typename codeplay_policy::event_t
Executor<PolicyHandler<codeplay_policy>>::execute(
    Gemm<input_t, output_t, DoubleBuffer, NbcA, NbcB, ClSize, tile_type, TransA,
         TransB, element_t, is_beta_zero, GemmMemoryType, GemmAlgorithm,
         VectorSize>
        gemm_tree) {
  using gemm_t = Gemm<input_t, output_t, DoubleBuffer, NbcA, NbcB, ClSize,
                      tile_type, TransA, TransB, element_t, is_beta_zero,
                      GemmMemoryType, GemmAlgorithm, VectorSize>;
  auto rng = gemm_t::get_nd_range(gemm_tree.m_, gemm_tree.n_,
                                  policy_handler_.get_num_compute_units());
  return {execute_tree<
      Choose<GemmMemoryType == static_cast<int>(gemm_memory_t::local), int,
             using_local_memory::enabled, using_local_memory::disabled>::type>(
      policy_handler_.get_queue(), gemm_tree, rng.get_local_range()[0],
      rng.get_global_range()[0], gemm_t::local_memory_size)};
}

/* Tall and skinny Gemm */
template <>
template <typename input_t, typename output_t, bool DoubleBuffer, bool NbcA,
          bool NbcB, int ClSize, typename tile_type, bool TransA, bool TransB,
          typename element_t, bool is_beta_zero, int GemmMemoryType,
          int VectorSize>
inline typename codeplay_policy::event_t
Executor<PolicyHandler<codeplay_policy>>::execute(
    Gemm<input_t, output_t, DoubleBuffer, NbcA, NbcB, ClSize, tile_type, TransA,
         TransB, element_t, is_beta_zero, GemmMemoryType,
         static_cast<int>(gemm_algorithm_t::tall_skinny), VectorSize>
        gemm_wrapper) {
  using index_t = typename std::make_signed<typename input_t::index_t>::type;

  const index_t rows = gemm_wrapper.m_;
  const index_t cols = gemm_wrapper.n_;
  const index_t ldc = gemm_wrapper.ldc_;

  /* Depth of the cube buffer */
  const index_t depth = GemmPartial<input_t, output_t, DoubleBuffer, NbcA, NbcB,
                                    ClSize, tile_type, TransA, TransB, false,
                                    is_beta_zero, element_t, GemmMemoryType>::
      get_ideal_cube_depth(policy_handler_.get_num_compute_units(), rows, cols,
                           gemm_wrapper.k_);

  /* In some cases, use the tsgemm kernel as a normal gemm operation */
  if (depth == 1 || gemm_wrapper.k_ <= 2048) {
    GemmPartial<input_t, output_t, DoubleBuffer, NbcA, NbcB, ClSize, tile_type,
                TransA, TransB, true, is_beta_zero, element_t, GemmMemoryType>
        gemm_partial(gemm_wrapper.a_, gemm_wrapper.b_, gemm_wrapper.c_,
                     gemm_wrapper.alpha_, gemm_wrapper.beta_, 1);
    auto events = execute(gemm_partial);

    return events;
  }
  /* Else use the tall and skinny algorithm */

  /* First step: partial gemm */
  /* Create the cube buffer that will hold the output of the partial gemm */
  auto cube_buffer = make_sycl_iterator_buffer<element_t>(rows * cols * depth);

  /* Create a first matrix view used for the partial gemm */
  auto cube_gemm =
      make_matrix_view<col_major>(*this, cube_buffer, rows, cols * depth, rows);
  /* Execute the partial gemm operation */
  /* Note: we set is_beta_zero to true regardless of the value of beta
   * because this option is meant for use with a simple Gemm only */
  GemmPartial<input_t, output_t, DoubleBuffer, NbcA, NbcB, ClSize, tile_type,
              TransA, TransB, false, true, element_t, GemmMemoryType>
      gemm_partial(gemm_wrapper.a_, gemm_wrapper.b_, cube_gemm,
                   gemm_wrapper.alpha_, gemm_wrapper.beta_, depth);
  auto events = execute(gemm_partial);

  /* Create a second view used for the reduction */
  auto cube_reduction = make_matrix_view<col_major>(
      *this, cube_buffer, rows * cols, depth, rows * cols);

  /* Second step: reduction */
  /* Best case: we can reduce directly in C */
  if (is_beta_zero && ldc == rows) {
    constexpr int work_group_size = tile_type::wg_rows * tile_type::wg_cols;
    Reduction<blas::AddOperator, input_t, output_t, ClSize, work_group_size,
              element_t, static_cast<int>(Reduction_t::partial_rows)>
        reduction(cube_reduction, gemm_wrapper.c_, rows * cols, depth);
    events = concatenate_vectors(events, execute(reduction));
  }
  /* Otherwise we reduce to a temporary buffer */
  else {
    /* Create a temporary buffer to hold alpha * A * B */
    auto temp_buffer = make_sycl_iterator_buffer<element_t>(rows * cols);
    auto temp =
        make_matrix_view<col_major>(*this, temp_buffer, rows, cols, rows);

    /* Execute the reduction */
    constexpr int work_group_size = tile_type::wg_rows * tile_type::wg_cols;
    Reduction<blas::AddOperator, input_t, output_t, ClSize, work_group_size,
              element_t, static_cast<int>(Reduction_t::partial_rows)>
        reduction(cube_reduction, temp, rows * cols, depth);
    events = concatenate_vectors(events, execute(reduction));

    /* If beta is zero, simply do a 2D copy from the temp buffer to C */
    if (is_beta_zero) {
      auto assignOp = make_op<Assign>(gemm_wrapper.c_, temp);
      events = concatenate_vectors(events, execute(assignOp));
    }
    /* Else add temp and beta * C and then assign to C */
    else {
      auto scalOp = make_op<ScalarOp, ProductOperator>(gemm_wrapper.beta_,
                                                       gemm_wrapper.c_);
      auto addOp = make_op<BinaryOp, AddOperator>(temp, scalOp);
      auto assignOp = make_op<Assign>(gemm_wrapper.c_, addOp);
      events = concatenate_vectors(events, execute(assignOp));
    }
  }

  return events;
}

/* GemmPartial */
template <>
template <typename input_t, typename output_t, bool DoubleBuffer, bool NbcA,
          bool NbcB, int ClSize, typename tile_type, bool TransA, bool TransB,
          bool IsFinal, bool IsBetaZero, typename element_t, int GemmMemoryType>
inline typename codeplay_policy::event_t
Executor<PolicyHandler<codeplay_policy>>::execute(
    GemmPartial<input_t, output_t, DoubleBuffer, NbcA, NbcB, ClSize, tile_type,
                TransA, TransB, IsFinal, IsBetaZero, element_t, GemmMemoryType>
        gemm_partial) {
  auto gemm_partial_range =
      gemm_partial.get_nd_range(policy_handler_.get_num_compute_units());
  return {execute_tree<
      Choose<GemmMemoryType == static_cast<int>(gemm_memory_t::local), int,
             using_local_memory::enabled, using_local_memory::disabled>::type>(
      policy_handler_.get_queue(), gemm_partial,
      gemm_partial_range.get_local_range()[0],
      gemm_partial_range.get_global_range()[0],
      gemm_partial.local_memory_size)};
}

/* Utility function used by the ReductionPartialRows specialization */
template <typename operator_t, int ClSize, int WgSize, typename element_t,
          typename input_t, typename output_t, typename index_t,
          typename queue_t>
static inline cl::sycl::event launch_row_reduction_step(
    queue_t queue, input_t& in, output_t& out, index_t group_count_cols,
    index_t local_memory_size, index_t num_compute_units) {
  ReductionPartialRows<operator_t, input_t, output_t, ClSize, WgSize, element_t>
      reduction_step(in, out, group_count_cols);
  auto step_range = reduction_step.get_nd_range(num_compute_units);
  return execute_tree<using_local_memory::enabled>(
      queue, reduction_step, step_range.get_local_range()[0],
      step_range.get_global_range()[0], local_memory_size);
}

/* ReductionPartialRows */
template <>
template <typename operator_t, typename input_t, typename output_t, int ClSize,
          int WgSize, typename element_t>
inline typename codeplay_policy::event_t
Executor<PolicyHandler<codeplay_policy>>::execute(
    Reduction<operator_t, input_t, output_t, ClSize, WgSize, element_t,
              static_cast<int>(Reduction_t::partial_rows)>
        reduction_wrapper) {
  using index_t = typename input_t::index_t;
  using params_t =
      blas::ReductionRows_Params<index_t, element_t, ClSize, WgSize>;

  /* Extract data from the reduction wrapper */
  const index_t rows_ = reduction_wrapper.rows_,
                cols_ = reduction_wrapper.cols_;
  input_t& in_ = reduction_wrapper.in_;
  output_t& out_ = reduction_wrapper.out_;

  const index_t num_compute_units = policy_handler_.get_num_compute_units();

  /* Choose at run-time whether to do a one-step or two-step reduction.
   * These heuristics have been selected empirically by benchmarking one-step
   * against two-step reduction */
  const bool two_step_reduction = (cols_ > 2048);

  /* Create an empty event vector */
  typename codeplay_policy::event_t reduction_event;

  /* 2-step reduction */
  if (two_step_reduction) {
    static const index_t max_group_count_col =
        (cols_ - 1) / params_t::work_group_cols + 1;
    static const index_t group_count_cols =
        params_t::work_group_cols < max_group_count_col
            ? params_t::work_group_cols
            : max_group_count_col;

    /* Create a temporary buffer */
    auto temp_buffer =
        make_sycl_iterator_buffer<element_t>(rows_ * group_count_cols);
    auto temp_ = make_matrix_view<col_major>(*this, temp_buffer, rows_,
                                             group_count_cols, rows_);

    /* 1st step */
    reduction_event.push_back(
        launch_row_reduction_step<operator_t, ClSize, WgSize, element_t>(
            policy_handler_.get_queue(), in_, temp_, group_count_cols,
            params_t::local_memory_size, num_compute_units));

    /* 2nd step */
    reduction_event.push_back(
        launch_row_reduction_step<operator_t, ClSize, WgSize, element_t>(
            policy_handler_.get_queue(), temp_, out_, 1,
            params_t::local_memory_size, num_compute_units));
  }
  /* 1-step reduction */
  else {
    reduction_event.push_back(
        launch_row_reduction_step<operator_t, ClSize, WgSize, element_t>(
            policy_handler_.get_queue(), in_, out_, 1,
            params_t::local_memory_size, num_compute_units));
  }

  return reduction_event;
}

}  // namespace blas

#endif  // EXECUTOR_SYCL_HPP
