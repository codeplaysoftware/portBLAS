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

#include "blas_meta.h"
#include "executors/executor.h"
#include "executors/kernel_constructor.h"
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
          typename element_t, bool is_beta_zero, int Gemm_type>
inline typename codeplay_policy::event_t
Executor<PolicyHandler<codeplay_policy>>::execute(
    Gemm<input_t, output_t, DoubleBuffer, NbcA, NbcB, ClSize, tile_type, TransA,
         TransB, element_t, is_beta_zero, Gemm_type>
        gemm_tree) {
  auto rng =
      Gemm<input_t, output_t, DoubleBuffer, NbcA, NbcB, ClSize, tile_type,
           TransA, TransB, element_t, is_beta_zero,
           Gemm_type>::get_nd_range(gemm_tree.m_, gemm_tree.n_,
                                    policy_handler_.get_num_compute_units());
  return {execute_tree<
      Choose<Gemm_type == static_cast<int>(Gemm_t::local_memory),
             using_local_memory::enabled, using_local_memory::disabled>::type>(
      policy_handler_.get_queue(), gemm_tree, rng.get_local_range()[0],
      rng.get_global_range()[0],
      Gemm<input_t, output_t, DoubleBuffer, NbcA, NbcB, ClSize, tile_type,
           TransA, TransB, element_t, is_beta_zero,
           Gemm_type>::local_memory_size)};
}

}  // namespace blas

#endif  // EXECUTOR_SYCL_HPP
