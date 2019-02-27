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
/*! Executor<Policy_Handler<BLAS_SYCL_Policy>>.
 * @brief Executes an Expression Tree using SYCL.
 */
template class Executor<Policy_Handler<BLAS_SYCL_Policy>>;
/*!
 * @brief Constructs a SYCL executor using the given queue.
 * @param q A SYCL queue.
 */
template <>
inline Executor<Policy_Handler<BLAS_SYCL_Policy>>::Executor(
    typename BLAS_SYCL_Policy::queue_type q)
    : policy_handler_(Policy_Handler<BLAS_SYCL_Policy>(q)) {}

template <>
inline Policy_Handler<BLAS_SYCL_Policy>
Executor<Policy_Handler<BLAS_SYCL_Policy>>::get_policy_handler() const {
  return policy_handler_;
}

/*!
 * @brief Executes the tree without defining required shared memory.
 */
template <>
template <typename Tree>
inline typename BLAS_SYCL_Policy::event_type
Executor<Policy_Handler<BLAS_SYCL_Policy>>::execute(Tree t) {
  const auto localSize = policy_handler_.get_work_group_size();
  auto _N = t.getSize();
  auto nWG = (_N + localSize - 1) / localSize;
  auto globalSize = nWG * localSize;

  return {execute_tree<using_shared_mem::disabled>(
      policy_handler_.get_queue(), t, localSize, globalSize, 0)};
};

/*!
 * @brief Executes the tree fixing the localSize but without defining
 * required shared memory.
 */
template <>
template <typename Tree, typename IndexType>
inline typename BLAS_SYCL_Policy::event_type
Executor<Policy_Handler<BLAS_SYCL_Policy>>::execute(Tree t,
                                                    IndexType localSize) {
  auto _N = t.getSize();
  auto nWG = (_N + localSize - 1) / localSize;
  auto globalSize = nWG * localSize;
  return {execute_tree<using_shared_mem::disabled>(
      policy_handler_.get_queue(), t, localSize, globalSize, 0)};
};

/*!
 * @brief Executes the tree fixing the localSize but without defining
 * required shared memory.
 */
template <>
template <typename Tree, typename IndexType>
inline typename BLAS_SYCL_Policy::event_type
Executor<Policy_Handler<BLAS_SYCL_Policy>>::execute(Tree t, IndexType localSize,
                                                    IndexType globalSize) {
  return {execute_tree<using_shared_mem::disabled>(
      policy_handler_.get_queue(), t, localSize, globalSize, 0)};
}

/*!
 * @brief Executes the tree with specific local, global and shared
 * memory values.
 */
template <>
template <typename Tree, typename IndexType>
inline typename BLAS_SYCL_Policy::event_type
Executor<Policy_Handler<BLAS_SYCL_Policy>>::execute(Tree t, IndexType localSize,
                                                    IndexType globalSize,
                                                    IndexType shMem) {
  return {execute_tree<using_shared_mem::enabled>(
      policy_handler_.get_queue(), t, localSize, globalSize, shMem)};
}

/*!
 * @brief Applies a reduction to a tree.
 */
template <>
template <typename Op, typename LHS, typename RHS>
inline typename BLAS_SYCL_Policy::event_type
Executor<Policy_Handler<BLAS_SYCL_Policy>>::execute(
    AssignReduction<Op, LHS, RHS> t) {
  using Tree = AssignReduction<Op, LHS, RHS>;
  auto _N = t.getSize();
  auto localSize = t.blqS;
  // IF THERE ARE ENOUGH ELEMENTS, EACH BLOCK PROCESS TWO BLOCKS OF
  // ELEMENTS THEREFORE, 2*GLOBALSIZE ELEMENTS ARE PROCESSED IN A STEP
  // MOREOVER, A LOOP ALLOWS TO REPEAT THE PROCESS UNTIL
  // ALL THE ELEMENTS ARE PROCESSED
  auto nWG = (t.grdS + (2 * localSize) - 1) / (2 * localSize);
  auto lhs = t.l;
  auto rhs = t.r;

  // Two accessors to local memory
  auto sharedSize = ((nWG < localSize) ? localSize : nWG);
  auto shMem1 = make_sycl_iterator_buffer<typename LHS::value_type>(sharedSize);
  auto shMem2 = make_sycl_iterator_buffer<typename LHS::value_type>(sharedSize);
  auto opShMem1 = LHS(shMem1, 1, sharedSize);
  auto opShMem2 = LHS(shMem2, 1, sharedSize);
  typename BLAS_SYCL_Policy::event_type event;
  bool frst = true;
  bool even = false;
  do {
    auto globalSize = nWG * localSize;
    if (frst) {
      // THE FIRST CASE USES THE ORIGINAL BINARY/TERNARY FUNCTION
      auto localTree =
          Tree(((nWG == 1) ? lhs : opShMem1), rhs, localSize, globalSize);
      event.push_back(execute_tree<using_shared_mem::enabled>(
          policy_handler_.get_queue(), localTree, localSize, globalSize,
          sharedSize));
    } else {
      // THE OTHER CASES ALWAYS USE THE BINARY FUNCTION
      auto localTree = AssignReduction<Op, LHS, LHS>(
          ((nWG == 1) ? lhs : (even ? opShMem2 : opShMem1)),
          (even ? opShMem1 : opShMem2), localSize, globalSize);
      event.push_back(execute_tree<using_shared_mem::enabled>(
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
 * buffer_iterator.
 */
template <>
template <typename Operator, typename LHS, typename RHS, typename Scratch>
inline typename BLAS_SYCL_Policy::event_type
Executor<Policy_Handler<BLAS_SYCL_Policy>>::execute(
    AssignReduction<Operator, LHS, RHS> t, Scratch scr) {
  using Tree = AssignReduction<Operator, LHS, RHS>;
  auto _N = t.getSize();
  auto localSize = t.blqS;
  // IF THERE ARE ENOUGH ELEMENTS, EACH BLOCK PROCESS TWO BLOCKS OF
  // ELEMENTS THEREFORE, 2*GLOBALSIZE ELEMENTS ARE PROCESSED IN A STEP
  // MOREOVER, A LOOP ALLOWS TO REPEAT THE PROCESS UNTIL
  // ALL THE ELEMENTS ARE PROCESSED
  auto nWG = (t.grdS + (2 * localSize) - 1) / (2 * localSize);
  auto lhs = t.l;
  auto rhs = t.r;
  typename BLAS_SYCL_Policy::event_type event;
  // Two accessors to local memory
  auto sharedSize = ((nWG < localSize) ? localSize : nWG);
  auto opShMem1 = LHS(scr, 1, sharedSize);
  auto opShMem2 = LHS(scr + sharedSize, 1, sharedSize);

  bool frst = true;
  bool even = false;
  do {
    auto globalSize = nWG * localSize;
    if (frst) {
      // THE FIRST CASE USES THE ORIGINAL BINARY/TERNARY FUNCTION
      auto localTree =
          Tree(((nWG == 1) ? lhs : opShMem1), rhs, localSize, globalSize);
      event.push_back(execute_tree<using_shared_mem::enabled>(
          policy_handler_.get_queue(), localTree, localSize, globalSize,
          sharedSize));
    } else {
      // THE OTHER CASES ALWAYS USE THE BINARY FUNCTION
      auto localTree = AssignReduction<Operator, LHS, LHS>(
          ((nWG == 1) ? lhs : (even ? opShMem2 : opShMem1)),
          (even ? opShMem1 : opShMem2), localSize, globalSize);
      event.push_back(execute_tree<using_shared_mem::enabled>(
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
template <typename RHS0, typename RHS1, bool DoubleBuffer, bool NbcA, bool NbcB,
          int ClSize, typename tile_type, bool TransA, bool TransB, typename T,
          bool is_beta_zero, int Gemm_type>
inline typename BLAS_SYCL_Policy::event_type
Executor<Policy_Handler<BLAS_SYCL_Policy>>::execute(
    Gemm<RHS0, RHS1, DoubleBuffer, NbcA, NbcB, ClSize, tile_type, TransA,
         TransB, T, is_beta_zero, Gemm_type>
        gemm_tree) {
  auto rng =
      Gemm<RHS0, RHS1, DoubleBuffer, NbcA, NbcB, ClSize, tile_type, TransA,
           TransB, T, is_beta_zero,
           Gemm_type>::get_nd_range(gemm_tree.m, gemm_tree.n,
                                    policy_handler_.get_num_compute_units());
  return {execute_tree<
      Choose<Gemm_type == static_cast<int>(Gemm_t::local_memory),
             using_shared_mem::enabled, using_shared_mem::disabled>::type>(
      policy_handler_.get_queue(), gemm_tree, rng.get_local_range()[0],
      rng.get_global_range()[0],
      Gemm<RHS0, RHS1, DoubleBuffer, NbcA, NbcB, ClSize, tile_type, TransA,
           TransB, T, is_beta_zero, Gemm_type>::local_memory_size)};
}

}  // namespace blas

#endif  // EXECUTOR_SYCL_HPP
