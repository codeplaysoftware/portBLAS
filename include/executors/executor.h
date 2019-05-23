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
 *  @filename executor.h
 *
 **************************************************************************/

#ifndef SYCL_BLAS_EXECUTOR_H
#define SYCL_BLAS_EXECUTOR_H
#include "../blas_meta.h"
#include "../operations/blas1_trees.h"
#include "../operations/blas2_trees.h"
#include "../operations/blas3_trees.h"
#include "../policy/policy_handler.h"
namespace blas {

/** Executor.
 * @brief Primary template for the Executor specializations.
 * The Executor represents the object that executes a tree on
 * a specific backend.
 * Executors have state, and they must be instantiated
 * before using them.
 * Only one method is mandatory, the Execute method.
 */
template <typename policy_handler_t>
class Executor {
 public:
  using policy_t = typename policy_handler_t::policy_t;
  inline Executor(typename policy_t::queue_t q)
      : policy_handler_(policy_handler_t(q)) {}
  inline policy_handler_t get_policy_handler() const { return policy_handler_; }

  template <typename expression_tree_t>
  typename policy_t::event_t execute(expression_tree_t tree);

  template <typename expression_tree_t, typename index_t>
  typename policy_t::event_t execute(expression_tree_t tree, index_t localSize);

  template <typename expression_tree_t, typename index_t>
  typename policy_t::event_t execute(expression_tree_t tree, index_t localSize,
                                     index_t globalSize);
  template <typename expression_tree_t, typename index_t>
  typename policy_t::event_t execute(expression_tree_t tree, index_t localSize,
                                     index_t globalSize,
                                     index_t local_memory_size);

  template <typename operator_t, typename lhs_t, typename rhs_t>
  typename policy_t::event_t execute(AssignReduction<operator_t, lhs_t, rhs_t>);

  template <typename operator_t, typename lhs_t, typename rhs_t,
            typename local_memory_t>
  typename policy_t::event_t execute(
      AssignReduction<operator_t, lhs_t, rhs_t> t, local_memory_t scr);
  template <typename input_t, typename output_t, bool DoubleBuffer, bool NbcA,
            bool NbcB, int ClSize, typename tile_type, bool TransA, bool TransB,
            typename element_t, bool is_beta_zero, int Gemm_type>
  typename policy_t::event_t execute(
      Gemm<input_t, output_t, DoubleBuffer, NbcA, NbcB, ClSize, tile_type,
           TransA, TransB, element_t, is_beta_zero, Gemm_type>
          gemm_tree);

 private:
  policy_handler_t policy_handler_;
};

}  // namespace blas

#endif  // SYCL_BLAS_EXECUTOR_H
