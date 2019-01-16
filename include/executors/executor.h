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
#include "blas_meta.h"
#include "operations/blas1_trees.h"
#include "operations/blas2_trees.h"
#include "operations/blas3_trees.h"
namespace blas {

/** Executor.
 * @brief Primary template for the Executor specializations.
 * The Executor represents the object that executes a tree on
 * a specific backend.
 * Executors have state, and they must be instantiated
 * before using them.
 * Only one method is mandatory, the Execute method.
 */
template <class PolicyHandler>
class Executor {
 public:
  using Policy = typename PolicyHandler::Policy;
  Executor(typename Policy::queue_type q);
  PolicyHandler get_policy_handler() const;

  template <typename Tree>
  typename Policy::event_type execute(Tree tree);

  template <typename Tree, typename IndexType>
  typename Policy::event_type execute(Tree tree, IndexType localSize);

  template <typename Tree, typename IndexType>
  typename Policy::event_type execute(Tree tree, IndexType localSize,
                                      IndexType globalSize);
  template <typename Tree, typename IndexType>
  typename Policy::event_type execute(Tree tree, IndexType localSize,
                                      IndexType globalSize,
                                      IndexType local_memory_size);

  template <typename Op, typename LHS, typename RHS>
  typename Policy::event_type execute(AssignReduction<Op, LHS, RHS>);

  template <typename Operator, typename LHS, typename RHS, typename Scratch>
  typename Policy::event_type execute(AssignReduction<Operator, LHS, RHS> t,
                                      Scratch scr);
  template <typename RHS0, typename RHS1, bool DoubleBuffer, bool NbcA,
            bool NbcB, int ClSize, typename tile_type, bool TransA, bool TransB,
            typename T, bool is_beta_zero, int Gemm_type>
  typename Policy::event_type execute(
      Gemm<RHS0, RHS1, DoubleBuffer, NbcA, NbcB, ClSize, tile_type, TransA,
           TransB, T, is_beta_zero, Gemm_type>
          gemm_tree);

 private:
  PolicyHandler policy_handler_;
};

}  // namespace blas

#endif  // SYCL_BLAS_EXECUTOR_H
