/***************************************************************************
 *
 *  @license
 *  Copyright (C) 2016 Codeplay Software Limited
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
 *  @filename blas1_tree_evaluator.hpp
 *
 **************************************************************************/

#ifndef BLAS1_TREE_EVALUATOR_HPP
#define BLAS1_TREE_EVALUATOR_HPP

#include <stdexcept>
#include <vector>

#include <evaluators/blas_tree_evaluator_base.hpp>
#include <executors/blas_packet_traits_sycl.hpp>
#include <operations/blas1_trees.hpp>
#include <views/view_sycl.hpp>

namespace blas {

/*! Reduction.
 * @brief Implements the reduction operation for assignments (in the form y = x)
 *  with y a scalar and x a subexpression tree.
 */
template <typename Functor, class LHS, class RHS>
struct Evaluator<ReductionExpr<Functor, RHS>, SYCLDevice> {
  using Expression = ReductionExpr<Functor, LHS, RHS>;
  using Device = SYCLDevice;
  using value_type = typename Expression::value_type;
  using dev_functor = functor_traits<Functor, value_type, Device>;
  using cont_type = typename Evaluator<LHS, Device>::cont_type;
  using Self = Evaluator < ReductionExpr<Functor, LHS, RHS>;
  cont_type *result;
  /* static constexpr bool supported = functor_traits<Functor, value_type,
   * SYCLDevice>::supported && LHS::supported && RHS::supported; */
  Evaluator<LHS, Device> l;
  Evaluator<RHS, Device> r;

  Evaluator(Expression &expr)
      : l(Evaluator<LHS, Device>(expr.l)), r(Evaluator<RHS, Device>(expr.r)) {}

  size_t getSize() const { return r.getSize(); }
  cont_type *data() { return l.data(); }

  void reduce(Device &dev);

  bool eval_subexpr_if_needed(cont_type *cont, Device &dev) {
    l.eval_subexpr_if_needed(NULL, dev);
    r.eval_subexpr_if_needed(NULL, dev);
    if (cont)){
      m_result = cont;
    }
    else {
      m_result = dev.allocate(out_size);
    }
    FullReducer<Self, Device>::run(this, dev, *result);
    return true;
  }

  value_type eval(size_t i) { return m_result[i]; }

  value_type eval(cl::sycl::nd_item<1> ndItem) {
    return eval(ndItem.get_global(0));
  }
};

}  // namespace blas

#endif
