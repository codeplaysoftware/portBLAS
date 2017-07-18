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
 *  @filename blas1_trees.hpp
 *
 **************************************************************************/

#ifndef BLAS1_TREE_EXPR_HPP
#define BLAS1_TREE_EXPR_HPP

#include <stdexcept>
#include <vector>

#include <operations/blas_operators.hpp>
#include <views/view_sycl.hpp>

namespace blas {

/*! Reduction.
 * @brief Implements the reduction operation for assignments (in the form y = x)
 *  with y a scalar and x a subexpression tree.
 */
template <typename Functor, class LHS, class RHS>
struct ReductionExpr {
  using value_type = typename LHS::value_type;

  LHS l;
  RHS r;

  ReductionExpr(LHS &_l, RHS &_r) : l(_l), r(_r) {}

  size_t getSize() const { return r.getSize(); }
};

template <typename Functor, typename LHS, typename RHS>
ReductionExpr<Functor, LHS, RHS> make_ReductionExpr(LHS &l, RHS &r) {
  return ReductionExpr<Functor, LHS, RHS>(l, r);
}

template <typename LHS, typename RHS>
ReductionExpr<addOp2_struct, LHS, RHS> make_addReductionExpr(LHS &l, RHS &r) {
  return make_ReductionExpr<addOp2_struct>(l, r);
}

template <typename LHS, typename RHS>
ReductionExpr<prdOp2_struct, LHS, RHS> make_prdReductionExpr(LHS &l, RHS &r) {
  return make_ReductionExpr<prdOp2_struct>(l, r);
}

template <typename LHS, typename RHS>
ReductionExpr<addAbsOp2_struct, LHS, RHS> make_addAbsReductionExpr(LHS &l,
                                                                   RHS &r) {
  return make_ReductionExpr<addAbsOp2_struct>(l, r);
}

template <typename LHS, typename RHS>
ReductionExpr<maxIndOp2_struct, LHS, RHS> make_maxIndReductionExpr(LHS &l,
                                                                   RHS &r) {
  return make_ReductionExpr<maxIndOp2_struct>(l, r);
}

template <typename LHS, typename RHS>
ReductionExpr<minIndOp2_struct, LHS, RHS> make_minIndReductionExpr(LHS &l,
                                                                   RHS &r) {
  return make_ReductionExpr<minIndOp2_struct>(l, r);
}

}  // namespace blas

#endif
