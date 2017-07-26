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

#include <executors/blas_pointer_struct.hpp>
#include <operations/blas_operators.hpp>
#include <views/operview_base.hpp>

namespace blas {

/*! Reduction.
 * @brief Implements the reduction operation for assignments (in the form y = x)
 *  with y a scalar and x a subexpression tree.
 */
template <typename Functor, class RHS,
          template <class> class MakePointer = MakeHostPointer>
struct ReductionExpr {
  using value_type = typename RHS::value_type;
  RHS r;
  ReductionExpr(RHS &_r) : r(_r) {}

  size_t getSize() const { return r.getSize(); }
};

template <typename Functor, typename RHS>
ReductionExpr<Functor, RHS> make_ReductionExpr(RHS &r) {
  return ReductionExpr<Functor, RHS>(r);
}

template <typename RHS>
ReductionExpr<addOp2_struct, RHS> make_addReductionExpr(RHS &r) {
  return make_ReductionExpr<addOp2_struct>(r);
}

template <typename RHS>
ReductionExpr<prdOp2_struct, RHS> make_prdReductionExpr(RHS &r) {
  return make_ReductionExpr<prdOp2_struct>(r);
}

template <typename RHS>
ReductionExpr<addAbsOp2_struct, RHS> make_addAbsReductionExpr(RHS &r) {
  return make_ReductionExpr<addAbsOp2_struct>(r);
}

template <typename RHS>
ReductionExpr<maxIndOp2_struct, RHS> make_maxIndReductionExpr(RHS &r) {
  return make_ReductionExpr<maxIndOp2_struct>(r);
}

template <typename RHS>
ReductionExpr<minIndOp2_struct, RHS> make_minIndReductionExpr(RHS &r) {
  return make_ReductionExpr<minIndOp2_struct>(r);
}

}  // namespace blas

#endif
