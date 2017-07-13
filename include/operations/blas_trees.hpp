/***************************************************************************
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
*  @filename blas_trees_expr.hpp
*
**************************************************************************/

#ifndef BLAS_TREE_EXPR_HPP
#define BLAS_TREE_EXPR_HPP

namespace blas {

/** Join.
 * @brief Joins both sides of the expression in the single kernel.
 */
template <class LHS, class RHS>
struct JoinExpr {
  using value_type = typename RHS::value_type;

  LHS l;
  RHS r;

  JoinExpr(LHS &_l, RHS _r) : l(_l), r(_r){};

  // PROBLEM: Only the RHS size is considered. If LHS size is different??
  // If it is smaller, eval function will crash
  size_t getSize() const { return r.getSize(); }
};

/** Assign.
 */
template <class LHS, class RHS>
struct AssignExpr {
  using value_type = typename RHS::value_type;

  LHS l;
  RHS r;

  AssignExpr(LHS &_l, RHS _r) : l(_l), r(_r){};

  // PROBLEM: Only the RHS size is considered. If LHS size is different??
  // If it is smaller, eval function will crash
  size_t getSize() const { return r.getSize(); }
};

/*! DoubleAssign.
 */
template <class LHS1, class LHS2, class RHS1, class RHS2>
struct DoubleAssignExpr {
  LHS1 l1;
  LHS2 l2;
  RHS1 r1;
  RHS2 r2;

 public:
  using value_type = typename RHS1::value_type;

  DoubleAssignExpr(LHS1 &_l1, LHS2 &_l2, RHS1 _r1, RHS2 _r2)
      : l1(_l1), l2(_l2), r1(_r1), r2(_r2){};

  // PROBLEM: Only the RHS size is considered. If LHS size is different??
  // If it is smaller, eval function will crash
  size_t getSize() const { return r2.getSize(); }
};

/*!ScalarOp.
 * @brief Implements an scalar operation.
 * (e.g alpha OP x, with alpha scalar and x vector)
 */
template <typename Functor, typename SCL, typename RHS>
struct ScalarExpr {
  using value_type = typename RHS::value_type;

  SCL scl;
  RHS r;

  ScalarExpr(SCL _scl, RHS &_r) : scl(_scl), r(_r){};

  size_t getSize() const { return r.getSize(); }
};

/*! UnaryOp.
 * Implements a Unary Operation ( Op(z), e.g. z++), with z a vector.
 */
template <typename Functor, typename RHS>
struct UnaryExpr {
  using value_type = typename RHS::value_type;

  RHS r;

  UnaryExpr(RHS &_r) : r(_r){};

  size_t getSize() const { return r.getSize(); }
};

/*! BinaryOp.
 * @brief Implements a Binary Operation (x OP z) with x and z vectors.
 */
template <typename Functor, typename LHS, typename RHS>
struct BinaryExpr {
  using value_type = typename RHS::value_type;

  LHS l;
  RHS r;

  BinaryExpr(LHS &_l, RHS &_r) : l(_l), r(_r){};

  // PROBLEM: Only the RHS size is considered. If LHS size is different??
  // If it is smaller, eval function will crash
  size_t getSize() const { return r.getSize(); }
};

/*! TupleOp.
 * @brief Implements a Tuple Operation (map (\x -> [i, x]) vector).
 */
template <typename RHS>
struct TupleExpr {
  using value_type = IndVal<typename RHS::value_type>;

  RHS r;

  TupleExpr(RHS &_r) : r(_r) {}

  size_t getSize() const { return r.getSize(); }
};
}  // namespace blas

#endif
