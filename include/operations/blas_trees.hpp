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
 *  @filename blas_trees_expr.hpp
 *
 **************************************************************************/

#ifndef BLAS_TREE_EXPR_HPP
#define BLAS_TREE_EXPR_HPP

#include <operations/blas_constants.hpp>
#include <operations/blas_operators.hpp>

namespace blas {

/*!
 * JoinExpr.
 * @brief An expression for two expressions.
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

/*!
 * AssignExpr.
 * @brief Expression for assignment.
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

/*!
 * DoubleAssignExpr.
 * @brief Expression used to swap two vector views.
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

/*!
 * ScalarExpr.
 * @brief Expression for scalar operation.
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

/*!
 * UnaryExpr.
 * @brief Expression for Unary Operation ( Op(z), e.g. z++), with z a vector.
 */
template <typename Functor, typename RHS>
struct UnaryExpr {
  using value_type = typename RHS::value_type;

  RHS r;

  UnaryExpr(RHS &_r) : r(_r){};

  size_t getSize() const { return r.getSize(); }
};

/*!
 * BinaryExpr.
 * @brief Expression for a Binary Operation (x OP z) with x and z vectors.
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

/*!
 * TupleExpr.
 * @brief Expression for index-dependent operations.
 */
template <typename RHS>
struct TupleExpr {
  using value_type = IndVal<typename RHS::value_type>;

  RHS r;

  TupleExpr(RHS &_r) : r(_r) {}

  size_t getSize() const { return r.getSize(); }
};

/*!
 * BreakExpr
 * @brief Expression for separating execution of a subexpression from the rest
 * of the execution.
 */
template <typename RHS>
struct EmptyExpr {
  using value_type = typename RHS::value_type;

  RHS r;

  EmptyExpr(RHS &_r) : r(_r) {}
  EmptyExpr(RHS &&_r) : EmptyExpr(_r) {}

  size_t getSize() const { return r.getSize(); }
};

/*!
 * BreakExpr
 * @brief Expression for separating execution of a subexpression from the rest
 * of the execution.
 */
template <typename RHS, template <class> class MakePointer>
struct BreakExpr {
  using value_type = typename RHS::value_type;

  RHS r;
  bool use_rhs_result;

  BreakExpr(RHS &_r, bool use_rhs_result = false)
      : r(_r), use_rhs_result(use_rhs_result) {}

  BreakExpr(RHS &&_r, bool use_rhs_result = false)
      : BreakExpr(_r, use_rhs_result) {}

  size_t getSize() const { return r.getSize(); }
};

template <typename RHS, template <class> class MakePointer>
struct BreakIfExpr {
  using value_type = typename RHS::value_type;

  EmptyExpr<RHS> r_empty;
  BreakExpr<RHS, MakePointer> r_break;
  bool to_break;

  BreakIfExpr(RHS &_r, bool to_break)
      : r_empty(_r), r_break(_r), to_break(to_break) {}

  size_t getSize() const { return r_empty.getSize(); }
};

template <typename RHS, template <class> class MakePointer>
struct StrideExpr {
  using value_type = typename RHS::value_type;

  RHS r;
  long offt;
  long strd;
  size_t N;

  StrideExpr(RHS &_r, long offt, long strd, size_t N)
      : r(_r), offt(offt), strd(strd), N(N) {}

  StrideExpr(BreakIfExpr<RHS, MakePointer> &_br, long offt, long strd, size_t N)
      : StrideExpr(_br.r, offt, strd, N) {}

  size_t getSize() const { return N; }
};

template <typename RHS>
StrideExpr<RHS, MakeHostPointer> make_strdExpr(RHS r, long offset, long stride,
                                               size_t N) {
  return StrideExpr<RHS, MakeHostPointer>(r, offset, stride, N);
}

/*!
@brief Template function for constructing expression nodes based on input
template and function arguments. Non-specialised case for N reference
subexpressions.
@tparam expressionT Expression type of the expression node.
@tparam subexprsTN Subexpression types of the oeration node.
@param subexpressions Reference subexpressions of the expression node.
@return Constructed expression node.
*/

template <template <class...> class expressionT, typename... subexprsTN>
expressionT<subexprsTN...> make_expr(subexprsTN... subexprs) {
  return expressionT<subexprsTN...>(subexprs...);
}

/*!
@brief Template function for constructing expression nodes based on input
template and function arguments. Specialised case for an operator and N
reference subexpressions.
@tparam expressionT Expression type of the expression node.
@tparam exprT Expr type of the expression node.
@tparam subexprsTN subexpression types of the expression node.
@param Subexpressions Reference subexpressions of the expression node.
@return Constructed expression node.
*/

template <template <class...> class expressionT, typename exprT,
          typename... subexprsTN>
expressionT<exprT, subexprsTN...> make_expr(subexprsTN... subexprs) {
  return expressionT<exprT, subexprsTN...>(subexprs...);
}

/*!
@brief Template function for constructing expression nodes based on input
template and function arguments. Specialised case for an expression, a single by
value subexpression and N reference subexpressions.
@tparam expressionT Expression type of the expression node.
@tparam exprT Expr type of the expression node.
@tparam subexprT0 Subexpression type of the first by value subexpression of the
expression node.
@tparam subexprsTN Subexpression types of the subsequent reference
subexpressions of
the expression node.
@param subexpr0 First by value subexpression of the expression node.
@param subexprs Subsequent reference subexpressions of the expression node.
@return Constructed expression node.
*/

template <template <class...> class expressionT, typename exprT,
          typename subexprT0, typename... subexprsTN>
expressionT<exprT, subexprT0, subexprsTN...> make_expr(subexprT0 subexpr0,
                                                       subexprsTN... subexprs) {
  return expressionT<exprT, subexprT0, subexprsTN...>(subexpr0, subexprs...);
}

}  // namespace blas

#endif
