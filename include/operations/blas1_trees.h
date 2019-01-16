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
 *  @filename blas1_trees.h
 *
 **************************************************************************/

#ifndef SYCL_BLAS_BLAS1_TREES_H
#define SYCL_BLAS_BLAS1_TREES_H
#include "operations/blas_constants.h"
#include <CL/sycl.hpp>
#include <stdexcept>
#include <vector>

namespace blas {
/** Join.
 * @brief Joins both sides of the expression in the single kernel.
 */
template <class LHS, class RHS>
struct Join {
  using IndexType = typename RHS::IndexType;
  using value_type = typename RHS::value_type;
  LHS l;
  RHS r;

  Join(LHS &_l, RHS _r);
  IndexType getSize() const;
  bool valid_thread(cl::sycl::nd_item<1> ndItem) const;
  value_type eval(IndexType i);
  value_type eval(cl::sycl::nd_item<1> ndItem);
  void bind(cl::sycl::handler &h);
};

/** Assign.
 */
template <class LHS, class RHS>
struct Assign {
  using IndexType = typename LHS::IndexType;
  using value_type = typename RHS::value_type;
  LHS l;
  RHS r;
  Assign(LHS &_l, RHS _r);
  IndexType getSize() const;
  bool valid_thread(cl::sycl::nd_item<1> ndItem) const;
  value_type eval(IndexType i);
  value_type eval(cl::sycl::nd_item<1> ndItem);
  void bind(cl::sycl::handler &h);
};

/*! DoubleAssign.
 */
template <class LHS1, class LHS2, class RHS1, class RHS2>
struct DoubleAssign {
  using IndexType = typename LHS1::IndexType;
  using value_type = typename RHS1::value_type;
  LHS1 l1;
  LHS2 l2;
  RHS1 r1;
  RHS2 r2;
  DoubleAssign(LHS1 &_l1, LHS2 &_l2, RHS1 _r1, RHS2 _r2);
  IndexType getSize() const;
  bool valid_thread(cl::sycl::nd_item<1> ndItem) const;
  value_type eval(IndexType i);
  value_type eval(cl::sycl::nd_item<1> ndItem);
  void bind(cl::sycl::handler &h);
};

/*!ScalarOp.
 * @brief Implements an scalar operation.
 * (e.g alpha OP x, with alpha scalar and x vector)
 */
template <typename Operator, typename SCL, typename RHS>
struct ScalarOp {
  using IndexType = typename RHS::IndexType;
  using value_type = typename RHS::value_type;
  SCL scl;
  RHS r;
  ScalarOp(SCL _scl, RHS &_r);
  IndexType getSize() const;
  bool valid_thread(cl::sycl::nd_item<1> ndItem) const;
  value_type eval(IndexType i);
  value_type eval(cl::sycl::nd_item<1> ndItem);
  void bind(cl::sycl::handler &h);
};

/*! UnaryOp.
 * Implements a Unary Operation ( Op(z), e.g. z++), with z a vector.
 */
template <typename Operator, typename RHS>
struct UnaryOp {
  using IndexType = typename RHS::IndexType;
  using value_type = typename RHS::value_type;
  RHS r;
  UnaryOp(RHS &_r);
  IndexType getSize() const;
  bool valid_thread(cl::sycl::nd_item<1> ndItem) const;
  value_type eval(IndexType i);
  value_type eval(cl::sycl::nd_item<1> ndItem);
  void bind(cl::sycl::handler &h);
};

/*! BinaryOp.
 * @brief Implements a Binary Operation (x OP z) with x and z vectors.
 */
template <typename Operator, typename LHS, typename RHS>
struct BinaryOp {
  using IndexType = typename RHS::IndexType;
  using value_type = typename RHS::value_type;
  LHS l;
  RHS r;
  BinaryOp(LHS &_l, RHS &_r);
  IndexType getSize() const;
  bool valid_thread(cl::sycl::nd_item<1> ndItem) const;
  value_type eval(IndexType i);
  value_type eval(cl::sycl::nd_item<1> ndItem);
  void bind(cl::sycl::handler &h);
};

/*! TupleOp.
 * @brief Implements a Tuple Operation (map (\x -> [i, x]) vector).
 */
template <typename RHS>
struct TupleOp {
  using IndexType = typename RHS::IndexType;
  using value_type = IndexValueTuple<typename RHS::value_type, IndexType>;
  RHS r;
  TupleOp(RHS &_r);
  IndexType getSize() const;
  bool valid_thread(cl::sycl::nd_item<1> ndItem) const;
  value_type eval(IndexType i);
  value_type eval(cl::sycl::nd_item<1> ndItem);
  void bind(cl::sycl::handler &h);
};

/*! AssignReduction.
 * @brief Implements the reduction operation for assignments (in the form y
 * = x) with y a scalar and x a subexpression tree.
 */
template <typename Operator, class LHS, class RHS>
struct AssignReduction {
  using value_type = typename RHS::value_type;
  using IndexType = typename RHS::IndexType;
  LHS l;
  RHS r;
  IndexType blqS;  // block  size
  IndexType grdS;  // grid  size
  AssignReduction(LHS &_l, RHS &_r, IndexType _blqS, IndexType _grdS);
  IndexType getSize() const;
  bool valid_thread(cl::sycl::nd_item<1> ndItem) const;
  value_type eval(IndexType i);
  value_type eval(cl::sycl::nd_item<1> ndItem);
  template <typename sharedT>
  value_type eval(sharedT scratch, cl::sycl::nd_item<1> ndItem);
  void bind(cl::sycl::handler &h);
};

template <typename Operator, typename LHS, typename RHS, typename IndexType>
inline AssignReduction<Operator, LHS, RHS> make_AssignReduction(
    LHS &l, RHS &r, IndexType blqS, IndexType grdS) {
  return AssignReduction<Operator, LHS, RHS>(l, r, blqS, grdS);
}

/*!
@brief Template function for constructing operation nodes based on input
template and function arguments. Non-specialized case for N reference operands.
@tparam operationT Operation type of the operation node.
@tparam operandsTN Operand types of the operation node.
@param operands Reference operands of the operation node.
@return Constructed operation node.
*/
template <template <class...> class operationT, typename... operandsTN>
inline operationT<operandsTN...> make_op(operandsTN &... operands) {
  return operationT<operandsTN...>(operands...);
}

/*!
@brief Template function for constructing operation nodes based on input
template and function arguments. Specialized case for an operator and N
reference operands.
@tparam operationT Operation type of the operation node.
@tparam operatorT Operator type of the operation node.
@tparam operandsTN Operand types of the operation node.
@param operands Reference operands of the operation node.
@return Constructed operation node.
*/
template <template <class...> class operationT, typename operatorT,
          typename... operandsTN>
inline operationT<operatorT, operandsTN...> make_op(operandsTN &... operands) {
  return operationT<operatorT, operandsTN...>(operands...);
}

/*!
@brief Template function for constructing operation nodes based on input
template and function arguments. Specialized case for an operator, a single by
value operand and N reference operands.
@tparam operationT Operation type of the operation node.
@tparam operatorT Operator type of the operation node.
@tparam operandT0 Operand type of the first by value operand of the operation
node.
@tparam operandsTN Operand types of the subsequent reference operands of the
operation node.
@param operand0 First by value operand of the operation node.
@param operands Subsequent reference operands of the operation node.
@return Constructed operation node.
*/
template <template <class...> class operationT, typename operatorT,
          typename operandT0, typename... operandsTN>
inline operationT<operatorT, operandT0, operandsTN...> make_op(
    operandT0 operand0, operandsTN &... operands) {
  return operationT<operatorT, operandT0, operandsTN...>(operand0, operands...);
}

template <typename RHS>
inline TupleOp<RHS> make_tuple_op(RHS &r) {
  return TupleOp<RHS>(r);
}

}  // namespace blas

#endif  // BLAS1_TREES_H
