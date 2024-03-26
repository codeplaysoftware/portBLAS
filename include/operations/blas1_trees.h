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
 *  portBLAS: BLAS implementation using SYCL
 *
 *  @filename blas1_trees.h
 *
 **************************************************************************/

#ifndef PORTBLAS_BLAS1_TREES_H
#define PORTBLAS_BLAS1_TREES_H
#include "operations/blas_constants.h"
#include "operations/blas_operators.h"
#include <CL/sycl.hpp>
#include <stdexcept>
#include <vector>

namespace blas {
/** Join.
 * @brief Joins both sides of the expression in the single kernel.
 */
template <typename lhs_t, typename rhs_t>
struct Join {
  using index_t = typename rhs_t::index_t;
  using value_t = typename rhs_t::value_t;
  lhs_t lhs_;
  rhs_t rhs_;

  Join(lhs_t &_l, rhs_t _r);
  index_t get_size() const;
  bool valid_thread(cl::sycl::nd_item<1> ndItem) const;
  value_t eval(index_t i);
  value_t eval(cl::sycl::nd_item<1> ndItem);
  void bind(cl::sycl::handler &h);
  void adjust_access_displacement();
};

/** Assign.
 */
template <typename lhs_t, typename rhs_t>
struct Assign {
  using index_t = typename lhs_t::index_t;
  using value_t = typename rhs_t::value_t;
  lhs_t lhs_;
  rhs_t rhs_;
  Assign(lhs_t &_l, rhs_t _r);
  index_t get_size() const;
  bool valid_thread(cl::sycl::nd_item<1> ndItem) const;
  value_t eval(index_t i);
  value_t eval(cl::sycl::nd_item<1> ndItem);
  void bind(cl::sycl::handler &h);
  void adjust_access_displacement();
};

/*! DoubleAssign.
 */
template <typename lhs_1_t, typename lhs_2_t, typename rhs_1_t,
          typename rhs_2_t>
struct DoubleAssign {
  using index_t = typename lhs_1_t::index_t;
  using value_t = typename rhs_1_t::value_t;
  lhs_1_t lhs_1_;
  lhs_2_t lhs_2_;
  rhs_1_t rhs_1_;
  rhs_2_t rhs_2_;
  DoubleAssign(lhs_1_t &_l1, lhs_2_t &_l2, rhs_1_t _r1, rhs_2_t _r2);
  index_t get_size() const;
  bool valid_thread(cl::sycl::nd_item<1> ndItem) const;
  value_t eval(index_t i);
  value_t eval(cl::sycl::nd_item<1> ndItem);
  void bind(cl::sycl::handler &h);
  void adjust_access_displacement();
};

/*!ScalarOp.
 * @brief Implements an scalar operation.
 * (e.g alpha OP x, with alpha scalar and x vector)
 */
template <typename operator_t, typename scalar_t, typename rhs_t>
struct ScalarOp {
  using index_t = typename rhs_t::index_t;
  using value_t = typename ResolveReturnType<operator_t, rhs_t>::type::value_t;
  scalar_t scalar_;
  rhs_t rhs_;
  ScalarOp(scalar_t _scl, rhs_t &_r);
  index_t get_size() const;
  bool valid_thread(cl::sycl::nd_item<1> ndItem) const;
  value_t eval(index_t i);
  value_t eval(cl::sycl::nd_item<1> ndItem);
  void bind(cl::sycl::handler &h);
  void adjust_access_displacement();
};

/*! UnaryOp.
 * Implements a Unary Operation ( operator_t(z), e.g. z++), with z a vector.
 */
template <typename operator_t, typename rhs_t>
struct UnaryOp {
  using index_t = typename rhs_t::index_t;
  using value_t = typename ResolveReturnType<operator_t, rhs_t>::type::value_t;
  rhs_t rhs_;
  UnaryOp(rhs_t &_r);
  index_t get_size() const;
  bool valid_thread(cl::sycl::nd_item<1> ndItem) const;
  value_t eval(index_t i);
  value_t eval(cl::sycl::nd_item<1> ndItem);
  void bind(cl::sycl::handler &h);
  void adjust_access_displacement();
};

/*! BinaryOp.
 * @brief Implements a Binary Operation (x OP z) with x and z vectors.
 */
template <typename operator_t, typename lhs_t, typename rhs_t>
struct BinaryOp {
  using index_t = typename rhs_t::index_t;
  using value_t = typename ResolveReturnType<operator_t, rhs_t>::type::value_t;
  lhs_t lhs_;
  rhs_t rhs_;
  BinaryOp(lhs_t &_l, rhs_t &_r);
  index_t get_size() const;
  bool valid_thread(cl::sycl::nd_item<1> ndItem) const;
  value_t eval(index_t i);
  value_t eval(cl::sycl::nd_item<1> ndItem);
  void bind(cl::sycl::handler &h);
  void adjust_access_displacement();
};

/*! BinaryOpConst.
 * @brief Implements a const Binary Operation (x OP z) with x and z vectors.
 */
template <typename operator_t, typename lhs_t, typename rhs_t>
struct BinaryOpConst {
  using index_t = typename rhs_t::index_t;
  using value_t = typename ResolveReturnType<operator_t, rhs_t>::type::value_t;
  lhs_t lhs_;
  rhs_t rhs_;
  BinaryOpConst(lhs_t &_l, rhs_t &_r);
  index_t get_size() const;
  bool valid_thread(cl::sycl::nd_item<1> ndItem) const;
  value_t eval(index_t i) const;
  value_t eval(cl::sycl::nd_item<1> ndItem) const;
  void bind(cl::sycl::handler &h);
  void adjust_access_displacement();
};

/*! TupleOp.
 * @brief Implements a Tuple Operation (map (\x -> [i, x]) vector).
 */
template <typename rhs_t>
struct TupleOp {
  using index_t = typename rhs_t::index_t;
  using value_t = IndexValueTuple<index_t, typename rhs_t::value_t>;
  rhs_t rhs_;
  TupleOp(rhs_t &_r);
  index_t get_size() const;
  bool valid_thread(cl::sycl::nd_item<1> ndItem) const;
  value_t eval(index_t i);
  value_t eval(cl::sycl::nd_item<1> ndItem);
  void bind(cl::sycl::handler &h);
  void adjust_access_displacement();
};

/*! AssignReduction.
 * @brief Implements the reduction operation for assignments (in the form y
 * = x) with y a scalar and x a subexpression tree.
 */
template <typename operator_t, typename lhs_t, typename rhs_t>
struct AssignReduction {
  using value_t = typename ResolveReturnType<operator_t, rhs_t>::type::value_t;
  using index_t = typename rhs_t::index_t;
  lhs_t lhs_;
  rhs_t rhs_;
  index_t local_num_thread_;   // block  size
  index_t global_num_thread_;  // grid  size
  AssignReduction(lhs_t &_l, rhs_t &_r, index_t _blqS, index_t _grdS);
  index_t get_size() const;
  bool valid_thread(cl::sycl::nd_item<1> ndItem) const;
  value_t eval(index_t i);
  value_t eval(cl::sycl::nd_item<1> ndItem);
  template <typename sharedT>
  value_t eval(sharedT scratch, cl::sycl::nd_item<1> ndItem);
  void bind(cl::sycl::handler &h);
  void adjust_access_displacement();
};

/*! .
 * @brief Generic implementation for operators that require a
 * reduction inside kernel code. (i.e. asum)
 *
 * The class is constructed using the make_wg_atomic_reduction
 * function below.
 *
 */
template <typename operator_t, bool usmManagedMem, typename lhs_t,
          typename rhs_t>
struct WGAtomicReduction {
  using value_t = typename lhs_t::value_t;
  using index_t = typename rhs_t::index_t;
  lhs_t lhs_;
  rhs_t rhs_;
  WGAtomicReduction(lhs_t &_l, rhs_t &_r);
  index_t get_size() const;
  bool valid_thread(cl::sycl::nd_item<1> ndItem) const;
  value_t eval(cl::sycl::nd_item<1> ndItem);
  template <typename sharedT>
  value_t eval(sharedT scratch, cl::sycl::nd_item<1> ndItem);
  void bind(cl::sycl::handler &h);
  void adjust_access_displacement();
};

/**
 * @brief Generic implementation for operators that require a
 * reduction inside kernel code for computing index of max/min value within the
 * input (i.e. iamax and iamin).
 *
 * The class is constructed using the make_index_max_min
 * function below.
 *
 * @tparam is_max Whether the operator is iamax or iamin
 * @tparam is_step0 Decides whether to write IndexValueTuple to output or final
 * index output
 * @tparam lhs_t Buffer or USM memory object type for output memory
 * @tparam rhs_t Buffer or USM memory object type for input memory
 */
template <bool is_max, bool is_step0, typename lhs_t, typename rhs_t>
struct IndexMaxMin {
  using value_t = typename rhs_t::value_t;
  using index_t = typename rhs_t::index_t;
  lhs_t lhs_;
  rhs_t rhs_;
  IndexMaxMin(lhs_t &_l, rhs_t &_r);
  index_t get_size() const;
  bool valid_thread(cl::sycl::nd_item<1> ndItem) const;
  void eval(cl::sycl::nd_item<1> ndItem);
  template <typename sharedT>
  void eval(sharedT scratch, cl::sycl::nd_item<1> ndItem);
  void bind(cl::sycl::handler &h);
  void adjust_access_displacement();
};

/*! Rotg.
 * @brief Implements the rotg (blas level 1 api)
 */
template <typename operand_t>
struct Rotg {
  using value_t = typename operand_t::value_t;
  using index_t = typename operand_t::index_t;
  operand_t a_;
  operand_t b_;
  operand_t c_;
  operand_t s_;
  Rotg(operand_t &a, operand_t &b, operand_t &c, operand_t &s);
  index_t get_size() const;
  value_t eval(index_t i);
  value_t eval(cl::sycl::nd_item<1> ndItem);
  bool valid_thread(cl::sycl::nd_item<1> ndItem) const;
  void bind(cl::sycl::handler &h);
  void adjust_access_displacement();
};

/*! Rotmg.
 * @brief Implements the rotmg (blas level 1 api)
 */
template <typename operand_t>
struct Rotmg {
  using value_t = typename operand_t::value_t;
  using index_t = typename operand_t::index_t;
  operand_t d1_;
  operand_t d2_;
  operand_t x1_;
  operand_t y1_;
  operand_t param_;
  Rotmg(operand_t &d1, operand_t &d2, operand_t &x1, operand_t &y1,
        operand_t &param);
  index_t get_size() const;
  value_t eval(index_t i);
  value_t eval(cl::sycl::nd_item<1> ndItem);
  bool valid_thread(cl::sycl::nd_item<1> ndItem) const;
  void bind(cl::sycl::handler &h);
  void adjust_access_displacement();
};

template <typename operator_t, typename lhs_t, typename rhs_t, typename index_t>
inline AssignReduction<operator_t, lhs_t, rhs_t> make_assign_reduction(
    lhs_t &lhs_, rhs_t &rhs_, index_t local_num_thread_,
    index_t global_num_thread_) {
  return AssignReduction<operator_t, lhs_t, rhs_t>(
      lhs_, rhs_, local_num_thread_, global_num_thread_);
}

template <typename operator_t, bool usmManagedMem = false, typename lhs_t,
          typename rhs_t>
inline WGAtomicReduction<operator_t, usmManagedMem, lhs_t, rhs_t>
make_wg_atomic_reduction(lhs_t &lhs_, rhs_t &rhs_) {
  return WGAtomicReduction<operator_t, usmManagedMem, lhs_t, rhs_t>(lhs_, rhs_);
}

template <bool is_max, bool is_step0, typename lhs_t, typename rhs_t>
inline IndexMaxMin<is_max, is_step0, lhs_t, rhs_t> make_index_max_min(
    lhs_t &lhs_, rhs_t &rhs_) {
  return IndexMaxMin<is_max, is_step0, lhs_t, rhs_t>(lhs_, rhs_);
}

/*!
@brief Template function for constructing operation nodes based on input
template and function arguments. Non-specialized case for N reference operands.
@tparam operation_t Operation type of the operation node.
@tparam operand_t Operand types of the operation node.
@param operands Reference operands of the operation node.
@return Constructed operation node.
*/
template <template <class...> class operation_t, typename... operand_t>
inline operation_t<operand_t...> make_op(operand_t &... operands) {
  return operation_t<operand_t...>(operands...);
}

/*!
@brief Template function for constructing operation nodes based on input
template and function arguments. Specialized case for an operator and N
reference operands.
@tparam operation_t Operation type of the operation node.
@tparam operator_t operator_t type of the operation node.
@tparam operand_t Operand types of the operation node.
@param operands Reference operands of the operation node.
@return Constructed operation node.
*/
template <template <class...> class operation_t, typename operator_t,
          typename... operand_t>
inline operation_t<operator_t, operand_t...> make_op(operand_t &... operands) {
  return operation_t<operator_t, operand_t...>(operands...);
}

/*!
@brief Template function for constructing operation nodes based on input
template and function arguments. Specialized case for an operator, a single by
value operand and N reference operands.
@tparam operation_t Operation type of the operation node.
@tparam operator_t operator_t type of the operation node.
@tparam first_operand_t Operand type of the first by value operand of the
operation node.
@tparam operand_t Operand types of the subsequent reference operands of the
operation node.
@param operand0 First by value operand of the operation node.
@param operands Subsequent reference operands of the operation node.
@return Constructed operation node.
*/
template <template <class...> class operation_t, typename operator_t,
          typename first_operand_t, typename... operand_t>
inline operation_t<operator_t, first_operand_t, operand_t...> make_op(
    first_operand_t operand0, operand_t &... operands) {
  return operation_t<operator_t, first_operand_t, operand_t...>(operand0,
                                                                operands...);
}

template <typename rhs_t>
inline TupleOp<rhs_t> make_tuple_op(rhs_t &rhs_) {
  return TupleOp<rhs_t>(rhs_);
}

}  // namespace blas

#endif  // BLAS1_TREES_H
