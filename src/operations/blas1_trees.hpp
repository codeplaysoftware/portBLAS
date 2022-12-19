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
 *  @filename blas1_trees.hpp
 *
 **************************************************************************/

#ifndef SYCL_BLAS_BLAS1_TREES_HPP
#define SYCL_BLAS_BLAS1_TREES_HPP

#include "operations/blas1_trees.h"
#include "operations/blas_operators.hpp"
#include "views/view_sycl.hpp"
#include <stdexcept>
#include <vector>

namespace blas {
namespace internal {

/*! DetectScalar.
 * @brief Class specialization used to detect scalar values in ScalarOp nodes.
 * When the value is not an integral basic type,
 * it is assumed to be a vector and the first value
 * is used.
 */
template <typename element_t>
struct DetectScalar {
  static typename element_t::value_t get_scalar(element_t &opSCL) {
    return opSCL.eval(0);
  }
};

/*! DetectScalar.
 * @brief See Detect Scalar.
 */
template <>
struct DetectScalar<int> {
  using element_t = int;
  static element_t get_scalar(element_t &scalar) { return scalar; }
};

/*! DetectScalar.
 * @brief See Detect Scalar.
 */
template <>
struct DetectScalar<float> {
  using element_t = float;
  static element_t get_scalar(element_t &scalar) { return scalar; }
};

/*! DetectScalar.
 * @brief See Detect Scalar.
 */
template <>
struct DetectScalar<double> {
  using element_t = double;
  static element_t get_scalar(element_t &scalar) { return scalar; }
};

#ifdef BLAS_DATA_TYPE_HALF
/*! DetectScalar.
 * @brief See Detect Scalar.
 */
template <>
struct DetectScalar<cl::sycl::half> {
  using element_t = cl::sycl::half;
  static element_t get_scalar(element_t &scalar) { return scalar; }
};
#endif  // BLAS_DATA_TYPE_HALF

/*! DetectScalar.
 * @brief See Detect Scalar.
 */
template <>
struct DetectScalar<std::complex<float>> {
  using element_t = std::complex<float>;
  static element_t get_scalar(element_t &scalar) { return scalar; }
};

/*! DetectScalar.
 * @brief See Detect Scalar.
 */
template <>
struct DetectScalar<std::complex<double>> {
  using element_t = std::complex<double>;
  static element_t get_scalar(element_t &scalar) { return scalar; }
};

/*! get_scalar.
 * @brief Template autodecuction function for DetectScalar.
 */
template <typename element_t>
auto get_scalar(element_t &scalar_)
    -> decltype(DetectScalar<element_t>::get_scalar(scalar_)) {
  return DetectScalar<element_t>::get_scalar(scalar_);
}
}  // namespace internal

/** Join.
 * @brief Joins both sides of the expression in the single kernel.
 */
template <typename lhs_t, typename rhs_t>
Join<lhs_t, rhs_t>::Join(lhs_t &_l, rhs_t _r) : lhs_(_l), rhs_(_r) {}

template <typename lhs_t, typename rhs_t>
SYCL_BLAS_INLINE typename Join<lhs_t, rhs_t>::index_t
Join<lhs_t, rhs_t>::get_size() const {
  return rhs_.get_size();
}

template <typename lhs_t, typename rhs_t>
SYCL_BLAS_INLINE bool Join<lhs_t, rhs_t>::valid_thread(
    cl::sycl::nd_item<1> ndItem) const {
  return ((ndItem.get_global_id(0) < Join<lhs_t, rhs_t>::get_size()));
}

template <typename lhs_t, typename rhs_t>
SYCL_BLAS_INLINE typename Join<lhs_t, rhs_t>::value_t Join<lhs_t, rhs_t>::eval(
    typename Join<lhs_t, rhs_t>::index_t i) {
  lhs_.eval(i);
  return rhs_.eval(i);
}

template <typename lhs_t, typename rhs_t>
SYCL_BLAS_INLINE typename Join<lhs_t, rhs_t>::value_t Join<lhs_t, rhs_t>::eval(
    cl::sycl::nd_item<1> ndItem) {
  return Join<lhs_t, rhs_t>::eval(ndItem.get_global_id(0));
}
template <typename lhs_t, typename rhs_t>
SYCL_BLAS_INLINE void Join<lhs_t, rhs_t>::bind(cl::sycl::handler &h) {
  lhs_.bind(h);
  rhs_.bind(h);
}
template <typename lhs_t, typename rhs_t>
SYCL_BLAS_INLINE void Join<lhs_t, rhs_t>::adjust_access_displacement() {
  lhs_.adjust_access_displacement();
  rhs_.adjust_access_displacement();
}

/** Assign.
 */

template <typename lhs_t, typename rhs_t>
Assign<lhs_t, rhs_t>::Assign(lhs_t &_l, rhs_t _r) : lhs_(_l), rhs_(_r){};

template <typename lhs_t, typename rhs_t>
SYCL_BLAS_INLINE typename Assign<lhs_t, rhs_t>::index_t
Assign<lhs_t, rhs_t>::get_size() const {
  return rhs_.get_size();
}

template <typename lhs_t, typename rhs_t>
SYCL_BLAS_INLINE bool Assign<lhs_t, rhs_t>::valid_thread(
    cl::sycl::nd_item<1> ndItem) const {
  using index_t = typename Assign<lhs_t, rhs_t>::index_t;
  return (static_cast<index_t>(ndItem.get_global_id(0)) <
          Assign<lhs_t, rhs_t>::get_size());
}

template <typename lhs_t, typename rhs_t>
SYCL_BLAS_INLINE typename Assign<lhs_t, rhs_t>::value_t
Assign<lhs_t, rhs_t>::eval(typename Assign<lhs_t, rhs_t>::index_t i) {
  auto val = lhs_.eval(i) = rhs_.eval(i);
  return val;
}

template <typename lhs_t, typename rhs_t>
SYCL_BLAS_INLINE typename Assign<lhs_t, rhs_t>::value_t
Assign<lhs_t, rhs_t>::eval(cl::sycl::nd_item<1> ndItem) {
  return Assign<lhs_t, rhs_t>::eval(ndItem.get_global_id(0));
}

template <typename lhs_t, typename rhs_t>
SYCL_BLAS_INLINE void Assign<lhs_t, rhs_t>::bind(cl::sycl::handler &h) {
  lhs_.bind(h);
  rhs_.bind(h);
}
template <typename lhs_t, typename rhs_t>
SYCL_BLAS_INLINE void Assign<lhs_t, rhs_t>::adjust_access_displacement() {
  lhs_.adjust_access_displacement();
  rhs_.adjust_access_displacement();
}

/*! DoubleAssign.
 */
template <typename lhs_1_t, typename lhs_2_t, typename rhs_1_t,
          typename rhs_2_t>
SYCL_BLAS_INLINE DoubleAssign<lhs_1_t, lhs_2_t, rhs_1_t, rhs_2_t>::DoubleAssign(
    lhs_1_t &_l1, lhs_2_t &_l2, rhs_1_t _r1, rhs_2_t _r2)
    : lhs_1_(_l1), lhs_2_(_l2), rhs_1_(_r1), rhs_2_(_r2){};

template <typename lhs_1_t, typename lhs_2_t, typename rhs_1_t,
          typename rhs_2_t>
SYCL_BLAS_INLINE
    typename DoubleAssign<lhs_1_t, lhs_2_t, rhs_1_t, rhs_2_t>::index_t
    DoubleAssign<lhs_1_t, lhs_2_t, rhs_1_t, rhs_2_t>::get_size() const {
  return rhs_2_.get_size();
}

template <typename lhs_1_t, typename lhs_2_t, typename rhs_1_t,
          typename rhs_2_t>
SYCL_BLAS_INLINE bool
DoubleAssign<lhs_1_t, lhs_2_t, rhs_1_t, rhs_2_t>::valid_thread(
    cl::sycl::nd_item<1> ndItem) const {
  return ((ndItem.get_global_id(0) < get_size()));
}

template <typename lhs_1_t, typename lhs_2_t, typename rhs_1_t,
          typename rhs_2_t>
SYCL_BLAS_INLINE
    typename DoubleAssign<lhs_1_t, lhs_2_t, rhs_1_t, rhs_2_t>::value_t
    DoubleAssign<lhs_1_t, lhs_2_t, rhs_1_t, rhs_2_t>::eval(
        typename DoubleAssign<lhs_1_t, lhs_2_t, rhs_1_t, rhs_2_t>::index_t i) {
  auto val1 = rhs_1_.eval(i);
  auto val2 = rhs_2_.eval(i);
  lhs_1_.eval(i) = val1;
  lhs_2_.eval(i) = val2;
  return val1;
}

template <typename lhs_1_t, typename lhs_2_t, typename rhs_1_t,
          typename rhs_2_t>
SYCL_BLAS_INLINE
    typename DoubleAssign<lhs_1_t, lhs_2_t, rhs_1_t, rhs_2_t>::value_t
    DoubleAssign<lhs_1_t, lhs_2_t, rhs_1_t, rhs_2_t>::eval(
        cl::sycl::nd_item<1> ndItem) {
  return DoubleAssign<lhs_1_t, lhs_2_t, rhs_1_t, rhs_2_t>::eval(
      ndItem.get_global_id(0));
}
template <typename lhs_1_t, typename lhs_2_t, typename rhs_1_t,
          typename rhs_2_t>
SYCL_BLAS_INLINE void DoubleAssign<lhs_1_t, lhs_2_t, rhs_1_t, rhs_2_t>::bind(
    cl::sycl::handler &h) {
  lhs_1_.bind(h);
  rhs_1_.bind(h);
  lhs_2_.bind(h);
  rhs_2_.bind(h);
}

template <typename lhs_1_t, typename lhs_2_t, typename rhs_1_t,
          typename rhs_2_t>
SYCL_BLAS_INLINE void
DoubleAssign<lhs_1_t, lhs_2_t, rhs_1_t, rhs_2_t>::adjust_access_displacement() {
  lhs_1_.adjust_access_displacement();
  rhs_1_.adjust_access_displacement();
  lhs_2_.adjust_access_displacement();
  rhs_2_.adjust_access_displacement();
}

/*!ScalarOp.
 * @brief Implements an scalar operation.
 * (e.g alpha OP x, with alpha scalar and x vector)
 */
template <typename operator_t, typename scalar_t, typename rhs_t>
ScalarOp<operator_t, scalar_t, rhs_t>::ScalarOp(scalar_t _scl, rhs_t &_r)
    : scalar_(_scl), rhs_(_r) {}

template <typename operator_t, typename scalar_t, typename rhs_t>
SYCL_BLAS_INLINE typename ScalarOp<operator_t, scalar_t, rhs_t>::index_t
ScalarOp<operator_t, scalar_t, rhs_t>::get_size() const {
  return rhs_.get_size();
}
template <typename operator_t, typename scalar_t, typename rhs_t>
SYCL_BLAS_INLINE bool ScalarOp<operator_t, scalar_t, rhs_t>::valid_thread(
    cl::sycl::nd_item<1> ndItem) const {
  return ((ndItem.get_global_id(0) <
           ScalarOp<operator_t, scalar_t, rhs_t>::get_size()));
}
template <typename operator_t, typename scalar_t, typename rhs_t>
SYCL_BLAS_INLINE typename ScalarOp<operator_t, scalar_t, rhs_t>::value_t
ScalarOp<operator_t, scalar_t, rhs_t>::eval(
    typename ScalarOp<operator_t, scalar_t, rhs_t>::index_t i) {
  return operator_t::eval(internal::get_scalar(scalar_), rhs_.eval(i));
}
template <typename operator_t, typename scalar_t, typename rhs_t>
SYCL_BLAS_INLINE typename ScalarOp<operator_t, scalar_t, rhs_t>::value_t
ScalarOp<operator_t, scalar_t, rhs_t>::eval(cl::sycl::nd_item<1> ndItem) {
  return ScalarOp<operator_t, scalar_t, rhs_t>::eval(ndItem.get_global_id(0));
}
template <typename operator_t, typename scalar_t, typename rhs_t>
SYCL_BLAS_INLINE void ScalarOp<operator_t, scalar_t, rhs_t>::bind(
    cl::sycl::handler &h) {
  rhs_.bind(h);
}

template <typename operator_t, typename scalar_t, typename rhs_t>
SYCL_BLAS_INLINE void
ScalarOp<operator_t, scalar_t, rhs_t>::adjust_access_displacement() {
  rhs_.adjust_access_displacement();
}
/*! UnaryOp.
 * Implements a Unary Operation ( operator_t(z), e.g. z++), with z a vector.
 */
template <typename operator_t, typename rhs_t>
UnaryOp<operator_t, rhs_t>::UnaryOp(rhs_t &_r) : rhs_(_r) {}

template <typename operator_t, typename rhs_t>
SYCL_BLAS_INLINE typename UnaryOp<operator_t, rhs_t>::index_t
UnaryOp<operator_t, rhs_t>::get_size() const {
  return rhs_.get_size();
}

template <typename operator_t, typename rhs_t>
SYCL_BLAS_INLINE bool UnaryOp<operator_t, rhs_t>::valid_thread(
    cl::sycl::nd_item<1> ndItem) const {
  return ((ndItem.get_global_id(0) < UnaryOp<operator_t, rhs_t>::get_size()));
}

template <typename operator_t, typename rhs_t>
SYCL_BLAS_INLINE typename UnaryOp<operator_t, rhs_t>::value_t
UnaryOp<operator_t, rhs_t>::eval(
    typename UnaryOp<operator_t, rhs_t>::index_t i) {
  return operator_t::eval(rhs_.eval(i));
}

template <typename operator_t, typename rhs_t>
SYCL_BLAS_INLINE typename UnaryOp<operator_t, rhs_t>::value_t
UnaryOp<operator_t, rhs_t>::eval(cl::sycl::nd_item<1> ndItem) {
  return UnaryOp<operator_t, rhs_t>::eval(ndItem.get_global_id(0));
}
template <typename operator_t, typename rhs_t>
SYCL_BLAS_INLINE void UnaryOp<operator_t, rhs_t>::bind(cl::sycl::handler &h) {
  rhs_.bind(h);
}
template <typename operator_t, typename rhs_t>
SYCL_BLAS_INLINE void UnaryOp<operator_t, rhs_t>::adjust_access_displacement() {
  rhs_.adjust_access_displacement();
}

/*! BinaryOp.
 * @brief Implements a Binary Operation (x OP z) with x and z vectors.
 */
template <typename operator_t, typename lhs_t, typename rhs_t>
BinaryOp<operator_t, lhs_t, rhs_t>::BinaryOp(lhs_t &_l, rhs_t &_r)
    : lhs_(_l), rhs_(_r){};

template <typename operator_t, typename lhs_t, typename rhs_t>
SYCL_BLAS_INLINE typename BinaryOp<operator_t, lhs_t, rhs_t>::index_t
BinaryOp<operator_t, lhs_t, rhs_t>::get_size() const {
  return rhs_.get_size();
}
template <typename operator_t, typename lhs_t, typename rhs_t>
SYCL_BLAS_INLINE bool BinaryOp<operator_t, lhs_t, rhs_t>::valid_thread(
    cl::sycl::nd_item<1> ndItem) const {
  return ((ndItem.get_global_id(0) < get_size()));
}

template <typename operator_t, typename lhs_t, typename rhs_t>
SYCL_BLAS_INLINE typename BinaryOp<operator_t, lhs_t, rhs_t>::value_t
BinaryOp<operator_t, lhs_t, rhs_t>::eval(
    typename BinaryOp<operator_t, lhs_t, rhs_t>::index_t i) {
  return operator_t::eval(lhs_.eval(i), rhs_.eval(i));
}
template <typename operator_t, typename lhs_t, typename rhs_t>
SYCL_BLAS_INLINE typename BinaryOp<operator_t, lhs_t, rhs_t>::value_t
BinaryOp<operator_t, lhs_t, rhs_t>::eval(cl::sycl::nd_item<1> ndItem) {
  return BinaryOp<operator_t, lhs_t, rhs_t>::eval(ndItem.get_global_id(0));
}
template <typename operator_t, typename lhs_t, typename rhs_t>
SYCL_BLAS_INLINE void BinaryOp<operator_t, lhs_t, rhs_t>::bind(
    cl::sycl::handler &h) {
  lhs_.bind(h);
  rhs_.bind(h);
}

template <typename operator_t, typename lhs_t, typename rhs_t>
SYCL_BLAS_INLINE void
BinaryOp<operator_t, lhs_t, rhs_t>::adjust_access_displacement() {
  lhs_.adjust_access_displacement();
  rhs_.adjust_access_displacement();
}

/*! TupleOp.
 * @brief Implements a Tuple Operation (map (\x -> [i, x]) vector).
 */
template <typename rhs_t>
TupleOp<rhs_t>::TupleOp(rhs_t &_r) : rhs_(_r) {}

template <typename rhs_t>
SYCL_BLAS_INLINE typename TupleOp<rhs_t>::index_t TupleOp<rhs_t>::get_size()
    const {
  return rhs_.get_size();
}

template <typename rhs_t>
SYCL_BLAS_INLINE bool TupleOp<rhs_t>::valid_thread(
    cl::sycl::nd_item<1> ndItem) const {
  return ((ndItem.get_global_id(0) < get_size()));
}
template <typename rhs_t>
SYCL_BLAS_INLINE typename TupleOp<rhs_t>::value_t TupleOp<rhs_t>::eval(
    typename TupleOp<rhs_t>::index_t i) {
  return TupleOp<rhs_t>::value_t(i, rhs_.eval(i));
}

template <typename rhs_t>
SYCL_BLAS_INLINE typename TupleOp<rhs_t>::value_t TupleOp<rhs_t>::eval(
    cl::sycl::nd_item<1> ndItem) {
  return TupleOp<rhs_t>::eval(ndItem.get_global_id(0));
}
template <typename rhs_t>
SYCL_BLAS_INLINE void TupleOp<rhs_t>::bind(cl::sycl::handler &h) {
  rhs_.bind(h);
}
template <typename rhs_t>
SYCL_BLAS_INLINE void TupleOp<rhs_t>::adjust_access_displacement() {
  rhs_.adjust_access_displacement();
}

/*! AssignReduction.
 * @brief Implements the reduction operation for assignments (in the form y
 * = x) with y a scalar and x a subexpression tree.
 */
template <typename operator_t, typename lhs_t, typename rhs_t>
AssignReduction<operator_t, lhs_t, rhs_t>::AssignReduction(lhs_t &_l, rhs_t &_r,
                                                           index_t _blqS,
                                                           index_t _grdS)
    : lhs_(_l), rhs_(_r), local_num_thread_(_blqS), global_num_thread_(_grdS){};

template <typename operator_t, typename lhs_t, typename rhs_t>
SYCL_BLAS_INLINE typename AssignReduction<operator_t, lhs_t, rhs_t>::index_t
AssignReduction<operator_t, lhs_t, rhs_t>::get_size() const {
  return rhs_.get_size();
}

template <typename operator_t, typename lhs_t, typename rhs_t>
SYCL_BLAS_INLINE bool AssignReduction<operator_t, lhs_t, rhs_t>::valid_thread(
    cl::sycl::nd_item<1> ndItem) const {
  return true;
}
template <typename operator_t, typename lhs_t, typename rhs_t>
SYCL_BLAS_INLINE typename AssignReduction<operator_t, lhs_t, rhs_t>::value_t
AssignReduction<operator_t, lhs_t, rhs_t>::eval(
    typename AssignReduction<operator_t, lhs_t, rhs_t>::index_t i) {
  index_t vecS = rhs_.get_size();
  index_t frs_thrd = 2 * local_num_thread_ * i;
  index_t lst_thrd = ((frs_thrd + local_num_thread_) > vecS)
                         ? vecS
                         : (frs_thrd + local_num_thread_);
  // this will be computed at compile time
  static constexpr value_t init_val = operator_t::template init<rhs_t>();
  // Reduction across the grid
  value_t val = init_val;
  for (index_t j = frs_thrd; j < lst_thrd; j++) {
    value_t local_val = init_val;
    for (index_t k = j; k < vecS; k += 2 * global_num_thread_) {
      local_val = operator_t::eval(local_val, rhs_.eval(k));
      if (k + local_num_thread_ < vecS) {
        local_val =
            operator_t::eval(local_val, rhs_.eval(k + local_num_thread_));
      }
    }
    // Reduction inside the block
    val = operator_t::eval(val, local_val);
  }
  if (i < lhs_.get_size()) {
    lhs_.eval(i) = val;
  }
  return val;
}
template <typename operator_t, typename lhs_t, typename rhs_t>
SYCL_BLAS_INLINE typename AssignReduction<operator_t, lhs_t, rhs_t>::value_t
AssignReduction<operator_t, lhs_t, rhs_t>::eval(cl::sycl::nd_item<1> ndItem) {
  return AssignReduction<operator_t, lhs_t, rhs_t>::eval(
      ndItem.get_global_id(0));
}
template <typename operator_t, typename lhs_t, typename rhs_t>
template <typename sharedT>
SYCL_BLAS_INLINE typename AssignReduction<operator_t, lhs_t, rhs_t>::value_t
AssignReduction<operator_t, lhs_t, rhs_t>::eval(sharedT scratch,
                                                cl::sycl::nd_item<1> ndItem) {
  index_t localid = ndItem.get_local_id(0);
  index_t localSz = ndItem.get_local_range(0);
  index_t groupid = ndItem.get_group(0);

  index_t vecS = rhs_.get_size();
  index_t frs_thrd = 2 * groupid * localSz + localid;

  // Reduction across the grid
  // TODO(Peter): This should be constexpr once half supports it
  static const value_t init_val = operator_t::template init<rhs_t>();
  value_t val = init_val;
  for (index_t k = frs_thrd; k < vecS; k += 2 * global_num_thread_) {
    val = operator_t::eval(val, rhs_.eval(k));
    if ((k + local_num_thread_ < vecS)) {
      val = operator_t::eval(val, rhs_.eval(k + local_num_thread_));
    }
  }

  scratch[localid] = val;
  // This barrier is mandatory to be sure the data is on the shared memory
  ndItem.barrier(cl::sycl::access::fence_space::local_space);

  // Reduction inside the block
  for (index_t offset = localSz >> 1; offset > 0; offset >>= 1) {
    if (localid < offset) {
      scratch[localid] =
          operator_t::eval(scratch[localid], scratch[localid + offset]);
    }
    // This barrier is mandatory to be sure the data are on the shared memory
    ndItem.barrier(cl::sycl::access::fence_space::local_space);
  }
  if (localid == 0) {
    lhs_.eval(groupid) = scratch[localid];
  }
  return lhs_.eval(groupid);
}

template <typename operator_t, typename lhs_t, typename rhs_t>
SYCL_BLAS_INLINE void AssignReduction<operator_t, lhs_t, rhs_t>::bind(
    cl::sycl::handler &h) {
  lhs_.bind(h);
  rhs_.bind(h);
}

template <typename operator_t, typename lhs_t, typename rhs_t>
SYCL_BLAS_INLINE void
AssignReduction<operator_t, lhs_t, rhs_t>::adjust_access_displacement() {
  lhs_.adjust_access_displacement();
  rhs_.adjust_access_displacement();
}

template <typename operand_t>
Rotg<operand_t>::Rotg(operand_t &_a, operand_t &_b, operand_t &_c,
                      operand_t &_s)
    : a_{_a}, b_{_b}, c_{_c}, s_{_s} {}

template <typename operand_t>
SYCL_BLAS_INLINE typename Rotg<operand_t>::index_t Rotg<operand_t>::get_size()
    const {
  return static_cast<Rotg<operand_t>::index_t>(1);
}

template <typename operand_t>
SYCL_BLAS_INLINE typename Rotg<operand_t>::value_t Rotg<operand_t>::eval(
    typename Rotg<operand_t>::index_t i) {
  using zero = constant<value_t, const_val::zero>;
  using one = constant<value_t, const_val::one>;

  value_t &a_ref = a_.eval(i);
  value_t &b_ref = b_.eval(i);
  value_t &c_ref = c_.eval(i);
  value_t &s_ref = s_.eval(i);

  const value_t abs_a = AbsoluteValue::eval(a_ref);
  const value_t abs_b = AbsoluteValue::eval(b_ref);
  const value_t sigma =
      abs_a > abs_b ? SignOperator::eval(a_ref) : SignOperator::eval(b_ref);
  const value_t r =
      ProductOperator::eval(sigma, HypotenuseOperator::eval(a_ref, b_ref));

  if (r == zero::value()) {
    c_ref = one::value();
    s_ref = zero::value();
  } else {
    c_ref = DivisionOperator::eval(a_ref, r);
    s_ref = DivisionOperator::eval(b_ref, r);
  }
  a_ref = r;

  /* Calculate z and assign it to parameter b */
  if (abs_a >= abs_b) {
    /* Documentation says that the comparison should be ">" but reference
     * implementation seems to be using ">=" */
    b_ref = s_ref;
  } else if (c_ref != zero::value()) {
    b_ref = DivisionOperator::eval(one::value(), c_ref);
  } else {
    b_ref = one::value();
  }

  // The return value of rotg is void but eval expects something to be returned.
  return zero::value();
}

template <typename operand_t>
SYCL_BLAS_INLINE typename Rotg<operand_t>::value_t Rotg<operand_t>::eval(
    cl::sycl::nd_item<1> ndItem) {
  return Rotg<operand_t>::eval(ndItem.get_global_id(0));
}

template <typename operand_t>
SYCL_BLAS_INLINE bool Rotg<operand_t>::valid_thread(
    cl::sycl::nd_item<1> ndItem) const {
  return ((ndItem.get_global_id(0) < Rotg<operand_t>::get_size()));
}

template <typename operand_t>
SYCL_BLAS_INLINE void Rotg<operand_t>::bind(cl::sycl::handler &h) {
  a_.bind(h);
  b_.bind(h);
  c_.bind(h);
  s_.bind(h);
}

template <typename operand_t>
SYCL_BLAS_INLINE void Rotg<operand_t>::adjust_access_displacement() {
  a_.adjust_access_displacement();
  b_.adjust_access_displacement();
  c_.adjust_access_displacement();
  s_.adjust_access_displacement();
}

template <typename operand_t>
Rotmg<operand_t>::Rotmg(operand_t &_d1, operand_t &_d2, operand_t &_x1,
                        operand_t &_y1, operand_t &_param)
    : d1_{_d1}, d2_{_d2}, x1_{_x1}, y1_{_y1}, param_{_param} {}

template <typename operand_t>
SYCL_BLAS_INLINE typename Rotmg<operand_t>::index_t Rotmg<operand_t>::get_size()
    const {
  return static_cast<Rotmg<operand_t>::index_t>(1);
}

/**
 * For further details about the rotmg algorithm refer to:
 *
 * Hopkins, Tim (1998) Restructuring the BLAS Level 1 Routine for Computing the
 * Modified Givens Transformation. Technical report. , Canterbury, Kent, UK.
 *
 * and
 *
 * C. L. Lawson, R. J. Hanson, D. R. Kincaid, and F. T. Krogh. Basic linear
 * algebra subprograms for Fortran usage. ACM Trans. Math. Softw., 5:308-323,
 * 1979.
 */
template <typename operand_t>
SYCL_BLAS_INLINE typename Rotmg<operand_t>::value_t Rotmg<operand_t>::eval(
    typename Rotmg<operand_t>::index_t i) {
  using zero = constant<value_t, const_val::zero>;
  using one = constant<value_t, const_val::one>;
  using two = constant<value_t, const_val::two>;
  using m_one = constant<value_t, const_val::m_one>;
  using m_two = constant<value_t, const_val::m_two>;

  using error = two;

  using clts_flag = one;  /* co-sin less than sin */
  using sltc_flag = zero; /* sin less than co-sin */
  using rescaled_flag = m_one;
  using unit_flag = m_two;

  /* Gamma is a magic number used to re-scale the output and avoid underflows or
   * overflows. Consult the papers above for more info */
  constexpr value_t gamma = static_cast<value_t>(4096.0);

  /* Square of gamma. */
  constexpr value_t gamma_sq = gamma * gamma;

  /* Inverse of the square of gamma (i.e. 1 / (gamma * gamma)) */
  constexpr value_t inv_gamma_sq = static_cast<value_t>(1.0) / gamma_sq;

  value_t &d1_ref = d1_.eval(i);
  value_t &d2_ref = d2_.eval(i);
  value_t &x1_ref = x1_.eval(i);
  value_t &flag_ref = param_.eval(static_cast<index_t>(0));
  value_t &h11_ref = param_.eval(static_cast<index_t>(1));
  value_t &h21_ref = param_.eval(static_cast<index_t>(2));
  value_t &h12_ref = param_.eval(static_cast<index_t>(3));
  value_t &h22_ref = param_.eval(static_cast<index_t>(4));

  value_t d1 = d1_ref;
  value_t d2 = d2_ref;
  value_t x1 = x1_ref;
  const value_t y1 = y1_.eval(i);

  value_t flag;
  value_t h11;
  value_t h21;
  value_t h12;
  value_t h22;
  value_t swap_temp;

  /* d1 cannot be negative.
   * Note: negative d2 is valid as a way to implement row removal */
  if (d1 < zero::value()) {
    flag = error::value();
  }
  /* If the input is of the form (c, 0), then we already have the expected
   * output. No calculations needed in this case */
  else if (d2 == zero::value() || y1 == zero::value()) {
    flag = unit_flag::value();
  }
  /* If the input is of the form (0, c) - just need to swap the elements.
   * Scaling may be needed */
  else if ((d1 == zero::value() || x1 == zero::value()) && d2 > zero::value()) {
    flag = clts_flag::value();
    /* clts_flag assumes h12 and h21 values. But they still need to be set
     * because of possible re-scaling */
    h12 = one::value();
    h21 = m_one::value();

    h11 = zero::value();
    h22 = zero::value();

    x1 = y1;
    swap_temp = d1;
    d1 = d2;
    d2 = swap_temp;

  } else {
    const value_t p1 = d1 * x1;
    const value_t p2 = d2 * y1;
    const value_t c = p1 * x1;
    const value_t s = p2 * y1;
    const value_t abs_c = AbsoluteValue::eval(c);
    const value_t abs_s = AbsoluteValue::eval(s);
    value_t u;

    if (abs_c > abs_s) {
      flag = sltc_flag::value();
      /* sltc_flag assumes h11 and h22 values. But they still need to be set
       * because of possible re-scaling */
      h11 = one::value();
      h22 = one::value();

      h21 = -y1 / x1;
      h12 = p2 / p1;
      u = one::value() - h12 * h21;

      /* If u underflowed exit with error */
      if (u <= zero::value()) {
        flag = error::value();
      } else {
        d1 = d1 / u;
        d2 = d2 / u;
        x1 = x1 * u;
      }
    } else {
      if (s < zero::value()) {
        flag = error::value();
      } else {
        flag = clts_flag::value();

        /* clts_flag assumes h12 and h21 values. But they still need to be set
         * because of possible re-scaling */
        h12 = one::value();
        h21 = m_one::value();
        h11 = p1 / p2;
        h22 = x1 / y1;

        u = one::value() + h11 * h22;

        /* The original algorithm assumes that d2 will never be negative at this
         * point. However, that is not true for some inputs which cause p2 to
         * underflow (i.e. if d2 and y1 are very small) */
        if (u < one::value()) {
          flag = error::value();
        }

        swap_temp = d1 / u;
        d1 = d2 / u;
        d2 = swap_temp;
        x1 = y1 * u;
      }
    }
  }

  /* Rescale to avoid underflows and overflows:
   * If necessary, apply scaling to d1 and d2 to avoid underflows or overflows.
   * If rescaling happens, then x1 and the calculated matrix values also need
   * to be rescaled to keep the math valid */
  if (flag != error::value() && flag != unit_flag::value()) {
    /* Avoid d1 underflow */
    while (AbsoluteValue::eval(d1) <= inv_gamma_sq && d1 != zero::value()) {
      flag = rescaled_flag::value();
      d1 = (d1 * gamma) * gamma;
      x1 = x1 / gamma;
      h11 = h11 / gamma;
      h12 = h12 / gamma;
    }

    /* Avoid d1 overflow */
    while (AbsoluteValue::eval(d1) > gamma_sq) {
      flag = rescaled_flag::value();
      d1 = (d1 / gamma) / gamma;
      x1 = x1 * gamma;
      h11 = h11 * gamma;
      h12 = h12 * gamma;
    }

    /* Avoid d2 underflow */
    while (AbsoluteValue::eval(d2) <= inv_gamma_sq && d2 != zero::value()) {
      flag = rescaled_flag::value();
      d2 = (d2 * gamma) * gamma;
      h21 = h21 / gamma;
      h22 = h22 / gamma;
    }

    /* Avoid d2 overflow */
    while (AbsoluteValue::eval(d2) > gamma_sq) {
      flag = rescaled_flag::value();
      d2 = (d2 / gamma) / gamma;
      h21 = h21 * gamma;
      h22 = h22 * gamma;
    }
  }

  /* Copy algorithm output to global memory */
  if (flag == error::value()) {
    h11_ref = zero::value();
    h12_ref = zero::value();
    h21_ref = zero::value();
    h22_ref = zero::value();
    d1_ref = zero::value();
    d2_ref = zero::value();
    x1_ref = zero::value();
  } else {
    d1_ref = d1;
    d2_ref = d2;
    x1_ref = x1;

    if (flag == unit_flag::value()) {
      h11_ref = one::value();
      h12_ref = zero::value();
      h21_ref = zero::value();
      h22_ref = one::value();
    } else if (flag == sltc_flag::value()) {
      h11_ref = one::value();
      h12_ref = h12;
      h21_ref = h21;
      h22_ref = one::value();
    } else if (flag == clts_flag::value()) {
      h11_ref = h11;
      h12_ref = one::value();
      h21_ref = m_one::value();
      h22_ref = h22;
    } else {
      h11_ref = h11;
      h12_ref = h12;
      h21_ref = h21;
      h22_ref = h22;
    }
  }
  flag_ref = flag;

  // The return value of rotmg is void but eval expects something to be
  // returned.
  return zero::value();
}

template <typename operand_t>
SYCL_BLAS_INLINE typename Rotmg<operand_t>::value_t Rotmg<operand_t>::eval(
    cl::sycl::nd_item<1> ndItem) {
  return Rotmg<operand_t>::eval(ndItem.get_global_id(0));
}

template <typename operand_t>
SYCL_BLAS_INLINE bool Rotmg<operand_t>::valid_thread(
    cl::sycl::nd_item<1> ndItem) const {
  return ((ndItem.get_global_id(0) < Rotmg<operand_t>::get_size()));
}

template <typename operand_t>
SYCL_BLAS_INLINE void Rotmg<operand_t>::bind(cl::sycl::handler &h) {
  d1_.bind(h);
  d2_.bind(h);
  x1_.bind(h);
  y1_.bind(h);
  param_.bind(h);
}

template <typename operand_t>
SYCL_BLAS_INLINE void Rotmg<operand_t>::adjust_access_displacement() {
  d1_.adjust_access_displacement();
  d2_.adjust_access_displacement();
  x1_.adjust_access_displacement();
  y1_.adjust_access_displacement();
  param_.adjust_access_displacement();
}

}  // namespace blas

#endif  // BLAS1_TREES_HPP
