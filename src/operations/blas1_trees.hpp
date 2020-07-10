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

/*! DetectScalar.
 * @brief See Detect Scalar.
 */
template <>
struct DetectScalar<cl::sycl::half> {
  using element_t = cl::sycl::half;
  static element_t get_scalar(element_t &scalar) { return scalar; }
};

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
  return ((ndItem.get_global_id(0) < Assign<lhs_t, rhs_t>::get_size()));
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

}  // namespace blas

#endif  // BLAS1_TREES_HPP
