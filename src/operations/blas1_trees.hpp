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
template <typename T>
struct DetectScalar {
  static typename T::value_type get_scalar(T &opSCL) { return opSCL.eval(0); }
};

/*! DetectScalar.
 * @brief See Detect Scalar.
 */
template <>
struct DetectScalar<int> {
  using T = int;
  static T get_scalar(T &scalar) { return scalar; }
};

/*! DetectScalar.
 * @brief See Detect Scalar.
 */
template <>
struct DetectScalar<float> {
  using T = float;
  static T get_scalar(T &scalar) { return scalar; }
};

/*! DetectScalar.
 * @brief See Detect Scalar.
 */
template <>
struct DetectScalar<double> {
  using T = double;
  static T get_scalar(T &scalar) { return scalar; }
};

/*! DetectScalar.
 * @brief See Detect Scalar.
 */
template <>
struct DetectScalar<std::complex<float>> {
  using T = std::complex<float>;
  static T get_scalar(T &scalar) { return scalar; }
};

/*! DetectScalar.
 * @brief See Detect Scalar.
 */
template <>
struct DetectScalar<std::complex<double>> {
  using T = std::complex<double>;
  static T get_scalar(T &scalar) { return scalar; }
};

/*! get_scalar.
 * @brief Template autodecuction function for DetectScalar.
 */
template <typename T>
auto get_scalar(T &scl) -> decltype(DetectScalar<T>::get_scalar(scl)) {
  return DetectScalar<T>::get_scalar(scl);
}
}  // namespace internal

/** Join.
 * @brief Joins both sides of the expression in the single kernel.
 */
template <class LHS, class RHS>
Join<LHS, RHS>::Join(LHS &_l, RHS _r) : l(_l), r(_r) {}

template <class LHS, class RHS>
sycl_blas_inline typename Join<LHS, RHS>::IndexType Join<LHS, RHS>::getSize()
    const {
  return r.getSize();
}

template <class LHS, class RHS>
sycl_blas_inline bool Join<LHS, RHS>::valid_thread(
    cl::sycl::nd_item<1> ndItem) const {
  return ((ndItem.get_global_id(0) < Join<LHS, RHS>::getSize()));
}

template <class LHS, class RHS>
sycl_blas_inline typename Join<LHS, RHS>::value_type Join<LHS, RHS>::eval(
    typename Join<LHS, RHS>::IndexType i) {
  l.eval(i);
  return r.eval(i);
}

template <class LHS, class RHS>
sycl_blas_inline typename Join<LHS, RHS>::value_type Join<LHS, RHS>::eval(
    cl::sycl::nd_item<1> ndItem) {
  return Join<LHS, RHS>::eval(ndItem.get_global_id(0));
}
template <class LHS, class RHS>
sycl_blas_inline void Join<LHS, RHS>::bind(cl::sycl::handler &h) {
  l.bind(h);
  r.bind(h);
}

/** Assign.
 */

template <class LHS, class RHS>
Assign<LHS, RHS>::Assign(LHS &_l, RHS _r) : l(_l), r(_r){};

template <class LHS, class RHS>
sycl_blas_inline typename Assign<LHS, RHS>::IndexType
Assign<LHS, RHS>::getSize() const {
  return r.getSize();
}

template <class LHS, class RHS>
sycl_blas_inline bool Assign<LHS, RHS>::valid_thread(
    cl::sycl::nd_item<1> ndItem) const {
  return ((ndItem.get_global_id(0) < Assign<LHS, RHS>::getSize()));
}

template <class LHS, class RHS>
sycl_blas_inline typename Assign<LHS, RHS>::value_type Assign<LHS, RHS>::eval(
    typename Assign<LHS, RHS>::IndexType i) {
  auto val = l.eval(i) = r.eval(i);
  return val;
}

template <class LHS, class RHS>
sycl_blas_inline typename Assign<LHS, RHS>::value_type Assign<LHS, RHS>::eval(
    cl::sycl::nd_item<1> ndItem) {
  return Assign<LHS, RHS>::eval(ndItem.get_global_id(0));
}

template <class LHS, class RHS>
sycl_blas_inline void Assign<LHS, RHS>::bind(cl::sycl::handler &h) {
  l.bind(h);
  r.bind(h);
}

/*! DoubleAssign.
 */
template <class LHS1, class LHS2, class RHS1, class RHS2>
sycl_blas_inline DoubleAssign<LHS1, LHS2, RHS1, RHS2>::DoubleAssign(LHS1 &_l1,
                                                                    LHS2 &_l2,
                                                                    RHS1 _r1,
                                                                    RHS2 _r2)
    : l1(_l1), l2(_l2), r1(_r1), r2(_r2){};

template <class LHS1, class LHS2, class RHS1, class RHS2>
sycl_blas_inline typename DoubleAssign<LHS1, LHS2, RHS1, RHS2>::IndexType
DoubleAssign<LHS1, LHS2, RHS1, RHS2>::getSize() const {
  return r2.getSize();
}

template <class LHS1, class LHS2, class RHS1, class RHS2>
sycl_blas_inline bool DoubleAssign<LHS1, LHS2, RHS1, RHS2>::valid_thread(
    cl::sycl::nd_item<1> ndItem) const {
  return ((ndItem.get_global_id(0) < getSize()));
}

template <class LHS1, class LHS2, class RHS1, class RHS2>
sycl_blas_inline typename DoubleAssign<LHS1, LHS2, RHS1, RHS2>::value_type
DoubleAssign<LHS1, LHS2, RHS1, RHS2>::eval(
    typename DoubleAssign<LHS1, LHS2, RHS1, RHS2>::IndexType i) {
  auto val1 = r1.eval(i);
  auto val2 = r2.eval(i);
  l1.eval(i) = val1;
  l2.eval(i) = val2;
  return val1;
}

template <class LHS1, class LHS2, class RHS1, class RHS2>
sycl_blas_inline typename DoubleAssign<LHS1, LHS2, RHS1, RHS2>::value_type
DoubleAssign<LHS1, LHS2, RHS1, RHS2>::eval(cl::sycl::nd_item<1> ndItem) {
  return DoubleAssign<LHS1, LHS2, RHS1, RHS2>::eval(ndItem.get_global_id(0));
}
template <class LHS1, class LHS2, class RHS1, class RHS2>
sycl_blas_inline void DoubleAssign<LHS1, LHS2, RHS1, RHS2>::bind(
    cl::sycl::handler &h) {
  l1.bind(h);
  r1.bind(h);
  l2.bind(h);
  r2.bind(h);
}

/*!ScalarOp.
 * @brief Implements an scalar operation.
 * (e.g alpha OP x, with alpha scalar and x vector)
 */
template <typename Operator, typename SCL, typename RHS>
ScalarOp<Operator, SCL, RHS>::ScalarOp(SCL _scl, RHS &_r) : scl(_scl), r(_r) {}

template <typename Operator, typename SCL, typename RHS>
sycl_blas_inline typename ScalarOp<Operator, SCL, RHS>::IndexType
ScalarOp<Operator, SCL, RHS>::getSize() const {
  return r.getSize();
}
template <typename Operator, typename SCL, typename RHS>
sycl_blas_inline bool ScalarOp<Operator, SCL, RHS>::valid_thread(
    cl::sycl::nd_item<1> ndItem) const {
  return ((ndItem.get_global_id(0) < ScalarOp<Operator, SCL, RHS>::getSize()));
}
template <typename Operator, typename SCL, typename RHS>
sycl_blas_inline typename ScalarOp<Operator, SCL, RHS>::value_type
ScalarOp<Operator, SCL, RHS>::eval(
    typename ScalarOp<Operator, SCL, RHS>::IndexType i) {
  return Operator::eval(internal::get_scalar(scl), r.eval(i));
}
template <typename Operator, typename SCL, typename RHS>
sycl_blas_inline typename ScalarOp<Operator, SCL, RHS>::value_type
ScalarOp<Operator, SCL, RHS>::eval(cl::sycl::nd_item<1> ndItem) {
  return ScalarOp<Operator, SCL, RHS>::eval(ndItem.get_global_id(0));
}
template <typename Operator, typename SCL, typename RHS>
sycl_blas_inline void ScalarOp<Operator, SCL, RHS>::bind(cl::sycl::handler &h) {
  r.bind(h);
}

/*! UnaryOp.
 * Implements a Unary Operation ( Op(z), e.g. z++), with z a vector.
 */
template <typename Operator, typename RHS>
UnaryOp<Operator, RHS>::UnaryOp(RHS &_r) : r(_r) {}

template <typename Operator, typename RHS>
sycl_blas_inline typename UnaryOp<Operator, RHS>::IndexType
UnaryOp<Operator, RHS>::getSize() const {
  return r.getSize();
}

template <typename Operator, typename RHS>
sycl_blas_inline bool UnaryOp<Operator, RHS>::valid_thread(
    cl::sycl::nd_item<1> ndItem) const {
  return ((ndItem.get_global_id(0) < UnaryOp<Operator, RHS>::getSize()));
}

template <typename Operator, typename RHS>
sycl_blas_inline typename UnaryOp<Operator, RHS>::value_type
UnaryOp<Operator, RHS>::eval(typename UnaryOp<Operator, RHS>::IndexType i) {
  return Operator::eval(r.eval(i));
}

template <typename Operator, typename RHS>
sycl_blas_inline typename UnaryOp<Operator, RHS>::value_type
UnaryOp<Operator, RHS>::eval(cl::sycl::nd_item<1> ndItem) {
  return UnaryOp<Operator, RHS>::eval(ndItem.get_global_id(0));
}
template <typename Operator, typename RHS>
sycl_blas_inline void UnaryOp<Operator, RHS>::bind(cl::sycl::handler &h) {
  r.bind(h);
}

/*! BinaryOp.
 * @brief Implements a Binary Operation (x OP z) with x and z vectors.
 */
template <typename Operator, typename LHS, typename RHS>
BinaryOp<Operator, LHS, RHS>::BinaryOp(LHS &_l, RHS &_r) : l(_l), r(_r){};

template <typename Operator, typename LHS, typename RHS>
sycl_blas_inline typename BinaryOp<Operator, LHS, RHS>::IndexType
BinaryOp<Operator, LHS, RHS>::getSize() const {
  return r.getSize();
}
template <typename Operator, typename LHS, typename RHS>
sycl_blas_inline bool BinaryOp<Operator, LHS, RHS>::valid_thread(
    cl::sycl::nd_item<1> ndItem) const {
  return ((ndItem.get_global_id(0) < getSize()));
}

template <typename Operator, typename LHS, typename RHS>
sycl_blas_inline typename BinaryOp<Operator, LHS, RHS>::value_type
BinaryOp<Operator, LHS, RHS>::eval(
    typename BinaryOp<Operator, LHS, RHS>::IndexType i) {
  return Operator::eval(l.eval(i), r.eval(i));
}
template <typename Operator, typename LHS, typename RHS>
sycl_blas_inline typename BinaryOp<Operator, LHS, RHS>::value_type
BinaryOp<Operator, LHS, RHS>::eval(cl::sycl::nd_item<1> ndItem) {
  return BinaryOp<Operator, LHS, RHS>::eval(ndItem.get_global_id(0));
}
template <typename Operator, typename LHS, typename RHS>
sycl_blas_inline void BinaryOp<Operator, LHS, RHS>::bind(cl::sycl::handler &h) {
  l.bind(h);
  r.bind(h);
}

/*! TupleOp.
 * @brief Implements a Tuple Operation (map (\x -> [i, x]) vector).
 */
template <typename RHS>
TupleOp<RHS>::TupleOp(RHS &_r) : r(_r) {}

template <typename RHS>
sycl_blas_inline typename TupleOp<RHS>::IndexType TupleOp<RHS>::getSize()
    const {
  return r.getSize();
}

template <typename RHS>
sycl_blas_inline bool TupleOp<RHS>::valid_thread(
    cl::sycl::nd_item<1> ndItem) const {
  return ((ndItem.get_global_id(0) < getSize()));
}
template <typename RHS>
sycl_blas_inline typename TupleOp<RHS>::value_type TupleOp<RHS>::eval(
    typename TupleOp<RHS>::IndexType i) {
  return TupleOp<RHS>::value_type(i, r.eval(i));
}

template <typename RHS>
sycl_blas_inline typename TupleOp<RHS>::value_type TupleOp<RHS>::eval(
    cl::sycl::nd_item<1> ndItem) {
  return TupleOp<RHS>::eval(ndItem.get_global_id(0));
}
template <typename RHS>
sycl_blas_inline void TupleOp<RHS>::bind(cl::sycl::handler &h) {
  r.bind(h);
}

/*! AssignReduction.
 * @brief Implements the reduction operation for assignments (in the form y
 * = x) with y a scalar and x a subexpression tree.
 */
template <typename Operator, class LHS, class RHS>
AssignReduction<Operator, LHS, RHS>::AssignReduction(LHS &_l, RHS &_r,
                                                     IndexType _blqS,
                                                     IndexType _grdS)
    : l(_l), r(_r), blqS(_blqS), grdS(_grdS){};

template <typename Operator, class LHS, class RHS>
sycl_blas_inline typename AssignReduction<Operator, LHS, RHS>::IndexType
AssignReduction<Operator, LHS, RHS>::getSize() const {
  return r.getSize();
}

template <typename Operator, class LHS, class RHS>
sycl_blas_inline bool AssignReduction<Operator, LHS, RHS>::valid_thread(
    cl::sycl::nd_item<1> ndItem) const {
  return true;
}
template <typename Operator, class LHS, class RHS>
sycl_blas_inline typename AssignReduction<Operator, LHS, RHS>::value_type
AssignReduction<Operator, LHS, RHS>::eval(
    typename AssignReduction<Operator, LHS, RHS>::IndexType i) {
  IndexType vecS = r.getSize();
  IndexType frs_thrd = 2 * blqS * i;
  IndexType lst_thrd = ((frs_thrd + blqS) > vecS) ? vecS : (frs_thrd + blqS);
  // Reduction across the grid
  value_type val = Operator::init(r);
  for (IndexType j = frs_thrd; j < lst_thrd; j++) {
    value_type local_val = Operator::init(r);
    for (IndexType k = j; k < vecS; k += 2 * grdS) {
      local_val = Operator::eval(local_val, r.eval(k));
      if (k + blqS < vecS) {
        local_val = Operator::eval(local_val, r.eval(k + blqS));
      }
    }
    // Reduction inside the block
    val = Operator::eval(val, local_val);
  }
  if (i < l.getSize()) {
    l.eval(i) = val;
  }
  return val;
}
template <typename Operator, class LHS, class RHS>
sycl_blas_inline typename AssignReduction<Operator, LHS, RHS>::value_type
AssignReduction<Operator, LHS, RHS>::eval(cl::sycl::nd_item<1> ndItem) {
  return AssignReduction<Operator, LHS, RHS>::eval(ndItem.get_global_id(0));
}
template <typename Operator, class LHS, class RHS>
template <typename sharedT>
sycl_blas_inline typename AssignReduction<Operator, LHS, RHS>::value_type
AssignReduction<Operator, LHS, RHS>::eval(sharedT scratch,
                                          cl::sycl::nd_item<1> ndItem) {
  IndexType localid = ndItem.get_local_id(0);
  IndexType localSz = ndItem.get_local_range(0);
  IndexType groupid = ndItem.get_group(0);

  IndexType vecS = r.getSize();
  IndexType frs_thrd = 2 * groupid * localSz + localid;

  // Reduction across the grid
  value_type val = Operator::init(r);
  for (IndexType k = frs_thrd; k < vecS; k += 2 * grdS) {
    val = Operator::eval(val, r.eval(k));
    if ((k + blqS < vecS)) {
      val = Operator::eval(val, r.eval(k + blqS));
    }
  }

  scratch[localid] = val;
  // This barrier is mandatory to be sure the data is on the shared memory
  ndItem.barrier(cl::sycl::access::fence_space::local_space);

  // Reduction inside the block
  for (IndexType offset = localSz >> 1; offset > 0; offset >>= 1) {
    if (localid < offset) {
      scratch[localid] =
          Operator::eval(scratch[localid], scratch[localid + offset]);
    }
    // This barrier is mandatory to be sure the data are on the shared memory
    ndItem.barrier(cl::sycl::access::fence_space::local_space);
  }
  if (localid == 0) {
    l.eval(groupid) = scratch[localid];
  }
  return l.eval(groupid);
}

template <typename Operator, class LHS, class RHS>
sycl_blas_inline void AssignReduction<Operator, LHS, RHS>::bind(
    cl::sycl::handler &h) {
  l.bind(h);
  r.bind(h);
}

}  // namespace blas

#endif  // BLAS1_TREES_HPP
