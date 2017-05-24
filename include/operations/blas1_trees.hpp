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

#ifndef BLAS1_TREES_HPP
#define BLAS1_TREES_HPP

#include <complex>
#include <iostream>
#include <stdexcept>
#include <vector>

#include <operations/blas_operators.hpp>
#include <views/view_sycl.hpp>

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
  static typename T::value_type get_scalar(T& opSCL) { return opSCL.eval(0); }
};

/*! DetectScalar.
 * @brief See Detect Scalar.
 */
template <>
struct DetectScalar<int> {
  using T = int;
  static T get_scalar(T& scalar) { return scalar; }
};

/*! DetectScalar.
 * @brief See Detect Scalar.
 */
template <>
struct DetectScalar<float> {
  using T = float;
  static T get_scalar(T& scalar) { return scalar; }
};

/*! DetectScalar.
 * @brief See Detect Scalar.
*/
template <>
struct DetectScalar<double> {
  using T = double;
  static T get_scalar(T& scalar) { return scalar; }
};

/*! DetectScalar.
 * @brief See Detect Scalar.
*/
template <>
struct DetectScalar<std::complex<float>> {
  using T = std::complex<float>;
  static T get_scalar(T& scalar) { return scalar; }
};

/*! DetectScalar.
 * @brief See Detect Scalar.
*/
template <>
struct DetectScalar<std::complex<double>> {
  using T = std::complex<double>;
  static T get_scalar(T& scalar) { return scalar; }
};

/*! get_scalar.
 * @brief Template autodecuction function for DetectScalar.
*/
template <typename T>
auto get_scalar(T& scl) -> decltype(DetectScalar<T>::get_scalar(scl)) {
  return DetectScalar<T>::get_scalar(scl);
}
}  // namespace internal

/** Join.
 * @brief Joins both sides of the expression in the single kernel.
 */
template <class LHS, class RHS>
struct Join {
  LHS l;
  RHS r;

  using value_type = typename RHS::value_type;

  Join(LHS& _l, RHS _r) : l(_l), r(_r){};

  // PROBLEM: Only the RHS size is considered. If LHS size is different??
  // If it is smaller, eval function will crash
  size_t getSize() { return r.getSize(); }

  value_type eval(size_t i) {
    l.eval(i);
    return r.eval(i);
  }

  value_type eval(cl::sycl::nd_item<1> ndItem) {
    return eval(ndItem.get_global(0));
  }
};

/** Assign.
 */
template <class LHS, class RHS>
struct Assign {
  LHS l;
  RHS r;

  using value_type = typename RHS::value_type;

  Assign(LHS& _l, RHS _r) : l(_l), r(_r){};

  // PROBLEM: Only the RHS size is considered. If LHS size is different??
  // If it is smaller, eval function will crash
  size_t getSize() { return r.getSize(); }

  value_type eval(size_t i) {
    auto val = l.eval(i) = r.eval(i);
    return val;
  }

  value_type eval(cl::sycl::nd_item<1> ndItem) {
    return eval(ndItem.get_global(0));
  }
};

/*! DoubleAssign.
 */
template <class LHS1, class LHS2, class RHS1, class RHS2>
struct DobleAssign {
  LHS1 l1;
  LHS2 l2;
  RHS1 r1;
  RHS2 r2;

 public:
  using value_type = typename RHS1::value_type;

  DobleAssign(LHS1& _l1, LHS2& _l2, RHS1 _r1, RHS2 _r2)
      : l1(_l1), l2(_l2), r1(_r1), r2(_r2){};

  // PROBLEM: Only the RHS size is considered. If LHS size is different??
  // If it is smaller, eval function will crash
  size_t getSize() { return r2.getSize(); }

  value_type eval(size_t i) {
    auto val1 = r1.eval(i);
    auto val2 = r2.eval(i);
    l1.eval(i) = val1;
    l2.eval(i) = val2;
    return val1;
  }

  value_type eval(cl::sycl::nd_item<1> ndItem) {
    return eval(ndItem.get_global(0));
  }
};

/*!ScalarOp.
 * @brief Implements an scalar operation.
 * (e.g alpha OP x, with alpha scalar and x vector)
 */
template <typename Operator, typename SCL, typename RHS>
struct ScalarOp {
  using value_type = typename RHS::value_type;
  SCL scl;
  RHS r;

  ScalarOp(SCL _scl, RHS& _r) : scl(_scl), r(_r){};

  size_t getSize() { return r.getSize(); }
  value_type eval(size_t i) {
    return Operator::eval(internal::get_scalar(scl), r.eval(i));
  }

  value_type eval(cl::sycl::nd_item<1> ndItem) {
    return eval(ndItem.get_global(0));
  }
};

/*! UnaryOp.
 * Implements a Unary Operation ( Op(z), e.g. z++), with z a vector.
 */
template <typename Operator, typename RHS>
struct UnaryOp {
  using value_type = typename RHS::value_type;
  RHS r;

  UnaryOp(RHS& _r) : r(_r){};

  size_t getSize() { return r.getSize(); }

  value_type eval(size_t i) { return Operator::eval(r.eval(i)); }

  value_type eval(cl::sycl::nd_item<1> ndItem) {
    return eval(ndItem.get_global(0));
  }
};

/*! BinaryOp.
 * @brief Implements a Binary Operation (x OP z) with x and z vectors.
 */
template <typename Operator, typename LHS, typename RHS>
struct BinaryOp {
  using value_type = typename RHS::value_type;
  LHS l;
  RHS r;

  BinaryOp(LHS& _l, RHS& _r) : l(_l), r(_r){};

  // PROBLEM: Only the RHS size is considered. If LHS size is different??
  // If it is smaller, eval function will crash
  size_t getSize() { return r.getSize(); }

  value_type eval(size_t i) { return Operator::eval(l.eval(i), r.eval(i)); }

  value_type eval(cl::sycl::nd_item<1> ndItem) {
    return eval(ndItem.get_global(0));
  }
};

/*! ReducAssignNewOp2.
 * @brief Implements the reduction operation for assignments (in the form y = x)
 *  with y a scalar and x a BinaryOp.
 */
template <typename Operator, class LHS, class RHS>
struct ReducAssignNewOp2 {
  using value_type = typename RHS::value_type;
  LHS l;
  RHS r;
  size_t blqS;  // block  size
  size_t grdS;  // grid  size

  ReducAssignNewOp2(LHS& _l, RHS& _r, size_t _blqS, size_t _grdS)
      : l(_l), r(_r), blqS(_blqS), grdS(_grdS){};

  size_t getSize() { return r.getSize(); }

  value_type eval(size_t i) {
    size_t vecS = r.getSize();
    size_t frs_thrd = 2 * blqS * i;
    size_t lst_thrd = ((frs_thrd + blqS) > vecS) ? vecS : (frs_thrd + blqS);
    // Reduction across the grid
    value_type val = Operator::init(r);
    for (size_t j = frs_thrd; j < lst_thrd; j++) {
      value_type local_val = Operator::init(r);
      for (size_t k = j; k < vecS; k += 2 * grdS) {
        local_val = Operator::eval(local_val, r.eval(k));
        local_val = ((k + blqS) < vecS)
                        ? Operator::eval(local_val, r.eval(k + blqS))
                        : local_val;
      }
      // Reduction inside the block
      val = Operator::eval(val, local_val);
    }
    return l.eval(i) = val;
  }
  value_type eval(cl::sycl::nd_item<1> ndItem) {
    return eval(ndItem.get_global(0));
  }
  template <typename sharedT>
  value_type eval(sharedT scratch, cl::sycl::nd_item<1> ndItem) {
    size_t localid = ndItem.get_local(0);
    size_t localSz = ndItem.get_local_range(0);
    size_t groupid = ndItem.get_group(0);

    size_t vecS = r.getSize();
    size_t frs_thrd = 2 * groupid * localSz + localid;

    // Reduction across the grid
    value_type val = Operator::init(r);
    for (size_t k = frs_thrd; k < vecS; k += 2 * grdS) {
      val = Operator::eval(val, r.eval(k));
      if ((k + blqS < vecS)) {
        val = Operator::eval(val, r.eval(k + blqS));
      }
    }

    scratch[localid] = val;
    // This barrier is mandatory to be sure the data is on the shared memory
    ndItem.barrier(cl::sycl::access::fence_space::local_space);

    // Reduction inside the block
    for (size_t offset = localSz >> 1; offset > 0; offset >>= 1) {
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
};

template <typename Operator, typename LHS, typename RHS>
ReducAssignNewOp2<Operator, LHS, RHS> make_ReducAssignNewOp2(LHS& l, RHS& r,
                                                             size_t blqS,
                                                             size_t grdS) {
  return ReducAssignNewOp2<Operator, LHS, RHS>(l, r, blqS, grdS);
}

template <typename LHS, typename RHS>
auto make_addReducAssignNewOp2(LHS& l, RHS& r, size_t blqS, size_t grdS)
    -> decltype(make_ReducAssignNewOp2<addOp2_struct>(l, r, blqS, grdS)) {
  return make_ReducAssignNewOp2<addOp2_struct>(l, r, blqS, grdS);
}

template <typename LHS, typename RHS>
auto make_prdReducAssignNewOp2(LHS& l, RHS& r, size_t blqS, size_t grdS)
    -> decltype(make_ReducAssignNewOp2<prdOp2_struct>(l, r, blqS, grdS)) {
  return make_ReducAssignNewOp2<prdOp2_struct>(l, r, blqS, grdS);
}

template <typename LHS, typename RHS>
auto make_addAbsReducAssignNewOp2(LHS& l, RHS& r, size_t blqS, size_t grdS)
    -> decltype(make_ReducAssignNewOp2<addAbsOp2_struct>(l, r, blqS, grdS)) {
  return make_ReducAssignNewOp2<addAbsOp2_struct>(l, r, blqS, grdS);
}

template <typename LHS, typename RHS>
auto make_maxIndReducAssignNewOp2(LHS& l, RHS& r, size_t blqS, size_t grdS)
    -> decltype(make_ReducAssignNewOp2<maxIndOp3_struct>(l, r, blqS, grdS)) {
  return make_ReducAssignNewOp2<maxIndOp3_struct>(l, r, blqS, grdS);
}

/*! ReducAssignNewOp3.
 * @brief Implements the reduction operation for assignments (in the form y = x)
 *  with y a scalar and x a TernaryOp.
 */
template <typename Operator, class LHS, class RHS>
struct ReducAssignNewOp3 {
  using value_type = typename RHS::value_type;
  using res_type = typename LHS::value_type;
  LHS l;
  RHS r;
  size_t blqS;  // block  size
  size_t grdS;  // grid  size

  ReducAssignNewOp3(LHS& _l, RHS& _r, size_t _blqS, size_t _grdS)
      : l(_l), r(_r), blqS(_blqS), grdS(_grdS){};

  size_t getSize() { return r.getSize(); }

  res_type eval(size_t i) {
    size_t vecS = r.getSize();
    size_t frs_thrd = 2 * blqS * i;
    size_t lst_thrd = ((frs_thrd + blqS) > vecS) ? vecS : (frs_thrd + blqS);
    // Reduction across the grid
    res_type val = Operator::init(r, l);
    for (size_t j = frs_thrd; j < lst_thrd; j++) {
      res_type local_val = Operator::init(r, l);
      for (size_t k = j; k < vecS; k += 2 * grdS) {
        local_val = Operator::eval(local_val, k, r.eval(k));
        local_val = ((k + blqS) < vecS) ? Operator::eval(local_val, (k + blqS),
                                                         r.eval(k + blqS))
                                        : local_val;
      }
      // Reduction inside the block
      val = Operator::eval(val, local_val);
    }
    return l.eval(i) = val;
  }
  value_type eval(cl::sycl::nd_item<1> ndItem) {
    return eval(ndItem.get_global(0));
  }
  template <typename sharedT>
  res_type eval(sharedT scratch, cl::sycl::nd_item<1> ndItem) {
    size_t localid = ndItem.get_local(0);
    size_t localSz = ndItem.get_local_range(0);
    size_t groupid = ndItem.get_group(0);

    size_t vecS = r.getSize();
    size_t frs_thrd = 2 * groupid * localSz + localid;

    // Reduction across the grid
    res_type val = Operator::init(r, l);
    for (size_t k = frs_thrd; k < vecS; k += 2 * grdS) {
      val = Operator::eval(val, k, r.eval(k));
      val = ((k + blqS) < vecS)
                ? Operator::eval(val, (k + blqS), r.eval(k + blqS))
                : val;
    }
    scratch[localid] = val;
    // This barrier is mandatory to be sure the data is on the shared memory
    ndItem.barrier(cl::sycl::access::fence_space::local_space);

    // Reduction inside the block
    for (size_t offset = localSz >> 1; offset > 0; offset >>= 1) {
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
};

template <typename Operator, typename LHS, typename RHS>
ReducAssignNewOp3<Operator, LHS, RHS> make_ReducAssignNewOp3(LHS& l, RHS& r,
                                                             size_t blqS,
                                                             size_t grdS) {
  return ReducAssignNewOp3<Operator, LHS, RHS>(l, r, blqS, grdS);
}

template <typename LHS, typename RHS>
auto make_maxIndReducAssignNewOp3(LHS& l, RHS& r, size_t blqS, size_t grdS)
    -> decltype(make_ReducAssignNewOp3<maxIndOp3_struct>(l, r, blqS, grdS)) {
  return make_ReducAssignNewOp3<maxIndOp3_struct>(l, r, blqS, grdS);
}

/*!
@brief Template function for constructing operation nodes based on input
tempalte and function arguments. Non-specialised case for N reference operands.
@tparam operationT Operation type of the operation node.
@tparam operandsTN Operand types of the oeration node.
@param operands Reference operands of the operation node.
@return Constructed operation node.
*/
template <template <class...> class operationT, typename... operandsTN>
operationT<operandsTN...> make_op(operandsTN&... operands) {
  return operationT<operandsTN...>(operands...);
}

/*!
@brief Template function for constructing operation nodes based on input
tempalte and function arguments. Specialised case for an operator and N
reference operands.
@tparam operationT Operation type of the operation node.
@tparam operatorT Operator type of the operation node.
@tparam operandsTN Operand types of the operation node.
@param operands Reference operands of the operation node.
@return Constructed operation node.
*/
template <template <class...> class operationT, typename operatorT,
          typename... operandsTN>
operationT<operatorT, operandsTN...> make_op(operandsTN&... operands) {
  return operationT<operatorT, operandsTN...>(operands...);
}

/*!
@brief Template function for constructing operation nodes based on input
tempalte and function arguments. Specialised case for an operator, a single by
value operand and N reference operands.
@tparam operationT Operation type of the operation node.
@tparam operatorT Operator type of the operation node.
@tparam operandT0 Operand type of the first by value operand of the operation
node.
@tparam operandsTN Operand types of the subsequent reference operands of the
operation node.
@param operand0 First by value operand of the operation node.
@param operands Subsequent reference operands of the oepration node.
@return Constructed operation node.
*/
template <template <class...> class operationT, typename operatorT,
          typename operandT0, typename... operandsTN>
operationT<operatorT, operandT0, operandsTN...> make_op(
    operandT0 operand0, operandsTN&... operands) {
  return operationT<operatorT, operandT0, operandsTN...>(operand0, operands...);
}

}  // namespace blas

#endif  // BLAS1_TREES_HPP
