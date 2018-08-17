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

#ifndef BLAS1_TREES_HPP
#define BLAS1_TREES_HPP

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
struct Join {
  using IndexType = typename RHS::IndexType;
  using value_type = typename RHS::value_type;
  LHS l;
  RHS r;

  Join(LHS &_l, RHS _r) : l(_l), r(_r){};

  // PROBLEM: Only the RHS size is considered. If LHS size is different??
  // If it is smaller, eval function will crash
  inline IndexType getSize() const { return r.getSize(); }

  inline bool valid_thread(cl::sycl::nd_item<1> ndItem) const {
    return ((ndItem.get_global_id(0) < getSize()));
  }

  value_type eval(IndexType i) {
    l.eval(i);
    return r.eval(i);
  }

  value_type eval(cl::sycl::nd_item<1> ndItem) {
    return eval(ndItem.get_global_id(0));
  }
  void bind(cl::sycl::handler &h) {
    l.bind(h);
    r.bind(h);
  }
};

/** Assign.
 */
template <class LHS, class RHS>
struct Assign {
  using IndexType = typename LHS::IndexType;
  using value_type = typename RHS::value_type;
  LHS l;
  RHS r;

  Assign(LHS &_l, RHS _r) : l(_l), r(_r){};

  // PROBLEM: Only the RHS size is considered. If LHS size is different??
  // If it is smaller, eval function will crash
  inline IndexType getSize() const { return r.getSize(); }

  inline bool valid_thread(cl::sycl::nd_item<1> ndItem) const {
    return ((ndItem.get_global_id(0) < getSize()));
  }

  value_type eval(IndexType i) {
    auto val = l.eval(i) = r.eval(i);
    return val;
  }
  void bind(cl::sycl::handler &h) {
    l.bind(h);
    r.bind(h);
  }

  value_type eval(cl::sycl::nd_item<1> ndItem) {
    return eval(ndItem.get_global_id(0));
  }
};

/*! DoubleAssign.
 */
template <class LHS1, class LHS2, class RHS1, class RHS2>
struct DobleAssign {
  using IndexType = typename LHS1::IndexType;
  using value_type = typename RHS1::value_type;
  LHS1 l1;
  LHS2 l2;
  RHS1 r1;
  RHS2 r2;

  DobleAssign(LHS1 &_l1, LHS2 &_l2, RHS1 _r1, RHS2 _r2)
      : l1(_l1), l2(_l2), r1(_r1), r2(_r2){};

  // PROBLEM: Only the RHS size is considered. If LHS size is different??
  // If it is smaller, eval function will crash
  inline IndexType getSize() const { return r2.getSize(); }

  inline bool valid_thread(cl::sycl::nd_item<1> ndItem) const {
    return ((ndItem.get_global_id(0) < getSize()));
  }

  value_type eval(IndexType i) {
    auto val1 = r1.eval(i);
    auto val2 = r2.eval(i);
    l1.eval(i) = val1;
    l2.eval(i) = val2;
    return val1;
  }

  value_type eval(cl::sycl::nd_item<1> ndItem) {
    return eval(ndItem.get_global_id(0));
  }
  inline void bind(cl::sycl::handler &h) {
    l1.bind(h);
    r1.bind(h);
    l2.bind(h);
    r2.bind(h);
  }
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
  ScalarOp(SCL _scl, RHS &_r) : scl(_scl), r(_r){};

  inline IndexType getSize() const { return r.getSize(); }

  inline bool valid_thread(cl::sycl::nd_item<1> ndItem) const {
    return ((ndItem.get_global_id(0) < getSize()));
  }

  value_type eval(IndexType i) {
    return Operator::eval(internal::get_scalar(scl), r.eval(i));
  }

  value_type eval(cl::sycl::nd_item<1> ndItem) {
    return eval(ndItem.get_global_id(0));
  }
  inline void bind(cl::sycl::handler &h) { r.bind(h); }
};

/*! UnaryOp.
 * Implements a Unary Operation ( Op(z), e.g. z++), with z a vector.
 */
template <typename Operator, typename RHS>
struct UnaryOp {
  using IndexType = typename RHS::IndexType;
  using value_type = typename RHS::value_type;
  RHS r;
  UnaryOp(RHS &_r) : r(_r){};

  inline IndexType getSize() const { return r.getSize(); }

  inline bool valid_thread(cl::sycl::nd_item<1> ndItem) const {
    return ((ndItem.get_global_id(0) < getSize()));
  }

  value_type eval(IndexType i) { return Operator::eval(r.eval(i)); }

  value_type eval(cl::sycl::nd_item<1> ndItem) {
    return eval(ndItem.get_global_id(0));
  }
  void bind(cl::sycl::handler &h) { r.bind(h); }
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

  BinaryOp(LHS &_l, RHS &_r) : l(_l), r(_r){};

  // PROBLEM: Only the RHS size is considered. If LHS size is different??
  // If it is smaller, eval function will crash
  inline IndexType getSize() const { return r.getSize(); }

  inline bool valid_thread(cl::sycl::nd_item<1> ndItem) const {
    return ((ndItem.get_global_id(0) < getSize()));
  }
  value_type eval(IndexType i) { return Operator::eval(l.eval(i), r.eval(i)); }

  value_type eval(cl::sycl::nd_item<1> ndItem) {
    return eval(ndItem.get_global_id(0));
  }
  void bind(cl::sycl::handler &h) {
    l.bind(h);
    r.bind(h);
  }
};

/*! TupleOp.
 * @brief Implements a Tuple Operation (map (\x -> [i, x]) vector).
 */
template <typename RHS>
struct TupleOp {
  using IndexType = typename RHS::IndexType;
  using value_type = IndexValueTuple<typename RHS::value_type>;
  RHS r;

  TupleOp(RHS &_r) : r(_r) {}

  inline IndexType getSize() const { return r.getSize(); }

  inline bool valid_thread(cl::sycl::nd_item<1> ndItem) const {
    return ((ndItem.get_global_id(0) < getSize()));
  }

  value_type eval(IndexType i) { return value_type(i, r.eval(i)); }

  value_type eval(cl::sycl::nd_item<1> ndItem) {
    return eval(ndItem.get_global_id(0));
  }
  void bind(cl::sycl::handler &h) { r.bind(h); }
};

template <typename RHS>
inline TupleOp<RHS> make_tuple_op(RHS &r) {
  return TupleOp<RHS>(r);
}

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

  AssignReduction(LHS &_l, RHS &_r, IndexType _blqS, IndexType _grdS)
      : l(_l), r(_r), blqS(_blqS), grdS(_grdS){};

  inline IndexType getSize() const { return r.getSize(); }

  inline bool valid_thread(cl::sycl::nd_item<1> ndItem) const { return true; }

  value_type eval(IndexType i) {
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
  value_type eval(cl::sycl::nd_item<1> ndItem) {
    return eval(ndItem.get_global_id(0));
  }
  template <typename sharedT>
  value_type eval(sharedT scratch, cl::sycl::nd_item<1> ndItem) {
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
  void bind(cl::sycl::handler &h) {
    l.bind(h);
    r.bind(h);
  }
};

template <typename Operator, typename LHS, typename RHS, typename IndexType>
AssignReduction<Operator, LHS, RHS> make_AssignReduction(LHS &l, RHS &r,
                                                         IndexType blqS,
                                                         IndexType grdS) {
  return AssignReduction<Operator, LHS, RHS>(l, r, blqS, grdS);
}

template <typename LHS, typename RHS, typename IndexType>
auto make_addAssignReduction(LHS &l, RHS &r, IndexType blqS, IndexType grdS)
    -> decltype(make_AssignReduction<addOp2_struct>(l, r, blqS, grdS)) {
  return make_AssignReduction<addOp2_struct>(l, r, blqS, grdS);
}

template <typename LHS, typename RHS, typename IndexType>
auto make_prdAssignReduction(LHS &l, RHS &r, IndexType blqS, IndexType grdS)
    -> decltype(make_AssignReduction<prdOp2_struct>(l, r, blqS, grdS)) {
  return make_AssignReduction<prdOp2_struct>(l, r, blqS, grdS);
}

template <typename LHS, typename RHS, typename IndexType>
auto make_addAbsAssignReduction(LHS &l, RHS &r, IndexType blqS, IndexType grdS)
    -> decltype(make_AssignReduction<addAbsOp2_struct>(l, r, blqS, grdS)) {
  return make_AssignReduction<addAbsOp2_struct>(l, r, blqS, grdS);
}

template <typename LHS, typename RHS, typename IndexType>
auto make_maxIndAssignReduction(LHS &l, RHS &r, IndexType blqS, IndexType grdS)
    -> decltype(make_AssignReduction<maxIndOp2_struct>(l, r, blqS, grdS)) {
  return make_AssignReduction<maxIndOp2_struct>(l, r, blqS, grdS);
}

template <typename LHS, typename RHS, typename IndexType>
auto make_minIndAssignReduction(LHS &l, RHS &r, IndexType blqS, IndexType grdS)
    -> decltype(make_AssignReduction<minIndOp2_struct>(l, r, blqS, grdS)) {
  return make_AssignReduction<minIndOp2_struct>(l, r, blqS, grdS);
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
operationT<operandsTN...> make_op(operandsTN &... operands) {
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
operationT<operatorT, operandsTN...> make_op(operandsTN &... operands) {
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
    operandT0 operand0, operandsTN &... operands) {
  return operationT<operatorT, operandT0, operandsTN...>(operand0, operands...);
}

}  // namespace blas

#endif  // BLAS1_TREES_HPP
