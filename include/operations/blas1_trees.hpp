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

#include <stdexcept>
#include <vector>

#include <operations/blas_operators.hpp>
#include <operations/blas_trees.hpp>
#include <views/view_sycl.hpp>

namespace blas {

/*! AssignReduction.
 * @brief Implements the reduction operation for assignments (in the form y = x)
 *  with y a scalar and x a subexpression tree.
 */
template <typename Operator, class LHS, class RHS>
struct AssignReduction {
  using value_type = typename RHS::value_type;
  LHS l;
  RHS r;
  size_t blqS;  // block  size
  size_t grdS;  // grid  size

  AssignReduction(LHS &_l, RHS &_r, size_t _blqS, size_t _grdS)
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
        if (k + blqS < vecS) {
          local_val = Operator::eval(local_val, r.eval(k + blqS));
        }
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
AssignReduction<Operator, LHS, RHS> make_AssignReduction(LHS &l, RHS &r,
                                                             size_t blqS,
                                                             size_t grdS) {
  return AssignReduction<Operator, LHS, RHS>(l, r, blqS, grdS);
}

template <typename LHS, typename RHS>
auto make_addAssignReduction(LHS &l, RHS &r, size_t blqS, size_t grdS)
    -> decltype(make_AssignReduction<addOp2_struct>(l, r, blqS, grdS)) {
  return make_AssignReduction<addOp2_struct>(l, r, blqS, grdS);
}

template <typename LHS, typename RHS>
auto make_prdAssignReduction(LHS &l, RHS &r, size_t blqS, size_t grdS)
    -> decltype(make_AssignReduction<prdOp2_struct>(l, r, blqS, grdS)) {
  return make_AssignReduction<prdOp2_struct>(l, r, blqS, grdS);
}

template <typename LHS, typename RHS>
auto make_addAbsAssignReduction(LHS &l, RHS &r, size_t blqS, size_t grdS)
    -> decltype(make_AssignReduction<addAbsOp2_struct>(l, r, blqS, grdS)) {
  return make_AssignReduction<addAbsOp2_struct>(l, r, blqS, grdS);
}

template <typename LHS, typename RHS>
auto make_maxIndAssignReduction(LHS &l, RHS &r, size_t blqS, size_t grdS)
    -> decltype(make_AssignReduction<maxIndOp2_struct>(l, r, blqS, grdS)) {
  return make_AssignReduction<maxIndOp2_struct>(l, r, blqS, grdS);
}

template <typename LHS, typename RHS>
auto make_minIndAssignReduction(LHS &l, RHS &r, size_t blqS, size_t grdS)
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
