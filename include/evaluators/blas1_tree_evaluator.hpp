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
 *  @filename blas1_tree_evaluator.hpp
 *
 **************************************************************************/

#ifndef BLAS1_TREE_EVALUATOR_HPP
#define BLAS1_TREE_EVALUATOR_HPP

#include <stdexcept>
#include <vector>

#include <evaluators/blas_tree_evaluator_base.hpp>
#include <executors/blas_packet_traits_sycl.hpp>
#include <operations/blas1_trees.hpp>
#include <views/view_sycl.hpp>

namespace blas {

/*! Reduction.
 * @brief Implements the reduction operation for assignments (in the form y = x)
 *  with y a scalar and x a subexpression tree.
 */
template <typename Functor, class LHS, class RHS>
struct Evaluator<ReductionExpr<Functor, LHS, RHS>, SYCLDevice> {
  using Expression = ReductionExpr<Functor, LHS, RHS>;
  using Device = SYCLDevice;
  using value_type = typename Expression::value_type;
  using dev_functor = functor_traits<Functor, value_type, Device>;
  using cont_type = typename Evaluator<LHS, Device>::cont_type;
  /* static constexpr bool supported = functor_traits<Functor, value_type, SYCLDevice>::supported && LHS::supported && RHS::supported; */

  Evaluator<LHS, Device> l;
  Evaluator<RHS, Device> r;

  size_t blqS = 256;
  size_t grdS = 512;

  Evaluator(Expression &expr)
      : l(Evaluator<LHS, Device>(expr.l)), r(Evaluator<RHS, Device>(expr.r)) {
    setBlocksize(256);
    setGridsize(512);
  }

  size_t getSize() const { return r.getSize(); }
  cont_type *data() { return l.data(); }

  void setBlocksize(size_t blocksize) { blqS = blocksize; }

  void setGridsize(size_t gridsize) { grdS = gridsize; }

  void reduce(Device &dev);

  bool eval_subexpr_if_needed(cont_type *cont, Device &dev) {
    l.eval_subexpr_if_needed(NULL, dev);
    r.eval_subexpr_if_needed(NULL, dev);
    reduce(dev);
    return true;
  }

  value_type eval(size_t i) {
    size_t vecS = r.getSize();

    size_t frs_thrd = 2 * blqS * i;
    size_t lst_thrd = ((frs_thrd + blqS) > vecS) ? vecS : (frs_thrd + blqS);
    // Reduction across the grid
    value_type val = dev_functor::init(r);
    for (size_t j = frs_thrd; j < lst_thrd; j++) {
      value_type local_val = dev_functor::init(r);
      for (size_t k = j; k < vecS; k += 2 * grdS) {
        local_val = dev_functor::eval(local_val, r.eval(k));
        if (k + blqS < vecS) {
          local_val = dev_functor::eval(local_val, r.eval(k + blqS));
        }
      }
      // Reduction inside the block
      val = dev_functor::eval(val, local_val);
    }
    return l.eval(i) = val;
  }

  template <typename sharedT>
  value_type eval(sharedT scratch, cl::sycl::nd_item<1> ndItem) {
    size_t localid = ndItem.get_local(0);
    size_t localSz = ndItem.get_local_range(0);
    size_t groupid = ndItem.get_group(0);

    size_t vecS = r.getSize();
    size_t frs_thrd = 2 * groupid * localSz + localid;

    // Reduction across the grid
    value_type val = dev_functor::init(r);
    for (size_t k = frs_thrd; k < vecS; k += 2 * grdS) {
      val = dev_functor::eval(val, r.eval(k));
      if ((k + blqS < vecS)) {
        val = dev_functor::eval(val, r.eval(k + blqS));
      }
    }

    scratch[localid] = val;
    // This barrier is mandatory to be sure the data is on the shared memory
    ndItem.barrier(cl::sycl::access::fence_space::local_space);

    // Reduction inside the block
    for (size_t offset = localSz >> 1; offset > 0; offset >>= 1) {
      if (localid < offset) {
        scratch[localid] =
            dev_functor::eval(scratch[localid], scratch[localid + offset]);
      }
      // This barrier is mandatory to be sure the data are on the shared memory
      ndItem.barrier(cl::sycl::access::fence_space::local_space);
    }
    if (localid == 0) {
      l.eval(groupid) = scratch[localid];
    }
    return l.eval(groupid);
  }

  value_type eval(cl::sycl::nd_item<1> ndItem) {
    return eval(ndItem.get_global(0));
  }
};

}  // namespace blas

#endif
