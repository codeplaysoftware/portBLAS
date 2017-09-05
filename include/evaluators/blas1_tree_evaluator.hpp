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
#include <executors/blas_device_sycl.hpp>
#include <executors/blas_pointer_struct.hpp>
#include <operations/blas1_trees.hpp>

namespace blas {

template <class AssignAssignEvaluatorT, class AssignEvaluatorT, class Functor> struct GenericReducer;

// host side reduction expr
template <typename Functor, class RHS, template <class> class MakePointer>
struct Evaluator<ReductionExpr<Functor, RHS, MakePointer>, SYCLDevice> {
  using Expression = ReductionExpr<Functor, RHS, MakePointer>;
  using Device = SYCLDevice;
  using value_type = typename Expression::value_type;
  using dev_functor = functor_traits<Functor, value_type, Device>;
  using cont_type = typename Evaluator<RHS, Device>::cont_type;

  static constexpr bool needassign = false;

  bool allocated_result = false;
  typename MakePointer<value_type>::type result;
  Evaluator<RHS, Device> r;

  explicit Evaluator(Expression &expr):
    r(Evaluator<RHS, Device>(expr.r)),
    result(MakePointer<value_type>::init())
  {}
  size_t getSize() const { return r.getSize(); }
  cont_type *data() { return r.data(); }

  template <typename AssignEvaluatorT = void>
  bool eval_subexpr_if_needed(typename MakePointer<value_type>::type cont, AssignEvaluatorT *assign, Device &dev) {
    r.template eval_subexpr_if_needed<AssignEvaluatorT>(nullptr, nullptr, dev);
    if (cont) {
      result = cont;
    } else {
      allocated_result = true;
      result = dev.allocate<value_type>(1);
    }
    GenericReducer<AssignEvaluatorT, Evaluator<RHS, Device>, dev_functor>::run(dev, assign, r, *result);
    return (cont == nullptr);
  }

  void cleanup(SYCLDevice &dev) {
    r.cleanup(dev);
    if (allocated_result) {
      allocated_result = false;
      dev.deallocate<value_type>(result);
    }
  }
};

// device side reduction expr
template <typename Functor, class RHS>
struct Evaluator<ReductionExpr<Functor, RHS, MakeDevicePointer>, SYCLDevice> {
  using Expression = ReductionExpr<Functor, RHS, MakeDevicePointer>;
  using Device = SYCLDevice;
  using Self = Evaluator<Expression, Device>;
  using value_type = typename Expression::value_type;
  using dev_functor = functor_traits<Functor, value_type, Device>;
  using cont_type = typename Evaluator<RHS, Device>::cont_type;

  static constexpr bool needassign = false;

  typename MakeDevicePointer<value_type>::type result;
  Evaluator<RHS, Device> r;

  explicit Evaluator(Expression &expr):
    r(Evaluator<RHS, Device>(expr.r)),
    result(MakeDevicePointer<value_type>::init())
  {}
  size_t getSize() const { return 1; }
  cont_type *data() { return r.data(); }
  value_type eval(size_t i) { return result[0]; }
  value_type eval(cl::sycl::nd_item<1> ndItem) {
    return eval(ndItem.get_global(0));
  }
  value_type &evalref(size_t i) { return result[i]; }
  value_type &evalref(cl::sycl::nd_item<1> ndItem) {
    return evalref(ndItem.get_global(0));
  }
  void cleanup(SYCLDevice &dev) { r.cleanup(dev); }
};

}  // namespace blas

#endif
