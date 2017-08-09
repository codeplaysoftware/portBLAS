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
#include <executors/blas_pointer_struct.hpp>
#include <operations/blas1_trees.hpp>

namespace blas {

template <class EvaluatorT>
struct GenericReducer;
template <class EvaluatorT>
struct GenericReducerTwoStage;
template <class EvaluatorT, class Reducer>
struct FullReducer;
template <class EvaluatorT, class Reducer>
struct PartialReduction;

/*!
 * Evaluator<ReductionExpr>.
 * @brief Evaluates the reduction expression; breaks the kernel.
 */
template <typename Functor, class RHS, template <class> class MakePointer>
struct Evaluator<ReductionExpr<Functor, RHS, MakePointer>, SYCLDevice> {
  using Expression = ReductionExpr<Functor, RHS, MakePointer>;
  using Device = SYCLDevice;
  using value_type = typename Expression::value_type;
  using dev_functor = functor_traits<Functor, value_type, Device>;
  using cont_type = typename Evaluator<RHS, Device>::cont_type;
  /* static constexpr bool supported = functor_traits<Functor, value_type,
   * SYCLDevice>::supported && RHS::supported; */
  bool allocated_result = false;
  typename MakePointer<value_type>::type result;
  Evaluator<RHS, Device> r;

  Evaluator(Expression &expr)
      : r(Evaluator<RHS, Device>(expr.r)), result(nullptr) {}
  size_t getSize() const { return r.getSize(); }
  cont_type *data() { return r.data(); }

  bool eval_subexpr_if_needed(typename MakePointer<value_type>::type cont,
                              Device &dev) {
    r.eval_subexpr_if_needed(nullptr, dev);
    if (cont) {
      result = cont;
    } else {
      allocated_result = true;
      result = dev.allocate<value_type>(1);
    }
    FullReducer<Evaluator<RHS, Device>,
                GenericReducer<Evaluator<RHS, Device>>>::run(dev, r, *result);
    return true;
  }

  value_type eval(size_t i) { return result[i]; }
  value_type eval(cl::sycl::nd_item<1> ndItem) {
    return eval(ndItem.get_global(0));
  }
  value_type &evalref(size_t i) { return result[i]; }
  value_type &evalref(cl::sycl::nd_item<1> ndItem) {
    return evalref(ndItem.get_global(0));
  }

  void cleanup(SYCLDevice &dev) {
    r.cleanup(dev);
    if (allocated_result) {
      allocated_result = false;
      dev.deallocate<value_type>(result);
    }
  }
};

template <typename Functor, class RHS>
struct Evaluator<ReductionExpr<Functor, RHS, MakeDevicePointer>, SYCLDevice> {
  using Expression = ReductionExpr<Functor, RHS, MakeDevicePointer>;
  using Device = SYCLDevice;
  using Self = Evaluator<Expression, Device>;
  using value_type = typename Expression::value_type;
  using dev_functor = functor_traits<Functor, value_type, Device>;
  using cont_type = typename Evaluator<RHS, Device>::cont_type;

  typename MakeDevicePointer<value_type>::type result;
  Evaluator<RHS, Device> r;

  Evaluator(Expression &expr) : r(Evaluator<RHS, Device>(expr.r)) {}
  size_t getSize() const { return r.getSize(); }
  cont_type *data() { return r.data(); }
  value_type eval(size_t i) { return result[i]; }
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
