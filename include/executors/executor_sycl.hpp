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
 *  @filename executor_sycl.hpp
 *
 **************************************************************************/

#ifndef BLAS_EXECUTOR_SYCL_HPP
#define BLAS_EXECUTOR_SYCL_HPP

#include <stdexcept>

#include <CL/sycl.hpp>

#include <executors/blas1_tree_executor.hpp>
#include <executors/blas2_tree_executor.hpp>
#include <executors/blas3_tree_executor.hpp>
#include <executors/blas_device_sycl.hpp>

namespace blas {

/*! execute_tree.
@brief the functor for executing a tree in SYCL.
@tparam EvaluatorT Type of the tree.
@param EvaluatorT Tree object.
*/
template <typename EvaluatorT>
struct ExecTreeFunctor {
  EvaluatorT ev;
  ExecTreeFunctor(EvaluatorT ev) : ev(ev) {}
  void operator()(cl::sycl::nd_item<1> i) { ev.eval(i); }
};

template <typename EvaluatorT>
struct ExecTreeFunctorAlt {
  EvaluatorT ev;
  ExecTreeFunctorAlt(EvaluatorT ev) : ev(ev) {}
  void operator()(cl::sycl::nd_item<1> i) { ev.eval2(i); }
};

/*! execute_tree.
@brief Static function for executing a tree in SYCL.
@tparam Tree Type of the tree.
@param q_ SYCL queue.
@param t Tree object.
@param _localSize Local work group size.
@param _globalSize Global work size.
*/
template <typename ExpressionT>
struct execute_tree {
  static void run(SYCLDevice &dev, ExpressionT expr, size_t localSize,
                  size_t globalSize) {
    using Device = SYCLDevice;
    using EvaluatorT = Evaluator<ExpressionT, Device>;

    EvaluatorT ev(expr);
    ev.eval_subexpr_if_needed(nullptr, dev);

    dev.sycl_queue().submit([=](cl::sycl::handler &h) mutable {
      auto nTree = blas::make_accessor(ev, h);
      cl::sycl::nd_range<1> gridConfiguration = cl::sycl::nd_range<1>{
          cl::sycl::range<1>{globalSize}, cl::sycl::range<1>{localSize}};
      h.parallel_for(gridConfiguration,
                     ExecTreeFunctor<decltype(nTree)>(nTree));
    });

    ev.cleanup(dev);
  }
};

template <typename RHS, template <class> class MakePointer>
struct execute_tree<BreakIfExpr<RHS, MakePointer>> {
  using ExpressionT = BreakIfExpr<RHS, MakePointer>;
  static void run(SYCLDevice &dev, ExpressionT expr, size_t localSize,
                  size_t globalSize) {
    using Device = SYCLDevice;
    using EvaluatorT = Evaluator<ExpressionT, Device>;

    EvaluatorT ev(expr);
    ev.eval_subexpr_if_needed(nullptr, dev);
    dev.sycl_queue().submit([=](cl::sycl::handler &h) mutable {
      auto nTree = blas::make_accessor(ev, h);
      cl::sycl::nd_range<1> gridConfiguration = cl::sycl::nd_range<1>{
          cl::sycl::range<1>{globalSize}, cl::sycl::range<1>{localSize}};
      if (ev.to_break) {
        h.parallel_for(gridConfiguration,
                       ExecTreeFunctor<decltype(nTree)>(nTree));
      } else {
        h.parallel_for(gridConfiguration,
                       ExecTreeFunctorAlt<decltype(nTree)>(nTree));
      }
    });
    ev.cleanup(dev);
  }
};

/*!
 * @brief Executes the tree without defining required shared memory.
 */
template <typename ExpressionT>
void execute(SYCLDevice &dev, ExpressionT expr) {
  size_t localsize, nwg, globalsize;
  auto _N = expr.getSize();
  dev.parallel_for_setup(localsize, nwg, globalsize, _N);
  execute_tree<ExpressionT>::run(dev, expr, localsize, globalsize);
}

template <typename EvaluatorT>
struct ExecSubTreeFunctor {
  EvaluatorT ev;
  ExecSubTreeFunctor(EvaluatorT ev) : ev(ev) {}
  void operator()(cl::sycl::nd_item<1> i) {
    ev.result[i.get_global(0)] = ev.subeval(i);
  }
};
template <typename EvaluatorT>
struct SubExecutor {
  using Expression = typename EvaluatorT::Expression;
  using Device = typename EvaluatorT::Device;
  static void run(EvaluatorT &ev, Device &dev) {
    size_t localsize, nwg, globalsize;
    auto _N = ev.getSize();
    dev.parallel_for_setup(localsize, nwg, globalsize, _N);

    ev.eval_subexpr_if_needed(nullptr, dev);

    dev.sycl_queue().submit([=](cl::sycl::handler &h) mutable {
      auto nTree = blas::make_accessor(ev, h);
      cl::sycl::nd_range<1> gridConfiguration = cl::sycl::nd_range<1>{
          cl::sycl::range<1>{globalsize}, cl::sycl::range<1>{localsize}};
      h.parallel_for(gridConfiguration,
                     ExecSubTreeFunctor<decltype(nTree)>(nTree));
    });

    ev.cleanup(dev);
  }
};

}  // namespace blas

#endif  // BLAS_EXECUTOR_SYCL_HPP
