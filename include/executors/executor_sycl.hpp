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
  void operator()(cl::sycl::nd_item<1> i) {
    if(i.get_global(0) < ev.getSize()) {
      ev.eval(i);
    }
  }
};

/*! execute_tree.
 * @brief A wrapper for executing the main tree.
 */
template <typename ExpressionT>
struct execute_tree {
  static void run(SYCLDevice &dev, ExpressionT expr, size_t localsize, size_t globalsize) {
    using Device = SYCLDevice;
    using EvaluatorT = Evaluator<ExpressionT, Device>;

    EvaluatorT ev(expr);
    ev.eval_subexpr_if_needed(nullptr, dev);

    dev.sycl_queue().submit([=](cl::sycl::handler &h) mutable {
      auto nTree = blas::make_accessor(ev, h);
      cl::sycl::nd_range<1> gridConfiguration = cl::sycl::nd_range<1>{cl::sycl::range<1>{globalsize}, cl::sycl::range<1>{localsize}};
      h.parallel_for(gridConfiguration, ExecTreeFunctor<decltype(nTree)>(nTree));
    });
    dev.sycl_queue().wait_and_throw();

    ev.cleanup(dev);
  }
};

/*!
 * @param dev SYCLDevice& device which will execute the expression.
 * @param expr ExpressionT expression to be executed.
 * @brief Executes the expression on SYCLDevice.
 */
template <typename ExpressionT>
void execute(SYCLDevice &dev, ExpressionT expr) {
  size_t localsize, nwg, globalsize;
  auto _N = expr.getSize();
  dev.parallel_for_setup(localsize, nwg, globalsize, _N);
  execute_tree<ExpressionT>::run(dev, expr, localsize, globalsize);
}

/*! execute_tree.
@brief the functor for executing a subtree in SYCL.
@tparam EvaluatorT Type of the tree.
@param EvaluatorT Tree object.
*/
template <typename EvaluatorT>
struct ExecSubTreeFunctor {
  EvaluatorT ev;
  ExecSubTreeFunctor(EvaluatorT ev) : ev(ev) {}
  void operator()(cl::sycl::nd_item<1> i) {
    ev.subeval(i);
  }
};

/*!
 * @brief A wrapper for dispatching the kernel for a subtree.
 */
template <typename EvaluatorT>
struct SubExecutor {
  using Expression = typename EvaluatorT::Expression;
  using Device = typename EvaluatorT::Device;
  static void run(EvaluatorT &ev, Device &dev) {
    size_t localsize, nwg, globalsize;
    auto _N = ev.getSize();
    dev.parallel_for_setup(localsize, nwg, globalsize, _N);

    dev.sycl_queue().submit([=](cl::sycl::handler &h) mutable {
      auto nTree = blas::make_accessor(ev, h);
      cl::sycl::nd_range<1> gridConfiguration = cl::sycl::nd_range<1>{cl::sycl::range<1>{globalsize}, cl::sycl::range<1>{localsize}};
      h.parallel_for(gridConfiguration, ExecSubTreeFunctor<decltype(nTree)>(nTree));
    });
    dev.sycl_queue().wait_and_throw();
  }
};

}  // namespace blas

#endif  // BLAS_EXECUTOR_SYCL_HPP
