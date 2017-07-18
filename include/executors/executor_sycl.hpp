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
  ExecTreeFunctor(EvaluatorT ev_) : ev(ev_) {}
  void operator()(cl::sycl::nd_item<1> i) { ev.eval(i); }
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
static void execute_tree(SYCLDevice &dev, ExpressionT expr, size_t localSize,
                         size_t globalSize) {
  using Device = SYCLDevice;
  using EvaluatorT = Evaluator<ExpressionT, Device>;

  EvaluatorT ev(expr);
  ev.eval_subexpr_if_needed(NULL, dev);

  dev.sycl_queue().submit([=](cl::sycl::handler &h) mutable {
    auto nTree = blas::make_accessor(ev, h);
    cl::sycl::nd_range<1> gridConfiguration = cl::sycl::nd_range<1>{
        cl::sycl::range<1>{globalSize}, cl::sycl::range<1>{localSize}};
    h.parallel_for(gridConfiguration, ExecTreeFunctor<decltype(nTree)>(nTree));
  });
}

/*!
 * @brief Executes the tree without defining required shared memory.
 */
template <typename ExpressionT>
void execute(SYCLDevice &dev, ExpressionT expr) {
  size_t localSize, nWG, globalSize;
  auto _N = expr.getSize();
  dev.parallel_for_setup(localSize, nWG, globalSize, _N);
  execute_tree(dev, expr, localSize, globalSize);
}

}  // namespace blas

#endif  // BLAS_EXECUTOR_SYCL_HPP
