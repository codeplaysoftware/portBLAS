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

#include <executors/blas_device_sycl.hpp>

namespace blas {
/*! execute_tree.
@brief Static function for executing a tree in SYCL.
@tparam int usingSharedMem specifying whether shared memory is enabled.
@tparam Tree Type of the tree.
@param q_ SYCL queue.
@param t Tree object.
@param _localSize Local work group size.
@param _globalSize Global work size.
@param _shMem Size in elements of the shared memory (should be zero if
usingSharedMem == false).
*/
template <typename ExpressionT, typename Device>
static void execute_tree(ExpressionT expr, Device &dev, size_t localSize,
                         size_t globalSize) {
  using EvaluatorT = Evaluator<ExpressionT, Device> using value_type =
      typename EvaluatorT::type;

  EvaluatorT ev(expr);
  ev.eval_subexpr_if_needed(NULL, dev);

  dev.sycl_queue().submit([=](cl::sycl::handler &h) mutable {
    auto nTree = blas::make_accessor(ev, h);
    auto scratch = shared_mem<value_type, usingSharedMem>(sharedsize, h);
    cl::sycl::nd_range<1> gridConfiguration = cl::sycl::nd_range<1>{
        cl::sycl::range<1>{globalSize}, cl::sycl::range<1>{localSize}};
    h.parallel_for(
        gridConfiguration,
        ExecTreeFunctor<usingSharedMem, decltype(nTree), decltype(scratch),
                        value_type>(scratch, nTree));
  });
}

/*! execute_tree.
@brief the functor for executing a tree in SYCL.
@tparam EvaluatorT Type of the tree.
@param EvaluatorT Tree object.
*/
template <typename EvaluatorT>
struct ExecTreeFunctor {
  EvaluatorT ev;
  ExecTreeFunctor(EvaluatorT ev_) :, ev(ev_) {}
  void operator()(cl::sycl::nd_item<1> i) { ev.eval(i); }
};

/*!
 * @brief Executes the tree without defining required shared memory.
 */
void execute(ExpressionT expr, Device &dev) {
  size_t localSize, nWG, globalSize;
  auto _N = expr.getSize();
  dev.parallel_for_setup(localSize, nWG, globalSize, _N);
  execute_tree<using_shared_mem::disabled>(dev, expr, localSize, globalSize, 0);
}

}  // namespace blas

#endif  // BLAS_EXECUTOR_SYCL_HPP
