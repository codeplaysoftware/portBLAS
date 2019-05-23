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
 *  @filename kernel_constructor.hpp
 *
 **************************************************************************/
#ifndef SYCL_BLAS_KERNEL_CONSTRUCTOR_HPP
#define SYCL_BLAS_KERNEL_CONSTRUCTOR_HPP

#include "executors/kernel_constructor.h"
#include <CL/sycl.hpp>
#include <iostream>

namespace blas {
/*!
@brief A struct for containing a local accessor if shared memory is enabled.
Non-specialized case for using_local_memory == enabled, which contains a local
accessor.
@tparam value_t Value type of the local accessor.
@tparam using_local_memory Enum class specifying whether shared memory is
enabled.
*/
template <typename value_t, int using_local_memory>
struct LocalMemory {
  /*!
  @brief Constructor that creates a local accessor from a size and a SYCL
  command group handler.
  @param size Size in elements of the local accessor.
  @param cgh SYCL command group handler.
  */
  SYCL_BLAS_INLINE LocalMemory(size_t size, cl::sycl::handler &cgh)
      : localAcc(cl::sycl::range<1>(size), cgh) {}

  /*!
  @brief Subscirpt operator that forwards on to the local accessor subscript
  operator.
  @param id SYCL id.
  @return Reference to an element of the local accessor.
  */
  SYCL_BLAS_INLINE value_t &operator[](cl::sycl::id<1> id) {
    return localAcc[id];
  }

  /*!
  @brief Local accessor.
  */
  cl::sycl::accessor<value_t, 1, cl::sycl::access::mode::read_write,
                     cl::sycl::access::target::local>
      localAcc;
};

/*!
@brief A struct for containing a local accessor if shared memory is enabled.
Specialised case for using_local_memory == disabled, which does not contain a
local accessor.
@tparam value_t Value type of the accessor.
*/
template <typename value_t>
struct LocalMemory<value_t, using_local_memory::disabled> {
  /*!
  @brief Constructor that does nothing.
  @param size Size in elements of the local accessor.
  @param cgh SYCL command group handler.
  */
  SYCL_BLAS_INLINE LocalMemory(size_t size, cl::sycl::handler &cgh) {}
};

/*!
@brief Template struct for containing an eval function, which uses shared memory
if enabled. Non-specialised case for using_local_memory == enabled, which calls
eval on the tree with the shared_memory as well as an index.
@tparam using_local_memory Enum class specifying whether shared memory is
enabled.
@tparam expression_tree_t Type of the tree.
@tparam local_memory_t Value type of the shared memory.
*/
template <int using_local_memory, typename expression_tree_t,
          typename local_memory_t>
struct ExpressionTreeEvaluator {
  /*!
  @brief Static function that calls eval on a tree, passing the shared_memory
  object and the index.
  @param tree Tree object.
  @param scratch Shared memory object.
  @param index SYCL nd_item.
  */
  static SYCL_BLAS_INLINE void eval(
      expression_tree_t &tree,
      LocalMemory<local_memory_t, using_local_memory> scratch,
      cl::sycl::nd_item<1> index) {
    tree.eval(scratch, index);
  }
};

/*! tree.
@brief Template struct for containing an eval function, which uses shared memory
if enabled. Specialised case for using_local_memory == false, which calls eval
on the tree with the index only.
@tparam using_local_memory Enum class specifying whether shared memory is
enabled.
@tparam expression_tree_t Type of the tree.
@tparam local_memory_t Value type of the shared memory.
*/
template <typename expression_tree_t, typename local_memory_t>
struct ExpressionTreeEvaluator<using_local_memory::disabled, expression_tree_t,
                               local_memory_t> {
  /*!
  @brief Static function that calls eval on a tree, passing only the index.
  @param tree Tree object.
  @param scratch Shared memory object.
  @param index SYCL nd_item.
  */
  static SYCL_BLAS_INLINE void eval(
      expression_tree_t &tree,
      LocalMemory<local_memory_t, using_local_memory::disabled> scratch,
      cl::sycl::nd_item<1> index) {
    if (tree.valid_thread(index)) {
      tree.eval(index);
    }
  }
};

/*! ExpressionTreeFunctor.
@brief the functor for executing a tree in SYCL.
@tparam int using_local_memory  specifying whether shared memory is enabled.
@tparam expression Tree Type of the tree.
@tparam value_t type of elements in shared memory.
@tparam local_memory_t shared memory type (local accessor type).
@param scratch_ shared memory object (local accessor).
@param t_ Tree object.
*/
template <int using_local_memory, typename expression_tree_t,
          typename local_memory_t, typename value_t>
struct ExpressionTreeFunctor {
  local_memory_t scratch;
  expression_tree_t t;
  SYCL_BLAS_INLINE ExpressionTreeFunctor(local_memory_t scratch_,
                                         expression_tree_t t_)
      : scratch(scratch_), t(t_) {}
  SYCL_BLAS_INLINE void operator()(cl::sycl::nd_item<1> i) {
    ExpressionTreeEvaluator<using_local_memory, expression_tree_t,
                            value_t>::eval(t, scratch, i);
  }
};

template <int using_local_memory, typename queue_t, typename expression_tree_t>
static SYCL_BLAS_INLINE cl::sycl::event execute_tree(queue_t q_,
                                                     expression_tree_t t,
                                                     size_t _localSize,
                                                     size_t _globalSize,
                                                     size_t _shMem) {
  using value_t =
      typename LocalMemoryType<using_local_memory, expression_tree_t>::type;

  auto localSize = _localSize;
  auto globalSize = _globalSize;
  auto shMem = _shMem;
  cl::sycl::event ev;
  try {
    auto cg1 = [=](cl::sycl::handler &h) mutable {
      t.bind(h);
      auto scratch = LocalMemory<value_t, using_local_memory>(shMem, h);

      cl::sycl::nd_range<1> gridConfiguration = cl::sycl::nd_range<1>{
          cl::sycl::range<1>{globalSize}, cl::sycl::range<1>{localSize}};
      h.parallel_for(
          gridConfiguration,
          ExpressionTreeFunctor<using_local_memory, expression_tree_t,
                                decltype(scratch), value_t>(scratch, t));
    };

    ev = q_.submit(cg1);
    return ev;
  } catch (cl::sycl::exception e) {
    std::cerr << e.what() << std::endl;
    return ev;
  }
}
}  // namespace blas
#endif  // KERNEL_CONSTRUCTOR_HPP
