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
Non-specialized case for usingSharedMem == enabled, which contains a local
accessor.
@tparam valueT Value type of the local accessor.
@tparam usingSharedMem Enum class specifying whether shared memory is enabled.
*/
template <typename valueT, int usingSharedMem>
struct shared_mem {
  /*!
  @brief Constructor that creates a local accessor from a size and a SYCL
  command group handler.
  @param size Size in elements of the local accessor.
  @param cgh SYCL command group handler.
  */
  sycl_blas_inline shared_mem(size_t size, cl::sycl::handler &cgh)
      : localAcc(cl::sycl::range<1>(size), cgh) {}

  /*!
  @brief Subscirpt operator that forwards on to the local accessor subscript
  operator.
  @param id SYCL id.
  @return Reference to an element of the local accessor.
  */
  sycl_blas_inline valueT &operator[](cl::sycl::id<1> id) {
    return localAcc[id];
  }

  /*!
  @brief Local accessor.
  */
  cl::sycl::accessor<valueT, 1, cl::sycl::access::mode::read_write,
                     cl::sycl::access::target::local>
      localAcc;
};

/*!
@brief A struct for containing a local accessor if shared memory is enabled.
Specialised case for usingSharedMem == disabled, which does not contain a local
accessor.
@tparam valueT Value type of the accessor.
*/
template <typename valueT>
struct shared_mem<valueT, using_shared_mem::disabled> {
  /*!
  @brief Constructor that does nothing.
  @param size Size in elements of the local accessor.
  @param cgh SYCL command group handler.
  */
  sycl_blas_inline shared_mem(size_t size, cl::sycl::handler &cgh) {}
};

/*!
@brief Template struct for containing an eval function, which uses shared memory
if enabled. Non-specialised case for usingSharedMem == enabled, which calls eval
on the tree with the shared_memory as well as an index.
@tparam usingSharedMem Enum class specifying whether shared memory is enabled.
@tparam tree_t Type of the tree.
@tparam sharedMemT Value type of the shared memory.
*/
template <int usingSharedMem, typename tree_t, typename sharedMemT>
struct tree {
  /*!
  @brief Static function that calls eval on a tree, passing the shared_memory
  object and the index.
  @param tree Tree object.
  @param scratch Shared memory object.
  @param index SYCL nd_item.
  */
  static sycl_blas_inline void eval(
      tree_t &tree, shared_mem<sharedMemT, usingSharedMem> scratch,
      cl::sycl::nd_item<1> index) {
    tree.eval(scratch, index);
  }
};

/*! tree.
@brief Template struct for containing an eval function, which uses shared memory
if enabled. Specialised case for usingSharedMem == false, which calls eval on
the tree with the index only.
@tparam usingSharedMem Enum class specifying whether shared memory is enabled.
@tparam tree_t Type of the tree.
@tparam sharedMemT Value type of the shared memory.
*/
template <typename tree_t, typename sharedMemT>
struct tree<using_shared_mem::disabled, tree_t, sharedMemT> {
  /*!
  @brief Static function that calls eval on a tree, passing only the index.
  @param tree Tree object.
  @param scratch Shared memory object.
  @param index SYCL nd_item.
  */
  static sycl_blas_inline void eval(
      tree_t &tree, shared_mem<sharedMemT, using_shared_mem::disabled> scratch,
      cl::sycl::nd_item<1> index) {
    if (tree.valid_thread(index)) {
      tree.eval(index);
    }
  }
};

/*! ExecTreeFunctor.
@brief the functor for executing a tree in SYCL.
@tparam int usingSharedMem  specifying whether shared memory is enabled.
@tparam expression Tree Type of the tree.
@tparam value_type type of elements in shared memory.
@tparam SharedMem shared memory type (local accessor type).
@param scratch_ shared memory object (local accessor).
@param t_ Tree object.
*/
template <int usingSharedMem, typename Tree, typename SharedMem,
          typename value_type>
struct ExecTreeFunctor {
  SharedMem scratch;
  Tree t;
  sycl_blas_inline ExecTreeFunctor(SharedMem scratch_, Tree t_)
      : scratch(scratch_), t(t_) {}
  sycl_blas_inline void operator()(cl::sycl::nd_item<1> i) {
    tree<usingSharedMem, Tree, value_type>::eval(t, scratch, i);
  }
};

template <int usingSharedMem, typename Tree>
static sycl_blas_inline cl::sycl::event execute_tree(cl::sycl::queue q_, Tree t,
                                                     size_t _localSize,
                                                     size_t _globalSize,
                                                     size_t _shMem) {
  using value_type = typename shared_mem_type<usingSharedMem, Tree>::type;

  auto localSize = _localSize;
  auto globalSize = _globalSize;
  auto shMem = _shMem;
  cl::sycl::event ev;
  try {
    auto cg1 = [=](cl::sycl::handler &h) mutable {
      t.bind(h);
      auto scratch = shared_mem<value_type, usingSharedMem>(shMem, h);

      cl::sycl::nd_range<1> gridConfiguration = cl::sycl::nd_range<1>{
          cl::sycl::range<1>{globalSize}, cl::sycl::range<1>{localSize}};
      h.parallel_for(
          gridConfiguration,
          ExecTreeFunctor<usingSharedMem, Tree, decltype(scratch), value_type>(
              scratch, t));
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
