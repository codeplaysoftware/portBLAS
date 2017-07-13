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
 *  @filename executor_sycl_base.hpp
 *
 **************************************************************************/

#ifndef BASE_EXECUTOR_SYCL_BASE_HPP
#define BASE_EXECUTOR_SYCL_BASE_HPP

#include <stdexcept>

#include <CL/sycl.hpp>

#include <evaluators/blas1_tree_evaluator.hpp>
#include <evaluators/blas2_tree_evaluator.hpp>
#include <evaluators/blas3_tree_evaluator.hpp>
#include <evaluators/blas_tree_evaluator.hpp>
#include <executors/blas1_tree_executor.hpp>
#include <executors/blas2_tree_executor.hpp>
#include <executors/blas3_tree_executor.hpp>
#include <views/view_sycl.hpp>

namespace blas {

/*!using_shared_mem.
 * @brief Indicates whether if the kernel uses shared memory or not.
 This is a work-around for the following enum class.
 //enum class using_shared_mem { enabled, disabled };
 When the member of enum class is used as a template parameters of the
 kernel functor, our duplicator cannot capture those elements.
 One way to solve it is to replace enum with namespace and replace each member
 of enum with static if. the other way is changing the order of stub file which
 is not recommended.
 */
namespace using_shared_mem {
static const int enabled = 0;
static const int disabled = 1;
};

/*!
@brief Template alias for a local accessor.
@tparam valueT Value type of the accessor.
*/
template <typename valueT>
using local_accessor_t = cl::sycl::accessor<valueT, 1, cl::sycl::access::mode::read_write, cl::sycl::access::target::local>;

/*!
@brief A struct for containing a local accessor if shared memory is enabled.
Non-specialised case for usingSharedMem == enabled, which contains a local
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
  shared_mem(size_t size, cl::sycl::handler &cgh)
      : localAcc(cl::sycl::range<1>(size), cgh) {}

  /*!
  @brief Subscirpt operator that forwards on to the local accessor subscript
  operator.
  @param id SYCL id.
  @return Reference to an element of the local accessor.
  */
  valueT &operator[](cl::sycl::id<1> id) { return localAcc[id]; }

  /*!
  @brief Local accessor.
  */
  local_accessor_t<valueT> localAcc;
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
  shared_mem(size_t size, cl::sycl::handler &cgh) {}
};

/*!
@brief Template struct defining the value type of shared memory if enabled.
Non-specialised case for usingSharedMem == enabled, which defines the type as
the value type of the tree.
@tparam usingSharedMemory Enum class specifying whether shared memory is
enabled.
*/
template <int usingSharedMem, typename EvaluatorT>
struct shared_mem_type {
  using type = typename blas::Converter<EvaluatorT>::value_type;
};

/*!
@brief Template struct defining the value type of shared memory if enabled.
Specialised case for usingSharedMem == disabled, which defines the type as void.
@tparam usingSharedMemory Enum class specifying whether shared memory is
enabled.
*/
template <typename EvaluatorT>
struct shared_mem_type<using_shared_mem::disabled, EvaluatorT> {
  using type = void;
};

/*!
@brief Template struct for containing an eval function, which uses shared memory
if enabled. Non-specialised case for usingSharedMem == enabled, which calls eval
on the tree with the shared_memory as well as an index.
@tparam usingSharedMem Enum class specifying whether shared memory is enabled.
@tparam treeT Type of the tree.
@tparam sharedMemT Value type of the shared memory.
*/
template <int usingSharedMem, typename EvaluatorT, typename sharedMemT>
struct tree {
  /*!
  @brief Static function that calls eval on a tree, passing the shared_memory
  object and the index.
  @param tree Tree object.
  @param scratch Shared memory object.
  @param index SYCL nd_item.
  */
  static void eval(EvaluatorT &tree, shared_mem<sharedMemT, usingSharedMem> scratch, cl::sycl::nd_item<1> index) {
    tree.eval(scratch, index);
  }
};

/*! tree.
@brief Template struct for containing an eval function, which uses shared memory
if enabled. Specialised case for usingSharedMem == false, which calls eval on
the tree with the index only.
@tparam usingSharedMem Enum class specifying whether shared memory is enabled.
@tparam treeT Type of the tree.
@tparam sharedMemT Value type of the shared memory.
*/
template <typename EvaluatorT, typename sharedMemT>
struct tree<using_shared_mem::disabled, EvaluatorT, sharedMemT> {
  /*!
  @brief Static function that calls eval on a tree, passing only the index.
  @param tree Tree object.
  @param scratch Shared memory object.
  @param index SYCL nd_item.
  */
  static void eval(EvaluatorT &tree, shared_mem<sharedMemT, using_shared_mem::disabled> scratch, cl::sycl::nd_item<1> index) {
    if ((index.get_global(0) < tree.getSize())) {
      tree.eval(index);
    }
  }
};

/*! execute_tree.
@brief the functor for executing a tree in SYCL.
@tparam int usingSharedMem  specifying whether shared memory is enabled.
@tparam Evaluator Type of the tree.
@tparam value_type type of elements in shared memory.
@tparam SharedMem shared memory type (local accessor type).
@param scratch_ shared memory object (local accessor).
@param evaluator_ Tree object.
*/
template <int usingSharedMem, typename Tree, typename SharedMem, typename value_type>
struct ExecTreeFunctor {
  SharedMem scratch;
  Tree t;
  ExecTreeFunctor(SharedMem scratch_, Tree t_) : scratch(scratch_), t(t_) {}
  void operator()(cl::sycl::nd_item<1> i) {
    tree<usingSharedMem, Tree, value_type>::eval(t, scratch, i);
  }
};

}  // namespace blas

#endif  // BLAS_EXECUTOR_SYCL_BASE_HPP
