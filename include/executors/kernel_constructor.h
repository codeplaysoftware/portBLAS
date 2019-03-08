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
 *  @filename kernel_constructor.h
 *
 **************************************************************************/

#ifndef SYCL_BLAS_KERNEL_CONSTRUCTOR_H
#define SYCL_BLAS_KERNEL_CONSTRUCTOR_H

#include <CL/sycl.hpp>

namespace blas {

/*!using_local_memory.
 * @brief Indicates whether if the kernel uses shared memory or not.
 This is a work-around for the following enum class.
*/
namespace using_local_memory {
static const int enabled = 0;
static const int disabled = 1;
};  // namespace using_local_memory

/*!
@brief A struct for containing a local accessor if shared memory is enabled.
Non-specialised case for using_local_memory == enabled, which contains a local
accessor.
@tparam value_t Value type of the local accessor.
@tparam using_local_memory Enum class specifying whether shared memory is
enabled.
*/
template <typename value_t, int using_local_memory>
struct LocalMemory;

/*!
@brief Template struct defining the value type of shared memory if enabled.
Non-specialised case for using_local_memory == enabled, which defines the type
as the value type of the tree.
@tparam usingSharedMemory Enum class specifying whether shared memory is
enabled.
*/
template <int using_local_memory, typename expression_tree_t>
struct LocalMemoryType {
  using type = typename expression_tree_t::value_t;
};

/*!
@brief Template struct defining the value type of shared memory if enabled.
Specialised case for using_local_memory == disabled, which defines the type as
void.
@tparam usingSharedMemory Enum class specifying whether shared memory is
enabled.
*/
template <typename expression_tree_t>
struct LocalMemoryType<using_local_memory::disabled, expression_tree_t> {
  using type = void;
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
struct ExpressionTreeEvaluator;

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
struct ExpressionTreeFunctor;

/*! execute_tree.
@brief Static function for executing a tree in SYCL.
@tparam int using_local_memory specifying whether shared memory is enabled.
@tparam Tree Type of the tree.
@param q_ SYCL queue.
@param t Tree object.
@param _localSize Local work group size.
@param _globalSize Global work size.
@param _shMem Size in elements of the shared memory (should be zero if
using_local_memory == false).
*/
template <int using_local_memory, typename queue_t, typename expression_tree_t>
static cl::sycl::event execute_tree(queue_t q, expression_tree_t t,
                                    size_t _localSize, size_t _globalSize,
                                    size_t _shMem);

}  // namespace blas

#endif  // SYCL_BLAS_KERNEL_CONSTRUCTOR_H
