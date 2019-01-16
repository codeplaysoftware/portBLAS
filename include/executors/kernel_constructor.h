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

/*!using_shared_mem.
 * @brief Indicates whether if the kernel uses shared memory or not.
 This is a work-around for the following enum class.
*/
namespace using_shared_mem {
static const int enabled = 0;
static const int disabled = 1;
};  // namespace using_shared_mem

/*!
@brief A struct for containing a local accessor if shared memory is enabled.
Non-specialised case for usingSharedMem == enabled, which contains a local
accessor.
@tparam valueT Value type of the local accessor.
@tparam usingSharedMem Enum class specifying whether shared memory is enabled.
*/
template <typename valueT, int usingSharedMem>
struct shared_mem;

/*!
@brief Template struct defining the value type of shared memory if enabled.
Non-specialised case for usingSharedMem == enabled, which defines the type as
the value type of the tree.
@tparam usingSharedMemory Enum class specifying whether shared memory is
enabled.
*/
template <int usingSharedMem, typename tree_t>
struct shared_mem_type {
  using type = typename tree_t::value_type;
};

/*!
@brief Template struct defining the value type of shared memory if enabled.
Specialised case for usingSharedMem == disabled, which defines the type as void.
@tparam usingSharedMemory Enum class specifying whether shared memory is
enabled.
*/
template <typename tree_t>
struct shared_mem_type<using_shared_mem::disabled, tree_t> {
  using type = void;
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
struct tree;

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
struct ExecTreeFunctor;

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
template <int usingSharedMem, typename Tree>
static cl::sycl::event execute_tree(cl::sycl::queue q_, Tree t,
                                    size_t _localSize, size_t _globalSize,
                                    size_t _shMem);

}  // namespace blas

#endif  // SYCL_BLAS_KERNEL_CONSTRUCTOR_H
