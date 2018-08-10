/***************************************************************************
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
 *  @filename queue_base.hpp
 *
 **************************************************************************/

#ifndef QUEUE_BASE_HPP
#define QUEUE_BASE_HPP
class Sequential {};
class Parallel {};
class SYCL {};
namespace blas {
template <class ExecutionPolicy>
class Queue_Interface {
  Queue_Interface() = delete;
  /*
  @brief This class is to determine whether or not the underlying device has
  dedicated shared memory
  */
  inline bool has_local_memory() const;
  /*
   @brief This class is used to allocated the a regin of memory on the device
   @tparam T the type of the pointer
   @param num_elements number of elements of the buffer
  */
  template <typename T>
  inline T *allocate(size_t num_elements) const;
  /*
  @brief this class is to deallocate the provided region of memory on the device
  @tparam T the type of the pointer
  @param p the pointer to be deleted
   */
  template <typename T>
  inline void deallocate(T *p) const;

  // This function returns Work-group size which is equal to  maximum device
  // workgroup size.
  inline size_t get_work_group_size() const;
};  // namespace blastemplate<classExecutionPolicy>classQueue_Interface

template <>
class Queue_Interface<Sequential> {
  Queue_Interface() {}
  /*
  @brief This class is to determine whether or not the underlying device has
  dedicated shared memory
  */
  inline bool has_local_memory() { return false; }
  /*
   @brief This class is used to allocated the a regin of memory on the device
   @tparam T the type of the pointer
   @param num_elements number of elements of the buffer
  */
  template <typename T>
  inline T *allocate(size_t num_elements) const {
    return std::malloc(num_elements * sizeof(T));
  }
  /*
  @brief this class is to deallocate the provided region of memory on the device
  @tparam T the type of the pointer
  @param p the pointer to be deleted
   */
  template <typename T>
  inline void deallocate(T *p) const {
    std::free(p);
  }

  // This function returns Work-group size which is equal to  maximum device
  // workgroup size.
  inline size_t get_work_group_size() { return size_t(256); }
};  // namespace blastemplate<classExecutionPolicy>classQueue_Interface

template <>
class Queue_Interface<Parallel> : Queue_Interface<Sequential> {};
}  // namespace blas
#endif
