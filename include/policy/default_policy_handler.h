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
 *  @filename default_policy_handler.h
 *
 **************************************************************************/

#ifndef SYCL_BLAS_DEFAULT_POLICY_H
#define SYCL_BLAS_DEFAULT_POLICY_H

namespace blas {

template <typename blas_policy_t>
class PolicyHandler {
 public:
  using policy_t = blas_policy_t;
  /*!
   * @brief SYCL queue for execution of trees.
   */
  PolicyHandler() = delete;
  /*!
   * @brief SYCL queue for execution of trees.
   */

  explicit PolicyHandler(typename policy_t::queue_t q);

  const typename policy_t::device_type get_device_type() const;

  bool has_local_memory() const;

  typename policy_t::queue_t get_queue() const;

  // Force the system not to set this to bigger than 256. As it can be

  size_t get_work_group_size() const;

  /*  @brief waiting for a list of sycl events
    @param first_event  and next_events are instances of sycl::sycl::event
  */
  /*  @brief waiting for a list of sycl events
   @param first_event  and next_events are instances of sycl::sycl::event
 */

  template <typename first_event_t, typename... next_event_t>
  void wait(first_event_t first_event, next_event_t... next_events);

  template <typename element_t>
  element_t *allocate(size_t num_elements) const;

  template <typename element_t>
  void deallocate(element_t *p) const;

  /*
  @brief this class is to return the dedicated buffer to the user
  @ tparam element_t is the type of the buffer
  @tparam container_type<element_t> is the type of the buffer that user can
  apply arithmetic operation on the host side
  */

  template <typename element_t>
  BufferIterator<element_t, policy_t> get_buffer(
      BufferIterator<element_t, policy_t> buff) const;
  /*  @brief Getting range accessor from the buffer created by virtual pointer
      @tparam element_t is the type of the data
      @tparam acc_md_t is the access mode
      @param container is the  data we want to get range accessor
  */

  /*  @brief Getting range accessor from the buffer created by buffer iterator
      @tparam element_t is the type of the data
      @tparam acc_md_t is the access mode
      @param container is the  data we want to get range accessor
  */

  template <typename element_t>
  typename policy_t::template default_accessor_t<
      typename ValueType<element_t>::type>
  get_range_access(BufferIterator<element_t, policy_t> buff);

  /*
  @brief this function is to get the offset from the actual pointer
  @tparam element_t is the type of the pointer
  */

  template <typename element_t>
  ptrdiff_t get_offset(const BufferIterator<element_t, policy_t> ptr) const;
  /*
  @brief this function is to get the offset from the actual pointer
  @tparam element_t is the type of the container_type<element_t>
  */

  /*  @brief Copying the data back to device
    @tparam element_t is the type of the data
    @param src is the host pointer we want to copy from.
    @param dst is the BufferIterator we want to copy to.
    @param size is the number of elements to be copied
  */

  template <typename element_t>
  typename policy_t::event_t copy_to_device(
      const element_t *src, BufferIterator<element_t, policy_t> dst, size_t);

  /*  @brief Copying the data back to device
    @tparam element_t is the type of the data
    @param src is the BufferIterator we want to copy from.
    @param dst is the host pointer we want to copy to.
    @param size is the number of elements to be copied
  */

  template <typename element_t>
  typename policy_t::event_t copy_to_host(
      BufferIterator<element_t, policy_t> src, element_t *dst, size_t);

  /*  @brief waiting for a sycl::queue.wait()
   */

  void wait();
};
}  // namespace blas
#endif
