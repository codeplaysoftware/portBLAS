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

template <typename Blas_Policy>
class Policy_Handler {
 public:
  using Policy = Blas_Policy;
  /*!
   * @brief SYCL queue for execution of trees.
   */
  Policy_Handler() = delete;
  /*!
   * @brief SYCL queue for execution of trees.
   */

  explicit Policy_Handler(typename Blas_Policy::queue_type q);

  const typename Blas_Policy::device_type get_device_type() const;

  bool has_local_memory() const;

  typename Blas_Policy::queue_type get_queue() const;

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

  template <typename T>
  T *allocate(size_t num_elements) const;

  template <typename T>
  void deallocate(T *p) const;

  /*
  @brief this class is to return the dedicated buffer to the user
  @ tparam T is the type of the buffer
  @tparam container_type<T> is the type of the buffer that user can apply
  arithmetic operation on the host side
  */

  template <typename T>
  typename Blas_Policy::template container_type<T> get_buffer(
      typename Blas_Policy::template container_type<T> buff) const;
  /*  @brief Getting range accessor from the buffer created by virtual pointer
      @tparam T is the type of the data
      @tparam AcM is the access mode
      @param container is the  data we want to get range accessor
  */

  /*  @brief Getting range accessor from the buffer created by buffer iterator
      @tparam T is the type of the data
      @tparam AcM is the access mode
      @param container is the  data we want to get range accessor
  */

  template <typename T>
  typename Blas_Policy::template access_type<typename scalar_type<T>::type>
  get_range_access(typename Blas_Policy::template container_type<T> buff);

  /*
  @brief this function is to get the offset from the actual pointer
  @tparam T is the type of the pointer
  */

  template <typename T>
  ptrdiff_t get_offset(
      const typename Blas_Policy::template container_type<T> ptr) const;
  /*
  @brief this function is to get the offset from the actual pointer
  @tparam T is the type of the container_type<T>
  */

  /// FIXME: temporary disabling this method due to reinterpret cast bug for
  /// explicit copy between host and device when range accessor is used
  /*  @brief Copying the data back to device
      @tparam T is the type of the data
      @param src is the host pointer we want to copy from.
      @param dst is the device pointer we want to copy to.
      @param size is the number of elements to be copied
  */

  /*  @brief Copying the data back to device
      @tparam T is the type of the data
      @param src is the host pointer we want to copy from.
      @param dst is the device pointer we want to copy to.
      @param size is the number of elements to be copied
  */

  template <typename T>
  typename Blas_Policy::event_type copy_to_device(const T *src, T *dst,
                                                  size_t size);

  /*  @brief Copying the data back to device
    @tparam T is the type of the data
    @param src is the host pointer we want to copy from.
    @param dst is the buffer_iterator we want to copy to.
    @param size is the number of elements to be copied
  */

  template <typename T>
  typename Blas_Policy::event_type copy_to_device(
      const T *src, typename Blas_Policy::template container_type<T> dst,
      size_t);
  /*  @brief Copying the data back to device
      @tparam T is the type of the data
      @param src is the device pointer we want to copy from.
      @param dst is the host pointer we want to copy to.
      @param size is the number of elements to be copied
  */
  /// FIXME: temporary disabling this method due to reinterpret cast bug for
  /// explicit copy between host and device when range accessor is used
  /*  @brief Copying the data back to device
      @tparam T is the type of the data
      @param src is the device pointer we want to copy from.
      @param dst is the host pointer we want to copy to.
      @param size is the number of elements to be copied
  */
  template <typename T>
  typename Blas_Policy::event_type copy_to_host(T *src, T *dst, size_t size);
  /*  @brief Copying the data back to device
    @tparam T is the type of the data
    @param src is the buffer_iterator we want to copy from.
    @param dst is the host pointer we want to copy to.
    @param size is the number of elements to be copied
  */

  template <typename T>
  typename Blas_Policy::event_type copy_to_host(
      typename Blas_Policy::template container_type<T> src, T *dst, size_t);

  /*  @brief waiting for a sycl::queue.wait()
   */

  void wait();
};
}  // namespace blas
#endif
