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
 *  @filename sycl_policy_handler.hpp
 *
 **************************************************************************/

#ifndef SYCL_BLAS_SYCL_POLICY_HANDLER_HPP
#define SYCL_BLAS_SYCL_POLICY_HANDLER_HPP

#include "policy/sycl_policy_handler.h"

namespace blas {

inline Policy_Handler<BLAS_SYCL_Policy>::Policy_Handler(cl::sycl::queue q)
    : q_(q),
      pointerMapperPtr_(std::shared_ptr<cl::sycl::codeplay::PointerMapper>(
          new cl::sycl::codeplay::PointerMapper(),
          [](cl::sycl::codeplay::PointerMapper *p) {
            p->clear();
            delete p;
          })),
      workGroupSize_(BLAS_SYCL_Policy::get_work_group_size(q)),
      selectedDeviceType_(BLAS_SYCL_Policy::find_chosen_device_type(q)),
      localMemorySupport_(BLAS_SYCL_Policy::has_local_memory(q)),
      computeUnits_(BLAS_SYCL_Policy::get_num_compute_units(q)) {}

template <typename T>
inline T *Policy_Handler<BLAS_SYCL_Policy>::allocate(
    size_t num_elements) const {
  return static_cast<T *>(cl::sycl::codeplay::SYCLmalloc(
      num_elements * sizeof(T), *pointerMapperPtr_));
}

template <typename T>
inline void Policy_Handler<BLAS_SYCL_Policy>::deallocate(T *p) const {
  cl::sycl::codeplay::SYCLfree(static_cast<void *>(p), *pointerMapperPtr_);
}

/*
@brief this class is to return the dedicated buffer to the user
@ tparam T is the type of the pointer
@tparam bufferT<T> is the type of the buffer points to the data. on the host
side buffer<T> and T are the same
*/

template <typename T>
inline buffer_iterator<T, BLAS_SYCL_Policy>
Policy_Handler<BLAS_SYCL_Policy>::get_buffer(T *ptr) const {
  using pointer_t = typename std::remove_const<T>::type *;
  auto original_buffer = pointerMapperPtr_->get_buffer(
      static_cast<void *>(const_cast<pointer_t>(ptr)));
  auto typed_size = original_buffer.get_count() / sizeof(T);
  auto buff = original_buffer.reinterpret<T>(cl::sycl::range<1>(typed_size));
  auto offset = get_offset(ptr);

  return buffer_iterator<T, BLAS_SYCL_Policy>(buff, offset, ptr - offset);
}

/*
@brief this class is to return the dedicated buffer to the user
@ tparam T is the type of the buffer
@tparam buffer_iterator<T, BLAS_SYCL_Policy> is the type of the buffer that user
can apply arithmetic operation on the host side
*/

template <typename T>
inline buffer_iterator<T, BLAS_SYCL_Policy>
Policy_Handler<BLAS_SYCL_Policy>::get_buffer(
    buffer_iterator<T, BLAS_SYCL_Policy> buff) const {
  return buff;
}

/*  @brief Getting range accessor from the buffer created by virtual pointer
    @tparam T is the type of the data
    @tparam AcM is the access mode
    @param container is the  data we want to get range accessor
*/

template <typename BLAS_SYCL_Policy::access_mode_type AcM, typename T>
inline typename BLAS_SYCL_Policy::template access_type<
    typename scalar_type<T>::type, AcM>
Policy_Handler<BLAS_SYCL_Policy>::get_range_access(T *vptr) {
  return Policy_Handler<BLAS_SYCL_Policy>::template get_range_access<T, AcM>(
      get_buffer(vptr));
}

/*  @brief Getting range accessor from the buffer created by buffer iterator
    @tparam T is the type of the data
    @tparam AcM is the access mode
    @param container is the  data we want to get range accessor
*/

template <typename T, typename BLAS_SYCL_Policy::access_mode_type AcM>
inline typename BLAS_SYCL_Policy::template access_type<
    typename scalar_type<T>::type, AcM>
Policy_Handler<BLAS_SYCL_Policy>::get_range_access(
    buffer_iterator<T, BLAS_SYCL_Policy> buff) {
  return blas::get_range_accessor<AcM>(buff);
}

/*
@brief this function is to get the offset from the actual pointer
@tparam T is the type of the pointer
*/

template <typename T>
inline std::ptrdiff_t Policy_Handler<BLAS_SYCL_Policy>::get_offset(
    const T *ptr) const {
  return (pointerMapperPtr_->get_offset(static_cast<const void *>(ptr)) /
          sizeof(T));
}
/*
@brief this function is to get the offset from the actual pointer
@tparam T is the type of the buffer_iterator<T,BLAS_SYCL_Policy>
*/

template <typename T>
inline std::ptrdiff_t Policy_Handler<BLAS_SYCL_Policy>::get_offset(
    buffer_iterator<T, BLAS_SYCL_Policy> buff) const {
  return buff.get_offset();
}

/*  @brief Copying the data back to device
    @tparam T is the type of the data
    @param src is the host pointer we want to copy from.
    @param dst is the device pointer we want to copy to.
    @param size is the number of elements to be copied
*/
template <typename T>
inline typename BLAS_SYCL_Policy::event_type
Policy_Handler<BLAS_SYCL_Policy>::copy_to_device(const T *src, T *dst,
                                                 size_t size) {
  return copy_to_device(src, get_buffer(dst), size);
}

/*  @brief Copying the data back to device
  @tparam T is the type of the data
  @param src is the host pointer we want to copy from.
  @param dst is the buffer_iterator we want to copy to.
  @param size is the number of elements to be copied
*/
template <typename T>
inline typename BLAS_SYCL_Policy::event_type
Policy_Handler<BLAS_SYCL_Policy>::copy_to_device(
    const T *src, buffer_iterator<T, BLAS_SYCL_Policy> dst, size_t size) {
  auto event = q_.submit([&](cl::sycl::handler &cgh) {
    auto acc =
        blas::get_range_accessor<cl::sycl::access::mode::write>(dst, cgh, size);
    cgh.copy(src, acc);
  });
  return {event};
}

/*  @brief Copying the data back to device
    @tparam T is the type of the data
    @param src is the device pointer we want to copy from.
    @param dst is the host pointer we want to copy to.
    @param size is the number of elements to be copied
*/
template <typename T>
inline typename BLAS_SYCL_Policy::event_type
Policy_Handler<BLAS_SYCL_Policy>::copy_to_host(T *src, T *dst, size_t size) {
  return copy_to_host(get_buffer(src), dst, size);
}

/*  @brief Copying the data back to device
  @tparam T is the type of the data
  @param src is the buffer_iterator we want to copy from.
  @param dst is the host pointer we want to copy to.
  @param size is the number of elements to be copied
*/
template <typename T>
inline typename BLAS_SYCL_Policy::event_type
Policy_Handler<BLAS_SYCL_Policy>::copy_to_host(
    buffer_iterator<T, BLAS_SYCL_Policy> src, T *dst, size_t size) {
  auto event = q_.submit([&](cl::sycl::handler &cgh) {
    auto acc =
        blas::get_range_accessor<cl::sycl::access::mode::read>(src, cgh, size);
    cgh.copy(acc, dst);
  });
  return {event};
}
}  // namespace blas
#endif  // QUEUE_SYCL_HPP
