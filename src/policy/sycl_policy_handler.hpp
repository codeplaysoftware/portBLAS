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

template <typename element_t>
inline element_t *PolicyHandler<codeplay_policy>::allocate(
    size_t num_elements) const {
  return static_cast<element_t *>(cl::sycl::codeplay::SYCLmalloc(
      num_elements * sizeof(element_t), *pointerMapperPtr_));
}

template <typename element_t>
inline void PolicyHandler<codeplay_policy>::deallocate(element_t *p) const {
  cl::sycl::codeplay::SYCLfree(static_cast<void *>(p), *pointerMapperPtr_);
}

/*
@brief this class is to return the dedicated buffer to the user
@ tparam element_t is the type of the pointer
@tparam bufferT<element_t> is the type of the buffer points to the data. on the
host side buffer<element_t> and element_t are the same
*/

template <typename element_t>
inline BufferIterator<element_t, codeplay_policy>
PolicyHandler<codeplay_policy>::get_buffer(element_t *ptr) const {
  using pointer_t = typename std::remove_const<element_t>::type *;
  auto original_buffer = pointerMapperPtr_->get_buffer(
      static_cast<void *>(const_cast<pointer_t>(ptr)));
  auto typed_size = original_buffer.get_count() / sizeof(element_t);
  auto buff =
      original_buffer.reinterpret<element_t>(cl::sycl::range<1>(typed_size));
  auto offset = get_offset(ptr);

  return BufferIterator<element_t, codeplay_policy>(buff, offset);
}

/*
@brief this class is to return the dedicated buffer to the user
@ tparam element_t is the type of the buffer
@tparam BufferIterator<element_t, codeplay_policy> is the type of the buffer
that user can apply arithmetic operation on the host side
*/

template <typename element_t>
inline BufferIterator<element_t, codeplay_policy>
PolicyHandler<codeplay_policy>::get_buffer(
    BufferIterator<element_t, codeplay_policy> buff) const {
  return buff;
}

/*  @brief Getting range accessor from the buffer created by virtual pointer
    @tparam element_t is the type of the data
    @tparam acc_md_t is the access mode
    @param container is the  data we want to get range accessor
*/

template <typename codeplay_policy::access_mode_t acc_md_t, typename element_t>
inline typename codeplay_policy::template default_accessor_t<
    typename ValueType<element_t>::type, acc_md_t>
PolicyHandler<codeplay_policy>::get_range_access(element_t *vptr) {
  return PolicyHandler<codeplay_policy>::template get_range_access<element_t,
                                                                   acc_md_t>(
      get_buffer(vptr));
}

/*  @brief Getting range accessor from the buffer created by buffer iterator
    @tparam element_t is the type of the data
    @tparam acc_md_t is the access mode
    @param container is the  data we want to get range accessor
*/

template <typename element_t, typename codeplay_policy::access_mode_t acc_md_t>
inline typename codeplay_policy::template default_accessor_t<
    typename ValueType<element_t>::type, acc_md_t>
PolicyHandler<codeplay_policy>::get_range_access(
    BufferIterator<element_t, codeplay_policy> buff) {
  return blas::get_range_accessor<acc_md_t>(buff);
}

/*
@brief this function is to get the offset from the actual pointer
@tparam element_t is the type of the pointer
*/

template <typename element_t>
inline std::ptrdiff_t PolicyHandler<codeplay_policy>::get_offset(
    const element_t *ptr) const {
  return (pointerMapperPtr_->get_offset(static_cast<const void *>(ptr)) /
          sizeof(element_t));
}
/*
@brief this function is to get the offset from the actual pointer
@tparam element_t is the type of the BufferIterator<element_t,codeplay_policy>
*/

template <typename element_t>
inline std::ptrdiff_t PolicyHandler<codeplay_policy>::get_offset(
    BufferIterator<element_t, codeplay_policy> buff) const {
  return buff.get_offset();
}

/*  @brief Copying the data back to device
    @tparam element_t is the type of the data
    @param src is the host pointer we want to copy from.
    @param dst is the device pointer we want to copy to.
    @param size is the number of elements to be copied
*/
template <typename element_t>
inline typename codeplay_policy::event_t
PolicyHandler<codeplay_policy>::copy_to_device(const element_t *src,
                                               element_t *dst, size_t size) {
  return copy_to_device(src, get_buffer(dst), size);
}

/*  @brief Copying the data back to device
  @tparam element_t is the type of the data
  @param src is the host pointer we want to copy from.
  @param dst is the BufferIterator we want to copy to.
  @param size is the number of elements to be copied
*/
template <typename element_t>
inline typename codeplay_policy::event_t
PolicyHandler<codeplay_policy>::copy_to_device(
    const element_t *src, BufferIterator<element_t, codeplay_policy> dst,
    size_t size) {
  auto event = q_.submit([&](cl::sycl::handler &cgh) {
    auto acc =
        blas::get_range_accessor<cl::sycl::access::mode::write>(dst, cgh, size);
    cgh.copy(src, acc);
  });
  return {event};
}

/*  @brief Copying the data back to device
    @tparam element_t is the type of the data
    @param src is the device pointer we want to copy from.
    @param dst is the host pointer we want to copy to.
    @param size is the number of elements to be copied
*/
template <typename element_t>
inline typename codeplay_policy::event_t
PolicyHandler<codeplay_policy>::copy_to_host(element_t *src, element_t *dst,
                                             size_t size) {
  return copy_to_host(get_buffer(src), dst, size);
}

/*  @brief Copying the data back to device
  @tparam element_t is the type of the data
  @param src is the BufferIterator we want to copy from.
  @param dst is the host pointer we want to copy to.
  @param size is the number of elements to be copied
*/
template <typename element_t>
inline typename codeplay_policy::event_t
PolicyHandler<codeplay_policy>::copy_to_host(
    BufferIterator<element_t, codeplay_policy> src, element_t *dst,
    size_t size) {
  auto event = q_.submit([&](cl::sycl::handler &cgh) {
    auto acc =
        blas::get_range_accessor<cl::sycl::access::mode::read>(src, cgh, size);
    cgh.copy(acc, dst);
  });
  return {event};
}

template <typename element_t>
inline typename codeplay_policy::event_t PolicyHandler<codeplay_policy>::fill(
    BufferIterator<element_t, codeplay_policy> buff, element_t value, size_t size) {
  auto event = q_.submit([&](cl::sycl::handler &cgh) {
    auto acc = blas::get_range_accessor<cl::sycl::access::mode::write>(
        buff, cgh, size);
    cgh.fill(acc, value);
  });
  return {event};
}

}  // namespace blas
#endif  // QUEUE_SYCL_HPP
