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
 *  @filename sycl_iterator.hpp
 *
 **************************************************************************/
#ifndef SYCL_BLAS_BUFFER_ITERATOR_HPP
#define SYCL_BLAS_BUFFER_ITERATOR_HPP
#include "container/sycl_iterator.h"
#include "operations/blas_constants.h"

namespace blas {

template <typename element_t>
inline BufferIterator<element_t, codeplay_policy>::BufferIterator(
    const typename BufferIterator<element_t, codeplay_policy>::buff_t& buff,
    std::ptrdiff_t offset)
    : offset_(offset), buffer_(buff) {}

template <typename element_t>
inline BufferIterator<element_t, codeplay_policy>::BufferIterator(
    const typename BufferIterator<element_t, codeplay_policy>::buff_t& buff)
    : BufferIterator(buff, 0) {}

template <typename element_t>
template <typename other_scalar_t>
inline BufferIterator<element_t, codeplay_policy>::BufferIterator(
    const BufferIterator<other_scalar_t, codeplay_policy>& other)
    : BufferIterator(other.get_buffer(), other.get_offset()) {}

template <typename element_t>
inline BufferIterator<element_t, codeplay_policy>&
BufferIterator<element_t, codeplay_policy>::operator+=(std::ptrdiff_t offset) {
  offset_ += offset;
  return *this;
}

template <typename element_t>
inline BufferIterator<element_t, codeplay_policy>
BufferIterator<element_t, codeplay_policy>::operator+(
    std::ptrdiff_t offset) const {
  return BufferIterator<element_t, codeplay_policy>(buffer_, offset_ + offset);
}

template <typename element_t>
inline BufferIterator<element_t, codeplay_policy>
BufferIterator<element_t, codeplay_policy>::operator-(
    std::ptrdiff_t offset) const {
  return BufferIterator<element_t, codeplay_policy>(buffer_, offset_ - offset);
}

template <typename element_t>
inline BufferIterator<element_t, codeplay_policy>&
BufferIterator<element_t, codeplay_policy>::operator-=(std::ptrdiff_t offset) {
  offset_ -= offset;
  return *this;
}

// Prefix operator (Increment and return value)
template <typename element_t>
inline BufferIterator<element_t, codeplay_policy>&
BufferIterator<element_t, codeplay_policy>::operator++() {
  offset_++;
  return (*this);
}
// Postfix operator (Return value and increment)
template <typename element_t>
inline BufferIterator<element_t, codeplay_policy>
BufferIterator<element_t, codeplay_policy>::operator++(int i) {
  BufferIterator<element_t, codeplay_policy> temp_iterator(*this);
  offset_ += 1;
  return temp_iterator;
}

template <typename element_t>
inline std::ptrdiff_t BufferIterator<element_t, codeplay_policy>::get_size()
    const {
  return (buffer_.get_count() - offset_);
}

template <typename element_t>
inline std::ptrdiff_t BufferIterator<element_t, codeplay_policy>::get_offset()
    const {
  return offset_;
}

template <typename element_t>
inline typename BufferIterator<element_t, codeplay_policy>::buff_t
BufferIterator<element_t, codeplay_policy>::get_buffer() const {
  return buffer_;
}

template <typename element_t>
inline void BufferIterator<element_t, codeplay_policy>::set_offset(
    std::ptrdiff_t offset) {
  offset_ = offset;
}

}  // end namespace blas
#endif  // BLAS_SYCL_ITERATOR_HPP
