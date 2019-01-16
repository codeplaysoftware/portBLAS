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

template <typename T>
inline buffer_iterator<T, BLAS_SYCL_Policy>::buffer_iterator(
    const typename buffer_iterator<T, BLAS_SYCL_Policy>::buff_t& buff_,
    std::ptrdiff_t offset_)
    : m_offset(offset_), m_buffer(buff_) {}

template <typename T>
inline buffer_iterator<T, BLAS_SYCL_Policy>::buffer_iterator(
    const typename buffer_iterator<T, BLAS_SYCL_Policy>::buff_t& buff_,
    std::ptrdiff_t offset_, T* legacy_ptr)
    : m_offset(offset_), m_buffer(buff_) {}

template <typename T>
inline buffer_iterator<T, BLAS_SYCL_Policy>::buffer_iterator(
    const typename buffer_iterator<T, BLAS_SYCL_Policy>::buff_t& buff_)
    : buffer_iterator(buff_, 0) {}

template <typename T>
template <typename other_scalar_t>
inline buffer_iterator<T, BLAS_SYCL_Policy>::buffer_iterator(
    const buffer_iterator<other_scalar_t, BLAS_SYCL_Policy>& other)
    : buffer_iterator(other.get_buffer(), other.get_offset()) {}

template <typename T>
inline buffer_iterator<T, BLAS_SYCL_Policy>&
buffer_iterator<T, BLAS_SYCL_Policy>::operator+=(std::ptrdiff_t offset_) {
  m_offset += offset_;
  return *this;
}

template <typename T>
inline buffer_iterator<T, BLAS_SYCL_Policy>
buffer_iterator<T, BLAS_SYCL_Policy>::operator+(std::ptrdiff_t offset_) const {
  return buffer_iterator<T, BLAS_SYCL_Policy>(m_buffer, m_offset + offset_);
}

template <typename T>
inline buffer_iterator<T, BLAS_SYCL_Policy>
buffer_iterator<T, BLAS_SYCL_Policy>::operator-(std::ptrdiff_t offset_) const {
  return buffer_iterator<T, BLAS_SYCL_Policy>(m_buffer, m_offset - offset_);
}

template <typename T>
inline buffer_iterator<T, BLAS_SYCL_Policy>&
buffer_iterator<T, BLAS_SYCL_Policy>::operator-=(std::ptrdiff_t offset_) {
  m_offset -= offset_;
  return *this;
}

// Prefix operator (Increment and return value)
template <typename T>
inline buffer_iterator<T, BLAS_SYCL_Policy>&
buffer_iterator<T, BLAS_SYCL_Policy>::operator++() {
  m_offset++;
  return (*this);
}
// Postfix operator (Return value and increment)
template <typename T>
inline buffer_iterator<T, BLAS_SYCL_Policy>
buffer_iterator<T, BLAS_SYCL_Policy>::operator++(int i) {
  buffer_iterator<T, BLAS_SYCL_Policy> temp_iterator(*this);
  m_offset += 1;
  return temp_iterator;
}

template <typename T>
inline std::ptrdiff_t buffer_iterator<T, BLAS_SYCL_Policy>::get_size() const {
  return (m_buffer.get_count() - m_offset);
}

template <typename T>
inline std::ptrdiff_t buffer_iterator<T, BLAS_SYCL_Policy>::get_offset() const {
  return m_offset;
}

template <typename T>
inline typename buffer_iterator<T, BLAS_SYCL_Policy>::buff_t
buffer_iterator<T, BLAS_SYCL_Policy>::get_buffer() const {
  return m_buffer;
}

template <typename T>
inline void buffer_iterator<T, BLAS_SYCL_Policy>::set_offset(
    std::ptrdiff_t offset_) {
  m_offset = offset_;
}

}  // end namespace blas
#endif  // BLAS_SYCL_ITERATOR_HPP
