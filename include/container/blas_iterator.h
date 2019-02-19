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
 *  @filename blas_iterator.h
 *
 **************************************************************************/
#ifndef SYCL_BLAS_ITERATOR_H
#define SYCL_BLAS_ITERATOR_H
#include "blas_meta.h"
#include <CL/sycl.hpp>

namespace blas {

/*
 * @brief allows arithmetic operations on m_buff object and capture the offset
 * of the buffer to access a certain region of the memory
 * @tparam T is the element type of the m_buff object
 * @tparam Policy is the policy determining the type of m_buff object
 This is a work-around for the following enum class.
*/

template <typename T, typename Policy>
class buffer_iterator {
 public:
  using scalar_t = T;
  using self_t = buffer_iterator<scalar_t, Policy>;
  using buff_t = typename Policy::template buffer_t<scalar_t, 1>;

  /*
   * buffer iterator constructor
   */
  buffer_iterator(const buff_t& buff, std::ptrdiff_t offset);
  /*
   * buffer iterator constructor
   */
  buffer_iterator(const buff_t& buff, std::ptrdiff_t offset,
                  scalar_t* legacy_ptr);
  /*
   * buffer iterator constructor
   */
  buffer_iterator(const buff_t& buff);
  /*
   * buffer iterator constructor
   */
  template <typename other_scalar_t>
  buffer_iterator(const buffer_iterator<other_scalar_t, Policy>& other);
  /*
   * += operator on buffer
   */
  self_t& operator+=(std::ptrdiff_t offset);
  /*
   * + operator on buffer
   */
  self_t operator+(std::ptrdiff_t offset) const;
  /*
   * - operator on buffer
   */
  self_t operator-(std::ptrdiff_t offset) const;
  /*
   * -= operator on buffer
   */
  self_t& operator-=(std::ptrdiff_t offset);

  // Prefix operator (Increment and return value)
  self_t& operator++();

  // Postfix operator (Return value and increment)
  self_t operator++(int i);
  /*
   * return the size of the m_buffer (m_buff.size -m_offset)
   */
  std::ptrdiff_t get_size() const;
  /*
   * return the starting point of m_buffer
   */

  std::ptrdiff_t get_offset() const;

  /*
   * returns m_buff
   */
  buff_t get_buffer() const;

  /*
   * use to set the offset
   */
  void set_offset(std::ptrdiff_t offset);

  scalar_t& operator*() = delete;

  scalar_t& operator[](int) = delete;

  scalar_t* operator->() = delete;

 private:
  buff_t m_buff;
  ptrdiff_t m_offset;
};

/*
 * returns the element type of m_buff in buffer iterator
 */
template <typename T, typename Policy>
struct scalar_type<buffer_iterator<T, Policy>> {
  using type = T;
};
/*
 * rebind the buffer iterator<U, Policy> with buffer_iterator<T, Policy>
 */
template <typename T, typename U, typename Policy>
struct rebind_type<T, buffer_iterator<U, Policy>> {
  using type = buffer_iterator<T, Policy>;
};

}  // end namespace blas
#endif  // SYCL_BLAS_ITERATOR_H
