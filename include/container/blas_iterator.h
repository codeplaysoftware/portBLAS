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
#include "../blas_meta.h"
#include <CL/sycl.hpp>

namespace blas {

/*
 * @brief allows arithmetic operations on m_buff object and capture the offset
 * of the buffer to access a certain region of the memory
 * @tparam element_t is the element type of the m_buff object
 * @tparam policy_t is the policy determining the type of m_buff object
 This is a work-around for the following enum class.
*/

template <typename element_t, typename policy_t>
class BufferIterator {
 public:
  using scalar_t = element_t;
  using self_t = BufferIterator<scalar_t, policy_t>;
  using buff_t = typename policy_t::template buffer_t<scalar_t, 1>;

  /*
   * buffer iterator constructor
   */
  BufferIterator(const buff_t& buff, std::ptrdiff_t offset);
  /*
   * buffer iterator constructor
   */
  BufferIterator(const buff_t& buff, std::ptrdiff_t offset,
                 scalar_t* legacy_ptr);
  /*
   * buffer iterator constructor
   */
  BufferIterator(const buff_t& buff);
  /*
   * buffer iterator constructor
   */
  template <typename other_scalar_t>
  BufferIterator(const BufferIterator<other_scalar_t, policy_t>& other);
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
   * return the size of the buffer_ (m_buff.size -offset_)
   */
  std::ptrdiff_t get_size() const;
  /*
   * return the starting point of buffer_
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
  ptrdiff_t offset_;
};

/*
 * returns the element type of m_buff in buffer iterator
 */
template <typename element_t, typename policy_t>
struct ValueType<BufferIterator<element_t, policy_t>> {
  using type = element_t;
};
/*
 * rebind the buffer iterator<U, policy_t> with BufferIterator<element_t,
 * policy_t>
 */
template <typename element_t, typename U, typename policy_t>
struct RebindType<element_t, BufferIterator<U, policy_t>> {
  using type = BufferIterator<element_t, policy_t>;
};

}  // end namespace blas
#endif  // SYCL_BLAS_ITERATOR_H
