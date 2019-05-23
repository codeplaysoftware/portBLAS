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
 *  @filename sycl_iterator.h
 *
 **************************************************************************/
#ifndef SYCL_BLAS_BUFFER_ITERATOR_H
#define SYCL_BLAS_BUFFER_ITERATOR_H
#include "../blas_meta.h"
#include "../policy/sycl_policy.h"
#include "blas_iterator.h"
#include <CL/sycl.hpp>
namespace blas {
/*!
 * @brief See BufferIterator.
 */
template <typename element_t>
class BufferIterator<element_t, codeplay_policy> {
 public:
  using scalar_t = element_t;
  using self_t = BufferIterator<scalar_t, codeplay_policy>;
  using buff_t = typename codeplay_policy::template buffer_t<scalar_t, 1>;

  /*!
   * friend function to create a range accessor from m_buff
   */
  template <cl::sycl::access::mode acc_md_t, typename scal_t>
  friend inline typename codeplay_policy::template accessor_t<scal_t, acc_md_t>
  get_range_accessor(BufferIterator<scal_t, codeplay_policy> buff_iterator,
                     cl::sycl::handler& cgh, size_t size);
  /*!
   * friend function to create a range placeholder accessor from m_buff
   */
  template <cl::sycl::access::mode acc_md_t, typename scal_t>
  friend inline
      typename codeplay_policy::template placeholder_accessor_t<scal_t,
                                                                acc_md_t>
      get_range_accessor(BufferIterator<scal_t, codeplay_policy> buff_iterator,
                         size_t size);
  /*!
   * @brief See BufferIterator.
   */
  BufferIterator(const buff_t& buff, std::ptrdiff_t offset);

  /*!
   * @brief See BufferIterator.
   */
  BufferIterator(const buff_t& buff);
  /*!
   * @brief See BufferIterator.
   */
  template <
      typename other_scalar_t, typename U = element_t,
      class = typename std::enable_if<
          !std::is_const<other_scalar_t>::value ||
          (std::is_const<other_scalar_t>::value && std::is_const<U>::value &&
           !std::is_same<U, other_scalar_t>::value)>::type>
  BufferIterator(const BufferIterator<other_scalar_t, codeplay_policy>& other);

  /*!
   * @brief See BufferIterator.
   */
  self_t& operator+=(std::ptrdiff_t offset);
  /*!
   * @brief See BufferIterator.
   */
  self_t operator+(std::ptrdiff_t offset) const;
  /*!
   * @brief See BufferIterator.
   */
  self_t operator-(std::ptrdiff_t offset) const;
  /*!
   * @brief See BufferIterator.
   */
  self_t& operator-=(std::ptrdiff_t offset);

  // Prefix operator (Increment and return value)
  self_t& operator++();

  // Postfix operator (Return value and increment)
  self_t operator++(int i);
  /*!
   * @brief See BufferIterator.
   */
  std::ptrdiff_t get_size() const;
  /*!
   * @brief See BufferIterator.
   */
  std::ptrdiff_t get_offset() const;
  /*!
   * @brief See BufferIterator.
   */
  buff_t get_buffer() const;

  /*!
   * @brief See BufferIterator.
   */
  void set_offset(std::ptrdiff_t offset);

  scalar_t& operator*() = delete;

  scalar_t* operator->() = delete;

 private:
  std::ptrdiff_t offset_;
  buff_t buffer_;
};

/*!
 * @brief friend function to create a range accessor (offset, size)
 * @tparam acc_md_t memory access mode
 * @tparam scalar_t the element type of the buffer
 * @param buff_iterator BufferIterator
 * @param cgh cl::sycl::handler
 * @param size the region needed to be copied
 */
template <cl::sycl::access::mode acc_md_t = cl::sycl::access::mode::read_write,
          typename scalar_t>
inline typename codeplay_policy::template accessor_t<scalar_t, acc_md_t>
get_range_accessor(BufferIterator<scalar_t, codeplay_policy> buff_iterator,
                   cl::sycl::handler& cgh, size_t size) {
  return typename codeplay_policy::template accessor_t<scalar_t, acc_md_t>(
      buff_iterator.buffer_, cgh, cl::sycl::range<1>(size),
      cl::sycl::id<1>(buff_iterator.get_offset()));
}
/*!
 * @brief friend function to create a range accessor from (offset,
 * m_buff.size())
 * @tparam acc_md_t memory access mode
 * @tparam scalar_t the element type of the buffer
 * @param buff_iterator BufferIterator
 * @param cgh cl::sycl::handler
 * @param size the region needed to be copied
 */
template <cl::sycl::access::mode acc_md_t = cl::sycl::access::mode::read_write,
          typename scalar_t>
inline typename codeplay_policy::template accessor_t<scalar_t, acc_md_t>
get_range_accessor(BufferIterator<scalar_t, codeplay_policy> buff_iterator,
                   cl::sycl::handler& cgh) {
  return get_range_accessor<acc_md_t>(buff_iterator, cgh,
                                      buff_iterator.get_size());
}

/*!
 * @brief friend function to create a range placeholder accessor from (offset,
 * size)
 * @tparam acc_md_t memory access mode
 * @tparam scalar_t the element type of the buffer
 * @param buff_iterator BufferIterator
 * @param cgh cl::sycl::handler
 * @param size the region needed to be copied
 */

template <cl::sycl::access::mode acc_md_t = cl::sycl::access::mode::read_write,
          typename scalar_t>
inline typename codeplay_policy::template placeholder_accessor_t<scalar_t,
                                                                 acc_md_t>
get_range_accessor(BufferIterator<scalar_t, codeplay_policy> buff_iterator,
                   size_t size) {
  return typename codeplay_policy::template placeholder_accessor_t<scalar_t,
                                                                   acc_md_t>(
      buff_iterator.buffer_, cl::sycl::range<1>(size),
      cl::sycl::id<1>(buff_iterator.get_offset()));
}

/*!
 * @brief friend function to create a range placeholder accessor from (offset,
 * m_buff.size())
 * @tparam acc_md_t memory access mode
 * @tparam scalar_t the element type of the buffer
 * @param buff_iterator BufferIterator
 * @param cgh cl::sycl::handler
 * @param size the region needed to be copied
 */
template <cl::sycl::access::mode acc_md_t = cl::sycl::access::mode::read_write,
          typename scalar_t>
inline typename codeplay_policy::template placeholder_accessor_t<scalar_t,
                                                                 acc_md_t>
get_range_accessor(BufferIterator<scalar_t, codeplay_policy> buff_iterator) {
  return get_range_accessor<acc_md_t>(buff_iterator, buff_iterator.get_size());
}

/*!
 * @brief Helper function to build BufferIterator
 * @tparam scalar_t the type of element of the m_buff
 * @tparam index_t the type of the index
 * @param data the host pointer to the data
 * @param size the size of data
 */
template <typename scalar_t, typename index_t>
inline blas::BufferIterator<scalar_t, codeplay_policy>
make_sycl_iterator_buffer(scalar_t* data, index_t size) {
  using buff_t = typename blas::codeplay_policy::buffer_t<scalar_t, 1>;
  return blas::BufferIterator<scalar_t, codeplay_policy>{
      buff_t{data, cl::sycl::range<1>(size)}};
}

/*!
 * @brief Helper function to build BufferIterator
 * @tparam scalar_t the type of element of the m_buff
 * @tparam index_t the type of the index
 * @param data the host pointer to the data
 * @param size the size of data
 */
template <typename scalar_t, typename index_t>
inline BufferIterator<scalar_t, codeplay_policy> make_sycl_iterator_buffer(
    std::vector<scalar_t>& data, index_t size) {
  using buff_t = typename blas::codeplay_policy::buffer_t<scalar_t, 1>;
  return blas::BufferIterator<scalar_t, codeplay_policy>{
      buff_t{data.data(), cl::sycl::range<1>(size)}};
}

/*!
 * @brief Helper function to build BufferIterator
 * @tparam scalar_t the type of element of the m_buff
 * @tparam index_t the type of the index
 * @param data the host pointer to the data
 * @param size the size of data
 */
template <typename scalar_t, typename index_t>
inline blas::BufferIterator<scalar_t, codeplay_policy>
make_sycl_iterator_buffer(index_t size) {
  using buff_t = typename blas::codeplay_policy::buffer_t<scalar_t, 1>;
  return blas::BufferIterator<scalar_t, codeplay_policy>{
      buff_t{cl::sycl::range<1>(size)}};
}
/*!
 * @brief Helper function to build BufferIterator
 * @tparam scalar_t the type of element of the m_buff
 * @tparam index_t the type of the index
 * @param data the host pointer to the data
 * @param size the size of data
 */
template <typename scalar_t>
inline blas::BufferIterator<scalar_t, codeplay_policy>
make_sycl_iterator_buffer(
    typename blas::codeplay_policy::buffer_t<scalar_t, 1> buff_) {
  return blas::BufferIterator<scalar_t, codeplay_policy>{buff_};
}
template <typename element_t>
inline BufferIterator<element_t, codeplay_policy>::BufferIterator(
    const typename BufferIterator<element_t, codeplay_policy>::buff_t& buff,
    std::ptrdiff_t offset)
    : offset_(offset), buffer_(buff) {}

template <typename element_t>
inline BufferIterator<element_t, codeplay_policy>::BufferIterator(
    const typename BufferIterator<element_t, codeplay_policy>::buff_t& buff)
    : BufferIterator(buff, 0) {}

// copy constructor buffer
template <typename element_t>
template <typename other_scalar_t, typename U, typename>
inline BufferIterator<element_t, codeplay_policy>::BufferIterator(
    const BufferIterator<other_scalar_t, codeplay_policy>& other)
    : BufferIterator(other.get_buffer().template reinterpret<element_t>(
                         cl::sycl::range<1>(other.get_buffer().get_count())),
                     other.get_offset()) {}

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

#endif  // SYCL_BLAS_BUFFER_ITERATOR_H
