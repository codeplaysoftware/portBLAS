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
 *  portBLAS: BLAS implementation using SYCL
 *
 *  @filename sycl_iterator.h
 *
 **************************************************************************/
#ifndef PORTBLAS_BUFFER_ITERATOR_H
#define PORTBLAS_BUFFER_ITERATOR_H
#include "blas_meta.h"
#include <CL/sycl.hpp>
namespace blas {
/*!
 * @brief See BufferIterator.
 */
template <typename element_t>
class BufferIterator {
 public:
  using scalar_t = element_t;
  template <int dim = 1>
  using buffer_t = cl::sycl::buffer<scalar_t, dim>;
  using access_mode_t = cl::sycl::access::mode;
  template <cl::sycl::access::mode acc_md_t =
                cl::sycl::access::mode::read_write,
            cl::sycl::access::target access_t =
                cl::sycl::access::target::global_buffer,
            cl::sycl::access::placeholder place_holder_t =
                cl::sycl::access::placeholder::false_t>
  using accessor_t =
      cl::sycl::accessor<scalar_t, 1, acc_md_t, access_t, place_holder_t>;
  template <cl::sycl::access::mode acc_md_t =
                cl::sycl::access::mode::read_write,
            cl::sycl::access::target access_t =
                cl::sycl::access::target::global_buffer,
            cl::sycl::access::placeholder place_holder_t =
                cl::sycl::access::placeholder::true_t>
  using placeholder_accessor_t =
      cl::sycl::accessor<scalar_t, 1, acc_md_t, access_t, place_holder_t>;
  template <access_mode_t acc_md_t = cl::sycl::access::mode::read_write>
  using default_accessor_t = placeholder_accessor_t<acc_md_t>;
  using self_t = BufferIterator<scalar_t>;
  using buff_t = buffer_t<1>;

  /*!
   * @brief friend function to create a range accessor (offset, size)
   * @tparam acc_md_t memory access mode
   * @tparam scalar_t the element type of the buffer
   * @param buff_iterator BufferIterator
   * @param cgh cl::sycl::handler
   * @param size the region needed to be copied
   */
  template <
      cl::sycl::access::mode acc_md_t = cl::sycl::access::mode::read_write>
  inline accessor_t<acc_md_t> get_range_accessor(cl::sycl::handler& cgh,
                                                 size_t size);

  /*!
   * @brief  create a range placeholder accessor from (offset,
   * size)
   * @tparam acc_md_t memory access mode
   * @tparam scalar_t the element type of the buffer
   * @param buff_iterator BufferIterator
   * @param cgh cl::sycl::handler
   * @param size the region needed to be copied
   */

  template <
      cl::sycl::access::mode acc_md_t = cl::sycl::access::mode::read_write>
  inline placeholder_accessor_t<acc_md_t> get_range_accessor(size_t size);

  /*!
   * @brief friend function to create a range accessor from (offset,
   * m_buff.size())
   * @tparam acc_md_t memory access mode
   * @tparam scalar_t the element type of the buffer
   * @param buff_iterator BufferIterator
   * @param cgh cl::sycl::handler
   * @param size the region needed to be copied
   */
  template <
      cl::sycl::access::mode acc_md_t = cl::sycl::access::mode::read_write>
  inline accessor_t<acc_md_t> get_range_accessor(cl::sycl::handler& cgh);

  /*!
   * @brief  create a range placeholder accessor from (offset,
   * m_buff.size())
   * @tparam acc_md_t memory access mode
   * @tparam scalar_t the element type of the buffer
   * @param buff_iterator BufferIterator
   * @param cgh cl::sycl::handler
   * @param size the region needed to be copied
   */
  template <
      cl::sycl::access::mode acc_md_t = cl::sycl::access::mode::read_write>
  inline placeholder_accessor_t<acc_md_t> get_range_accessor();
  /*!
   * @brief Default construct a BufferIterator.
   * This can be used to provide a placeholder BufferIterator, but it is a user
   * error if passed into any of the portBLAS functions.
   *
   * Should be removed once SYCL specifies that buffers are default
   * constructible. See:
   * https://github.com/codeplaysoftware/standards-proposals/blob/master/default-constructed-buffers/default-constructed-buffers.md
   */
  BufferIterator() : offset_{0}, buffer_{cl::sycl::range<1>{1}} {}
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
  BufferIterator(const BufferIterator<other_scalar_t>& other);

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

template <typename element_t>
template <cl::sycl::access::mode acc_md_t>
inline typename BufferIterator<element_t>::template accessor_t<acc_md_t>
BufferIterator<element_t>::get_range_accessor(cl::sycl::handler& cgh,
                                              size_t size) {
  return typename BufferIterator<element_t>::template accessor_t<acc_md_t>(
      buffer_, cgh, cl::sycl::range<1>(size),
      cl::sycl::id<1>(BufferIterator<element_t>::get_offset()));
}

template <typename element_t>
template <cl::sycl::access::mode acc_md_t>
inline typename BufferIterator<element_t>::template accessor_t<acc_md_t>
BufferIterator<element_t>::get_range_accessor(cl::sycl::handler& cgh) {
  return BufferIterator<element_t>::get_range_accessor<acc_md_t>(
      cgh, BufferIterator<element_t>::get_size());
}

template <typename element_t>
template <cl::sycl::access::mode acc_md_t>
inline typename BufferIterator<element_t>::template placeholder_accessor_t<
    acc_md_t>
BufferIterator<element_t>::get_range_accessor(size_t size) {
  return typename BufferIterator<element_t>::template placeholder_accessor_t<
      acc_md_t>(buffer_, cl::sycl::range<1>(size),
                cl::sycl::id<1>(BufferIterator<element_t>::get_offset()));
}

template <typename element_t>
template <cl::sycl::access::mode acc_md_t>
inline typename BufferIterator<element_t>::template placeholder_accessor_t<
    acc_md_t>
BufferIterator<element_t>::get_range_accessor() {
  return BufferIterator<element_t>::get_range_accessor<acc_md_t>(
      BufferIterator<element_t>::get_size());
}

/*!
 * @brief Helper function to build BufferIterator
 * @tparam scalar_t the type of element of the m_buff
 * @tparam index_t the type of the index
 * @param data the host pointer to the data
 * @param size the size of data
 */
template <typename scalar_t, typename index_t>
inline blas::BufferIterator<scalar_t> make_sycl_iterator_buffer(scalar_t* data,
                                                                index_t size) {
  using buff_t = typename blas::BufferIterator<scalar_t>::buff_t;
  return blas::BufferIterator<scalar_t>{buff_t{data, cl::sycl::range<1>(size)}};
}

/*!
 * @brief Helper function to build BufferIterator
 * @tparam scalar_t the type of element of the m_buff
 * @tparam index_t the type of the index
 * @param data the host pointer to the data
 * @param size the size of data
 */
template <typename scalar_t, typename index_t>
inline BufferIterator<scalar_t> make_sycl_iterator_buffer(
    std::vector<scalar_t>& data, index_t size) {
  using buff_t = typename blas::BufferIterator<scalar_t>::buff_t;
  return blas::BufferIterator<scalar_t>{
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
inline blas::BufferIterator<scalar_t> make_sycl_iterator_buffer(index_t size) {
  using buff_t = typename blas::BufferIterator<scalar_t>::buff_t;
  return blas::BufferIterator<scalar_t>{buff_t{cl::sycl::range<1>(size)}};
}
/*!
 * @brief Helper function to build BufferIterator
 * @tparam scalar_t the type of element of the m_buff
 * @tparam index_t the type of the index
 * @param data the host pointer to the data
 * @param size the size of data
 */
template <typename scalar_t>
inline blas::BufferIterator<scalar_t> make_sycl_iterator_buffer(
    typename blas::BufferIterator<scalar_t>::buff_t buff_) {
  return blas::BufferIterator<scalar_t>{buff_};
}
template <typename element_t>
inline BufferIterator<element_t>::BufferIterator(
    const typename BufferIterator<element_t>::buff_t& buff,
    std::ptrdiff_t offset)
    : offset_(offset), buffer_(buff) {}

template <typename element_t>
inline BufferIterator<element_t>::BufferIterator(
    const typename BufferIterator<element_t>::buff_t& buff)
    : BufferIterator(buff, 0) {}

// copy constructor buffer
template <typename element_t>
template <typename other_scalar_t, typename U, typename>
inline BufferIterator<element_t>::BufferIterator(
    const BufferIterator<other_scalar_t>& other)
    : BufferIterator(other.get_buffer().template reinterpret<element_t>(
                         cl::sycl::range<1>(other.get_buffer().get_count())),
                     other.get_offset()) {}

template <typename element_t>
inline BufferIterator<element_t>& BufferIterator<element_t>::operator+=(
    std::ptrdiff_t offset) {
  offset_ += offset;
  return *this;
}

template <typename element_t>
inline BufferIterator<element_t> BufferIterator<element_t>::operator+(
    std::ptrdiff_t offset) const {
  return BufferIterator<element_t>(buffer_, offset_ + offset);
}

template <typename element_t>
inline BufferIterator<element_t> BufferIterator<element_t>::operator-(
    std::ptrdiff_t offset) const {
  return BufferIterator<element_t>(buffer_, offset_ - offset);
}

template <typename element_t>
inline BufferIterator<element_t>& BufferIterator<element_t>::operator-=(
    std::ptrdiff_t offset) {
  offset_ -= offset;
  return *this;
}

// Prefix operator (Increment and return value)
template <typename element_t>
inline BufferIterator<element_t>& BufferIterator<element_t>::operator++() {
  offset_++;
  return (*this);
}
// Postfix operator (Return value and increment)
template <typename element_t>
inline BufferIterator<element_t> BufferIterator<element_t>::operator++(int i) {
  BufferIterator<element_t> temp_iterator(*this);
  offset_ += 1;
  return temp_iterator;
}

template <typename element_t>
inline std::ptrdiff_t BufferIterator<element_t>::get_size() const {
  return (buffer_.get_count() - offset_);
}

template <typename element_t>
inline std::ptrdiff_t BufferIterator<element_t>::get_offset() const {
  return offset_;
}

template <typename element_t>
inline typename BufferIterator<element_t>::buff_t
BufferIterator<element_t>::get_buffer() const {
  return buffer_;
}

template <typename element_t>
inline void BufferIterator<element_t>::set_offset(std::ptrdiff_t offset) {
  offset_ = offset;
}

/*
 * returns the element type of m_buff in buffer iterator
 */
template <typename element_t>
struct ValueType<BufferIterator<element_t>> {
  using type = typename std::remove_cv<element_t>::type;
};
/*
 * rebind the buffer iterator<U> with BufferIterator<element_t>
 */
template <typename element_t, typename U>
struct RebindType<element_t, BufferIterator<U>> {
  using type = BufferIterator<element_t>;
};

}  // end namespace blas

#endif  // PORTBLAS_BUFFER_ITERATOR_H
