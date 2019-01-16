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
#include "blas_meta.h"
#include "container/blas_iterator.h"
#include "policy/sycl_policy.h"
#include <CL/sycl.hpp>
namespace blas {
/*!
 * @brief See buffer_iterator.
 */
template <typename T>
class buffer_iterator<T, BLAS_SYCL_Policy> {
 public:
  using scalar_t = T;
  using self_t = buffer_iterator<scalar_t, BLAS_SYCL_Policy>;
  using buff_t = typename BLAS_SYCL_Policy::template buffer_t<scalar_t, 1>;

  /*!
   * friend function to create a range accessor from m_buff
   */
  template <cl::sycl::access::mode AcM, typename scal_t>
  friend inline typename BLAS_SYCL_Policy::template SyclAccessor<scal_t, AcM>
  get_range_accessor(buffer_iterator<scal_t, BLAS_SYCL_Policy> buff_iterator,
                     cl::sycl::handler& cgh, size_t size);
  /*!
   * friend function to create a range placeholder accessor from m_buff
   */
  template <cl::sycl::access::mode AcM, typename scal_t>
  friend inline
      typename BLAS_SYCL_Policy::template placeholder_accessor_t<scal_t, AcM>
      get_range_accessor(
          buffer_iterator<scal_t, BLAS_SYCL_Policy> buff_iterator, size_t size);
  /*!
   * @brief See buffer_iterator.
   */
  buffer_iterator(const buff_t& buff, std::ptrdiff_t offset);
  /*!
   * @brief See buffer_iterator.
   */
  buffer_iterator(const buff_t& buff, std::ptrdiff_t offset,
                  scalar_t* legacy_ptr);
  /*!
   * @brief See buffer_iterator.
   */
  buffer_iterator(const buff_t& buff);
  /*!
   * @brief See buffer_iterator.
   */
  template <typename other_scalar_t>
  buffer_iterator(
      const buffer_iterator<other_scalar_t, BLAS_SYCL_Policy>& other);

  /*!
   * @brief See buffer_iterator.
   */
  self_t& operator+=(std::ptrdiff_t offset);
  /*!
   * @brief See buffer_iterator.
   */
  self_t operator+(std::ptrdiff_t offset) const;
  /*!
   * @brief See buffer_iterator.
   */
  self_t operator-(std::ptrdiff_t offset) const;
  /*!
   * @brief See buffer_iterator.
   */
  self_t& operator-=(std::ptrdiff_t offset);

  // Prefix operator (Increment and return value)
  self_t& operator++();

  // Postfix operator (Return value and increment)
  self_t operator++(int i);
  /*!
   * @brief See buffer_iterator.
   */
  std::ptrdiff_t get_size() const;
  /*!
   * @brief See buffer_iterator.
   */
  std::ptrdiff_t get_offset() const;
  /*!
   * @brief See buffer_iterator.
   */
  buff_t get_buffer() const;

  /*!
   * @brief See buffer_iterator.
   */
  void set_offset(std::ptrdiff_t offset);

  scalar_t& operator*() = delete;

  scalar_t* operator->() = delete;

 private:
  std::ptrdiff_t m_offset;
  buff_t m_buffer;
};

/*!
 * @brief friend function to create a range accessor (offset, size)
 * @tparam AcM memory access mode
 * @tparam scalar_t the element type of the buffer
 * @param buff_iterator buffer_iterator
 * @param cgh cl::sycl::handler
 * @param size the region needed to be copied
 */
template <cl::sycl::access::mode AcM = cl::sycl::access::mode::read_write,
          typename scalar_t>
inline typename BLAS_SYCL_Policy::template SyclAccessor<scalar_t, AcM>
get_range_accessor(buffer_iterator<scalar_t, BLAS_SYCL_Policy> buff_iterator,
                   cl::sycl::handler& cgh, size_t size) {
  return typename BLAS_SYCL_Policy::template SyclAccessor<scalar_t, AcM>(
      buff_iterator.m_buffer, cgh, cl::sycl::range<1>(size),
      cl::sycl::id<1>(buff_iterator.get_offset()));
}
/*!
 * @brief friend function to create a range accessor from (offset,
 * m_buff.size())
 * @tparam AcM memory access mode
 * @tparam scalar_t the element type of the buffer
 * @param buff_iterator buffer_iterator
 * @param cgh cl::sycl::handler
 * @param size the region needed to be copied
 */
template <cl::sycl::access::mode AcM = cl::sycl::access::mode::read_write,
          typename scalar_t>
inline typename BLAS_SYCL_Policy::template SyclAccessor<scalar_t, AcM>
get_range_accessor(buffer_iterator<scalar_t, BLAS_SYCL_Policy> buff_iterator,
                   cl::sycl::handler& cgh) {
  return get_range_accessor<AcM>(buff_iterator, cgh, buff_iterator.get_size());
}

/*!
 * @brief friend function to create a range placeholder accessor from (offset,
 * size)
 * @tparam AcM memory access mode
 * @tparam scalar_t the element type of the buffer
 * @param buff_iterator buffer_iterator
 * @param cgh cl::sycl::handler
 * @param size the region needed to be copied
 */

template <cl::sycl::access::mode AcM = cl::sycl::access::mode::read_write,
          typename scalar_t>
inline typename BLAS_SYCL_Policy::template placeholder_accessor_t<scalar_t, AcM>
get_range_accessor(buffer_iterator<scalar_t, BLAS_SYCL_Policy> buff_iterator,
                   size_t size) {
  return
      typename BLAS_SYCL_Policy::template placeholder_accessor_t<scalar_t, AcM>(
          buff_iterator.m_buffer, cl::sycl::range<1>(size),
          cl::sycl::id<1>(buff_iterator.get_offset()));
}

/*!
 * @brief friend function to create a range placeholder accessor from (offset,
 * m_buff.size())
 * @tparam AcM memory access mode
 * @tparam scalar_t the element type of the buffer
 * @param buff_iterator buffer_iterator
 * @param cgh cl::sycl::handler
 * @param size the region needed to be copied
 */
template <cl::sycl::access::mode AcM = cl::sycl::access::mode::read_write,
          typename scalar_t>
inline typename BLAS_SYCL_Policy::template placeholder_accessor_t<scalar_t, AcM>
get_range_accessor(buffer_iterator<scalar_t, BLAS_SYCL_Policy> buff_iterator) {
  return get_range_accessor<AcM>(buff_iterator, buff_iterator.get_size());
}

/*!
 * @brief Helper function to build buffer_iterator
 * @tparam scalar_t the type of element of the m_buff
 * @tparam index_t the type of the index
 * @param data the host pointer to the data
 * @param size the size of data
 */
template <typename scalar_t, typename index_t>
inline blas::buffer_iterator<scalar_t, BLAS_SYCL_Policy>
make_sycl_iterator_buffer(scalar_t* data, index_t size) {
  using buff_t = typename blas::BLAS_SYCL_Policy::buffer_t<scalar_t, 1>;
  return blas::buffer_iterator<scalar_t, BLAS_SYCL_Policy>{
      buff_t{data, cl::sycl::range<1>(size)}};
}

/*!
 * @brief Helper function to build buffer_iterator
 * @tparam scalar_t the type of element of the m_buff
 * @tparam index_t the type of the index
 * @param data the host pointer to the data
 * @param size the size of data
 */
template <typename scalar_t, typename index_t>
inline buffer_iterator<scalar_t, BLAS_SYCL_Policy> make_sycl_iterator_buffer(
    std::vector<scalar_t>& data, index_t size) {
  using buff_t = typename blas::BLAS_SYCL_Policy::buffer_t<scalar_t, 1>;
  return blas::buffer_iterator<scalar_t, BLAS_SYCL_Policy>{
      buff_t{data.data(), cl::sycl::range<1>(size)}};
}

/*!
 * @brief Helper function to build buffer_iterator
 * @tparam scalar_t the type of element of the m_buff
 * @tparam index_t the type of the index
 * @param data the host pointer to the data
 * @param size the size of data
 */
template <typename scalar_t, typename index_t>
inline blas::buffer_iterator<scalar_t, BLAS_SYCL_Policy>
make_sycl_iterator_buffer(index_t size) {
  using buff_t = typename blas::BLAS_SYCL_Policy::buffer_t<scalar_t, 1>;
  return blas::buffer_iterator<scalar_t, BLAS_SYCL_Policy>{
      buff_t{cl::sycl::range<1>(size)}};
}
/*!
 * @brief Helper function to build buffer_iterator
 * @tparam scalar_t the type of element of the m_buff
 * @tparam index_t the type of the index
 * @param data the host pointer to the data
 * @param size the size of data
 */
template <typename scalar_t>
inline blas::buffer_iterator<scalar_t, BLAS_SYCL_Policy>
make_sycl_iterator_buffer(
    typename blas::BLAS_SYCL_Policy::buffer_t<scalar_t, 1> buff_) {
  return blas::buffer_iterator<scalar_t, BLAS_SYCL_Policy>{buff_};
}
}  // end namespace blas
#endif  // SYCL_BLAS_BUFFER_ITERATOR_H
