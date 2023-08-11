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
 *  @filename view.h
 *
 **************************************************************************/

#ifndef SYCL_BLAS_VIEW_H
#define SYCL_BLAS_VIEW_H

#include "blas_meta.h"
#include <stdexcept>
#include <vector>

namespace blas {

/*!
@brief Alias to std::string.
*/
using string_class_t = std::string;

/*!
@brief Template struct for containing vector that can used within a compile-time
expression.
@tparam value_t Type of each element fo the vector.
@tparam container_t Type of the container that is stored inside.
*/

template <typename view_container_t, typename view_index_t,
          typename view_increment_t>
struct VectorView {
  using value_t = typename ValueType<view_container_t>::type;
  using container_t = view_container_t;
  using index_t = view_index_t;
  using increment_t = view_increment_t;
  using self_t = VectorView<container_t, index_t, increment_t>;
  container_t data_;
  index_t size_;
  increment_t strd_;  // never size_t, because it could be negative

  // Start of the vector.
  // If stride is negative, start at the end of the vector and move backward.
  container_t ptr_;

  VectorView(view_container_t data, view_increment_t strd, view_index_t size);
  VectorView(VectorView<view_container_t, view_index_t, view_increment_t> opV,
             view_increment_t strd, view_index_t size);

  /*!
   * @brief Returns a reference to the container
   */
  SYCL_BLAS_INLINE container_t get_data() const;

  /*!
   * @brief Returns a pointer containing the raw data of the container
   */
  SYCL_BLAS_INLINE container_t get_pointer() const;

  /*! adjust_access_displacement.
   * @brief this method adjust the position of the data access to point to the
   *  data_ + offset_ on the device side. This function will be called at the
   * begining of an expression so that the kernel wont repeat this operation at
   * every eval call.
   * For USM case, this method is not going to do anything as the library
   * doesn't allow pointer manipulation.
   */
  SYCL_BLAS_INLINE void adjust_access_displacement() const;

  /*!
   @brief Returns the size of the view
   */
  SYCL_BLAS_INLINE index_t get_size() const;

  /*!
   @brief Returns the stride of the view.
  */
  SYCL_BLAS_INLINE increment_t get_stride();

  SYCL_BLAS_INLINE void bind(cl::sycl::handler &h) const {}

  /**** EVALUATING ****/
  template <bool use_as_ptr = false>
  SYCL_BLAS_INLINE typename std::enable_if<!use_as_ptr, value_t &>::type eval(
      index_t i) {
    return (strd_ == 1) ? *(ptr_ + i) : *(ptr_ + i * strd_);
  }

  template <bool use_as_ptr = false>
  SYCL_BLAS_INLINE typename std::enable_if<!use_as_ptr, value_t>::type eval(
      index_t i) const {
    return (strd_ == 1) ? *(ptr_ + i) : *(ptr_ + i * strd_);
  }

  SYCL_BLAS_INLINE value_t &eval(cl::sycl::nd_item<1> ndItem) {
    return eval(ndItem.get_global_id(0));
  }

  SYCL_BLAS_INLINE value_t eval(cl::sycl::nd_item<1> ndItem) const {
    return eval(ndItem.get_global_id(0));
  }

  template <bool use_as_ptr = false>
  SYCL_BLAS_INLINE typename std::enable_if<use_as_ptr, value_t &>::type eval(
      index_t indx) {
    return *(ptr_ + indx);
  }

  template <bool use_as_ptr = false>
  SYCL_BLAS_INLINE typename std::enable_if<use_as_ptr, value_t>::type eval(
      index_t indx) const noexcept {
    return *(ptr_ + indx);
  }
};

/*! MatrixView
@brief Represents a Matrix on the given Container.
@tparam value_t Value type of the container.
@tparam container_t Type of the container.
@tparam has_inc bool for col/row internal increment different from 1.
 */
template <typename view_container_t, typename view_index_t, typename layout,
    bool has_inc = false>
struct MatrixView {
  // Information related to the data
  using access_layout_t = layout;
  using value_t = typename ValueType<view_container_t>::type;
  using container_t = view_container_t;
  using index_t = view_index_t;
  using self_t = MatrixView<container_t, index_t, layout, has_inc>;
  container_t data_;
  index_t sizeR_;  // number of rows
  index_t sizeC_;  // number of columns
  index_t sizeL_;  // size of the leading dimension
  const index_t inc_;    // internal increment between same row/column elements

  // UPLO, BAND(KU,KL), PACKED, SIDE ARE ONLY REQUIRED
  MatrixView(view_container_t data, view_index_t sizeR, view_index_t sizeC);
  MatrixView(view_container_t data, view_index_t sizeR, view_index_t sizeC,
             view_index_t sizeL);
  MatrixView(view_container_t data, view_index_t sizeR, view_index_t sizeC,
             view_index_t sizeL, view_index_t inc);
  MatrixView(MatrixView<view_container_t, view_index_t, layout, has_inc> opM,
             view_index_t sizeR, view_index_t sizeC, view_index_t sizeL);

  /*!
   * @brief Returns the container
   */
  SYCL_BLAS_INLINE container_t get_data() const;

  /*!
   * @brief Returns the container
   */
  SYCL_BLAS_INLINE container_t get_pointer() const;

  /*!
   * @brief Returns the size of the view.
   */
  SYCL_BLAS_INLINE index_t get_size() const;

  /*!
   * @brief Returns the leading dimension.
   */
  SYCL_BLAS_INLINE const index_t getSizeL() const;

  /*! get_size_row.
   * @brief Return the number of columns.
   * @bug This value should change depending on the access mode, but
   * is currently set to Rows.
   */
  SYCL_BLAS_INLINE index_t get_size_row() const;

  /*! get_size_col.
   * @brief Return the number of columns.
   * @bug This value should change depending on the access mode, but
   * is currently set to Rows.
   */
  SYCL_BLAS_INLINE index_t get_size_col() const;

  /*! adjust_access_displacement.
   * @brief set displacement from the origin.
   * This method allows to have a pointer arithmetic semantics for buffers
   * in the host code. The end result of the pointer arithmetic is passed
   * as an access displacement for the buffer.
   * In the case of USM, this method does nothing since the pointer
   * arithmetic is performed implicitly.
   */
  SYCL_BLAS_INLINE void adjust_access_displacement() const;

  SYCL_BLAS_INLINE void bind(cl::sycl::handler &h) const {}

  /*! eval.
   * @brief Evaluation for the pair of row/col.
   */
  SYCL_BLAS_INLINE value_t &eval(index_t i, index_t j) {
    if constexpr (has_inc) {
      if constexpr (layout::is_col_major()) {
        return *(data_ + i * inc_ + sizeL_ * j);
      } else {
        return *(data_ + j * inc_ + sizeL_ * i);
      }
    } else {
      if constexpr (layout::is_col_major()) {
        return *(data_ + i + sizeL_ * j);
      } else {
        return *(data_ + j + sizeL_ * i);
      }
    }
  }

  SYCL_BLAS_INLINE value_t eval(index_t i, index_t j) const noexcept {
    if constexpr (has_inc) {
      if constexpr (layout::is_col_major()) {
        return *(data_ + i * inc_ + sizeL_ * j);
      } else {
        return *(data_ + j * inc_ + sizeL_ * i);
      }
    } else {
      if constexpr (layout::is_col_major()) {
        return *(data_ + i + sizeL_ * j);
      } else {
        return *(data_ + j + sizeL_ * i);
      }
    }
  }

  template <bool use_as_ptr = false>
  SYCL_BLAS_INLINE typename std::enable_if<!use_as_ptr, value_t &>::type eval(
      index_t indx) {
    const index_t j = indx / sizeR_;
    const index_t i = indx - sizeR_ * j;
    return eval(i, j);
  }

  template <bool use_as_ptr = false>
  SYCL_BLAS_INLINE typename std::enable_if<!use_as_ptr, value_t>::type eval(
      index_t indx) const noexcept {
    const index_t j = indx / sizeR_;
    const index_t i = indx - sizeR_ * j;
    return eval(i, j);
  }

  SYCL_BLAS_INLINE value_t &eval(cl::sycl::nd_item<1> ndItem) {
    return eval(ndItem.get_global_id(0));
  }

  SYCL_BLAS_INLINE value_t eval(cl::sycl::nd_item<1> ndItem) const noexcept {
    return eval(ndItem.get_global_id(0));
  }

  template <bool use_as_ptr = false>
  SYCL_BLAS_INLINE typename std::enable_if<use_as_ptr, value_t &>::type eval(
      index_t indx) {
    return *(data_ + indx);
  }

  template <bool use_as_ptr = false>
  SYCL_BLAS_INLINE typename std::enable_if<use_as_ptr, value_t>::type eval(
      index_t indx) const noexcept {
    return *(data_ + indx);
  }
};

template <typename container_t, typename index_t, typename increment_t>
struct VectorViewType;

template <typename value_t, typename index_t, typename increment_t>
struct VectorViewType<BufferIterator<value_t>, index_t, increment_t> {
  static constexpr cl::sycl::access::mode access_mode_t =
      Choose<std::is_const<value_t>::value, cl::sycl::access::mode,
             cl::sycl::access::mode::read,
             cl::sycl::access::mode::read_write>::type;
  using type =
      VectorView<typename BufferIterator<value_t>::template default_accessor_t<
                     access_mode_t>,
                 index_t, increment_t>;
};

template <typename value_t, typename index_t, typename increment_t>
struct VectorViewType<value_t *, index_t, increment_t> {
  using type = VectorView<value_t *, index_t, increment_t>;
};

template <typename value_t, typename index_t, typename increment_t>
struct VectorViewType<const value_t *, index_t, increment_t> {
  using type = const VectorView<const value_t *, index_t, increment_t>;
};

template <typename container_t, typename index_t, typename access_layout_t, bool has_inc = false>
struct MatrixViewType;

template <typename value_t, typename index_t, typename access_layout_t, bool has_inc>
struct MatrixViewType<BufferIterator<value_t>, index_t, access_layout_t, has_inc> {
  static constexpr cl::sycl::access::mode access_mode_t =
      Choose<std::is_const<value_t>::value, cl::sycl::access::mode,
             cl::sycl::access::mode::read,
             cl::sycl::access::mode::read_write>::type;
  using type =
      MatrixView<typename BufferIterator<value_t>::template default_accessor_t<
                     access_mode_t>,
                 index_t, access_layout_t, has_inc>;
};

template <typename value_t, typename index_t, typename access_layout_t, bool has_inc>
struct MatrixViewType<value_t *, index_t, access_layout_t, has_inc> {
  using type = MatrixView<value_t *, index_t, access_layout_t, has_inc>;
};

template <typename value_t, typename index_t, typename access_layout_t, bool has_inc>
struct MatrixViewType<const value_t *, index_t, access_layout_t, has_inc> {
  using type = const MatrixView<const value_t *, index_t, access_layout_t, has_inc>;
};

template <typename value_t, typename increment_t, typename index_t>
static SYCL_BLAS_INLINE auto make_vector_view(BufferIterator<value_t> buff,
                                              increment_t inc, index_t sz) {
  static constexpr cl::sycl::access::mode access_mode_t =
      Choose<std::is_const<value_t>::value, cl::sycl::access::mode,
             cl::sycl::access::mode::read,
             cl::sycl::access::mode::read_write>::type;
  using leaf_node_t =
      VectorView<typename BufferIterator<value_t>::template default_accessor_t<
                     access_mode_t>,
                 index_t, increment_t>;
  return leaf_node_t{buff.template get_range_accessor<access_mode_t>(),
                     (index_t)buff.get_offset(), inc, sz};
}

template <typename access_layout_t, typename value_t, typename index_t,
          bool has_inc = false>
static SYCL_BLAS_INLINE auto make_matrix_view(BufferIterator<value_t> buff,
                                              index_t m, index_t n,
                                              index_t lda, index_t inc = 1) {
  static constexpr cl::sycl::access::mode access_mode_t =
      Choose<std::is_const<value_t>::value, cl::sycl::access::mode,
             cl::sycl::access::mode::read,
             cl::sycl::access::mode::read_write>::type;
  using leaf_node_t =
      MatrixView<typename BufferIterator<value_t>::template default_accessor_t<
                     access_mode_t>,
                 index_t, access_layout_t, has_inc>;
  return leaf_node_t{buff.template get_range_accessor<access_mode_t>(), m, n,
                     lda, inc, (index_t)buff.get_offset()};
}

template <typename value_t, typename increment_t, typename index_t>
static SYCL_BLAS_INLINE auto make_vector_view(value_t *usm_ptr, increment_t inc,
                                              index_t sz) {
  using leaf_node_t = VectorView<value_t *, index_t, increment_t>;
  return leaf_node_t{usm_ptr, inc, sz};
}

template <typename access_layout_t, typename value_t, typename index_t,
          bool has_inc = false>
static SYCL_BLAS_INLINE auto make_matrix_view(value_t *usm_ptr, index_t m,
                                              index_t n, index_t lda,
                                              index_t inc = 1) {
  using leaf_node_t = MatrixView<value_t *, index_t, access_layout_t, has_inc>;
  return leaf_node_t{usm_ptr, m, n, lda, inc};
}

template <typename value_t, typename increment_t, typename index_t>
static SYCL_BLAS_INLINE auto make_vector_view(const value_t *usm_ptr,
                                              increment_t inc, index_t sz) {
  using leaf_node_t = VectorView<const value_t *, index_t, increment_t>;
  return leaf_node_t{usm_ptr, inc, sz};
}


template <typename access_layout_t, typename value_t, typename index_t,
          bool has_inc = false>
static SYCL_BLAS_INLINE auto make_matrix_view(const value_t *usm_ptr, index_t m,
                                              index_t n, index_t lda,
                                              index_t inc = 1) {
  using leaf_node_t = MatrixView<const value_t *, index_t, access_layout_t, has_inc>;
  return leaf_node_t{usm_ptr, m, n, lda, inc};
}

}  // namespace blas

#endif  // VIEW_H
