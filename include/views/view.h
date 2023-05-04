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
#include <iostream>
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

template <typename view_value_t, typename view_container_t,
          typename view_index_t, typename view_increment_t>
struct VectorView {
  using value_t = view_value_t;
  using container_t = view_container_t;
  using index_t = view_index_t;
  using increment_t = view_increment_t;
  using self_t = VectorView<value_t, container_t, index_t, increment_t>;
  container_t &data_;
  index_t size_data_;
  index_t size_;
  index_t disp_;
  increment_t strd_;  // never size_t, because it could be negative

  VectorView(view_container_t &data, view_index_t disp = 0,
             view_increment_t strd = 1);
  VectorView(view_container_t &data, view_index_t disp, view_increment_t strd,
             view_index_t size);
  VectorView(
      VectorView<view_value_t, view_container_t, view_index_t, view_increment_t>
          opV,
      view_index_t disp, view_increment_t strd, view_index_t size);

  /*!
  @brief Initializes the view using the indexing values.
  @param originalSize The original size of the container
  */
  inline void initialize(index_t originalSize);

  /*!
   * @brief Returns a reference to the container
   */
  container_t &get_data();

  /*!
   * @brief Returns a pointer containing the raw data of the container
   */
  value_t *get_pointer();

  /*! get_access_displacement.
   * @brief get displacement from the origin.
   */
  index_t get_access_displacement();

  /*! adjust_access_displacement.
   * @brief this method adjust the position of the data access to point to the
   *  data_ + offset_ on the device side. This function will be called at the
   * begining of an expression so that the kernel wont repeat this operation at
   * every eval call
   */
  void adjust_access_displacement();

  /*!
   * @brief Returns the size of the underlying container.
   */
  index_t get_data_size();

  /*!
   @brief Returns the size of the view
   */
  inline index_t get_size() const;

  /*!
   @brief Returns the stride of the view.
  */
  increment_t get_stride();

  /**** EVALUATING ****/
  value_t &eval(index_t i);
};

/*! MatrixView
@brief Represents a Matrix on the given Container.
@tparam value_t Value type of the container.
@tparam container_t Type of the container.
 */
template <typename view_value_t, typename view_container_t,
          typename view_index_t, typename layout>
struct MatrixView {
  // Information related to the data
  using access_layout_t = layout;
  using value_t = view_value_t;
  using container_t = view_container_t;
  using index_t = view_index_t;
  using self_t = MatrixView<value_t, container_t, index_t, layout>;
  container_t &data_;
  index_t size_data_;  // real size of the data
  index_t sizeR_;      // number of rows
  index_t sizeC_;      // number of columns
  index_t sizeL_;      // size of the leading dimension
  index_t disp_;       // displacementt od the first element
  // UPLO, BAND(KU,KL), PACKED, SIDE ARE ONLY REQUIRED
  MatrixView(view_container_t &data, view_index_t sizeR, view_index_t sizeC);
  MatrixView(view_container_t &data, view_index_t sizeR, view_index_t sizeC,
             view_index_t sizeL, view_index_t disp);
  MatrixView(
      MatrixView<view_value_t, view_container_t, view_index_t, layout> opM,
      view_index_t sizeR, view_index_t sizeC, view_index_t sizeL,
      view_index_t disp);

  /*!
   * @brief Returns the container
   */
  container_t &get_data();

  /*!
   * @brief Returns the container
   */
  value_t *get_pointer();

  /*!
   * @brief Returns the data size
   */
  index_t get_data_size() const;

  /*!
   * @brief Returns the size of the view.
   */
  inline index_t get_size() const;

  /*! get_size_row.
   * @brief Return the number of columns.
   * @bug This value should change depending on the access mode, but
   * is currently set to Rows.
   */
  inline index_t get_size_row() const;

  /*! get_size_col.
   * @brief Return the number of columns.
   * @bug This value should change depending on the access mode, but
   * is currently set to Rows.
   */
  index_t get_size_col() const;

  /*! get_access_device.
   * @brief Access on the Device (e.g CPU: Row, GPU: Column).
   */
  int get_access_device() const;

  /*! adjust_access_displacement.
   * @brief set displacement from the origin.
   */
  void adjust_access_displacement();

  /*! get_access_displacement.
   * @brief get displacement from the origin.
   */
  index_t get_access_displacement() const;

  /*! eval.
   * @brief Evaluation for the given linear value.
   */
  value_t &eval(index_t k);

  /*! eval.
   * @brief Evaluation for the pair of row/col.
   */
  value_t &eval(index_t i, index_t j);
};

template <typename scalar_t, typename container_t, typename index_t,
          typename increment_t>
struct VectorViewTypeFactory {
  using output_t = VectorView<scalar_t, container_t, index_t, increment_t>;
};

template <typename scalar_t, typename container_t, typename index_t,
          typename access_mode_t>
struct MatrixViewTypeFactory {
  using output_t = MatrixView<scalar_t, container_t, index_t, access_mode_t>;
};

template <typename scalar_t, typename increment_t, typename index_t>
static inline auto make_vector_view(BufferIterator<scalar_t> buff,
                                    increment_t inc, index_t sz) {
  static constexpr cl::sycl::access::mode access_mode_t =
      Choose<std::is_const<scalar_t>::value, cl::sycl::access::mode,
             cl::sycl::access::mode::read,
             cl::sycl::access::mode::read_write>::type;
  using leaf_node_t = typename VectorViewTypeFactory<
      scalar_t,
      typename BufferIterator<scalar_t>::template default_accessor_t<
          access_mode_t>,
      index_t, increment_t>::output_t;
  return leaf_node_t{buff.template get_range_accessor<access_mode_t>(),
                     (index_t)buff.get_offset(), inc, sz};
}

template <typename access_layout_t, typename scalar_t, typename index_t>
static inline auto make_matrix_view(BufferIterator<scalar_t> buff, index_t m,
                                    index_t n, index_t lda,
                                    index_t stride = static_cast<index_t>(1)) {
  static constexpr cl::sycl::access::mode access_mode_t =
      Choose<std::is_const<scalar_t>::value, cl::sycl::access::mode,
             cl::sycl::access::mode::read,
             cl::sycl::access::mode::read_write>::type;
  using leaf_node_t = typename MatrixViewTypeFactory<
      scalar_t,
      typename BufferIterator<scalar_t>::template default_accessor_t<
          access_mode_t>,
      index_t, access_layout_t>::output_t;
  return leaf_node_t{buff.template get_range_accessor<access_mode_t>(), m, n,
                     lda, stride, (index_t)buff.get_offset()};
}

}  // namespace blas

#endif  // VIEW_H
