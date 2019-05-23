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

#include "../blas_meta.h"
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
   * @brief Returns the displacement
   */
  index_t get_access_displacement();

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

  /*!
   * @brief Adds a displacement to the view, creating a new view.
   */
  self_t operator+(index_t disp);

  /*!
   * @brief Adds a displacement to the view, creating a new view.
   */
  self_t operator()(index_t disp);

  /*!
   @brief Multiplies the view stride by the given one and returns a new one
  */
  self_t operator*(increment_t strd);
  /*!
   @brief
  */
  self_t operator%(index_t size);

  /**** EVALUATING ****/
  value_t &eval(index_t i);

  template <class X, class Y, typename IndxT, typename IncrT>
  friend std::ostream &operator<<(std::ostream &stream,
                                  VectorView<X, Y, IndxT, IncrT> opvS);

  void print_h(const char *name);
};

/*! MatrixView
@brief Represents a Matrix on the given Container.
@tparam value_t Value type of the container.
@tparam container_t Type of the container.
 */
template <typename view_value_t, typename view_container_t,
          typename view_index_t>
struct MatrixView {
  // Information related to the data
  using value_t = view_value_t;
  using container_t = view_container_t;
  using index_t = view_index_t;
  using self_t = MatrixView<value_t, container_t, index_t>;
  container_t &data_;
  int accessDev_;      // True for row-major, column-major otherwise
  index_t size_data_;  // real size of the data
  int accessOpr_;      // Operation Access Mode (True: Normal, False: Transpose)
  index_t sizeR_;      // number of rows
  index_t sizeC_;      // number of columns
  index_t sizeL_;      // size of the leading dimension
  index_t disp_;       // displacementt od the first element
  // UPLO, BAND(KU,KL), PACKED, SIDE ARE ONLY REQUIRED
  MatrixView(view_container_t &data, int accessDev, view_index_t sizeR,
             view_index_t sizeC);
  MatrixView(view_container_t &data, view_index_t sizeR, view_index_t sizeC);
  MatrixView(view_container_t &data, int accessDev, view_index_t sizeR,
             view_index_t sizeC, int accessOpr, view_index_t sizeL,
             view_index_t disp);
  MatrixView(view_container_t &data, view_index_t sizeR, view_index_t sizeC,
             int accessOpr, view_index_t sizeL, view_index_t disp);
  MatrixView(MatrixView<view_value_t, view_container_t, view_index_t> opM,
             int accessDev, view_index_t sizeR, view_index_t sizeC,
             int accessOpr, view_index_t sizeL, view_index_t disp);
  MatrixView(MatrixView<view_value_t, view_container_t, view_index_t> opM,
             view_index_t sizeR, view_index_t sizeC, int accessOpr,
             view_index_t sizeL, view_index_t disp);

  /*!
   * @brief Returns the container
   */
  container_t &get_data();

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

  /*! is_row_access.
   * @brief Access mode for the view.
   * Combination of the device access vs the operation mode.
   */
  int is_row_access() const;

  /*! get_access_device.
   * @brief Access on the Device (e.g CPU: Row, GPU: Column).
   */
  int get_access_device() const;

  /*! get_access_operation.
   * @brief Returns the operation access mode
   * @return True: Normal access, False: Transpose
   */
  int get_access_operation() const;

  /*! get_access_displacement.
   * @brief get displacement from the origin.
   */
  index_t get_access_displacement() const;

  /*!
   * @brief Adds a displacement to the view, creating a new view.
   */
  self_t operator+(index_t disp);

  /*!
   * @brief Adds a displacement to the view, creating a new view.
   */
  self_t operator()(index_t i, index_t j);

  /*! eval.
   * @brief Evaluation for the given linear value.
   */
  value_t &eval(index_t k);

  /*! eval.
   * @brief Evaluation for the pair of row/col.
   */
  value_t &eval(index_t i, index_t j);

  /*! print_h
   * @brief Display the contents of the matrix in stdout.
   */
  void print_h(const char *name);
};

template <typename policy_t, typename data_t, typename index_t,
          typename increment_t>
struct VectorViewTypeFactory {
  using scalar_t = typename ValueType<data_t>::type;
  using output_t =
      VectorView<scalar_t,
                 typename policy_t::template default_accessor_t<scalar_t>,
                 index_t, increment_t>;
};

template <typename policy_t, typename element_t, typename index_t>
struct MatrixViewTypeFactory {
  using scalar_t = typename ValueType<element_t>::type;
  using output_t =
      MatrixView<scalar_t,
                 typename policy_t::template default_accessor_t<scalar_t>,
                 index_t>;
};

template <typename executor_t, typename container_t, typename increment_t,
          typename index_t>
inline
    typename VectorViewTypeFactory<typename executor_t::policy_t, container_t,
                                   index_t, increment_t>::output_t
    make_vector_view(executor_t &ex, container_t buff, increment_t inc,
                     index_t sz) {
  using leaf_node_t =
      typename VectorViewTypeFactory<typename executor_t::policy_t, container_t,
                                     index_t, increment_t>::output_t;
  return leaf_node_t{ex.get_policy_handler().get_buffer(buff), inc, sz};
}

template <typename executor_t, typename container_t, typename index_t,
          typename operator_t>
inline typename MatrixViewTypeFactory<typename executor_t::policy_t,
                                      container_t, index_t>::output_t
make_matrix_view(executor_t &ex, container_t buff, index_t m, index_t n,
                 index_t lda, operator_t accessOpr) {
  using leaf_node_t =
      typename MatrixViewTypeFactory<typename executor_t::policy_t, container_t,
                                     index_t>::output_t;
  return leaf_node_t{ex.get_policy_handler().get_buffer(buff), m, n, accessOpr,
                     lda};
}

}  // namespace blas

#endif  // VIEW_H
