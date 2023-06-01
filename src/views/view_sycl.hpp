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
 *  @filename view_sycl.hpp
 *
 **************************************************************************/

#ifndef SYCL_BLAS_VIEW_SYCL_HPP
#define SYCL_BLAS_VIEW_SYCL_HPP

#include <CL/sycl.hpp>
#include <type_traits>

#include "blas_meta.h"
#include "container/sycl_iterator.h"
#include "views/view.h"

namespace blas {

/*!
 * @brief View of a vector with an accessor.
 * @tparam scalar_t Value type of accessor.
 */

template <typename ViewScalarT, int dim, cl::sycl::access::mode acc_mode_t,
          cl::sycl::access::target access_t,
          cl::sycl::access::placeholder place_holder_t, typename view_index_t,
          typename view_increment_t>
struct VectorView<
    ViewScalarT,
    cl::sycl::accessor<ViewScalarT, dim, acc_mode_t, access_t, place_holder_t>,
    view_index_t, view_increment_t> {
  using scalar_t = ViewScalarT;
  using value_t = scalar_t;
  using index_t = view_index_t;
  using increment_t = view_increment_t;
  static constexpr cl::sycl::access::mode access_mode_t = acc_mode_t;
  using container_t = cl::sycl::accessor<ViewScalarT, dim, acc_mode_t, access_t,
                                         place_holder_t>;
  using self_t = VectorView<scalar_t, container_t, index_t, increment_t>;

  // Accessor to the data containing the vector values.
  container_t data_;

  // Number of elements in the vector that will be read.
  const index_t size_;

  // Number of elements offset into the data buffer to start reading from.
  const index_t disp_;

  // Stride between data elements in memory.
  // If negative the data is read backwards, from
  //     data_.get_pointer() + (-size_ + 1) * stride_ + 1
  // up to
  //     data_.get_pointer()
  const increment_t stride_;

  // global pointer access inside the kernel
  cl::sycl::global_ptr<scalar_t> ptr_;

  // Round up the ration num / den, i.e. compute ceil(num / den)
  static SYCL_BLAS_INLINE index_t round_up_ratio(index_t num, index_t den) {
    return (num + den - 1) / den;
  }

  // Compute the number of elements to read from data. This is useful when a
  // VectorView is created without an explicit size, so that only the necessary
  // number of threads are launched.
  static SYCL_BLAS_INLINE index_t calculate_input_data_size(
      container_t &data, index_t, increment_t stride, index_t size) noexcept {
    increment_t const positive_stride = stride < 0 ? -stride : stride;
    index_t const calc_size = round_up_ratio(data.get_count(), positive_stride);
    return std::min(size, calc_size);
  }

  /*!
   * @brief See VectorView.
   */
  SYCL_BLAS_INLINE VectorView(container_t data, index_t disp, increment_t strd,
                              index_t size)
      : data_{data},
        size_(calculate_input_data_size(data, disp, strd, size)),
        disp_((strd > 0) ? disp : disp + (size_ - 1) * (-strd)),
        stride_(strd) {}

  /*!
   * @brief See VectorView.
   */
  SYCL_BLAS_INLINE VectorView(container_t data)
      : VectorView(data, 0, 1, data_.get_size()) {}

  /*!
   * @brief See VectorView.
   */
  SYCL_BLAS_INLINE VectorView(container_t data, index_t disp)
      : VectorView(data, disp, 1, data_.get_size()) {}

  /*!
   * @brief See VectorView.
   */
  SYCL_BLAS_INLINE VectorView(self_t &opV, index_t disp, increment_t strd,
                              index_t size)
      : VectorView(opV.get_data(), disp, strd, size) {}

  /*!
   * @brief See VectorView.
   */
  SYCL_BLAS_INLINE container_t &get_data() { return data_; }
  /*!
   * @brief See VectorView.
   */
  SYCL_BLAS_INLINE scalar_t *get_pointer() const { return ptr_; }

  /*!
   * @brief See VectorView.
   */
  SYCL_BLAS_INLINE index_t get_data_size() const { return data_.get_size(); }

  /*!
   * @brief See VectorView.
   */
  SYCL_BLAS_INLINE index_t get_size() const { return size_; }

  /*!
   * @brief See VectorView.
   */
  SYCL_BLAS_INLINE index_t get_access_displacement() const { return disp_; }

  /*!
   * @brief See VectorView.
   */
  SYCL_BLAS_INLINE increment_t get_stride() const { return stride_; }

  /**** EVALUATING ****/
  template <bool use_as_ptr = false>
  SYCL_BLAS_INLINE typename std::enable_if<!use_as_ptr, scalar_t &>::type eval(
      index_t i) {
    return (stride_ == 1) ? *(ptr_ + i) : *(ptr_ + i * stride_);
  }

  template <bool use_as_ptr = false>
  SYCL_BLAS_INLINE typename std::enable_if<!use_as_ptr, scalar_t>::type eval(
      index_t i) const {
    return (stride_ == 1) ? *(ptr_ + i) : *(ptr_ + i * stride_);
  }

  SYCL_BLAS_INLINE scalar_t &eval(cl::sycl::nd_item<1> ndItem) {
    return eval(ndItem.get_global_id(0));
  }

  SYCL_BLAS_INLINE const scalar_t eval(cl::sycl::nd_item<1> ndItem) const {
    return eval(ndItem.get_global_id(0));
  }

  template <bool use_as_ptr = false>
  SYCL_BLAS_INLINE typename std::enable_if<use_as_ptr, scalar_t &>::type eval(
      index_t indx) {
    return *(ptr_ + indx);
  }

  template <bool use_as_ptr = false>
  SYCL_BLAS_INLINE typename std::enable_if<use_as_ptr, scalar_t>::type eval(
      index_t indx) const noexcept {
    return *(ptr_ + indx);
  }

  SYCL_BLAS_INLINE void bind(cl::sycl::handler &h) { h.require(data_); }
  SYCL_BLAS_INLINE void adjust_access_displacement() {
    ptr_ = data_.get_pointer() + disp_;
  }
};

template <class ViewScalarT, int dim, cl::sycl::access::mode acc_mode_t,
          cl::sycl::access::target access_t,
          cl::sycl::access::placeholder place_holder_t, typename view_index_t,
          typename layout, bool is_inc>
struct MatrixView<
    ViewScalarT,
    cl::sycl::accessor<ViewScalarT, dim, acc_mode_t, access_t, place_holder_t>,
    view_index_t, layout, is_inc>;
/*!
 * @brief Specialization of an MatrixView with an accessor.
 */
template <class ViewScalarT, int dim, cl::sycl::access::mode acc_mode_t,
          cl::sycl::access::target access_t,
          cl::sycl::access::placeholder place_holder_t, typename view_index_t,
          typename layout, bool is_inc>
struct MatrixView<
    ViewScalarT,
    cl::sycl::accessor<ViewScalarT, dim, acc_mode_t, access_t, place_holder_t>,
    view_index_t, layout, is_inc> {
  using access_layout_t = layout;
  using scalar_t = ViewScalarT;
  using index_t = view_index_t;
  static constexpr cl::sycl::access::mode access_mode_t = acc_mode_t;
  using container_t = cl::sycl::accessor<ViewScalarT, dim, acc_mode_t, access_t,
                                         place_holder_t>;
  using self_t = MatrixView<scalar_t, container_t, index_t, layout>;

  using value_t = scalar_t;
  // Information related to the data
  container_t data_;
  // Information related to the operation
  const index_t sizeR_;  // number of rows
  const index_t sizeC_;  // number of columns
  const index_t sizeL_;  // size of the leading dimension
  const index_t inc_;    // internal increment between same row/column elements
  const index_t disp_;   // displacementt od the first element
  cl::sycl::global_ptr<scalar_t>
      ptr_;  // global pointer access inside the kernel

  /**** CONSTRUCTORS ****/
  SYCL_BLAS_INLINE MatrixView(container_t data, index_t sizeR, index_t sizeC,
                              index_t sizeL, index_t disp)
      : data_{data},
        sizeR_(sizeR),
        sizeC_(sizeC),
        sizeL_(sizeL),
        inc_(1),
        disp_(disp) {}

  SYCL_BLAS_INLINE MatrixView(container_t data, index_t sizeR, index_t sizeC)
      : MatrixView(data, sizeR, sizeC,
                   (layout::is_col_major() ? sizeR_ : sizeC_), 0) {}

  SYCL_BLAS_INLINE MatrixView(self_t opM, index_t sizeR, index_t sizeC,
                              index_t sizeL, index_t disp)
      : MatrixView(opM.data_, sizeR, sizeC, sizeL, disp) {}

  SYCL_BLAS_INLINE MatrixView(container_t data, index_t sizeR, index_t sizeC,
                              index_t sizeL, index_t inc, index_t disp)
      : data_{data},
        sizeR_(sizeR),
        sizeC_(sizeC),
        sizeL_(sizeL),
        inc_(inc),
        disp_(disp) {}

  /**** RETRIEVING DATA ****/
  SYCL_BLAS_INLINE container_t &get_data() { return data_; }

  SYCL_BLAS_INLINE const index_t get_size() const { return sizeR_ * sizeC_; }

  SYCL_BLAS_INLINE index_t get_data_size() const { return data_.get_size(); }

  SYCL_BLAS_INLINE const index_t getSizeL() const { return sizeL_; }

  SYCL_BLAS_INLINE const index_t get_size_row() const { return sizeR_; }

  SYCL_BLAS_INLINE const index_t get_size_col() const { return sizeC_; }

  SYCL_BLAS_INLINE index_t get_access_displacement() const { return disp_; }

  SYCL_BLAS_INLINE scalar_t *get_pointer() const { return ptr_; }

  /**** EVALUATING ***/

  template <bool intern_inc = is_inc>
  SYCL_BLAS_INLINE typename std::enable_if<intern_inc, scalar_t &>::type eval(
      index_t i, index_t j) {
    return ((layout::is_col_major()) ? *(ptr_ + i * inc_ + sizeL_ * j)
                                     : *(ptr_ + j * inc_ + sizeL_ * i));
  }

  template <bool intern_inc = is_inc>
  SYCL_BLAS_INLINE typename std::enable_if<!intern_inc, scalar_t &>::type eval(
      index_t i, index_t j) {
    return ((layout::is_col_major()) ? *(ptr_ + i + sizeL_ * j)
                                     : *(ptr_ + j + sizeL_ * i));
  }

  template <bool intern_inc = is_inc>
  SYCL_BLAS_INLINE typename std::enable_if<intern_inc, scalar_t &>::type eval(
      index_t i, index_t j) const noexcept {
    return ((layout::is_col_major()) ? *(ptr_ + i * inc_ + sizeL_ * j)
                                     : *(ptr_ + j * inc_ + sizeL_ * i));
  }

  template <bool intern_inc = is_inc>
  SYCL_BLAS_INLINE typename std::enable_if<!intern_inc, scalar_t &>::type eval(
      index_t i, index_t j) const noexcept {
    return ((layout::is_col_major()) ? *(ptr_ + i + sizeL_ * j)
                                     : *(ptr_ + j + sizeL_ * i));
  }

  template <bool use_as_ptr = false>
  SYCL_BLAS_INLINE typename std::enable_if<!use_as_ptr, scalar_t &>::type eval(
      index_t indx) {
    const index_t j = indx / sizeR_;
    const index_t i = indx - sizeR_ * j;
    return eval(i, j);
  }

  template <bool use_as_ptr = false>
  SYCL_BLAS_INLINE typename std::enable_if<!use_as_ptr, scalar_t>::type eval(
      index_t indx) const noexcept {
    const index_t j = indx / sizeR_;
    const index_t i = indx - sizeR_ * j;
    return eval(i, j);
  }

  SYCL_BLAS_INLINE scalar_t &eval(cl::sycl::nd_item<1> ndItem) {
    return eval(ndItem.get_global_id(0));
  }

  SYCL_BLAS_INLINE scalar_t eval(cl::sycl::nd_item<1> ndItem) const noexcept {
    return eval(ndItem.get_global_id(0));
  }

  template <bool use_as_ptr = false>
  SYCL_BLAS_INLINE typename std::enable_if<use_as_ptr, scalar_t &>::type eval(
      index_t indx) {
    return *(ptr_ + indx);
  }

  template <bool use_as_ptr = false>
  SYCL_BLAS_INLINE typename std::enable_if<use_as_ptr, scalar_t>::type eval(
      index_t indx) const noexcept {
    return *(ptr_ + indx);
  }

  SYCL_BLAS_INLINE void bind(cl::sycl::handler &h) { h.require(data_); }

  SYCL_BLAS_INLINE void adjust_access_displacement() {
    ptr_ = data_.get_pointer() + disp_;
  }
};

}  // namespace blas

#endif  // VIEW_SYCL_HPP
