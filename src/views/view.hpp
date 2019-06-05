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
 *  @filename view.hpp
 *
 **************************************************************************/

#ifndef SYCL_BLAS_VIEW_HPP
#define SYCL_BLAS_VIEW_HPP

#include "views/view.h"
#include <iostream>
#include <stdexcept>
#include <vector>

namespace blas {

/*!
@brief Constructs a view from the given container re-using the container size.
@param data
@param disp
@param strd
*/
template <class _value_t, class _container_t, typename _IndexType,
          typename _IncrementType>
VectorView<_value_t, _container_t, _IndexType, _IncrementType>::VectorView(
    _container_t &data, _IndexType disp, _IncrementType strd)
    : data_(data),
      size_data_(data_.size()),
      size_(data_.size()),
      disp_(disp),
      strd_(strd) {}

/*!
@brief Creates a view with a size smaller than the container size.
@param data
@param disp
@param strd
@param size
*/
template <class _value_t, class _container_t, typename _IndexType,
          typename _IncrementType>
VectorView<_value_t, _container_t, _IndexType, _IncrementType>::VectorView(
    _container_t &data, _IndexType disp, _IncrementType strd, _IndexType size)
    : data_(data),
      size_data_(data_.size()),
      size_(0),
      disp_(disp),
      strd_(strd) {
  initialize(size);
}

/*!
 @brief Creates a view from an existing view.
*/
template <class _value_t, class _container_t, typename _IndexType,
          typename _IncrementType>
VectorView<_value_t, _container_t, _IndexType, _IncrementType>::VectorView(
    VectorView<_value_t, _container_t, _IndexType, _IncrementType> opV,
    _IndexType disp, _IncrementType strd, _IndexType size)
    : data_(opV.get_data()),
      size_data_(opV.get_data().size()),
      size_(0),
      disp_(disp),
      strd_(strd) {
  initialize(size);
}

/*!
@brief Initializes the view using the indexing values.
@param originalSize The original size of the container
*/
template <class _value_t, class _container_t, typename _IndexType,
          typename _IncrementType>
inline void VectorView<_value_t, _container_t, _IndexType,
                       _IncrementType>::initialize(_IndexType originalSize) {
  if (strd_ > 0) {
    auto sizeV = (size_data_ - disp_);
    auto quot = (sizeV + strd_ - 1) / strd_;  // ceiling
    size_ = quot;
  } else if (strd_ > 0) {
    auto nstrd = -strd_;
    auto quot = (disp_ + nstrd) / nstrd;  // ceiling
    size_ = quot;
  } else {
    // Stride is zero, not valid!
    throw std::invalid_argument("Cannot create view with 0 stride");
  }
  if (originalSize < size_) size_ = originalSize;
  if (strd_ < 0) disp_ += (size_ - 1) * strd_;
}

/*!
 * @brief Returns a reference to the container
 */
template <class _value_t, class _container_t, typename _IndexType,
          typename _IncrementType>
inline _container_t &
VectorView<_value_t, _container_t, _IndexType, _IncrementType>::get_data() {
  return data_;
}

/*!
 * @brief Returns a reference to the container
 */
template <class _value_t, class _container_t, typename _IndexType,
          typename _IncrementType>
inline _value_t *
VectorView<_value_t, _container_t, _IndexType, _IncrementType>::get_pointer() {
  return data_;
}
/*!
 * @brief Returns the displacement
 */
template <class _value_t, class _container_t, typename _IndexType,
          typename _IncrementType>
inline _IndexType VectorView<_value_t, _container_t, _IndexType,
                             _IncrementType>::get_access_displacement() {
  return disp_;
}

/*!
 * @brief Returns the displacement
 */
template <class _value_t, class _container_t, typename _IndexType,
          typename _IncrementType>
inline void VectorView<_value_t, _container_t, _IndexType,
                       _IncrementType>::set_access_displacement() {
  return data_ += disp_;
}

/*!
 * @brief Returns the size of the underlying container.
 */
template <class _value_t, class _container_t, typename _IndexType,
          typename _IncrementType>
inline _IndexType VectorView<_value_t, _container_t, _IndexType,
                             _IncrementType>::get_data_size() {
  return size_data_;
}

/*!
 @brief Returns the size of the view
 */
template <class _value_t, class _container_t, typename _IndexType,
          typename _IncrementType>
inline _IndexType VectorView<_value_t, _container_t, _IndexType,
                             _IncrementType>::get_size() const {
  return size_;
}

/*!
 @brief Returns the stride of the view.
*/
template <class _value_t, class _container_t, typename _IndexType,
          typename _IncrementType>
inline _IncrementType
VectorView<_value_t, _container_t, _IndexType, _IncrementType>::get_stride() {
  return strd_;
}

/**** EVALUATING ****/
template <class _value_t, class _container_t, typename _IndexType,
          typename _IncrementType>
_value_t &VectorView<_value_t, _container_t, _IndexType, _IncrementType>::eval(
    index_t i) {
  auto ind = disp_;
  if (strd_ > 0) {
    ind += strd_ * i;
  } else {
    ind -= strd_ * (size_ - i - 1);
  }
  if (ind >= size_data_) {
    // out of range access
    throw std::invalid_argument("Out of range access");
  }
  return data_[ind];
}
template <class _value_t, class _container_t, typename _IndexType,
          typename _IncrementType>
void VectorView<_value_t, _container_t, _IndexType, _IncrementType>::print_h(
    const char *name) {
  int frst = 1;
  std::cout << name << " = [ ";
  for (index_t i = 0; i < size_; i++) {
    if (frst)
      std::cout << eval(i);
    else
      std::cout << " , " << eval(i);
    frst = 0;
  }
  std::cout << " ]" << std::endl;
}

/*!
 * @brief Constructs a matrix view on the container.
 * @param data Reference to the container.
 * @param sizeR Number of rows.
 * @param sizeC Nummber of columns.
 */
template <class _value_t, class _container_t, typename _IndexType,
          typename layout>
MatrixView<_value_t, _container_t, _IndexType, layout>::MatrixView(
    _container_t &data, _IndexType sizeR, _IndexType sizeC)
    : data_(data),
      size_data_(data_.get_size()),
      sizeR_(sizeR),
      sizeC_(sizeC),
      sizeL_((MatrixView<_value_t, _container_t, _IndexType, layout>::is_col_major())
          ? sizeC_
          : sizeR_)),
      disp_(0) {}

/*!
 * @brief Constructs a matrix view on the container.
 * @param data Reference to the container.
 * @param accessDev Row-major or column-major.
 * @param sizeR Number of rows.
 * @param sizeC Number of columns.
 * @param accessOpr
 * @param sizeL Size of the leading dimension.
 * @param disp Displacement from the start.
 */
template <class _value_t, class _container_t, typename _IndexType,
          typename layout>
MatrixView<_value_t, _container_t, _IndexType, layout>::MatrixView(
    _container_t &data, _IndexType sizeR, _IndexType sizeC, _IndexType sizeL,
    _IndexType disp)
    : data_(data + disp),
      size_data_(data_.size()),
      sizeR_(sizeR),
      sizeC_(sizeC),
      sizeL_(sizeL),
      disp_(0) {}

/*!
 * @brief Constructs a matrix view on the container.
 * @param data Reference to the container.
 * @param sizeR Number of rows.
 * @param sizeC Number of columns.
 * @param sizeL Size of the leading dimension.
 * @param disp Displacement from the start.
 */
template <class _value_t, class _container_t, typename _IndexType,
          typename layout>
MatrixView<_value_t, _container_t, _IndexType, layout>::MatrixView(
    _container_t &data, _IndexType sizeR, _IndexType sizeC, _IndexType sizeL,
    _IndexType disp)
    : data_(data + disp),
      size_data_(data_.size()),
      sizeR_(sizeR),
      sizeC_(sizeC),
      sizeL_(sizeL),
      disp_(0) {}

/*!
 *@brief Creates a matrix view from the given one but with different access
 * parameters.
 * @param opM Matrix view.
 * @param accessDev Row-major or column-major.
 * @param sizeR Number of rows.
 * @param sizeC Number of columns.
 * @param accessorOpr
 * @param sizeL Size of the leading dimension.
 * @param disp Displacement from the start.
 */
template <class _value_t, class _container_t, typename _IndexType,
          typename layout>
MatrixView<_value_t, _container_t, _IndexType, layout>::MatrixView(
    MatrixView<_value_t, _container_t, _IndexType, layout> opM,
    _IndexType sizeR, _IndexType sizeC, _IndexType sizeL, _IndexType disp)
    : data_(opM.data_ + disp),
      size_data_(opM.size_data_),
      sizeR_(sizeR),
      sizeC_(sizeC),
      sizeL_(sizeL),
      disp_(0) {}

/*!
 * @brief Creates a matrix view from the given one.
 * @param opM Matrix view.
 * @param sizeR Number of rows.
 * @param sizeC Number of columns.
 * @param accessorOpr
 * @param sizeL Size of leading dimension.
 * @param disp Displacement from the start.
 */
template <class _value_t, class _container_t, typename _IndexType,
          typename layout>
MatrixView<_value_t, _container_t, _IndexType, layout>::MatrixView(
    MatrixView<_value_t, _container_t, _IndexType, layout> opM,
    _IndexType sizeR, _IndexType sizeC, _IndexType sizeL, _IndexType disp)
    : data_(opM.data_ + disp),
      size_data_(opM.size_data_),
      sizeR_(sizeR),
      sizeC_(sizeC),
      sizeL_(sizeL),
      disp_(0) {}

/*!
 * @brief Returns the container
 */
template <class _value_t, class _container_t, typename _IndexType,
          typename layout>
inline _container_t &
MatrixView<_value_t, _container_t, _IndexType, layout>::get_data() {
  return data_;
}

/*!
 * @brief Returns the data size
 */
template <class _value_t, class _container_t, typename _IndexType,
          typename layout>
inline _IndexType
MatrixView<_value_t, _container_t, _IndexType, layout>::get_data_size() const {
  return size_data_;
}

/*!
 * @brief Returns the size of the view.
 */
template <class _value_t, class _container_t, typename _IndexType,
          typename layout>
inline _IndexType
MatrixView<_value_t, _container_t, _IndexType, layout>::get_size() const {
  return sizeR_ * sizeC_;
}

/*! get_size_row.
 * @brief Return the number of columns.
 * @bug This value should change depending on the access mode, but
 * is currently set to Rows.
 */
template <class _value_t, class _container_t, typename _IndexType,
          typename layout>
inline _IndexType
MatrixView<_value_t, _container_t, _IndexType, layout>::get_size_row() const {
  return sizeR_;
}

/*! get_size_col.
 * @brief Return the number of columns.
 * @bug This value should change depending on the access mode, but
 * is currently set to Rows.
 */
template <class _value_t, class _container_t, typename _IndexType,
          typename layout>
inline _IndexType
MatrixView<_value_t, _container_t, _IndexType, layout>::get_size_col() const {
  return sizeC_;
}

/*! get_access_displacement.
 * @brief get displacement from the origin.
 */
template <class _value_t, class _container_t, typename _IndexType,
          typename layout>
inline _IndexType MatrixView<_value_t, _container_t, _IndexType,
                             layout>::get_access_displacement() const {
  return disp_;
}

/*! get_access_displacement.
 * @brief get displacement from the origin.
 */
template <class _value_t, class _container_t, typename _IndexType,
          typename layout>
inline void MatrixView<_value_t, _container_t, _IndexType,
                       layout>::set_access_displacement() const {
  return data_ += disp_;
}

/*! eval.
 * @brief Evaluation for the given linear value.
 */
template <class _value_t, class _container_t, typename _IndexType,
          typename layout>
_value_t &MatrixView<_value_t, _container_t, _IndexType, layout>::eval(
    _IndexType ind) {
  return data_[ind];
}

/*! eval.
 * @brief Evaluation for the pair of row/col.
 */
template <class _value_t, class _container_t, typename _IndexType,
          typename layout>
_value_t &MatrixView<_value_t, _container_t, _IndexType, layout>::eval(
    _IndexType i, _IndexType j) {
  return ((layout::is_col_major()) ? data_[(sizeL_ * i) + j]
                                   : data_[(sizeL_ * j) + i]);
}

}  // namespace blas

#endif  // VIEW_HPP
