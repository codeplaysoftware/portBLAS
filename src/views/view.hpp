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
template <class _ValueT, class _ContainerT, typename _IndexType,
          typename _IncrementType>
vector_view<_ValueT, _ContainerT, _IndexType, _IncrementType>::vector_view(
    _ContainerT &data, _IndexType disp, _IncrementType strd)
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
template <class _ValueT, class _ContainerT, typename _IndexType,
          typename _IncrementType>
vector_view<_ValueT, _ContainerT, _IndexType, _IncrementType>::vector_view(
    _ContainerT &data, _IndexType disp, _IncrementType strd, _IndexType size)
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
template <class _ValueT, class _ContainerT, typename _IndexType,
          typename _IncrementType>
vector_view<_ValueT, _ContainerT, _IndexType, _IncrementType>::vector_view(
    vector_view<_ValueT, _ContainerT, _IndexType, _IncrementType> opV,
    _IndexType disp, _IncrementType strd, _IndexType size)
    : data_(opV.getData()),
      size_data_(opV.getData().size()),
      size_(0),
      disp_(disp),
      strd_(strd) {
  initialize(size);
}

/*!
@brief Initializes the view using the indexing values.
@param originalSize The original size of the container
*/
template <class _ValueT, class _ContainerT, typename _IndexType,
          typename _IncrementType>
inline void vector_view<_ValueT, _ContainerT, _IndexType,
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
template <class _ValueT, class _ContainerT, typename _IndexType,
          typename _IncrementType>
inline _ContainerT &
vector_view<_ValueT, _ContainerT, _IndexType, _IncrementType>::getData() {
  return data_;
}

/*!
 * @brief Returns the displacement
 */
template <class _ValueT, class _ContainerT, typename _IndexType,
          typename _IncrementType>
inline _IndexType
vector_view<_ValueT, _ContainerT, _IndexType, _IncrementType>::getDisp() {
  return disp_;
}

/*!
 * @brief Returns the size of the underlying container.
 */
template <class _ValueT, class _ContainerT, typename _IndexType,
          typename _IncrementType>
inline _IndexType
vector_view<_ValueT, _ContainerT, _IndexType, _IncrementType>::getDataSize() {
  return size_data_;
}

/*!
 @brief Returns the size of the view
 */
template <class _ValueT, class _ContainerT, typename _IndexType,
          typename _IncrementType>
inline _IndexType
vector_view<_ValueT, _ContainerT, _IndexType, _IncrementType>::getSize() const {
  return size_;
}

/*!
 @brief Returns the stride of the view.
*/
template <class _ValueT, class _ContainerT, typename _IndexType,
          typename _IncrementType>
inline _IncrementType
vector_view<_ValueT, _ContainerT, _IndexType, _IncrementType>::getStrd() {
  return strd_;
}

/*!
 * @brief Adds a displacement to the view, creating a new view.
 */
template <class _ValueT, class _ContainerT, typename _IndexType,
          typename _IncrementType>
vector_view<_ValueT, _ContainerT, _IndexType, _IncrementType>
vector_view<_ValueT, _ContainerT, _IndexType, _IncrementType>::operator+(
    _IndexType disp) {
  if (this->strd_ > 0) {
    return vector_view<_ValueT, _ContainerT, _IndexType, _IncrementType>(
        this->data_, this->disp_ + (disp * this->strd_), this->strd_,
        this->size_ - disp);
  } else {
    return vector_view<_ValueT, _ContainerT, _IndexType, _IncrementType>(
        this->data_, this->disp_ - ((this->size_ - 1) - disp) * this->strd_,
        this->strd_, this->size_ - disp);
  }
}

/*!
 * @brief Adds a displacement to the view, creating a new view.
 */
template <class _ValueT, class _ContainerT, typename _IndexType,
          typename _IncrementType>
vector_view<_ValueT, _ContainerT, _IndexType, _IncrementType>
vector_view<_ValueT, _ContainerT, _IndexType, _IncrementType>::operator()(
    _IndexType disp) {
  if (this->strd_ > 0) {
    return vector_view<_ValueT, _ContainerT, _IndexType, _IncrementType>(
        this->data_, this->disp_ + (disp * this->strd_), this->strd_,
        this->size_ - disp);
  } else {
    return vector_view<_ValueT, _ContainerT, _IndexType, _IncrementType>(
        this->data_, this->disp_ - ((this->size_ - 1) - disp) * this->strd_,
        this->strd_, this->size_ - disp);
  }
}

/*!
 @brief Multiplies the view stride by the given one and returns a new one
*/
template <class _ValueT, class _ContainerT, typename _IndexType,
          typename _IncrementType>
vector_view<_ValueT, _ContainerT, _IndexType, _IncrementType>
    vector_view<_ValueT, _ContainerT, _IndexType, _IncrementType>::operator*(
        _IncrementType strd) {
  return vector_view<_ValueT, _ContainerT, _IndexType, _IncrementType>(
      this->data_, this->disp_, this->strd_ * strd);
}

/*!
 @brief
*/
template <class _ValueT, class _ContainerT, typename _IndexType,
          typename _IncrementType>
vector_view<_ValueT, _ContainerT, _IndexType, _IncrementType>
vector_view<_ValueT, _ContainerT, _IndexType, _IncrementType>::operator%(
    _IndexType size) {
  if (this->strd_ > 0) {
    return vector_view<_ValueT, _ContainerT, _IndexType, _IncrementType>(
        this->data_, this->disp_, this->strd_, size);
  } else {
    return vector_view<_ValueT, _ContainerT, _IndexType, _IncrementType>(
        this->data_, this->disp_ - (this->size_ - 1) * this->strd_, this->strd_,
        size);
  }
}

/**** EVALUATING ****/
template <class _ValueT, class _ContainerT, typename _IndexType,
          typename _IncrementType>
_ValueT &vector_view<_ValueT, _ContainerT, _IndexType, _IncrementType>::eval(
    IndexType i) {
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
template <class _ValueT, class _ContainerT, typename _IndexType,
          typename _IncrementType>
void vector_view<_ValueT, _ContainerT, _IndexType, _IncrementType>::printH(
    const char *name) {
  int frst = 1;
  std::cout << name << " = [ ";
  for (IndexType i = 0; i < size_; i++) {
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
 * @param accessDev Row-major or column-major.
 * @param sizeR Number of rows.
 * @param sizeC Number of columns.
 */
template <class _ValueT, class _ContainerT, typename _IndexType>
matrix_view<_ValueT, _ContainerT, _IndexType>::matrix_view(_ContainerT &data,
                                                           int accessDev,
                                                           _IndexType sizeR,
                                                           _IndexType sizeC)
    : data_(data),
      accessDev_(accessDev),
      size_data_(data_.get_size()),
      accessOpr_(1),
      sizeR_(sizeR),
      sizeC_(sizeC),
      sizeL_(0),
      disp_(0) {
  sizeL_ = (!(accessDev_ ^ accessOpr_)) ? sizeC_ : sizeR_;
}

/*!
 * @brief Constructs a matrix view on the container.
 * @param data Reference to the container.
 * @param sizeR Number of rows.
 * @param sizeC Nummber of columns.
 */
template <class _ValueT, class _ContainerT, typename _IndexType>
matrix_view<_ValueT, _ContainerT, _IndexType>::matrix_view(_ContainerT &data,
                                                           _IndexType sizeR,
                                                           _IndexType sizeC)
    : data_(data),
      accessDev_(1),
      size_data_(data_.get_size()),
      accessOpr_(1),
      sizeR_(sizeR),
      sizeC_(sizeC),
      sizeL_(0),
      disp_(0) {
  sizeL_ = (!(accessDev_ ^ accessOpr_)) ? sizeC_ : sizeR_;
}

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
template <class _ValueT, class _ContainerT, typename _IndexType>
matrix_view<_ValueT, _ContainerT, _IndexType>::matrix_view(
    _ContainerT &data, int accessDev, _IndexType sizeR, _IndexType sizeC,
    int accessOpr, _IndexType sizeL, _IndexType disp)
    : data_(data),
      accessDev_(accessDev),
      size_data_(data_.size()),
      accessOpr_(accessOpr),
      sizeR_(sizeR),
      sizeC_(sizeC),
      sizeL_(sizeL),
      disp_(disp) {}

/*!
 * @brief Constructs a matrix view on the container.
 * @param data Reference to the container.
 * @param sizeR Number of rows.
 * @param sizeC Number of columns.
 * @param accessOpr
 * @param sizeL Size of the leading dimension.
 * @param disp Displacement from the start.
 */
template <class _ValueT, class _ContainerT, typename _IndexType>
matrix_view<_ValueT, _ContainerT, _IndexType>::matrix_view(
    _ContainerT &data, _IndexType sizeR, _IndexType sizeC, int accessOpr,
    _IndexType sizeL, _IndexType disp)
    : data_(data),
      accessDev_(1),
      size_data_(data_.size()),
      accessOpr_(accessOpr),
      sizeR_(sizeR),
      sizeC_(sizeC),
      sizeL_(sizeL),
      disp_(disp) {}

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
template <class _ValueT, class _ContainerT, typename _IndexType>
matrix_view<_ValueT, _ContainerT, _IndexType>::matrix_view(
    matrix_view<_ValueT, _ContainerT, _IndexType> opM, int accessDev,
    _IndexType sizeR, _IndexType sizeC, int accessOpr, _IndexType sizeL,
    _IndexType disp)
    : data_(opM.data_),
      accessDev_(accessDev),
      size_data_(opM.size_data_),
      accessOpr_(!(opM.accessOpr_ ^ accessOpr)),
      sizeR_(sizeR),
      sizeC_(sizeC),
      sizeL_(sizeL),
      disp_(disp) {}

/*!
 * @brief Creates a matrix view from the given one.
 * @param opM Matrix view.
 * @param sizeR Number of rows.
 * @param sizeC Number of columns.
 * @param accessorOpr
 * @param sizeL Size of leading dimension.
 * @param disp Displacement from the start.
 */
template <class _ValueT, class _ContainerT, typename _IndexType>
matrix_view<_ValueT, _ContainerT, _IndexType>::matrix_view(
    matrix_view<_ValueT, _ContainerT, _IndexType> opM, _IndexType sizeR,
    _IndexType sizeC, int accessOpr, _IndexType sizeL, _IndexType disp)
    : data_(opM.data_),
      accessDev_(opM.accessDev_),
      size_data_(opM.size_data_),
      accessOpr_(!(opM.accessOpr_ ^ accessOpr)),
      sizeR_(sizeR),
      sizeC_(sizeC),
      sizeL_(sizeL),
      disp_(disp) {}

/*!
 * @brief Returns the container
 */
template <class _ValueT, class _ContainerT, typename _IndexType>
inline _ContainerT &matrix_view<_ValueT, _ContainerT, _IndexType>::getData() {
  return data_;
}

/*!
 * @brief Returns the data size
 */
template <class _ValueT, class _ContainerT, typename _IndexType>
inline _IndexType matrix_view<_ValueT, _ContainerT, _IndexType>::getDataSize()
    const {
  return size_data_;
}

/*!
 * @brief Returns the size of the view.
 */
template <class _ValueT, class _ContainerT, typename _IndexType>
inline _IndexType matrix_view<_ValueT, _ContainerT, _IndexType>::getSize()
    const {
  return sizeR_ * sizeC_;
}

/*! getSizeR.
 * @brief Return the number of columns.
 * @bug This value should change depending on the access mode, but
 * is currently set to Rows.
 */
template <class _ValueT, class _ContainerT, typename _IndexType>
inline _IndexType matrix_view<_ValueT, _ContainerT, _IndexType>::getSizeR()
    const {
  return sizeR_;
}

/*! getSizeC.
 * @brief Return the number of columns.
 * @bug This value should change depending on the access mode, but
 * is currently set to Rows.
 */
template <class _ValueT, class _ContainerT, typename _IndexType>
inline _IndexType matrix_view<_ValueT, _ContainerT, _IndexType>::getSizeC()
    const {
  return sizeC_;
}

/*! is_row_access.
 * @brief Access mode for the view.
 * Combination of the device access vs the operation mode.
 */
template <class _ValueT, class _ContainerT, typename _IndexType>
inline int matrix_view<_ValueT, _ContainerT, _IndexType>::is_row_access()
    const {
  return !(accessDev_ ^ accessOpr_);
}

/*! getAccessDev.
 * @brief Access on the Device (e.g CPU: Row, GPU: Column).
 */
template <class _ValueT, class _ContainerT, typename _IndexType>
inline int matrix_view<_ValueT, _ContainerT, _IndexType>::getAccessDev() const {
  return accessDev_;
}

/*! getAccessOpr.
 * @brief Returns the operation access mode
 * @return True: Normal access, False: Transpose
 */
template <class _ValueT, class _ContainerT, typename _IndexType>
inline int matrix_view<_ValueT, _ContainerT, _IndexType>::getAccessOpr() const {
  return accessOpr_;
}

/*! getDisp.
 * @brief get displacement from the origin.
 */
template <class _ValueT, class _ContainerT, typename _IndexType>
inline _IndexType matrix_view<_ValueT, _ContainerT, _IndexType>::getDisp()
    const {
  return disp_;
}

/*!
 * @brief Adds a displacement to the view, creating a new view.
 */
template <class _ValueT, class _ContainerT, typename _IndexType>
matrix_view<_ValueT, _ContainerT, _IndexType>
matrix_view<_ValueT, _ContainerT, _IndexType>::operator+(_IndexType disp) {
  return matrix_view<_ValueT, _ContainerT, _IndexType>(
      this->data_, this->accessDev_, this->sizeR_, this->sizeC_,
      this->accessOpr_, this->sizeL_, this->disp_ + disp);
}

/*!
 * @brief Adds a displacement to the view, creating a new view.
 */
template <class _ValueT, class _ContainerT, typename _IndexType>
matrix_view<_ValueT, _ContainerT, _IndexType>
matrix_view<_ValueT, _ContainerT, _IndexType>::operator()(_IndexType i,
                                                          _IndexType j) {
  if (!(accessDev_ ^ accessOpr_)) {
    // ACCESING BY ROWS
    return matrix_view<_ValueT, _ContainerT, _IndexType>(
        this->data_, this->accessDev_, this->sizeR_, this->sizeC_,
        this->accessOpr_, this->sizeL_, this->disp_ + i * this->sizeL_ + j);
  } else {
    // ACCESING BY COLUMN
    return matrix_view<_ValueT, _ContainerT, _IndexType>(
        this->data_, this->accessDev_, this->sizeR_, this->sizeC_,
        this->accessOpr_, this->sizeL_, this->disp_ + i + this->sizeL_ * j);
  }
}

/*! eval.
 * @brief Evaluation for the given linear value.
 */
template <class _ValueT, class _ContainerT, typename _IndexType>
_ValueT &matrix_view<_ValueT, _ContainerT, _IndexType>::eval(_IndexType k) {
  auto ind = disp_;
  auto access = (!(accessDev_ ^ accessOpr_));
  auto size = (access) ? sizeC_ : sizeR_;
  auto i = (access) ? (k / size) : (k % size);
  auto j = (access) ? (k % size) : (k / size);
  ;
  return eval(i, j);
}

/*! eval.
 * @brief Evaluation for the pair of row/col.
 */
template <class _ValueT, class _ContainerT, typename _IndexType>
_ValueT &matrix_view<_ValueT, _ContainerT, _IndexType>::eval(_IndexType i,
                                                             _IndexType j) {
  auto ind = disp_;
  if (!(accessDev_ ^ accessOpr_)) {
    ind += (sizeL_ * i) + j;
  } else {
    ind += (sizeL_ * j) + i;
  }
  if (ind >= size_data_) {
    std::cout << "ind = " << ind << std::endl;
    throw std::invalid_argument("Out of range access");
  }
  return data_[ind];
}

/*! printH
 * @brief Display the contents of the matrix in stdout.
 */
template <class _ValueT, class _ContainerT, typename _IndexType>
void matrix_view<_ValueT, _ContainerT, _IndexType>::printH(const char *name) {
  std::cout << name << " = [ " << std::endl;
  for (IndexType i = 0; i < ((accessOpr_) ? sizeR_ : sizeC_); i++) {
    int frst = 1;
    for (IndexType j = 0; j < ((accessOpr_) ? sizeC_ : sizeR_); j++) {
      if (frst)
        std::cout << eval(i, j);
      else
        std::cout << " , " << eval(i, j);
      frst = 0;
    }
    std::cout << " ;" << std::endl;
  }
  std::cout << " ]" << std::endl;
}

}  // namespace blas

#endif  // VIEW_HPP
