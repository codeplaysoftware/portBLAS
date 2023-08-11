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
@brief Creates a view with a size smaller than the container size.
@param data
@param disp
@param strd
@param size
*/
template <class _value_t, class _container_t, typename _IndexType,
          typename _IncrementType>
SYCL_BLAS_INLINE VectorView<_value_t, _container_t, _IndexType,
                            _IncrementType>::VectorView(_container_t data,
                                                        _IncrementType strd,
                                                        _IndexType size)
    : data_(data), size_(size), strd_(strd), ptr_(data) {}

/*!
 @brief Creates a view from an existing view.
*/
template <class _value_t, class _container_t, typename _IndexType,
          typename _IncrementType>
SYCL_BLAS_INLINE
VectorView<_value_t, _container_t, _IndexType, _IncrementType>::VectorView(
    VectorView<_value_t, _container_t, _IndexType, _IncrementType> opV,
    _IncrementType strd, _IndexType size)
    : data_(opV.get_data()), size_(size), strd_(strd), ptr_(opV.get_data()) {}

/*!
 * @brief Returns a reference to the container
 */
template <class _value_t, class _container_t, typename _IndexType,
          typename _IncrementType>
SYCL_BLAS_INLINE _container_t
VectorView<_value_t, _container_t, _IndexType, _IncrementType>::get_data() {
  return data_;
}

/*!
 * @brief Returns a reference to the container
 */
template <class _value_t, class _container_t, typename _IndexType,
          typename _IncrementType>
SYCL_BLAS_INLINE _value_t*
VectorView<_value_t, _container_t, _IndexType, _IncrementType>::get_pointer() const {
  return ptr_;
}

/*! adjust_access_displacement
 * @brief adjust pointer offset
 * The user is responsible to adjust pointer offset for USM.
 */
template <class _value_t, class _container_t, typename _IndexType,
          typename _IncrementType>
SYCL_BLAS_INLINE void VectorView<_value_t, _container_t, _IndexType,
                                 _IncrementType>::adjust_access_displacement() {
}

/*!
 @brief Returns the size of the view
 */
template <class _value_t, class _container_t, typename _IndexType,
          typename _IncrementType>
SYCL_BLAS_INLINE _IndexType VectorView<_value_t, _container_t, _IndexType,
                                       _IncrementType>::get_size() const {
  return size_;
}

/*!
 @brief Returns the stride of the view.
*/
template <class _value_t, class _container_t, typename _IndexType,
          typename _IncrementType>
SYCL_BLAS_INLINE _IncrementType
VectorView<_value_t, _container_t, _IndexType, _IncrementType>::get_stride() {
  return strd_;
}

/*!
 * @brief Constructs a matrix view on the container.
 * @param data Pointer to memory.
 * @param sizeR Number of rows.
 * @param sizeC Nummber of columns.
 */
template <class _value_t, class _container_t, typename _IndexType, typename layout, bool has_inc>
SYCL_BLAS_INLINE MatrixView<_value_t, _container_t, _IndexType, layout, has_inc>::MatrixView(_container_t data, _IndexType sizeR, _IndexType sizeC) : data_(data),
      sizeR_(sizeR),
      sizeC_(sizeC),
      sizeL_((layout::is_col_major()) ? sizeR_ : sizeC_),
      ptr_(data) {}

/*!
 * @brief Constructs a matrix view on the container.
 * @param data Pointer to memory.
 * @param sizeR Number of rows.
 * @param sizeC Number of columns.
 * @param sizeL Size of the leading dimension.
 * @param disp Displacement from the start.
 */
template <class _value_t, class _container_t, typename _IndexType,
          typename layout, bool has_inc>
SYCL_BLAS_INLINE
MatrixView<_value_t, _container_t, _IndexType, layout, has_inc>::MatrixView(
    _container_t data, _IndexType sizeR, _IndexType sizeC, _IndexType sizeL)
    : data_(data), sizeR_(sizeR), sizeC_(sizeC), sizeL_(sizeL), ptr_(data) {}

/*!
 *@brief Creates a matrix view from the given one but with different access
 * parameters.
 * @param opM Matrix view.
 * @param sizeR Number of rows.
 * @param sizeC Number of columns.
 * @param sizeL Size of the leading dimension.
 * @param disp Displacement from the start.
 */
template <class _value_t, class _container_t, typename _IndexType,
          typename layout, bool has_inc>
SYCL_BLAS_INLINE
MatrixView<_value_t, _container_t, _IndexType, layout, has_inc>::MatrixView(
    MatrixView<_value_t, _container_t, _IndexType, layout, has_inc> opM,
    _IndexType sizeR, _IndexType sizeC, _IndexType sizeL)
    : data_(opM.get_data()),
      sizeR_(sizeR),
      sizeC_(sizeC),
      sizeL_(sizeL),
      ptr_(opM.get_data()) {}

/*!
 * @brief Returns the container
 */
template <class _value_t, class _container_t, typename _IndexType,
          typename layout, bool has_inc>
SYCL_BLAS_INLINE _container_t
MatrixView<_value_t, _container_t, _IndexType, layout, has_inc>::get_data() {
  return data_;
}

/*!
 * @brief Returns the size of the view.
 */
template <class _value_t, class _container_t, typename _IndexType,
          typename layout, bool has_inc>
SYCL_BLAS_INLINE _IndexType
MatrixView<_value_t, _container_t, _IndexType, layout, has_inc>::get_size() const {
  return sizeR_ * sizeC_;
}

/*!
 * @brief Returns a pointer to the container
 */
template <class _value_t, class _container_t, typename _IndexType,
          typename layout, bool has_inc>
SYCL_BLAS_INLINE _value_t*
MatrixView<_value_t, _container_t, _IndexType, layout, has_inc>::get_pointer() const {
  return ptr_;
}

/*! get_size_row.
 * @brief Return the number of columns.
 * @bug This value should change depending on the access mode, but
 * is currently set to Rows.
 */
template <class _value_t, class _container_t, typename _IndexType,
          typename layout, bool has_inc>
SYCL_BLAS_INLINE _IndexType
MatrixView<_value_t, _container_t, _IndexType, layout, has_inc>::get_size_row() const {
  return sizeR_;
}

/*! get_size_col.
 * @brief Return the number of columns.
 * @bug This value should change depending on the access mode, but
 * is currently set to Rows.
 */
template <class _value_t, class _container_t, typename _IndexType,
          typename layout, bool has_inc>
SYCL_BLAS_INLINE _IndexType
MatrixView<_value_t, _container_t, _IndexType, layout, has_inc>::get_size_col() const {
  return sizeC_;
}

/*! getSizeL.
 * @brief Return the leading dimension.
 */
template <class _value_t, class _container_t, typename _IndexType,
          typename layout, bool has_inc>
SYCL_BLAS_INLINE const _IndexType
MatrixView<_value_t, _container_t, _IndexType, layout, has_inc>::getSizeL() const {
  return sizeL_;
}

/*! adjust_access_displacement.
 * @brief adjust pointer offset
 * The user is responsible to adjust pointer offset for USM.
 */
template <class _value_t, class _container_t, typename _IndexType,
          typename layout, bool has_inc>
SYCL_BLAS_INLINE void MatrixView<_value_t, _container_t, _IndexType,
                                 layout, has_inc>::adjust_access_displacement() {
}

}  // namespace blas

#endif  // VIEW_HPP
