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

#include "blas_meta.h"
#include "container/sycl_iterator.h"
#include "types/access_types.h"
#include "views/view.h"

namespace blas {

template <typename ViewScalarT, typename view_index_t,
          typename view_increment_t>
struct VectorView<
    ViewScalarT,
    typename codeplay_policy::template placeholder_accessor_t<ViewScalarT>,
    view_index_t, view_increment_t>;
/*!
 * @brief View of a vector with an accessor.
 * @tparam scalar_t Value type of accessor.
 */
template <typename ViewScalarT, typename view_index_t,
          typename view_increment_t>
struct VectorView<
    ViewScalarT,
    typename codeplay_policy::template placeholder_accessor_t<ViewScalarT>,
    view_index_t, view_increment_t> {
  using scalar_t = ViewScalarT;
  using index_t = view_index_t;
  using increment_t = view_increment_t;
  using container_t =
      typename codeplay_policy::template placeholder_accessor_t<scalar_t>;
  using self_t = VectorView<scalar_t, container_t, index_t, increment_t>;
  container_t data_;
  const index_t size_;
  const index_t disp_;
  const increment_t strd_;  // never size_t, because it could be negative
  cl::sycl::global_ptr<scalar_t>
      ptr_;  // global pointer access inside the kernel
  using value_t = scalar_t;

  static SYCL_BLAS_INLINE const index_t calc_size(container_t &data,
                                                  index_t disp,
                                                  increment_t strd,
                                                  index_t size) noexcept {
    const index_t sz = (strd > 0)
                           ? (((data.get_size() - disp) + strd - 1) / strd)
                           : ((disp + (-strd)) / (-strd));
    return (size < sz) ? size : sz;
  }

  /*!
   * @brief See VectorView.
   */
  SYCL_BLAS_INLINE VectorView(container_t data, index_t disp, increment_t strd,
                              index_t size)
      : data_{data},
        size_(calc_size(data, disp, strd, size)),
        disp_((strd > 0) ? disp : disp + (size_ - 1) * (-strd)),
        strd_(strd) {}

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
  SYCL_BLAS_INLINE VectorView(BufferIterator<scalar_t, codeplay_policy> data,
                              increment_t strd, index_t size)
      : VectorView(get_range_accessor(data), data.get_offset(), strd, size) {}

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
  SYCL_BLAS_INLINE increment_t get_stride() const { return strd_; }

  SYCL_BLAS_INLINE scalar_t eval(index_t i) const {
    return *(ptr_ + i * strd_);
  }
  /**** EVALUATING ****/
  SYCL_BLAS_INLINE scalar_t &eval(index_t i) { return *(ptr_ + i * strd_); }

  SYCL_BLAS_INLINE scalar_t &eval(cl::sycl::nd_item<1> ndItem) {
    return eval(ndItem.get_global_id(0));
  }

  SYCL_BLAS_INLINE scalar_t eval(cl::sycl::nd_item<1> ndItem) const {
    return eval(ndItem.get_global_id(0));
  }

  SYCL_BLAS_INLINE void bind(cl::sycl::handler &h) { h.require(data_); }
  SYCL_BLAS_INLINE void set_access_displacement() {
    ptr_ = data_.get_pointer() + disp_;
  }
};

template <class ViewScalarT, typename view_index_t, typename layout>
struct MatrixView<
    ViewScalarT,
    typename codeplay_policy::template placeholder_accessor_t<ViewScalarT>,
    view_index_t, layout>;
/*!
 * @brief Specialization of an MatrixView with an accessor.
 */
template <class ViewScalarT, typename view_index_t, typename layout>
struct MatrixView<
    ViewScalarT,
    typename codeplay_policy::template placeholder_accessor_t<ViewScalarT>,
    view_index_t, layout> {
  using access_layout_t = layout;
  using scalar_t = ViewScalarT;
  using index_t = view_index_t;
  using container_t =
      typename codeplay_policy::template placeholder_accessor_t<scalar_t>;
  using self_t = MatrixView<scalar_t, container_t, index_t, layout>;

  using value_t = scalar_t;
  // Information related to the data
  container_t data_;
  // Information related to the operation
  const index_t sizeR_;  // number of rows
  const index_t sizeC_;  // number of columns
  const index_t sizeL_;  // size of the leading dimension
  const index_t disp_;   // displacementt od the first element
  cl::sycl::global_ptr<scalar_t>
      ptr_;  // global pointer access inside the kernel

  /**** CONSTRUCTORS ****/
  SYCL_BLAS_INLINE MatrixView(container_t data, index_t sizeR, index_t sizeC,
                              index_t sizeL, index_t disp)
      : data_{data}, sizeR_(sizeR), sizeC_(sizeC), sizeL_(sizeL), disp_(disp) {}

  SYCL_BLAS_INLINE MatrixView(container_t data, index_t sizeR, index_t sizeC)
      : MatrixView(data, sizeR, sizeC,
                   (layout::is_col_major() ? sizeR_ : sizeC_), 0) {}

  SYCL_BLAS_INLINE MatrixView(BufferIterator<scalar_t, codeplay_policy> data,
                              index_t sizeR, index_t sizeC, index_t sizeL)
      : MatrixView(get_range_accessor(data), sizeR, sizeC, sizeL,
                   data.get_offset()) {}

  SYCL_BLAS_INLINE MatrixView(self_t opM, index_t sizeR, index_t sizeC,
                              index_t sizeL, index_t disp)
      : data_{opM.data_},
        sizeR_(sizeR),
        sizeC_(sizeC),
        sizeL_(sizeL),
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
  SYCL_BLAS_INLINE scalar_t &eval(index_t ind) { return data_[ind]; }

  SYCL_BLAS_INLINE const scalar_t eval(index_t indx) const noexcept {
    return *(ptr_ + indx);
  }

  SYCL_BLAS_INLINE scalar_t &eval(index_t i, index_t j) {
    return ((layout::is_col_major()) ? *(ptr_ + i + sizeL_ * j)
                                     : *(ptr_ + j + sizeL_ * i));
  }

  SYCL_BLAS_INLINE scalar_t &eval(cl::sycl::nd_item<1> ndItem) {
    return eval(ndItem.get_global_id(0));
  }

  SYCL_BLAS_INLINE const scalar_t eval(cl::sycl::nd_item<1> ndItem) const
      noexcept {
    return eval(ndItem.get_global_id(0));
  }

  SYCL_BLAS_INLINE void bind(cl::sycl::handler &h) { h.require(data_); }

  SYCL_BLAS_INLINE void set_access_displacement() {
    ptr_ = data_.get_pointer() + disp_;
  }
};  // namespace blas

}  // namespace blas

#endif  // VIEW_SYCL_HPP
