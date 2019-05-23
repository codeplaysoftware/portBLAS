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

#include "blas_meta.h"
#include "container/sycl_iterator.h"
#include "types/access_types.h"
#include "views/view.h"
#include <CL/sycl.hpp>

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
  index_t size_data_;
  index_t size_;
  index_t disp_;
  increment_t strd_;  // never size_t, because it could negative

  using value_t = scalar_t;

  /*!
   * @brief See VectorView.
   */
  SYCL_BLAS_INLINE VectorView(container_t data)
      : data_{data},
        size_data_(data_.get_size()),
        size_(data_.get_size()),
        disp_(0),
        strd_(1) {}

  /*!
   * @brief See VectorView.
   */
  SYCL_BLAS_INLINE VectorView(container_t data, index_t disp)
      : data_{data},
        size_data_(data_.get_size()),
        size_(data_.get_size()),
        disp_(disp),
        strd_(1) {}

  SYCL_BLAS_INLINE VectorView(BufferIterator<scalar_t, codeplay_policy> data,
                              increment_t strd, index_t size)
      : VectorView(get_range_accessor(data), data.get_offset(), strd, size) {}
  /*!
   * @brief See VectorView.
   */
  SYCL_BLAS_INLINE VectorView(container_t data, index_t disp, increment_t strd,
                              index_t size)
      : data_{data},
        size_data_(data_.get_size()),
        size_(0),
        disp_(disp),
        strd_(strd) {
    if (strd_ > 0) {
      auto sizeV = (size_data_ - disp);
      auto quot = (sizeV + strd - 1) / strd;  // ceiling
      size_ = quot;
    } else if (strd_ < 0) {
      auto nstrd = -strd;
      auto quot = (disp + nstrd) / nstrd;  // ceiling
      size_ = quot;
#ifndef __SYCL_DEVICE_ONLY__
    } else {
// Stride is zero, not valid!
#ifdef VERBOSE
      printf("std = 0 \n");
#endif  // VERBOSE
      throw std::invalid_argument("Cannot create view with 0 stride");
#endif  //__SYCL_DEVICE_ONLY__
    }
    if (size < size_) size_ = size;
    if (strd_ < 0) disp_ += (size_ - 1) * strd_;
  }

  /*!
   * @brief See VectorView.
   */
  SYCL_BLAS_INLINE VectorView(self_t &opV, index_t disp, increment_t strd,
                              index_t size)
      : data_{opV.get_data()},
        size_data_(opV.get_data().get_size()),
        size_(0),
        disp_(disp),
        strd_(strd) {
    if (strd_ > 0) {
      auto sizeV = (size_data_ - disp);
      auto quot = (sizeV + strd - 1) / strd;  // ceiling
      size_ = quot;
    } else if (strd_ < 0) {
      auto nstrd = -strd;
      auto quot = (disp + nstrd) / nstrd;  // ceiling
      size_ = quot;
#ifndef __SYCL_DEVICE_ONLY__
    } else {
// Stride is zero, not valid!
#ifdef VERBOSE
      printf("std = 0 \n");
#endif  //  VERBOSE
      throw std::invalid_argument("Cannot create view with 0 stride");
#endif  //__SYCL_DEVICE_ONLY__
    }
    if (size < size_) size_ = size;
    if (strd_ < 0) disp_ += (size_ - 1) * strd_;
  }

  /*!
   * @brief See VectorView.
   */
  SYCL_BLAS_INLINE container_t &get_data() { return data_; }

  /*!
   * @brief See VectorView.
   */
  SYCL_BLAS_INLINE index_t get_data_size() const { return size_data_; }

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

  /*!
   * @brief See VectorView.
   */
  SYCL_BLAS_INLINE self_t operator+(index_t disp) {
    if (this->strd_ > 0)
      return self_t(this->data_, this->disp_ + (disp * this->strd_),
                    this->strd_, this->size_ - disp);
    else
      return self_t(this->data_,
                    this->disp_ - ((this->size_ - 1) - disp) * this->strd_,
                    this->strd_, this->size_ - disp);
  }

  /*!
   * @brief See VectorView.
   */
  SYCL_BLAS_INLINE self_t operator()(index_t disp) {
    if (this->strd_ > 0)
      return self_t(this->data_, this->disp_ + (disp * this->strd_),
                    this->strd_, this->size_ - disp);
    else
      return self_t(this->data_,
                    this->disp_ - ((this->size_ - 1) - disp) * this->strd_,
                    this->strd_, this->size_ - disp);
  }

  /*!
   * @brief See VectorView.
   */
  SYCL_BLAS_INLINE self_t operator*(long strd) {
    return self_t(this->data_, this->disp_, this->strd_ * strd);
  }

  /*!
   * @brief See VectorView.
   */
  SYCL_BLAS_INLINE self_t operator%(index_t size) {
    if (this->strd_ > 0) {
      return self_t(this->data_, this->disp_, this->strd_, size);
    } else {
      return self_t(this->data_, this->disp_ - (this->size_ - 1) * this->strd_,
                    this->strd_, size);
    }
  }

  /**** EVALUATING ****/
  SYCL_BLAS_INLINE scalar_t &eval(index_t i) {
    auto ind = disp_;
    if (strd_ == 1) {
      ind += i;
    } else if (strd_ > 0) {
      ind += strd_ * i;
    } else {
      ind -= strd_ * (size_ - i - 1);
    }
#ifndef __SYCL_DEVICE_ONLY__
    if (ind >= size_data_) {
      // out of range access
      throw std::invalid_argument("Out of range access");
    }
#endif  //__SYCL_DEVICE_ONLY__
    return data_[ind];
  }

  SYCL_BLAS_INLINE scalar_t &eval(cl::sycl::nd_item<1> ndItem) {
    return eval(ndItem.get_global_id(0));
  }

  /**** PRINTING ****/
  template <class X, class Y, typename IndxT, typename IncrT>
  friend std::ostream &operator<<(std::ostream &stream,
                                  VectorView<X, Y, IndxT, IncrT> opvS);

  void print_h(const char *name) {
    int frst = 1;
    printf("%s = [ ", name);
    for (size_t i = 0; i < size_; i++) {
      if (frst)
        printf("%f", eval(i));
      else
        printf(" , %f", eval(i));
      frst = 0;
    }
    printf(" ]\n");
  }

  SYCL_BLAS_INLINE void bind(cl::sycl::handler &h) { h.require(data_); }
};
template <class ViewScalarT, typename view_index_t>
struct MatrixView<
    ViewScalarT,
    typename codeplay_policy::template placeholder_accessor_t<ViewScalarT>,
    view_index_t>;
/*!
 * @brief Specialization of an MatrixView with an accessor.
 */
template <class ViewScalarT, typename view_index_t>
struct MatrixView<
    ViewScalarT,
    typename codeplay_policy::template placeholder_accessor_t<ViewScalarT>,
    view_index_t> {
  using scalar_t = ViewScalarT;
  using index_t = view_index_t;
  using container_t =
      typename codeplay_policy::template placeholder_accessor_t<scalar_t>;
  using self_t = MatrixView<scalar_t, container_t, index_t>;
  // Information related to the data
  container_t data_;
  // int accessDev_;  // row-major or column-major value for the device/language

  Access deviceAccess_;
  Access operationAccess_;
  Access overallAccess_;

  index_t size_data_;  // real size of the data
  // Information related to the operation
  // int accessOpr_;    // row-major or column-major
  index_t sizeR_;  // number of rows
  index_t sizeC_;  // number of columns
  index_t sizeL_;  // size of the leading dimension
  index_t disp_;   // displacementt od the first element
  // UPLO, BAND(KU,KL), PACKED, SIDE ARE ONLY REQUIRED
  using value_t = scalar_t;

  /**** CONSTRUCTORS ****/

  SYCL_BLAS_INLINE MatrixView(container_t data, Access accessDev, index_t sizeR,
                              index_t sizeC)
      : data_{data},
        deviceAccess_(accessDev),
        operationAccess_(Access::row_major()),
        overallAccess_(deviceAccess_, operationAccess_),
        size_data_(data_.get_size()),
        sizeR_(sizeR),
        sizeC_(sizeC),
        sizeL_(0),
        disp_(0) {
    sizeL_ = overallAccess_.is_row_major() ? sizeC_ : sizeR_;
  }

  SYCL_BLAS_INLINE MatrixView(container_t data, index_t sizeR, index_t sizeC)
      : data_{data},
        deviceAccess_(Access::col_major()),
        operationAccess_(Access::row_major()),
        overallAccess_(deviceAccess_, operationAccess_),
        size_data_(data_.get_size()),
        sizeR_(sizeR),
        sizeC_(sizeC),
        sizeL_(0),
        disp_(0) {
    sizeL_ = Access(deviceAccess_, operationAccess_).is_row_major() ? sizeC_
                                                                    : sizeR_;
  }

  SYCL_BLAS_INLINE MatrixView(container_t data, Access accessDev, index_t sizeR,
                              index_t sizeC, Access accessOpr, index_t sizeL,
                              index_t disp)
      : data_{data},
        deviceAccess_(accessDev),
        operationAccess_(accessOpr),
        overallAccess_(deviceAccess_, operationAccess_),
        size_data_(data_.get_size()),
        sizeR_(sizeR),
        sizeC_(sizeC),
        sizeL_(sizeL),
        disp_(disp) {}

  SYCL_BLAS_INLINE MatrixView(container_t data, index_t sizeR, index_t sizeC,
                              Access accessOpr, index_t sizeL, index_t disp)
      : data_{data},
        deviceAccess_(Access::col_major()),
        operationAccess_(accessOpr),
        overallAccess_(deviceAccess_, operationAccess_),
        size_data_(data_.get_size()),
        sizeR_(sizeR),
        sizeC_(sizeC),
        sizeL_(sizeL),
        disp_(disp) {}

  SYCL_BLAS_INLINE MatrixView(BufferIterator<scalar_t, codeplay_policy> data,
                              index_t sizeR, index_t sizeC, Access accessOpr,
                              index_t sizeL)
      : MatrixView(get_range_accessor(data), sizeR, sizeC, accessOpr, sizeL,
                   data.get_offset()) {}

  SYCL_BLAS_INLINE MatrixView(self_t opM, Access accessDev, index_t sizeR,
                              index_t sizeC, Access accessOpr, index_t sizeL,
                              index_t disp)
      : data_{opM.data_},
        deviceAccess_(accessDev),
        operationAccess_(opM.operationAccess_, accessOpr),
        overallAccess_(deviceAccess_, operationAccess_),
        size_data_(opM.size_data_),
        sizeR_(sizeR),
        sizeC_(sizeC),
        sizeL_(sizeL),
        disp_(disp) {}

  SYCL_BLAS_INLINE MatrixView(self_t opM, index_t sizeR, index_t sizeC,
                              Access accessOpr, index_t sizeL, index_t disp)
      : data_{opM.data_},
        deviceAccess_(opM.accessDev_),
        operationAccess_(opM.operationAccess_, accessOpr),
        overallAccess_(deviceAccess_, operationAccess_),
        size_data_(opM.size_data_),
        sizeR_(sizeR),
        sizeC_(sizeC),
        sizeL_(sizeL),
        disp_(disp) {}

  /**** RETRIEVING DATA ****/
  SYCL_BLAS_INLINE container_t &get_data() { return data_; }

  SYCL_BLAS_INLINE index_t get_data_size() const { return size_data_; }

  SYCL_BLAS_INLINE index_t get_size() const { return sizeR_ * sizeC_; }

  SYCL_BLAS_INLINE index_t getSizeL() const { return sizeL_; }

  SYCL_BLAS_INLINE index_t get_size_row() const { return sizeR_; }

  SYCL_BLAS_INLINE index_t get_size_col() const { return sizeC_; }

  SYCL_BLAS_INLINE bool is_row_access() const {
    return overallAccess_.is_row_major();
  }

  SYCL_BLAS_INLINE Access get_access_device() const { return deviceAccess_; }

  SYCL_BLAS_INLINE Access get_access_operation() const {
    return operationAccess_;
  }

  SYCL_BLAS_INLINE index_t get_access_displacement() const { return disp_; }

  /**** OPERATORS ****/
  SYCL_BLAS_INLINE MatrixView<scalar_t, container_t, view_index_t> operator+(
      index_t disp) {
    return MatrixView<scalar_t, container_t, view_index_t>(
        this->data_, this->accessDev_, this->sizeR_, this->sizeC_,
        this->accessOpr_, this->sizeL_, this->disp_ + disp);
  }

  SYCL_BLAS_INLINE MatrixView<scalar_t, container_t, view_index_t> operator()(
      index_t i, index_t j) {
    if (overallAccess_.is_row_major()) {
      // ACCESING BY ROWS
      return MatrixView<scalar_t, container_t, view_index_t>(
          this->data_, this->accessDev_, this->sizeR_, this->sizeC_,
          this->accessOpr_, this->sizeL_, this->disp_ + i * this->sizeL_ + j);
    } else {
      // ACCESING BY COLUMNS
      return MatrixView<scalar_t, container_t, view_index_t>(
          this->data_, this->accessDev_, this->sizeR_, this->sizeC_,
          this->accessOpr_, this->sizeL_, this->disp_ + i + this->sizeL_ * j);
    }
  }

  /**** EVALUATING ***/
  SYCL_BLAS_INLINE scalar_t &eval(index_t k) {
    bool access = overallAccess_.is_row_major();
    auto size = (access) ? sizeC_ : sizeR_;
    auto i = (access) ? (k / size) : (k % size);
    auto j = (access) ? (k % size) : (k / size);

    return eval(i, j);
  }

  SYCL_BLAS_INLINE scalar_t &eval(index_t i, index_t j) {
    auto ind = disp_;

    if (overallAccess_.is_row_major()) {
      ind += (sizeL_ * i) + j;
    } else {
      ind += (sizeL_ * j) + i;
    }
    return data_[ind + disp_];
  }

  SYCL_BLAS_INLINE scalar_t &eval(cl::sycl::nd_item<1> ndItem) {
    return eval(ndItem.get_global_id(0));
  }

  SYCL_BLAS_INLINE void bind(cl::sycl::handler &h) { h.require(data_); }
};

}  // namespace blas

#endif  // VIEW_SYCL_HPP
