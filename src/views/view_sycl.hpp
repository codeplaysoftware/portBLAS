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

template <typename _ScalarT, typename _IndexType, typename _IncrementType>
struct vector_view<
    _ScalarT,
    typename BLAS_SYCL_Policy::template placeholder_accessor_t<_ScalarT>,
    _IndexType, _IncrementType>;
/*!
 * @brief View of a vector with an accessor.
 * @tparam ScalarT Value type of accessor.
 */
template <typename _ScalarT, typename _IndexType, typename _IncrementType>
struct vector_view<
    _ScalarT,
    typename BLAS_SYCL_Policy::template placeholder_accessor_t<_ScalarT>,
    _IndexType, _IncrementType> {
  using ScalarT = _ScalarT;
  using IndexType = _IndexType;
  using IncrementType = _IncrementType;
  using ContainerT =
      typename BLAS_SYCL_Policy::template placeholder_accessor_t<ScalarT>;
  using Self = vector_view<ScalarT, ContainerT, IndexType, IncrementType>;
  ContainerT data_;
  IndexType size_data_;
  IndexType size_;
  IndexType disp_;
  IncrementType strd_;  // never size_t, because it could negative

  using value_type = ScalarT;

  /*!
   * @brief See vector_view.
   */
  sycl_blas_inline vector_view(ContainerT data)
      : data_{data},
        size_data_(data_.get_size()),
        size_(data_.get_size()),
        disp_(0),
        strd_(1) {}

  /*!
   * @brief See vector_view.
   */
  sycl_blas_inline vector_view(ContainerT data, IndexType disp)
      : data_{data},
        size_data_(data_.get_size()),
        size_(data_.get_size()),
        disp_(disp),
        strd_(1) {}

  sycl_blas_inline vector_view(buffer_iterator<ScalarT, BLAS_SYCL_Policy> data,
                               IncrementType strd, IndexType size)
      : vector_view(get_range_accessor(data), data.get_offset(), strd, size) {}
  /*!
   * @brief See vector_view.
   */
  sycl_blas_inline vector_view(ContainerT data, IndexType disp,
                               IncrementType strd, IndexType size)
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
   * @brief See vector_view.
   */
  sycl_blas_inline vector_view(Self &opV, IndexType disp, IncrementType strd,
                               IndexType size)
      : data_{opV.getData()},
        size_data_(opV.getData().get_size()),
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
   * @brief See vector_view.
   */
  sycl_blas_inline ContainerT &getData() { return data_; }

  /*!
   * @brief See vector_view.
   */
  sycl_blas_inline IndexType getDataSize() const { return size_data_; }

  /*!
   * @brief See vector_view.
   */
  sycl_blas_inline IndexType getSize() const { return size_; }

  /*!
   * @brief See vector_view.
   */
  sycl_blas_inline IndexType getDisp() const { return disp_; }

  /*!
   * @brief See vector_view.
   */
  sycl_blas_inline IncrementType getStrd() const { return strd_; }

  /*!
   * @brief See vector_view.
   */
  sycl_blas_inline Self operator+(IndexType disp) {
    if (this->strd_ > 0)
      return Self(this->data_, this->disp_ + (disp * this->strd_), this->strd_,
                  this->size_ - disp);
    else
      return Self(this->data_,
                  this->disp_ - ((this->size_ - 1) - disp) * this->strd_,
                  this->strd_, this->size_ - disp);
  }

  /*!
   * @brief See vector_view.
   */
  sycl_blas_inline Self operator()(IndexType disp) {
    if (this->strd_ > 0)
      return Self(this->data_, this->disp_ + (disp * this->strd_), this->strd_,
                  this->size_ - disp);
    else
      return Self(this->data_,
                  this->disp_ - ((this->size_ - 1) - disp) * this->strd_,
                  this->strd_, this->size_ - disp);
  }

  /*!
   * @brief See vector_view.
   */
  sycl_blas_inline Self operator*(long strd) {
    return Self(this->data_, this->disp_, this->strd_ * strd);
  }

  /*!
   * @brief See vector_view.
   */
  sycl_blas_inline Self operator%(IndexType size) {
    if (this->strd_ > 0) {
      return Self(this->data_, this->disp_, this->strd_, size);
    } else {
      return Self(this->data_, this->disp_ - (this->size_ - 1) * this->strd_,
                  this->strd_, size);
    }
  }

  /**** EVALUATING ****/
  sycl_blas_inline ScalarT &eval(IndexType i) {
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

  sycl_blas_inline ScalarT &eval(cl::sycl::nd_item<1> ndItem) {
    return eval(ndItem.get_global_id(0));
  }

  /**** PRINTING ****/
  template <class X, class Y, typename IndxT, typename IncrT>
  friend std::ostream &operator<<(std::ostream &stream,
                                  vector_view<X, Y, IndxT, IncrT> opvS);

  void printH(const char *name) {
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

  sycl_blas_inline void bind(cl::sycl::handler &h) { h.require(data_); }
};
template <class _ScalarT, typename _IndexType>
struct matrix_view<
    _ScalarT,
    typename BLAS_SYCL_Policy::template placeholder_accessor_t<_ScalarT>,
    _IndexType>;
/*!
 * @brief Specialization of an matrix_view with an accessor.
 */
template <class _ScalarT, typename _IndexType>
struct matrix_view<
    _ScalarT,
    typename BLAS_SYCL_Policy::template placeholder_accessor_t<_ScalarT>,
    _IndexType> {
  using ScalarT = _ScalarT;
  using IndexType = _IndexType;
  using ContainerT =
      typename BLAS_SYCL_Policy::template placeholder_accessor_t<ScalarT>;
  using Self = matrix_view<ScalarT, ContainerT, IndexType>;
  // Information related to the data
  ContainerT data_;
  // int accessDev_;  // row-major or column-major value for the device/language

  Access deviceAccess_;
  Access operationAccess_;
  Access overallAccess_;

  IndexType size_data_;  // real size of the data
  // Information related to the operation
  // int accessOpr_;    // row-major or column-major
  IndexType sizeR_;  // number of rows
  IndexType sizeC_;  // number of columns
  IndexType sizeL_;  // size of the leading dimension
  IndexType disp_;   // displacementt od the first element
  // UPLO, BAND(KU,KL), PACKED, SIDE ARE ONLY REQUIRED
  using value_type = ScalarT;

  /**** CONSTRUCTORS ****/

  sycl_blas_inline matrix_view(ContainerT data, Access accessDev,
                               IndexType sizeR, IndexType sizeC)
      : data_{data},
        deviceAccess_(accessDev),
        operationAccess_(Access::RowMajor()),
        overallAccess_(deviceAccess_, operationAccess_),
        size_data_(data_.get_size()),
        sizeR_(sizeR),
        sizeC_(sizeC),
        sizeL_(0),
        disp_(0) {
    sizeL_ = overallAccess_.isRowMajor() ? sizeC_ : sizeR_;
  }

  sycl_blas_inline matrix_view(ContainerT data, IndexType sizeR,
                               IndexType sizeC)
      : data_{data},
        deviceAccess_(Access::ColMajor()),
        operationAccess_(Access::RowMajor()),
        overallAccess_(deviceAccess_, operationAccess_),
        size_data_(data_.get_size()),
        sizeR_(sizeR),
        sizeC_(sizeC),
        sizeL_(0),
        disp_(0) {
    sizeL_ =
        Access(deviceAccess_, operationAccess_).isRowMajor() ? sizeC_ : sizeR_;
  }

  sycl_blas_inline matrix_view(ContainerT data, Access accessDev,
                               IndexType sizeR, IndexType sizeC,
                               Access accessOpr, IndexType sizeL,
                               IndexType disp)
      : data_{data},
        deviceAccess_(accessDev),
        operationAccess_(accessOpr),
        overallAccess_(deviceAccess_, operationAccess_),
        size_data_(data_.get_size()),
        sizeR_(sizeR),
        sizeC_(sizeC),
        sizeL_(sizeL),
        disp_(disp) {}

  sycl_blas_inline matrix_view(ContainerT data, IndexType sizeR,
                               IndexType sizeC, Access accessOpr,
                               IndexType sizeL, IndexType disp)
      : data_{data},
        deviceAccess_(Access::ColMajor()),
        operationAccess_(accessOpr),
        overallAccess_(deviceAccess_, operationAccess_),
        size_data_(data_.get_size()),
        sizeR_(sizeR),
        sizeC_(sizeC),
        sizeL_(sizeL),
        disp_(disp) {}

  sycl_blas_inline matrix_view(buffer_iterator<ScalarT, BLAS_SYCL_Policy> data,
                               IndexType sizeR, IndexType sizeC,
                               Access accessOpr, IndexType sizeL)
      : matrix_view(get_range_accessor(data), sizeR, sizeC, accessOpr, sizeL,
                    data.get_offset()) {}

  sycl_blas_inline matrix_view(Self opM, Access accessDev, IndexType sizeR,
                               IndexType sizeC, Access accessOpr,
                               IndexType sizeL, IndexType disp)
      : data_{opM.data_},
        deviceAccess_(accessDev),
        operationAccess_(opM.operationAccess_, accessOpr),
        overallAccess_(deviceAccess_, operationAccess_),
        size_data_(opM.size_data_),
        sizeR_(sizeR),
        sizeC_(sizeC),
        sizeL_(sizeL),
        disp_(disp) {}

  sycl_blas_inline matrix_view(Self opM, IndexType sizeR, IndexType sizeC,
                               Access accessOpr, IndexType sizeL,
                               IndexType disp)
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
  sycl_blas_inline ContainerT &getData() { return data_; }

  sycl_blas_inline IndexType getDataSize() const { return size_data_; }

  sycl_blas_inline IndexType getSize() const { return sizeR_ * sizeC_; }

  sycl_blas_inline IndexType getSizeL() const { return sizeL_; }

  sycl_blas_inline IndexType getSizeR() const { return sizeR_; }

  sycl_blas_inline IndexType getSizeC() const { return sizeC_; }

  sycl_blas_inline bool is_row_access() const {
    return overallAccess_.isRowMajor();
  }

  sycl_blas_inline Access getAccessDev() const { return deviceAccess_; }

  sycl_blas_inline Access getAccessOpr() const { return operationAccess_; }

  sycl_blas_inline IndexType getDisp() const { return disp_; }

  /**** OPERATORS ****/
  sycl_blas_inline matrix_view<ScalarT, ContainerT, _IndexType> operator+(
      IndexType disp) {
    return matrix_view<ScalarT, ContainerT, _IndexType>(
        this->data_, this->accessDev_, this->sizeR_, this->sizeC_,
        this->accessOpr_, this->sizeL_, this->disp_ + disp);
  }

  sycl_blas_inline matrix_view<ScalarT, ContainerT, _IndexType> operator()(
      IndexType i, IndexType j) {
    if (overallAccess_.isRowMajor()) {
      // ACCESING BY ROWS
      return matrix_view<ScalarT, ContainerT, _IndexType>(
          this->data_, this->accessDev_, this->sizeR_, this->sizeC_,
          this->accessOpr_, this->sizeL_, this->disp_ + i * this->sizeL_ + j);
    } else {
      // ACCESING BY COLUMNS
      return matrix_view<ScalarT, ContainerT, _IndexType>(
          this->data_, this->accessDev_, this->sizeR_, this->sizeC_,
          this->accessOpr_, this->sizeL_, this->disp_ + i + this->sizeL_ * j);
    }
  }

  /**** EVALUATING ***/
  sycl_blas_inline ScalarT &eval(IndexType k) {
    bool access = overallAccess_.isRowMajor();
    auto size = (access) ? sizeC_ : sizeR_;
    auto i = (access) ? (k / size) : (k % size);
    auto j = (access) ? (k % size) : (k / size);

    return eval(i, j);
  }

  sycl_blas_inline ScalarT &eval(IndexType i, IndexType j) {
    auto ind = disp_;

    if (overallAccess_.isRowMajor()) {
      ind += (sizeL_ * i) + j;
    } else {
      ind += (sizeL_ * j) + i;
    }
    return data_[ind + disp_];
  }

  sycl_blas_inline ScalarT &eval(cl::sycl::nd_item<1> ndItem) {
    return eval(ndItem.get_global_id(0));
  }

  sycl_blas_inline void bind(cl::sycl::handler &h) { h.require(data_); }
};

}  // namespace blas

#endif  // VIEW_SYCL_HPP
