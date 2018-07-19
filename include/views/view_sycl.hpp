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

#ifndef VIEW_SYCL_HPP
#define VIEW_SYCL_HPP

#include <CL/sycl.hpp>

#include <queue/helper.hpp>
#include <queue/sycl_iterator.hpp>
#include <types/sycl_types.hpp>
#include <views/operview_base.hpp>

namespace blas {

template <typename Executor, typename T>
struct ViewTypeTrace<Executor, buffer_iterator<T>> {
  using VectorView = vector_view<T, typename Executor::template ContainerT<T>>;
  using MatrixView = matrix_view<T, typename Executor::template ContainerT<T>>;
};

template <typename ScalarT, int dim = 1,
          typename Allocator = cl::sycl::default_allocator<uint8_t>>
using bufferT = cl::sycl::buffer<ScalarT, dim, Allocator>;

template <typename ContainerT>
struct get_size_struct {
  static inline auto get_size(ContainerT &c) -> decltype(c.size()) {
    return c.size();
  }
};

template <typename ScalarT, int dim, typename Allocator>
struct get_size_struct<bufferT<ScalarT, dim, Allocator>> {
  static inline auto get_size(bufferT<ScalarT> &b) -> decltype(b.get_size()) {
    return b.get_size();
  }
};

template <typename ContainerT>
auto get_size(ContainerT &c)
    -> decltype(get_size_struct<ContainerT>::get_size(c)) {
  return get_size_struct<ContainerT>::get_size(c);
}

template <typename ScalarT, int dim = 1,
          typename Allocator = cl::sycl::default_allocator<ScalarT>>
using BufferVectorView = vector_view<ScalarT, bufferT<ScalarT, dim, Allocator>>;

template <typename ScalarT>
using BufferMatrixView = matrix_view<ScalarT, bufferT<ScalarT>>;

/*!
 * @brief Alias to a read_write host accessor.
 */
template <
    typename ScalarT,
    cl::sycl::access::mode AcM = cl::sycl::access::mode::read_write,
    cl::sycl::access::target AcT = cl::sycl::access::target::global_buffer,
    cl::sycl::access::placeholder AcP = cl::sycl::access::placeholder::true_t>
using PaccessorT = cl::sycl::accessor<ScalarT, 1, AcM, AcT, AcP>;

/*!
 * @brief View of a vector with an accessor.
 * @tparam ScalarT Value type of accessor.
 */
template <typename ScalarT_, typename IndexType_, typename IncrementType_>
struct vector_view<ScalarT_, PaccessorT<ScalarT_>, IndexType_, IncrementType_> {
  using ScalarT = ScalarT_;
  using IndexType = IndexType_;
  using IncrementType = IncrementType_;
  using ContainerT = PaccessorT<ScalarT>;
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
  vector_view(ContainerT data)
      : data_{data},
        size_data_(data_.get_size()),
        size_(data_.get_size()),
        disp_(0),
        strd_(1) {}

  /*!
   * @brief See vector_view.
   */
  vector_view(ContainerT data, IndexType disp)
      : data_{data},
        size_data_(data_.get_size()),
        size_(data_.get_size()),
        disp_(disp),
        strd_(1) {}

  vector_view(buffer_iterator<ScalarT> data, IncrementType strd, IndexType size)
      : vector_view(get_range_accessor(data), data.get_offset(), strd, size) {}
  /*!
   * @brief See vector_view.
   */
  vector_view(ContainerT data, IndexType disp, IncrementType strd,
              IndexType size)
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
  vector_view(Self &opV, IndexType disp, IncrementType strd, IndexType size)
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
  ContainerT &getData() { return data_; }

  /*!
   * @brief See vector_view.
   */
  IndexType getDataSize() { return size_data_; }

  /*!
   * @brief See vector_view.
   */
  IndexType getSize() { return size_; }

  /*!
   * @brief See vector_view.
   */
  IndexType getDisp() { return disp_; }

  /*!
   * @brief See vector_view.
   */
  IncrementType getStrd() { return strd_; }

  /*!
   * @brief See vector_view.
   */
  Self operator+(IndexType disp) {
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
  Self operator()(IndexType disp) {
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
  Self operator*(long strd) {
    return Self(this->data_, this->disp_, this->strd_ * strd);
  }

  /*!
   * @brief See vector_view.
   */
  Self operator%(IndexType size) {
    if (this->strd_ > 0) {
      return Self(this->data_, this->disp_, this->strd_, size);
    } else {
      return Self(this->data_, this->disp_ - (this->size_ - 1) * this->strd_,
                  this->strd_, size);
    }
  }

  /**** EVALUATING ****/
  ScalarT &eval(IndexType i) {
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
      //    printf("(E) ind = %ld , size_data_ = %ld \n", ind, size_data_);
      // out of range access
      //      throw std::invalid_argument("Out of range access");
    }
#endif  //__SYCL_DEVICE_ONLY__
    return data_[ind];
  }

  ScalarT &eval(cl::sycl::nd_item<1> ndItem) {
    return eval(ndItem.get_global_id(0));
  }

  /**** PRINTING ****/
  template <class X, class Y>
  friend std::ostream &operator<<(std::ostream &stream, vector_view<X, Y> opvS);

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

  void bind(cl::sycl::handler &h) { h.require(data_); }
};

/*!
 * @brief Specialization of an matrix_view with an accessor.
 */
template <class ScalarT_, typename IndexType_>
struct matrix_view<ScalarT_, PaccessorT<ScalarT_>, IndexType_> {
  using ScalarT = ScalarT_;
  using IndexType = IndexType_;
  using ContainerT = PaccessorT<ScalarT>;
  using Self = matrix_view<ScalarT, ContainerT, IndexType>;
  // Information related to the data
  ContainerT data_;
  int accessDev_;  // row-major or column-major value for the device/language
  IndexType size_data_;  // real size of the data
  // Information related to the operation
  int accessOpr_;    // row-major or column-major
  IndexType sizeR_;  // number of rows
  IndexType sizeC_;  // number of columns
  IndexType sizeL_;  // size of the leading dimension
  IndexType disp_;   // displacementt od the first element
  // UPLO, BAND(KU,KL), PACKED, SIDE ARE ONLY REQUIRED
  using value_type = ScalarT;

  /**** CONSTRUCTORS ****/

  matrix_view(ContainerT data, int accessDev, IndexType sizeR, IndexType sizeC)
      : data_{data},
        accessDev_(accessDev),
        size_data_(data_.get_size()),
        accessOpr_(1),
        sizeR_(sizeR),
        sizeC_(sizeC),
        sizeL_(0),
        disp_(0) {
    sizeL_ = (!(accessDev_ ^ accessOpr_)) ? sizeC_ : sizeR_;
  }

  matrix_view(ContainerT data, IndexType sizeR, IndexType sizeC)
      : data_{data},
        accessDev_(0),
        size_data_(data_.get_size()),
        accessOpr_(1),
        sizeR_(sizeR),
        sizeC_(sizeC),
        sizeL_(0),
        disp_(0) {
    sizeL_ = (!(accessDev_ ^ accessOpr_)) ? sizeC_ : sizeR_;
  }

  matrix_view(ContainerT data, int accessDev, IndexType sizeR, IndexType sizeC,
              int accessOpr, IndexType sizeL, IndexType disp)
      : data_{data},
        accessDev_(accessDev),
        size_data_(data_.get_size()),
        accessOpr_(accessOpr),
        sizeR_(sizeR),
        sizeC_(sizeC),
        sizeL_(sizeL),
        disp_(disp) {}

  matrix_view(ContainerT data, IndexType sizeR, IndexType sizeC, int accessOpr,
              IndexType sizeL, IndexType disp)
      : data_{data},
        accessDev_(0),
        size_data_(data_.get_size()),
        accessOpr_(accessOpr),
        sizeR_(sizeR),
        sizeC_(sizeC),
        sizeL_(sizeL),
        disp_(disp) {}

  matrix_view(buffer_iterator<ScalarT> data, IndexType sizeR, IndexType sizeC,
              int accessOpr, IndexType sizeL)
      : matrix_view(get_range_accessor(data), sizeR, sizeC, accessOpr, sizeL,
                    data.get_offset()) {}

  matrix_view(Self opM, int accessDev, IndexType sizeR, IndexType sizeC,
              int accessOpr, IndexType sizeL, IndexType disp)
      : data_{opM.data_},
        accessDev_(accessDev),
        size_data_(opM.size_data_),
        accessOpr_(!(opM.accessOpr_ ^ accessOpr)),
        sizeR_(sizeR),
        sizeC_(sizeC),
        sizeL_(sizeL),
        disp_(disp) {}

  matrix_view(Self opM, IndexType sizeR, IndexType sizeC, int accessOpr,
              IndexType sizeL, IndexType disp)
      : data_{opM.data_},
        accessDev_(opM.accessDev_),
        size_data_(opM.size_data_),
        accessOpr_(!(opM.accessOpr_ ^ accessOpr)),
        sizeR_(sizeR),
        sizeC_(sizeC),
        sizeL_(sizeL),
        disp_(disp) {}

  /**** RETRIEVING DATA ****/
  inline ContainerT &getData() { return data_; }

  inline IndexType getDataSize() { return size_data_; }

  inline IndexType getSize() { return sizeR_ * sizeC_; }

  inline IndexType getSizeL() { return sizeL_; }

  inline IndexType getSizeR() { return sizeR_; }

  inline IndexType getSizeC() { return sizeC_; }

  inline int getAccess() { return !(accessDev_ ^ accessOpr_); }

  inline int getAccessDev() { return accessDev_; }

  inline int getAccessOpr() { return accessOpr_; }

  inline long getDisp() { return disp_; }

  /**** OPERATORS ****/
  matrix_view<ScalarT, ContainerT> operator+(IndexType disp) {
    return matrix_view<ScalarT, ContainerT>(
        this->data_, this->accessDev_, this->sizeR_, this->sizeC_,
        this->accessOpr_, this->sizeL_, this->disp_ + disp);
  }

  matrix_view<ScalarT, ContainerT> operator()(IndexType i, IndexType j) {
    if (!(accessDev_ ^ accessOpr_)) {
      // ACCESING BY ROWS
      return matrix_view<ScalarT, ContainerT>(
          this->data_, this->accessDev_, this->sizeR_, this->sizeC_,
          this->accessOpr_, this->sizeL_, this->disp_ + i * this->sizeL_ + j);
    } else {
      // ACCESING BY COLUMNS
      return matrix_view<ScalarT, ContainerT>(
          this->data_, this->accessDev_, this->sizeR_, this->sizeC_,
          this->accessOpr_, this->sizeL_, this->disp_ + i + this->sizeL_ * j);
    }
  }

  /**** EVALUATING ***/
  inline ScalarT &eval(IndexType k) {
    int access = (!(accessDev_ ^ accessOpr_));
    auto size = (access) ? sizeC_ : sizeR_;
    auto i = (access) ? (k / size) : (k % size);
    auto j = (access) ? (k % size) : (k / size);

    return eval(i, j);
  }

  inline ScalarT &eval(IndexType i, IndexType j) {  // -> decltype(data_[i]) {
    auto ind = disp_;
    int accessMode = !(accessDev_ ^ accessOpr_);

    if (accessMode) {
      ind += (sizeL_ * i) + j;
    } else {
      ind += (sizeL_ * j) + i;
    }
    return data_[ind + disp_];
  }

  inline ScalarT &eval(cl::sycl::nd_item<1> ndItem) {
    return eval(ndItem.get_global_id(0));
  }

  void bind(cl::sycl::handler &h) { h.require(data_); }
};

}  // namespace blas

#endif  // VIEW_SYCL_HPP
