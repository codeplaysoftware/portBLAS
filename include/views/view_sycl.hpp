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

#include <views/operview_base.hpp>

namespace blas {

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

/*! vector_view<ScalarT, bufferT<Scalar>>
 * @brief Specialization of the vector view to operate with buffers.
 * Note that the buffer class cannot be accessed on the host.
 */
template <typename ScalarT_, int dim, typename Allocator_, typename IndexType_,
          typename IncrementType_>
struct vector_view<ScalarT_, bufferT<ScalarT_, dim, Allocator_>, IndexType_,
                   IncrementType_> {
  using ScalarT = ScalarT_;
  using ContainerT = bufferT<ScalarT, dim, Allocator_>;
  using IndexType = IndexType_;
  using IncrementType = IncrementType_;
  using Self = vector_view<ScalarT, ContainerT, IndexType, IncrementType>;
  ContainerT &data_;
  IndexType size_data_;
  IndexType size_;
  IndexType disp_;
  IncrementType strd_;  // never size_t, because it could negative

  using value_type = ScalarT;

  /*! initialize.
   * Initializes the vector view with the information from.
   */
  inline void initialize(IndexType originalSize) {
    if (strd_ > 0) {
      auto sizeV = (size_data_ - disp_);
      auto quot = (sizeV + strd_ - 1) / strd_;  // ceiling
      size_ = quot;
    } else if (strd_ < 0) {
      auto nstrd = -strd_;
      auto quot = (disp_ + nstrd) / nstrd;  // ceiling
      size_ = quot;
#ifndef __SYCL_DEVICE_ONLY__
    } else {
      // Stride is zero, not valid!
      printf("std = 0 \n");
      throw std::invalid_argument("Cannot create view with 0 stride");
#endif  //__SYCL_DEVICE_ONLY__
    }
    if (originalSize < size_) size_ = originalSize;
    if (strd_ < 0) disp_ += (size_ - 1) * strd_;
  }

  /*! vector_view.
   * See vector_view.
   */
  vector_view(ContainerT &data, IndexType disp = 0, IncrementType strd = 1)
      : data_(data),
        size_data_(data_.get_size()),
        size_(data_.get_size() / sizeof(ScalarT)),
        disp_(disp),
        strd_(strd) {}

  /*! vector_view.
   * See vector_view.
   */
  vector_view(ContainerT &data, IndexType disp, IncrementType strd,
              IndexType size)
      : data_(data),
        size_data_(data_.get_size()),
        size_(0),
        disp_(disp),
        strd_(strd) {
    initialize(size);
  }

  /*! vector_view.
   * See vector_view.
   */
  vector_view(Self opV, IndexType disp, IncrementType strd, IndexType size)
      : data_(opV.getData()),
        size_data_(opV.getData().get_size()),
        size_(0),
        disp_(disp),
        strd_(strd) {
    initialize(size);
  }

  /*! vector_view.
   * See vector_view.
   */
  ContainerT &getData() { return data_; }

  /*! vector_view.
   * See vector_view.
   */
  IndexType getDataSize() { return size_data_; }

  /*! vector_view.
   * See vector_view.
   */
  IndexType getSize() { return size_; }

  /*! vector_view.
   * See vector_view.
   */
  IndexType getDisp() { return disp_; }

  /*! vector_view.
   * See vector_view.
   */
  IncrementType getStrd() { return strd_; }

  /*! vector_view.
   * See vector_view.
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

  /*! vector_view.
   * See vector_view.
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

  /*! vector_view.
   * See vector_view.
   */
  Self operator*(long strd) {
    return Self(this->data_, this->disp_, this->strd_ * strd);
  }

  /*! vector_view.
   * See vector_view.
   */
  Self operator%(IndexType size) {
    if (this->strd_ > 0) {
      return Self(this->data_, this->disp_, this->strd_, size);
    } else {
      return Self(this->data_, this->disp_ - (this->size_ - 1) * this->strd_,
                  this->strd_, size);
    }
  }

  /*! eval.
   * See vector_view::eval.
   */
  ScalarT &eval(IndexType i) {
    //  auto eval(size_t i) -> decltype(data_[i]) {
    auto ind = disp_;  // The disp has been integrated in range_accessor
    if (strd_ == 1) {
      ind += i;
    } else if (strd_ > 0) {
      ind += strd_ * i;
    } else {
      ind -= strd_ * (size_ - i - 1);
    }
#ifndef __SYCL_DEVICE_ONLY__
    if (ind >= size_data_) {
#ifdef VERBOSE
      // out of range access
      printf("(A) ind = %ld , size_data_ = %ld \n", ind, size_data_);
#endif  //  VERBOSE
      throw std::invalid_argument("Out of range access");
    }
#endif  //__SYCL_DEVICE_ONLY__
    ScalarT retVal;
    {
      auto hostPtr = data_.template get_access<cl::sycl::access::mode::read>();
      retVal = hostPtr[ind + disp_];
    }

    return retVal;
  }

  /*! eval.
   * See eval.
   */
  ScalarT &eval(cl::sycl::nd_item<1> ndItem) {
    return eval(ndItem.get_global(0));
  }

  /*! val.
   * @brief Allows printing information on the host.
   */
  ScalarT val(IndexType i) {
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
#ifdef VERBOSE
      printf("(B) ind = %ld , size_data_ = %ld \n", ind, size_data_);
#endif  //  VERBOSE
      // out of range access
      throw std::invalid_argument("Out of range access");
    }
#endif  //__SYCL_DEVICE_ONLY__
    ScalarT retVal;
    {
      auto hostPtr = data_.template get_access<cl::sycl::access::mode::read>();
      retVal = hostPtr[ind + disp_];
    }

    return retVal;
  }

  /**** PRINTING ****/
  template <class X, class Y>
  friend std::ostream &operator<<(std::ostream &stream, vector_view<X, Y> opvS);

  void printH(const char *name) {
    int frst = 1;
    printf("%s = [ ", name);
    for (size_t i = 0; i < size_; i++) {
      if (frst) {
        printf("%f", val(i));
      } else {
        printf(" , %f", val(i));
      }
      frst = 0;
    }
    printf(" ]\n");
  }
};

template <typename ScalarT>
using BufferMatrixView = matrix_view<ScalarT, bufferT<ScalarT>>;

/*! matrix_view.
 * @brief Specialization for matrix_view with a buffer, implementing
 * the evaluation function on the host as a host accessor.
 * @bug This class should share method implementation via a third one with the
 *  original specialization.
 * @tparam ScalarT Value type of the SYCL buffer.
 */
template <typename ScalarT_, int dim, typename Allocator_, typename IndexType_>
struct matrix_view<ScalarT_, bufferT<ScalarT_, dim, Allocator_>, IndexType_> {
  using ScalarT = ScalarT_;
  using IndexType = IndexType_;
  using ContainerT = bufferT<ScalarT, dim, Allocator_>;
  using Self = matrix_view<ScalarT, ContainerT, IndexType>;
  // Information related to the data
  ContainerT &data_;
  int accessDev_;  // row-major or column-major value for the device/language.
  IndexType size_data_;  // real size of the data
  // Information related to the operation
  int accessOpr_;    // row-major or column-major.
  IndexType sizeR_;  // number of rows
  IndexType sizeC_;  // number of columns
  IndexType sizeL_;  // size of the leading dimension
  IndexType disp_;   // displacementt from the first element
  // UPLO, BAND(KU,KL), PACKED, SIDE ARE ONLY REQUIRED
  using value_type = ScalarT;

  /*! matrix_view.
   * @brief See matrix_view.
   */
  matrix_view(ContainerT &data, int accessDev, IndexType sizeR, IndexType sizeC)
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

  /*! matrix_view.
   * @brief See matrix_view.
   */
  matrix_view(ContainerT &data, IndexType sizeR, IndexType sizeC)
      : data_(data),
        accessDev_(0),
        size_data_(data_.get_size()),
        accessOpr_(1),
        sizeR_(sizeR),
        sizeC_(sizeC),
        sizeL_(0),
        disp_(0) {
    sizeL_ = (!(accessDev_ ^ accessOpr_)) ? sizeC_ : sizeR_;
  }

  /*! matrix_view.
   * @brief See matrix_view.
   */
  matrix_view(ContainerT &data, int accessDev, IndexType sizeR, IndexType sizeC,
              int accessOpr, IndexType sizeL, IndexType disp)
      : data_(data),
        accessDev_(accessDev),
        size_data_(data_.get_size()),
        accessOpr_(accessOpr),
        sizeR_(sizeR),
        sizeC_(sizeC),
        sizeL_(sizeL),
        disp_(disp) {}

  /*! matrix_view.
   * @brief See matrix_view.
   */
  matrix_view(ContainerT &data, IndexType sizeR, IndexType sizeC, int accessOpr,
              IndexType sizeL, IndexType disp)
      : data_(data),
        accessDev_(0),
        size_data_(data_.get_size()),
        accessOpr_(accessOpr),
        sizeR_(sizeR),
        sizeC_(sizeC),
        sizeL_(sizeL),
        disp_(disp) {}

  /*! matrix_view.
   * @brief See matrix_view.
   */
  matrix_view(Self opM, int accessDev, IndexType sizeR, IndexType sizeC,
              int accessOpr, IndexType sizeL, IndexType disp)
      : data_(opM.data_),
        accessDev_(accessDev),
        size_data_(opM.size_data_),
        accessOpr_(!(opM.accessOpr_ ^ accessOpr)),
        sizeR_(sizeR),
        sizeC_(sizeC),
        sizeL_(sizeL),
        disp_(disp) {}

  /*! matrix_view.
   * @brief See matrix_view.
   */
  matrix_view(Self opM, IndexType sizeR, IndexType sizeC, int accessOpr,
              IndexType sizeL, IndexType disp)
      : data_(opM.data_),
        accessDev_(opM.accessDev_),
        size_data_(opM.size_data_),
        accessOpr_(!(opM.accessOpr_ ^ accessOpr)),
        sizeR_(sizeR),
        sizeC_(sizeC),
        sizeL_(sizeL),
        disp_(disp) {}

  /*!
   * @brief See matrix_view.
   */
  ContainerT &getData() { return data_; }

  /*!
   * @brief See matrix_view.
   */
  IndexType getDataSize() { return size_data_; }

  /*!
   * @brief See matrix_view.
   */
  IndexType getSize() { return sizeR_ * sizeC_; }

  /*!
   * @brief See matrix_view.
   */
  IndexType getSizeR() { return sizeR_; }

  /*!
   * @brief See matrix_view.
   */
  IndexType getSizeC() { return sizeC_; }

  inline IndexType getSizeL() { return sizeL_; }

  /*!
   * @brief See matrix_view.
   */
  int getAccess() { return !(accessDev_ ^ accessOpr_); }

  /*!
   * @brief See matrix_view.
   */
  int getAccessDev() { return accessDev_; }

  /*!
   * @brief See matrix_view.
   */
  int getAccessOpr() { return accessOpr_; }

  /*!
   * @brief See matrix_view.
   */
  IndexType getDisp() { return disp_; }

  /*!
   * @brief See matrix_view.
   */
  Self operator+(IndexType disp) {
    return matrix_view<ScalarT, ContainerT>(
        this->data_, this->accessDev_, this->sizeR_, this->sizeC_,
        this->accessOpr_, this->sizeL_, this->disp_ + disp);
  }

  /*!
   * @brief See matrix_view.
   */
  Self operator()(IndexType i, IndexType j) {
    if (!(accessDev_ ^ accessOpr_)) {
      // ACCESING BY ROWS
      return Self(this->data_, this->accessDev_, this->sizeR_, this->sizeC_,
                  this->accessOpr_, this->sizeL_,
                  this->disp_ + i * this->sizeL_ + j);
    } else {
      // ACCESING BY COLUMNS
      return Self(this->data_, this->accessDev_, this->sizeR_, this->sizeC_,
                  this->accessOpr_, this->sizeL_,
                  this->disp_ + i + this->sizeL_ * j);
    }
  }

  /*!
   * @brief See matrix_view.
   */
  ScalarT &eval(IndexType k) {  // -> decltype(data_[i]) {
    auto ind = disp_;  // The disp has been integrated in range_accessor
    int access = (!(accessDev_ ^ accessOpr_));
    auto size = (access) ? sizeC_ : sizeR_;
    auto i = (access) ? (k / size) : (k % size);
    auto j = (access) ? (k % size) : (k / size);

    return eval(i, j);
  }

  /*!
   * @brief See matrix_view.
   */
  ScalarT &eval(IndexType i, IndexType j) {
    auto ind = disp_;  // The disp has been integrated in range_accessor;

    if (!(accessDev_ ^ accessOpr_)) {
      ind += (sizeL_ * i) + j;
    } else {
      ind += (sizeL_ * j) + i;
    }
#ifndef __SYCL_DEVICE_ONLY__
    if (ind >= size_data_) {
#ifdef VERBOSE
      printf("(C) ind = %ld , size_data_ = %ld \n", ind, size_data_);
#endif  // VERBOSE
      // out of range access
      throw std::invalid_argument("Out of range access");
    }
#endif  // __SYCL_DEVICE_ONLY__
    ScalarT retVal;
    {
      // however for the host accessor it can be used as we did not use range
      // accessor here
      auto hostPtr = data_.template get_access<cl::sycl::access::mode::read>();
      retVal = hostPtr[ind + disp_];
    }

    return retVal;
  }

  ScalarT &eval(cl::sycl::nd_item<1> ndItem) {
    return eval(ndItem.get_global(0));
  }

  /*! val.
   * @brief Used to print the values on the host.
   */
  ScalarT val(IndexType i, IndexType j) {
    auto ind = disp_;  // The disp has been integrated in range_accessor

    if (!(accessDev_ ^ accessOpr_)) {
      ind += (sizeL_ * i) + j;
    } else {
      ind += (sizeL_ * j) + i;
    }
#ifndef __SYCL_DEVICE_ONLY__
    if (ind >= size_data_) {
#ifdef VERBOSE
      printf("(D) ind = %ld , size_data_ = %ld \n", ind, size_data_);
#endif  // VERBOSE
      // out of range access
      throw std::invalid_argument("Out of range access");
    }
#endif  //__SYCL_DEVICE_ONLY__
    ScalarT retVal;
    {
      auto hostPtr = data_.template get_access<cl::sycl::access::mode::read>();
      retVal = hostPtr[ind + disp_];
    }

    return retVal;
  }

  /*!
   * @brief
   */
  void printH(const char *name) {
    printf("%s = [ \n", name);
    for (size_t i = 0; i < ((accessOpr_) ? sizeR_ : sizeC_); i++) {
      int frst = 1;
      for (size_t j = 0; j < ((accessOpr_) ? sizeC_ : sizeR_); j++) {
        if (frst)
          printf("%f", val(i, j));
        else
          printf(" , %f", val(i, j));
        frst = 0;
      }
      printf(" ; \n");
    }
    printf(" ]\n");
  }
};

/*!
 * @brief Alias to a read_write host accessor.
 */
template <typename ScalarT>
using accessorT =
    cl::sycl::accessor<ScalarT, 1, cl::sycl::access::mode::read_write,
                       cl::sycl::access::target::global_buffer>;

/*!
 * @brief View of a vector with an accessor.
 * @tparam ScalarT Value type of accessor.
 */
template <typename ScalarT_, typename IndexType_, typename IncrementType_>
struct vector_view<ScalarT_, accessorT<ScalarT_>, IndexType_, IncrementType_> {
  using ScalarT = ScalarT_;
  using IndexType = IndexType_;
  using IncrementType = IncrementType_;
  using ContainerT = accessorT<ScalarT>;
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
  vector_view(ContainerT &data)
      : data_{data},
        size_data_(data_.get_size()),
        size_(data_.get_size()),
        disp_(0),
        strd_(1) {}

  /*!
   * @brief See vector_view.
   */
  vector_view(ContainerT &data, IndexType disp)
      : data_{data},
        size_data_(data_.get_size()),
        size_(data_.get_size()),
        disp_(disp),
        strd_(1) {}

  /*!
   * @brief See vector_view.
   */
  vector_view(ContainerT &data, IndexType disp, IncrementType strd,
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
    auto ind = disp_;  // The disp has been integrated in range_accessor
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
    return eval(ndItem.get_global(0));
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
};

/*!
 * @brief Specialization of an matrix_view with an accessor.
 */
template <class ScalarT_, typename IndexType_>
struct matrix_view<ScalarT_, accessorT<ScalarT_>, IndexType_> {
  using ScalarT = ScalarT_;
  using IndexType = IndexType_;
  using ContainerT = accessorT<ScalarT>;
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

  matrix_view(ContainerT &data, int accessDev, IndexType sizeR, IndexType sizeC)
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

  matrix_view(ContainerT &data, IndexType sizeR, IndexType sizeC)
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

  matrix_view(ContainerT &data, int accessDev, IndexType sizeR, IndexType sizeC,
              int accessOpr, IndexType sizeL, IndexType disp)
      : data_{data},
        accessDev_(accessDev),
        size_data_(data_.get_size()),
        accessOpr_(accessOpr),
        sizeR_(sizeR),
        sizeC_(sizeC),
        sizeL_(sizeL),
        disp_(disp) {}

  matrix_view(ContainerT &data, IndexType sizeR, IndexType sizeC, int accessOpr,
              IndexType sizeL, IndexType disp)
      : data_{data},
        accessDev_(0),
        size_data_(data_.get_size()),
        accessOpr_(accessOpr),
        sizeR_(sizeR),
        sizeC_(sizeC),
        sizeL_(sizeL),
        disp_(disp) {}

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
    auto ind = disp_;  // The disp has been integrated in range_accessor
    int accessMode = !(accessDev_ ^ accessOpr_);

    if (accessMode) {
      ind += (sizeL_ * i) + j;
    } else {
      ind += (sizeL_ * j) + i;
    }
#ifndef __SYCL_DEVICE_ONLY__
    if (ind >= size_data_) {
      printf("(G) ind = %ld , size_data_ = %ld \n", static_cast<size_t>(ind),
             static_cast<size_t>(size_data_));
    }
#endif  //__SYCL_DEVICE_ONLY__
    return data_[ind + disp_];
  }

  inline ScalarT &eval(cl::sycl::nd_item<1> ndItem) {
    return eval(ndItem.get_global(0));
  }

  /**** PRINTING ****/
  void printH(const char *name) {
    printf("%s = [ \n", name);
    for (size_t i = 0; i < ((accessOpr_) ? sizeR_ : sizeC_); i++) {
      int frst = 1;
      for (size_t j = 0; j < ((accessOpr_) ? sizeC_ : sizeR_); j++) {
        if (frst)
          printf("%f", eval(i, j));
        else
          printf(" , %f", eval(i, j));
        frst = 0;
      }
      printf(" ; \n");
    }
    printf(" ]\n");
  }
};

}  // namespace blas

#endif  // VIEW_SYCL_HPP
