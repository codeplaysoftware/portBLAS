/***************************************************************************
 *
 *  @license
 *  Copyright (C) 2016 Codeplay Software Limited
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

template <typename ContainerT>
struct get_size_struct {
  static inline size_t get_size(ContainerT &c) { return c.size(); }
};

template <typename ContainerT>
size_t get_size(ContainerT &c) {
  return get_size_struct<ContainerT>::get_size(c);
}

template <typename ScalarT, int dim = 1, typename Allocator = cl::sycl::default_allocator<ScalarT>>
using bufferT = cl::sycl::buffer<ScalarT, dim, Allocator>;

template <typename ScalarT, int dim = 1, typename Allocator = cl::sycl::default_allocator<ScalarT>>
using BufferVectorView = vector_view<ScalarT, bufferT<ScalarT, dim, Allocator> >;

template <typename ScalarT, int dim , typename Allocator >
struct get_size_struct< bufferT<ScalarT, dim, Allocator> > {
  static inline size_t get_size(bufferT<ScalarT> &b) { return b.get_size(); }
};

/*! vector_view<ScalarT, bufferT<Scalar>>
 * @brief Specialization of the vector view to operate with buffers.
 * Note that the buffer class cannot be accessed on the host.
 */
template <typename ScalarT, int dim, typename Allocator >
struct vector_view<ScalarT, bufferT<ScalarT, dim, Allocator>> {
  using ContainerT = bufferT<ScalarT, dim, Allocator>;
  ContainerT &data_;
  size_t size_data_;
  size_t size_;
  size_t disp_;
  long strd_;  // never size_t, because it could negative

  using value_type = ScalarT;

  /*! initialize.
   * Initializes the vector view with the information from.
   */
  inline void initialize(size_t originalSize) {
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
  vector_view(ContainerT &data, size_t disp = 0, long strd = 1)
      : data_(data),
        size_data_(data_.get_size()),
        size_(data_.get_size() / sizeof(ScalarT)),
        disp_(disp),
        strd_(strd) {}

  /*! vector_view.
   * See vector_view.
   */
  vector_view(ContainerT &data, size_t disp, long strd, size_t size)
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
  vector_view(vector_view<ScalarT, ContainerT> opV, size_t disp, long strd,
              size_t size)
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
  size_t getDataSize() { return size_data_; }

  /*! vector_view.
   * See vector_view.
   */
  size_t getSize() { return size_; }

  /*! vector_view.
   * See vector_view.
   */
  size_t getDisp() { return disp_; }

  /*! vector_view.
   * See vector_view.
   */
  long getStrd() { return strd_; }

  /*! vector_view.
   * See vector_view.
   */
  vector_view<ScalarT, ContainerT> operator+(size_t disp) {
    if (this->strd_ > 0)
      return vector_view<ScalarT, ContainerT>(
          this->data_, this->disp_ + (disp * this->strd_), this->strd_,
          this->size_ - disp);
    else
      return vector_view<ScalarT, ContainerT>(
          this->data_, this->disp_ - ((this->size_ - 1) - disp) * this->strd_,
          this->strd_, this->size_ - disp);
  }

  /*! vector_view.
   * See vector_view.
   */
  vector_view<ScalarT, ContainerT> operator()(size_t disp) {
    if (this->strd_ > 0)
      return vector_view<ScalarT, ContainerT>(
          this->data_, this->disp_ + (disp * this->strd_), this->strd_,
          this->size_ - disp);
    else
      return vector_view<ScalarT, ContainerT>(
          this->data_, this->disp_ - ((this->size_ - 1) - disp) * this->strd_,
          this->strd_, this->size_ - disp);
  }

  /*! vector_view.
    * See vector_view.
    */
  vector_view<ScalarT, ContainerT> operator*(long strd) {
    return vector_view<ScalarT, ContainerT>(this->data_, this->disp_,
                                            this->strd_ * strd);
  }

  /*! vector_view.
   * See vector_view.
   */
  vector_view<ScalarT, ContainerT> operator%(size_t size) {
    if (this->strd_ > 0) {
      return vector_view<ScalarT, ContainerT>(this->data_, this->disp_,
                                              this->strd_, size);
    } else {
      return vector_view<ScalarT, ContainerT>(
          this->data_, this->disp_ - (this->size_ - 1) * this->strd_,
          this->strd_, size);
    }
  }

  /*! eval.
    * See vector_view::eval.
    */
  ScalarT &eval(size_t i) {
    //  auto eval(size_t i) -> decltype(data_[i]) {
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
      // out of range access
      printf("(A) ind = %ld , size_data_ = %ld \n", ind, size_data_);
#endif  //  VERBOSE
      throw std::invalid_argument("Out of range access");
    }
#endif  //__SYCL_DEVICE_ONLY__
    ScalarT retVal;
    {
      auto hostPtr =
          data_.template get_access<cl::sycl::access::mode::read_write,
                                    cl::sycl::access::target::host_buffer>();
      retVal = hostPtr[ind];
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
  ScalarT val(size_t i) {
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
      auto hostPtr =
          data_.template get_access<cl::sycl::access::mode::read_write,
                                    cl::sycl::access::target::host_buffer>();
      retVal = hostPtr[ind];
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
template <class ScalarT>
struct matrix_view<ScalarT, bufferT<ScalarT>> {
  using ContainerT = bufferT<ScalarT>;
  // Information related to the data
  ContainerT &data_;
  int accessDev_;     // row-major or column-major value for the device/language
  size_t size_data_;  // real size of the data
  // Information related to the operation
  int accessOpr_;  // row-major or column-major
  size_t sizeR_;   // number of rows
  size_t sizeC_;   // number of columns
  size_t sizeL_;   // size of the leading dimension
  size_t disp_;    // displacementt od the first element
  // UPLO, BAND(KU,KL), PACKED, SIDE ARE ONLY REQUIRED
  using value_type = ScalarT;

  /*! matrix_view.
   * @brief See matrix_view.
   */
  matrix_view(ContainerT &data, int accessDev, size_t sizeR, size_t sizeC)
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
  matrix_view(ContainerT &data, size_t sizeR, size_t sizeC)
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
  matrix_view(ContainerT &data, int accessDev, size_t sizeR, size_t sizeC,
              int accessOpr, size_t sizeL, size_t disp)
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
  matrix_view(ContainerT &data, size_t sizeR, size_t sizeC, int accessOpr,
              size_t sizeL, size_t disp)
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
  matrix_view(matrix_view<ScalarT, ContainerT> opM, int accessDev, size_t sizeR,
              size_t sizeC, int accessOpr, size_t sizeL, size_t disp)
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
  matrix_view(matrix_view<ScalarT, ContainerT> opM, size_t sizeR, size_t sizeC,
              int accessOpr, size_t sizeL, size_t disp)
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
  size_t getDataSize() { return size_data_; }

  /*!
   * @brief See matrix_view.
   */
  size_t getSize() { return sizeR_ * sizeC_; }

  /*!
   * @brief See matrix_view.
   */
  size_t getSizeR() { return sizeR_; }

  /*!
   * @brief See matrix_view.
   */
  size_t getSizeC() { return sizeC_; }

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
  long getDisp() { return disp_; }

  /*!
   * @brief See matrix_view.
   */
  matrix_view<ScalarT, ContainerT> operator+(size_t disp) {
    return matrix_view<ScalarT, ContainerT>(
        this->data_, this->accessDev_, this->sizeR_, this->sizeC_,
        this->accessOpr_, this->sizeL_, this->disp_ + disp);
  }

  /*!
   * @brief See matrix_view.
   */
  matrix_view<ScalarT, ContainerT> operator()(size_t i, size_t j) {
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

  /*!
   * @brief See matrix_view.
   */
  ScalarT &eval(size_t k) {  // -> decltype(data_[i]) {
    auto ind = disp_;
    int access = (!(accessDev_ ^ accessOpr_));
    auto size = (access) ? sizeC_ : sizeR_;
    auto i = (access) ? (k / size) : (k % size);
    auto j = (access) ? (k % size) : (k / size);

    return eval(i, j);
  }

  /*!
   * @brief See matrix_view.
   */
  ScalarT &eval(size_t i, size_t j) {
    auto ind = disp_;

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
      auto hostPtr =
          data_.template get_access<cl::sycl::access::mode::read_write,
                                    cl::sycl::access::target::host_buffer>();
      retVal = hostPtr[ind];
    }

    return retVal;
  }

  ScalarT &eval(cl::sycl::nd_item<1> ndItem) {
    return eval(ndItem.get_global(0));
  }

  /*! val.
   * @brief Used to print the values on the host.
   */
  ScalarT val(size_t i, size_t j) {
    auto ind = disp_;

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
      auto hostPtr =
          data_.template get_access<cl::sycl::access::mode::read_write,
                                    cl::sycl::access::target::host_buffer>();
      retVal = hostPtr[ind];
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
template <typename ScalarT>
struct vector_view<ScalarT, accessorT<ScalarT>> {
  using ContainerT = accessorT<ScalarT>;

  ContainerT data_;
  size_t size_data_;
  size_t size_;
  size_t disp_;
  long strd_;  // never size_t, because it could negative

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
  vector_view(ContainerT &data, size_t disp)
      : data_{data},
        size_data_(data_.get_size()),
        size_(data_.get_size()),
        disp_(disp),
        strd_(1) {}

  /*!
   * @brief See vector_view.
   */
  vector_view(ContainerT &data, size_t disp, long strd, size_t size)
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
  vector_view(vector_view<ScalarT, ContainerT> &opV, size_t disp, long strd,
              size_t size)
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
  size_t getDataSize() { return size_data_; }

  /*!
   * @brief See vector_view.
   */
  size_t getSize() { return size_; }

  /*!
   * @brief See vector_view.
   */
  size_t getDisp() { return disp_; }

  /*!
   * @brief See vector_view.
   */
  long getStrd() { return strd_; }

  /*!
   * @brief See vector_view.
   */
  vector_view<ScalarT, ContainerT> operator+(size_t disp) {
    if (this->strd_ > 0)
      return vector_view<ScalarT, ContainerT>(
          this->data_, this->disp_ + (disp * this->strd_), this->strd_,
          this->size_ - disp);
    else
      return vector_view<ScalarT, ContainerT>(
          this->data_, this->disp_ - ((this->size_ - 1) - disp) * this->strd_,
          this->strd_, this->size_ - disp);
  }

  /*!
   * @brief See vector_view.
   */
  vector_view<ScalarT, ContainerT> operator()(size_t disp) {
    if (this->strd_ > 0)
      return vector_view<ScalarT, ContainerT>(
          this->data_, this->disp_ + (disp * this->strd_), this->strd_,
          this->size_ - disp);
    else
      return vector_view<ScalarT, ContainerT>(
          this->data_, this->disp_ - ((this->size_ - 1) - disp) * this->strd_,
          this->strd_, this->size_ - disp);
  }

  /*!
   * @brief See vector_view.
   */
  vector_view<ScalarT, ContainerT> operator*(long strd) {
    return vector_view<ScalarT, ContainerT>(this->data_, this->disp_,
                                            this->strd_ * strd);
  }

  /*!
   * @brief See vector_view.
   */
  vector_view<ScalarT, ContainerT> operator%(size_t size) {
    if (this->strd_ > 0) {
      return vector_view<ScalarT, ContainerT>(this->data_, this->disp_,
                                              this->strd_, size);
    } else {
      return vector_view<ScalarT, ContainerT>(
          this->data_, this->disp_ - (this->size_ - 1) * this->strd_,
          this->strd_, size);
    }
  }

  /**** EVALUATING ****/
  ScalarT &eval(size_t i) {
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
      printf("(E) ind = %ld , size_data_ = %ld \n", ind, size_data_);
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
template <class ScalarT>
struct matrix_view<ScalarT, accessorT<ScalarT>> {
  using ContainerT = accessorT<ScalarT>;
  // Information related to the data
  ContainerT data_;
  int accessDev_;     // row-major or column-major value for the device/language
  size_t size_data_;  // real size of the data
  // Information related to the operation
  int accessOpr_;  // row-major or column-major
  size_t sizeR_;   // number of rows
  size_t sizeC_;   // number of columns
  size_t sizeL_;   // size of the leading dimension
  size_t disp_;    // displacementt od the first element
  // UPLO, BAND(KU,KL), PACKED, SIDE ARE ONLY REQUIRED
  using value_type = ScalarT;

  /**** CONSTRUCTORS ****/

  matrix_view(ContainerT &data, int accessDev, size_t sizeR, size_t sizeC)
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

  matrix_view(ContainerT &data, size_t sizeR, size_t sizeC)
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

  matrix_view(ContainerT &data, int accessDev, size_t sizeR, size_t sizeC,
              int accessOpr, size_t sizeL, size_t disp)
      : data_{data},
        accessDev_(accessDev),
        size_data_(data_.get_size()),
        accessOpr_(accessOpr),
        sizeR_(sizeR),
        sizeC_(sizeC),
        sizeL_(sizeL),
        disp_(disp) {}

  matrix_view(ContainerT &data, size_t sizeR, size_t sizeC, int accessOpr,
              size_t sizeL, size_t disp)
      : data_{data},
        accessDev_(0),
        size_data_(data_.get_size()),
        accessOpr_(accessOpr),
        sizeR_(sizeR),
        sizeC_(sizeC),
        sizeL_(sizeL),
        disp_(disp) {}

  matrix_view(matrix_view<ScalarT, ContainerT> opM, int accessDev, size_t sizeR,
              size_t sizeC, int accessOpr, size_t sizeL, size_t disp)
      : data_{opM.data_},
        accessDev_(accessDev),
        size_data_(opM.size_data_),
        accessOpr_(!(opM.accessOpr_ ^ accessOpr)),
        sizeR_(sizeR),
        sizeC_(sizeC),
        sizeL_(sizeL),
        disp_(disp) {}

  matrix_view(matrix_view<ScalarT, ContainerT> opM, size_t sizeR, size_t sizeC,
              int accessOpr, size_t sizeL, size_t disp)
      : data_{opM.data_},
        accessDev_(opM.accessDev_),
        size_data_(opM.size_data_),
        accessOpr_(!(opM.accessOpr_ ^ accessOpr)),
        sizeR_(sizeR),
        sizeC_(sizeC),
        sizeL_(sizeL),
        disp_(disp) {}

  /**** RETRIEVING DATA ****/
  ContainerT &getData() { return data_; }

  size_t getDataSize() { return size_data_; }

  size_t getSize() { return sizeR_ * sizeC_; }

  size_t getSizeR() { return sizeR_; }

  size_t getSizeC() { return sizeC_; }

  int getAccess() { return !(accessDev_ ^ accessOpr_); }

  int getAccessDev() { return accessDev_; }

  int getAccessOpr() { return accessOpr_; }

  long getDisp() { return disp_; }

  /**** OPERATORS ****/
  matrix_view<ScalarT, ContainerT> operator+(size_t disp) {
    return matrix_view<ScalarT, ContainerT>(
        this->data_, this->accessDev_, this->sizeR_, this->sizeC_,
        this->accessOpr_, this->sizeL_, this->disp_ + disp);
  }

  matrix_view<ScalarT, ContainerT> operator()(size_t i, size_t j) {
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
  ScalarT &eval(size_t k) {
    int access = (!(accessDev_ ^ accessOpr_));
    auto size = (access) ? sizeC_ : sizeR_;
    auto i = (access) ? (k / size) : (k % size);
    auto j = (access) ? (k % size) : (k / size);

    return eval(i, j);
  }

  ScalarT &eval(size_t i, size_t j) {  // -> decltype(data_[i]) {
    auto ind = disp_;
    int accessMode = !(accessDev_ ^ accessOpr_);

    if (accessMode) {
      ind += (sizeL_ * i) + j;
    } else {
      ind += (sizeL_ * j) + i;
    }
#ifndef __SYCL_DEVICE_ONLY__
    if (ind >= size_data_) {
      printf("(G) ind = %ld , size_data_ = %ld \n", ind, size_data_);
    }
#endif  //__SYCL_DEVICE_ONLY__
    return data_[ind];
  }

  ScalarT &eval(cl::sycl::nd_item<1> ndItem) {
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
