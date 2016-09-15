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
 *  @filename operview_base.hpp
 *
 **************************************************************************/

#ifndef OPERVIEW_BASE_HPP
#define OPERVIEW_BASE_HPP

#include <iostream>
#include <stdexcept>
#include <vector>

namespace blas {

/*!
@brief Alias to std::string.
*/
using string_class = std::string;

/*!
@brief Template struct for containing vector that can used within a compile-time
expression.
@tparam valueT Type of each element fo the vector.
@tparam containerT Type of the container that is stored inside.
*/
template <class valueT, class containerT>
struct vector_view {
  containerT& data_;
  size_t size_data_;
  size_t size_;
  size_t disp_;
  long strd_;  // never size_t, because it could be negative

  using value_type = valueT;

  /*!
  @brief Initializes the view using the indexing values.
  @param originalSize The original size of the container
  */
  inline void initialize(size_t originalSize) {
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
  @brief Constructs a view from the given container re-using the container size.
  @param data
  @param disp
  @param strd
  */
  vector_view(containerT& data, size_t disp = 0, long strd = 1)
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
  vector_view(containerT& data, size_t disp, long strd, size_t size)
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
  vector_view(vector_view<valueT, containerT> opV, size_t disp, long strd,
              size_t size)
      : data_(opV.getData()),
        size_data_(opV.getData().size()),
        size_(0),
        disp_(disp),
        strd_(strd) {
    initialize(size);
  }

  /*!
   * @brief Returns a reference to the container
   */
  containerT& getData() { return data_; }

  /*!
   * @brief Returns the displacement
   */
  size_t getDisp() { return disp_; }

  /*!
   * @brief Returns the size of the underlying container.
   */
  size_t getDataSize() { return size_data_; }

  /*!
   @brief Returns the size of the view
   */
  size_t getSize() { return size_; }

  /*!
   @brief Returns the stride of the view.
  */
  long getStrd() { return strd_; }

  /*!
   * @brief Adds a displacement to the view, creating a new view.
   */
  vector_view<valueT, containerT> operator+(size_t disp) {
    if (this->strd_ > 0) {
      return vector_view<valueT, containerT>(this->data_,
                                             this->disp_ + (disp * this->strd_),
                                             this->strd_, this->size_ - disp);
    } else {
      return vector_view<valueT, containerT>(
          this->data_, this->disp_ - ((this->size_ - 1) - disp) * this->strd_,
          this->strd_, this->size_ - disp);
    }
  }

  /*!
   * @brief Adds a displacement to the view, creating a new view.
   */
  vector_view<valueT, containerT> operator()(size_t disp) {
    if (this->strd_ > 0) {
      return vector_view<valueT, containerT>(this->data_,
                                             this->disp_ + (disp * this->strd_),
                                             this->strd_, this->size_ - disp);
    } else {
      return vector_view<valueT, containerT>(
          this->data_, this->disp_ - ((this->size_ - 1) - disp) * this->strd_,
          this->strd_, this->size_ - disp);
    }
  }

  /*!
   @brief Multiplies the view stride by the given one and returns a new one
  */
  vector_view<valueT, containerT> operator*(long strd) {
    return vector_view<valueT, containerT>(this->data_, this->disp_,
                                           this->strd_ * strd);
  }

  /*!
   @brief
  */
  vector_view<valueT, containerT> operator%(size_t size) {
    if (this->strd_ > 0) {
      return vector_view<valueT, containerT>(this->data_, this->disp_,
                                             this->strd_, size);
    } else {
      return vector_view<valueT, containerT>(
          this->data_, this->disp_ - (this->size_ - 1) * this->strd_,
          this->strd_, size);
    }
  }

  /**** EVALUATING ****/
  valueT& eval(size_t i) {
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

  template <class X, class Y>
  friend std::ostream& operator<<(std::ostream& stream, vector_view<X, Y> opvS);

  void printH(const char* name) {
    bool frst = true;
    std::cout << name << " = [ ";
    for (size_t i = 0; i < size_; i++) {
      if (frst)
        std::cout << eval(i);
      else
        std::cout << " , " << eval(i);
      frst = false;
    }
    std::cout << " ]" << std::endl;
  }
};

/*! matrix_view
@brief Represents a Matrix on the given Container.
@tparam valueT Value type of the container.
@tparam containerT Type of the container.
 */
template <class valueT, class containerT>
struct matrix_view {
  // Information related to the data
  containerT& data_;
  bool accessDev_;    // True for row-major, column-major otherwise
  size_t size_data_;  // real size of the data
  bool accessOpr_;    // Operation Access Mode (True: Normal, False: Transpose)
  size_t sizeR_;      // number of rows
  size_t sizeC_;      // number of columns
  size_t sizeL_;      // size of the leading dimension
  size_t disp_;       // displacementt od the first element
  // UPLO, BAND(KU,KL), PACKED, SIDE ARE ONLY REQUIRED
  using value_type = valueT;

  /*!
   * @brief Constructs a matrix view on the container.
   * @param data Reference to the container.
   * @param accessDev Row-major or column-major.
   * @param sizeR Number of rows.
   * @param sizeC Number of columns.
   */
  matrix_view(containerT& data, bool accessDev, size_t sizeR, size_t sizeC)
      : data_(data),
        accessDev_(accessDev),
        size_data_(data_.get_size()),
        accessOpr_(true),
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
  matrix_view(containerT& data, size_t sizeR, size_t sizeC)
      : data_(data),
        accessDev_(true),
        size_data_(data_.get_size()),
        accessOpr_(true),
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
  matrix_view(containerT& data, bool accessDev, size_t sizeR, size_t sizeC,
              bool accessOpr, size_t sizeL, size_t disp)
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
  matrix_view(containerT& data, size_t sizeR, size_t sizeC, bool accessOpr,
              size_t sizeL, size_t disp)
      : data_(data),
        accessDev_(true),
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
  matrix_view(matrix_view<valueT, containerT> opM, bool accessDev, size_t sizeR,
              size_t sizeC, bool accessOpr, size_t sizeL, size_t disp)
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
  matrix_view(matrix_view<valueT, containerT> opM, size_t sizeR, size_t sizeC,
              bool accessOpr, size_t sizeL, size_t disp)
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
  containerT& getData() { return data_; }

  /*!
   * @brief Returns the data size
   */
  size_t getDataSize() { return size_data_; }

  /*!
   * @brief Returns the size of the view.
   */
  size_t getSize() { return sizeR_ * sizeC_; }

  /*! getSizeR.
   * @brief Return the number of columns.
   * @bug This value should change depending on the access mode, but
   * is currently set to Rows.
   */
  size_t getSizeR() { return sizeR_; }

#if BLAS_EXPERIMENTAL
  // These implementationS are currently not working
  size_t getSizeR() { return getAccess() ? sizeR_ : sizeC_; }
  size_t getSizeR() { return accessOpr_ ? sizeR_ : sizeC_; }
#endif  // BLAS_EXPERIMENTAL

  /*! getSizeC.
   * @brief Return the number of columns.
   * @bug This value should change depending on the access mode, but 
   * is currently set to Rows. 
   */ size_t getSizeC() {
    return sizeC_;
  }
#if BLAS_EXPERIMENTAL
  // This implementations are currently not working
  size_t getSizeC() { return getAccess() ? sizeC_ : sizeR_; }
  size_t getSizeC() { return accessOpr_ ? sizeC_ : sizeR_; }
#endif  // BLAS_EXPERIMENTAL

  /*! getAccess.
   * @brief Access mode for the view.
   * Combination of the device access vs the operation mode.
   */
  bool getAccess() { return !(accessDev_ ^ accessOpr_); }

  /*! getAccessDev.
   * @brief Access on the Device (e.g CPU: Row, GPU: Column).
   */
  bool getAccessDev() { return accessDev_; }

  /*! getAccessOpr.
   * @brief Returns the operation access mode
   * @return True: Normal access, False: Transpose
   */
  bool getAccessOpr() { return accessOpr_; }

  /*! getDisp.
   * @brief get displacement from the origin.
   */
  long getDisp() { return disp_; }

  /*!
   * @brief Adds a displacement to the view, creating a new view.
   */
  matrix_view<valueT, containerT> operator+(size_t disp) {
    return matrix_view<valueT, containerT>(
        this->data_, this->accessDev_, this->sizeR_, this->sizeC_,
        this->accessOpr_, this->sizeL_, this->disp_ + disp);
  }

  /*!
   * @brief Adds a displacement to the view, creating a new view.
   */
  matrix_view<valueT, containerT> operator()(size_t i, size_t j) {
    if (!(accessDev_ ^ accessOpr_)) {
      // ACCESING BY ROWS
      return matrix_view<valueT, containerT>(
          this->data_, this->accessDev_, this->sizeR_, this->sizeC_,
          this->accessOpr_, this->sizeL_, this->disp_ + i * this->sizeL_ + j);
    } else {
      // ACCESING BY COLUMN
      return matrix_view<valueT, containerT>(
          this->data_, this->accessDev_, this->sizeR_, this->sizeC_,
          this->accessOpr_, this->sizeL_, this->disp_ + i + this->sizeL_ * j);
    }
  }

  /*! eval.
   * @brief Evaluation for the given linear value.
   */
  valueT& eval(size_t k) {
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
  valueT& eval(size_t i, size_t j) {
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
  void printH(const char* name) {
    std::cout << name << " = [ " << std::endl;
    for (size_t i = 0; i < ((accessOpr_) ? sizeR_ : sizeC_); i++) {
      bool frst = true;
      for (size_t j = 0; j < ((accessOpr_) ? sizeC_ : sizeR_); j++) {
        if (frst)
          std::cout << eval(i, j);
        else
          std::cout << " , " << eval(i, j);
        frst = false;
      }
      std::cout << " ;" << std::endl;
    }
    std::cout << " ]" << std::endl;
  }
};

}  // namespace blas

#endif  // OPERVIEW_BASE_HPP
