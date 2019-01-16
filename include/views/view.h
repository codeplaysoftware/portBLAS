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
 *  @filename view.h
 *
 **************************************************************************/

#ifndef SYCL_BLAS_VIEW_H
#define SYCL_BLAS_VIEW_H

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
@tparam ValueT Type of each element fo the vector.
@tparam ContainerT Type of the container that is stored inside.
*/

template <class ValueT_, class ContainerT_, typename IndexType_,
          typename IncrementType_>
struct vector_view {
  using ValueT = ValueT_;
  using ContainerT = ContainerT_;
  using IndexType = IndexType_;
  using IncrementType = IncrementType_;
  using Self = vector_view<ValueT, ContainerT, IndexType, IncrementType>;
  using value_type = ValueT;
  ContainerT &data_;
  IndexType size_data_;
  IndexType size_;
  IndexType disp_;
  IncrementType strd_;  // never size_t, because it could be negative

  vector_view(ContainerT_ &data, IndexType_ disp = 0, IncrementType_ strd = 1);
  vector_view(ContainerT_ &data, IndexType_ disp, IncrementType_ strd,
              IndexType_ size);
  vector_view(vector_view<ValueT_, ContainerT_, IndexType_, IncrementType_> opV,
              IndexType_ disp, IncrementType_ strd, IndexType_ size);

  /*!
  @brief Initializes the view using the indexing values.
  @param originalSize The original size of the container
  */
  inline void initialize(IndexType originalSize);

  /*!
   * @brief Returns a reference to the container
   */
  ContainerT &getData();

  /*!
   * @brief Returns the displacement
   */
  IndexType getDisp();

  /*!
   * @brief Returns the size of the underlying container.
   */
  IndexType getDataSize();

  /*!
   @brief Returns the size of the view
   */
  inline IndexType getSize() const;

  /*!
   @brief Returns the stride of the view.
  */
  IncrementType getStrd();

  /*!
   * @brief Adds a displacement to the view, creating a new view.
   */
  Self operator+(IndexType disp);

  /*!
   * @brief Adds a displacement to the view, creating a new view.
   */
  Self operator()(IndexType disp);

  /*!
   @brief Multiplies the view stride by the given one and returns a new one
  */
  Self operator*(IncrementType strd);
  /*!
   @brief
  */
  Self operator%(IndexType size);

  /**** EVALUATING ****/
  ValueT &eval(IndexType i);

  template <class X, class Y, typename IndxT, typename IncrT>
  friend std::ostream &operator<<(std::ostream &stream,
                                  vector_view<X, Y, IndxT, IncrT> opvS);

  void printH(const char *name);
};

/*! matrix_view
@brief Represents a Matrix on the given Container.
@tparam ValueT Value type of the container.
@tparam ContainerT Type of the container.
 */
template <class ValueT_, class ContainerT_, typename IndexType_>
struct matrix_view {
  // Information related to the data
  using ValueT = ValueT_;
  using ContainerT = ContainerT_;
  using IndexType = IndexType_;
  using Self = matrix_view<ValueT, ContainerT, IndexType>;
  ContainerT &data_;
  int accessDev_;        // True for row-major, column-major otherwise
  IndexType size_data_;  // real size of the data
  int accessOpr_;    // Operation Access Mode (True: Normal, False: Transpose)
  IndexType sizeR_;  // number of rows
  IndexType sizeC_;  // number of columns
  IndexType sizeL_;  // size of the leading dimension
  IndexType disp_;   // displacementt od the first element
  // UPLO, BAND(KU,KL), PACKED, SIDE ARE ONLY REQUIRED
  using value_type = ValueT;

  matrix_view(ContainerT_ &data, int accessDev, IndexType_ sizeR,
              IndexType_ sizeC);
  matrix_view(ContainerT_ &data, IndexType_ sizeR, IndexType_ sizeC);
  matrix_view(ContainerT_ &data, int accessDev, IndexType_ sizeR,
              IndexType_ sizeC, int accessOpr, IndexType_ sizeL,
              IndexType_ disp);
  matrix_view(ContainerT_ &data, IndexType_ sizeR, IndexType_ sizeC,
              int accessOpr, IndexType_ sizeL, IndexType_ disp);
  matrix_view(matrix_view<ValueT_, ContainerT_, IndexType_> opM, int accessDev,
              IndexType_ sizeR, IndexType_ sizeC, int accessOpr,
              IndexType_ sizeL, IndexType_ disp);
  matrix_view(matrix_view<ValueT_, ContainerT_, IndexType_> opM,
              IndexType_ sizeR, IndexType_ sizeC, int accessOpr,
              IndexType_ sizeL, IndexType_ disp);

  /*!
   * @brief Returns the container
   */
  ContainerT &getData();

  /*!
   * @brief Returns the data size
   */
  IndexType getDataSize() const;

  /*!
   * @brief Returns the size of the view.
   */
  inline IndexType getSize() const;

  /*! getSizeR.
   * @brief Return the number of columns.
   * @bug This value should change depending on the access mode, but
   * is currently set to Rows.
   */
  inline IndexType getSizeR() const;

  /*! getSizeC.
   * @brief Return the number of columns.
   * @bug This value should change depending on the access mode, but
   * is currently set to Rows.
   */
  IndexType getSizeC() const;

  /*! is_row_access.
   * @brief Access mode for the view.
   * Combination of the device access vs the operation mode.
   */
  int is_row_access() const;

  /*! getAccessDev.
   * @brief Access on the Device (e.g CPU: Row, GPU: Column).
   */
  int getAccessDev() const;

  /*! getAccessOpr.
   * @brief Returns the operation access mode
   * @return True: Normal access, False: Transpose
   */
  int getAccessOpr() const;

  /*! getDisp.
   * @brief get displacement from the origin.
   */
  IndexType getDisp() const;

  /*!
   * @brief Adds a displacement to the view, creating a new view.
   */
  Self operator+(IndexType disp);

  /*!
   * @brief Adds a displacement to the view, creating a new view.
   */
  Self operator()(IndexType i, IndexType j);

  /*! eval.
   * @brief Evaluation for the given linear value.
   */
  ValueT &eval(IndexType k);

  /*! eval.
   * @brief Evaluation for the pair of row/col.
   */
  ValueT &eval(IndexType i, IndexType j);

  /*! printH
   * @brief Display the contents of the matrix in stdout.
   */
  void printH(const char *name);
};

template <typename Policy, typename T, typename IndexType,
          typename IncrementType>
struct VectorViewTypeTrace {
  using ScalarT = typename scalar_type<T>::type;
  using Type =
      vector_view<ScalarT, typename Policy::template access_type<ScalarT>,
                  IndexType, IncrementType>;
};

template <typename Policy, typename T, typename IndexType>
struct MatrixViewTypeTrace {
  using ScalarT = typename scalar_type<T>::type;
  using Type =
      matrix_view<ScalarT, typename Policy::template access_type<ScalarT>,
                  IndexType>;
};

template <typename Executor, typename ContainerT, typename IncrementType,
          typename IndexType>
inline typename VectorViewTypeTrace<typename Executor::Policy, ContainerT,
                                    IndexType, IncrementType>::Type
make_vector_view(Executor &ex, ContainerT buff, IncrementType inc,
                 IndexType sz) {
  using LeafNode =
      typename VectorViewTypeTrace<typename Executor::Policy, ContainerT,
                                   IndexType, IncrementType>::Type;
  return LeafNode{ex.get_policy_handler().get_buffer(buff), inc, sz};
}

template <typename Executor, typename ContainerT, typename IndexType,
          typename Opertype>
inline typename MatrixViewTypeTrace<typename Executor::Policy, ContainerT,
                                    IndexType>::Type
make_matrix_view(Executor &ex, ContainerT buff, IndexType m, IndexType n,
                 IndexType lda, Opertype accessOpr) {
  using LeafNode = typename MatrixViewTypeTrace<typename Executor::Policy,
                                                ContainerT, IndexType>::Type;
  return LeafNode{ex.get_policy_handler().get_buffer(buff), m, n, accessOpr,
                  lda};
}

}  // namespace blas

#endif  // VIEW_H
