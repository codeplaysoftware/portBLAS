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
 *  @filename blas2_trees.h
 *
 **************************************************************************/

#ifndef SYCL_BLAS_BLAS2_TREES_H
#define SYCL_BLAS_BLAS2_TREES_H
namespace blas {
template <class Output_t, class Matrix_t, class Vector_t>
struct Gemv {
  using value_type = typename Vector_t::value_type;
  using IndexType = typename Vector_t::IndexType;
  Output_t l;
  Matrix_t matrix;
  Vector_t vector;

  Gemv(Output_t &_l, Matrix_t &_matrix, Vector_t &_vector);
  IndexType getSize() const;
  bool valid_thread(cl::sycl::nd_item<1> ndItem) const;
  value_type eval(IndexType i);
  value_type eval(cl::sycl::nd_item<1> ndItem);
  void bind(cl::sycl::handler &h);
};

template <typename Output_t, typename Matrix_t, typename Vector_t>
Gemv<Output_t, Matrix_t, Vector_t> make_gemv(Output_t &l, Matrix_t &matrix,
                                             Vector_t &vector) {
  return Gemv<Output_t, Matrix_t, Vector_t>(l, matrix, vector);
}

template <class RHS>
struct AddSetColumns {
  using value_type = typename RHS::value_type;
  using IndexType = typename RHS::IndexType;

  RHS r;

  AddSetColumns(RHS &_r);

  IndexType getSize() const;

  bool valid_thread(cl::sycl::nd_item<1> ndItem) const;

  value_type eval(IndexType i);

  value_type eval(cl::sycl::nd_item<1> ndItem);
  void bind(cl::sycl::handler &h);
};

/**
 * @fn make_addSetColumns
 * @brief Constructs an AddSetColumns structure.
 */
template <class RHS>
AddSetColumns<RHS> make_addSetColumns(RHS &r) {
  return AddSetColumns<RHS>(r);
}

/**
 * @struct Gemv_Col
 * @brief Tree node representing a Gemv, with parallel expressed across columns
 * *
 */
template <bool Lower, bool Diag, bool Upper, bool Unit, class LHS,
          class Matrix_t, class Vector_t>
struct Gemv_Col {
  using value_type = typename Vector_t::value_type;
  using IndexType = typename Vector_t::IndexType;
  LHS l;
  Matrix_t matrix;
  Vector_t vector;
  IndexType nWG_row;
  IndexType nWG_col;
  IndexType shrMemSize;

  Gemv_Col(LHS &_l, Matrix_t &_matrix, Vector_t &_vector, IndexType &_nWG_row,
           IndexType &_nWG_col, IndexType &_shrMemSize);
  IndexType getSize() const;
  bool valid_thread(cl::sycl::nd_item<1> ndItem) const;
  value_type eval(IndexType i);
  value_type eval(cl::sycl::nd_item<1> ndItem);
  template <typename sharedT>
  value_type eval(sharedT shrMem, cl::sycl::nd_item<1> ndItem);
  void bind(cl::sycl::handler &h);
};

// template <class LHS, class Matrix_t, class Vector_t>
template <bool Lower = true, bool Diag = true, bool Upper = true,
          bool Unit = false, class LHS, class Matrix_t, class Vector_t>
Gemv_Col<Lower, Diag, Upper, Unit, LHS, Matrix_t, Vector_t> make_Gemv_Col(
    LHS &l, Matrix_t &matrix, Vector_t &vector,
    typename Vector_t::IndexType nWG_row, typename Vector_t::IndexType nWG_col,
    typename Vector_t::IndexType shrMemSize) {
  return Gemv_Col<Lower, Diag, Upper, Unit, LHS, Matrix_t, Vector_t>(
      l, matrix, vector, nWG_row, nWG_col, shrMemSize);
}

/**
 * @struct Gemv_Row
 * @brief Tree node representing a row-based/row-parallel generalised matrix
 * vector multiplication.
 */
template <int interLoop, bool Lower, bool Diag, bool Upper, bool Unit,
          class LHS, class Matrix_t, class Vector_t>
struct Gemv_Row {
  using value_type = typename Vector_t::value_type;
  using IndexType = typename Vector_t::IndexType;

  LHS l;
  Matrix_t matrix;
  Vector_t vector;
  IndexType nWG_row;
  IndexType nWG_col;
  IndexType shrMemSize;

  Gemv_Row(LHS &_l, Matrix_t &_matrix, Vector_t &_vector, IndexType &_nWG_row,
           IndexType &_nWG_col, IndexType &_shrMemSize);
  IndexType getSize() const;
  bool valid_thread(cl::sycl::nd_item<1> ndItem) const;
  value_type eval(IndexType i);
  value_type eval(cl::sycl::nd_item<1> ndItem);
  template <typename sharedT>
  value_type eval(sharedT shrMem, cl::sycl::nd_item<1> ndItem);
  void bind(cl::sycl::handler &h);
};
/*!
 @brief Generator/factory for row-based GEMV trees.

 make_Gemv_Row(
    LHS &l,
    Matrix_t &matrix,
    Vector_t &vector,
    typename Vector_t::IndexType nWG_row,
    typename Vector_t::IndexType nWG_col,
    typename Vector_t::IndexType shrMemSize
 )
 */
template <int interLoop = 1, bool Lower = true, bool Diag = true,
          bool Upper = true, bool Unit = false, typename LHS, typename Matrix_t,
          typename Vector_t>
Gemv_Row<interLoop, Lower, Diag, Upper, Unit, LHS, Matrix_t, Vector_t>
make_Gemv_Row(LHS &l, Matrix_t &matrix, Vector_t &vector,
              typename Vector_t::IndexType nWG_row,
              typename Vector_t::IndexType nWG_col,
              typename Vector_t::IndexType shrMemSize) {
  return Gemv_Row<interLoop, Lower, Diag, Upper, Unit, LHS, Matrix_t, Vector_t>(
      l, matrix, vector, nWG_row, nWG_col, shrMemSize);
}

/**** GER BY ROWS M ROWS x N BLOCK USING PROPERLY THE SHARED MEMORY ****/
// template <class LHS, class RHS1, class RHS2>
template <bool Single, bool Lower, bool Diag, bool Upper, class LHS, class RHS1,
          class RHS2>
struct Ger_Row {
  using value_type = typename RHS2::value_type;
  using IndexType = typename RHS2::IndexType;
  LHS l;
  RHS1 r1;
  RHS2 r2;
  IndexType nWG_row;
  IndexType nWG_col;
  IndexType shrMemSize;
  value_type scl;
  Ger_Row(LHS &_l, value_type _scl, RHS1 &_r1, RHS2 &_r2, IndexType &_nWG_row,
          IndexType &_nWG_col, IndexType &_shrMemSize);
  IndexType getSize() const;
  bool valid_thread(cl::sycl::nd_item<1> ndItem) const;
  value_type eval(IndexType i);
  value_type eval(cl::sycl::nd_item<1> ndItem);
  template <typename sharedT>
  value_type eval(sharedT shrMem, cl::sycl::nd_item<1> ndItem);
  void bind(cl::sycl::handler &h);
};

template <bool Single = true, bool Lower = true, bool Diag = true,
          bool Upper = true, class LHS, class RHS1, class RHS2>
Ger_Row<Single, Lower, Diag, Upper, LHS, RHS1, RHS2> make_Ger_Row(
    LHS &l, typename LHS::value_type scl, RHS1 &r1, RHS2 &r2,
    typename RHS2::IndexType nWG_row, typename RHS2::IndexType nWG_col,
    typename RHS2::IndexType shrMemSize) {
  return Ger_Row<Single, Lower, Diag, Upper, LHS, RHS1, RHS2>(
      l, scl, r1, r2, nWG_row, nWG_col, shrMemSize);
}

/**** GER BY COLUMNS M ROWS x N BLOCK USING PROPERLY THE SHARED MEMORY ****/
// template <class LHS, class RHS1, class RHS2>
template <bool Single, bool Lower, bool Diag, bool Upper, class LHS, class RHS1,
          class RHS2>
struct Ger_Col {
  using value_type = typename RHS2::value_type;
  using IndexType = typename RHS2::IndexType;

  LHS l;
  RHS1 r1;
  RHS2 r2;
  IndexType nWG_row;
  IndexType nWG_col;
  IndexType shrMemSize;
  value_type scl;

  Ger_Col(LHS &_l, value_type _scl, RHS1 &_r1, RHS2 &_r2, IndexType &_nWG_row,
          IndexType &_nWG_col, IndexType &_shrMemSize);
  IndexType getSize() const;
  bool valid_thread(cl::sycl::nd_item<1> ndItem) const;
  value_type eval(IndexType i);
  value_type eval(cl::sycl::nd_item<1> ndItem);
  template <typename sharedT>
  value_type eval(sharedT shrMem, cl::sycl::nd_item<1> ndItem);
  void bind(cl::sycl::handler &h);
};

// template <class LHS, class RHS1, class RHS2>
template <bool Single = true, bool Lower = true, bool Diag = true,
          bool Upper = true, class LHS, class RHS1, class RHS2>
Ger_Col<Single, Lower, Diag, Upper, LHS, RHS1, RHS2> make_Ger_Col(
    LHS &l, typename LHS::value_type scl, RHS1 &r1, RHS2 &r2,
    typename RHS2::IndexType nWG_row, typename RHS2::IndexType nWG_col,
    typename RHS2::IndexType shrMemSize) {
  return Ger_Col<Single, Lower, Diag, Upper, LHS, RHS1, RHS2>(
      l, scl, r1, r2, nWG_row, nWG_col, shrMemSize);
}
}  // namespace blas
#endif  // BLAS2_TREES_H
