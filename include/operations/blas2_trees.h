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
template <typename output_t, typename matrix_t, typename vector_t>
struct Gemv {
  using value_t = typename vector_t::value_t;
  using index_t = typename vector_t::index_t;
  output_t lhs_;
  matrix_t matrix_;
  vector_t vector_;

  Gemv(output_t &_l, matrix_t &_matrix, vector_t &_vector);
  index_t get_size() const;
  bool valid_thread(cl::sycl::nd_item<1> ndItem) const;
  value_t eval(index_t i);
  value_t eval(cl::sycl::nd_item<1> ndItem);
  void bind(cl::sycl::handler &h);
};

template <typename output_t, typename matrix_t, typename vector_t>
Gemv<output_t, matrix_t, vector_t> make_gemv(output_t &lhs_, matrix_t &matrix_,
                                             vector_t &vector_) {
  return Gemv<output_t, matrix_t, vector_t>(lhs_, matrix_, vector_);
}

template <typename rhs_t>
struct AddSetColumns {
  using value_t = typename rhs_t::value_t;
  using index_t = typename rhs_t::index_t;

  rhs_t rhs_;

  AddSetColumns(rhs_t &_r);

  index_t get_size() const;

  bool valid_thread(cl::sycl::nd_item<1> ndItem) const;

  value_t eval(index_t i);

  value_t eval(cl::sycl::nd_item<1> ndItem);
  void bind(cl::sycl::handler &h);
};

/**
 * @fn make_addSetColumns
 * @brief Constructs an AddSetColumns structure.
 */
template <typename rhs_t>
AddSetColumns<rhs_t> make_addSetColumns(rhs_t &rhs_) {
  return AddSetColumns<rhs_t>(rhs_);
}

/**
 * @struct GemvCol
 * @brief Tree node representing a Gemv, with parallel expressed across columns
 * *
 */
template <bool Lower, bool Diag, bool Upper, bool Unit, typename lhs_t,
          typename matrix_t, typename vector_t>
struct GemvCol {
  using value_t = typename vector_t::value_t;
  using index_t = typename vector_t::index_t;
  lhs_t lhs_;
  matrix_t matrix_;
  vector_t vector_;
  index_t nWG_row_;
  index_t nWG_col_;
  index_t local_memory_size_;

  GemvCol(lhs_t &_l, matrix_t &_matrix, vector_t &_vector, index_t &_nWG_row,
          index_t &_nWG_col, index_t &_shrMemSize);
  index_t get_size() const;
  bool valid_thread(cl::sycl::nd_item<1> ndItem) const;
  value_t eval(index_t i);
  value_t eval(cl::sycl::nd_item<1> ndItem);
  template <typename sharedT>
  value_t eval(sharedT shrMem, cl::sycl::nd_item<1> ndItem);
  void bind(cl::sycl::handler &h);
};

// template <typename lhs_t, typename matrix_t,typename vector_t>
template <bool Lower = true, bool Diag = true, bool Upper = true,
          bool Unit = false, typename lhs_t, typename matrix_t,
          typename vector_t>
GemvCol<Lower, Diag, Upper, Unit, lhs_t, matrix_t, vector_t> make_Gemv_Col(
    lhs_t &lhs_, matrix_t &matrix_, vector_t &vector_,
    typename vector_t::index_t nWG_row_, typename vector_t::index_t nWG_col_,
    typename vector_t::index_t local_memory_size_) {
  return GemvCol<Lower, Diag, Upper, Unit, lhs_t, matrix_t, vector_t>(
      lhs_, matrix_, vector_, nWG_row_, nWG_col_, local_memory_size_);
}

/**
 * @struct GemvRow
 * @brief Tree node representing a row-based/row-parallel generalised matrix_
 * vector_ multiplication.
 */
template <int interLoop, bool Lower, bool Diag, bool Upper, bool Unit,
          typename lhs_t, typename matrix_t, typename vector_t>
struct GemvRow {
  using value_t = typename vector_t::value_t;
  using index_t = typename vector_t::index_t;

  lhs_t lhs_;
  matrix_t matrix_;
  vector_t vector_;
  index_t nWG_row_;
  index_t nWG_col_;
  index_t local_memory_size_;

  GemvRow(lhs_t &_l, matrix_t &_matrix, vector_t &_vector, index_t &_nWG_row,
          index_t &_nWG_col, index_t &_shrMemSize);
  index_t get_size() const;
  bool valid_thread(cl::sycl::nd_item<1> ndItem) const;
  value_t eval(index_t i);
  value_t eval(cl::sycl::nd_item<1> ndItem);
  template <typename sharedT>
  value_t eval(sharedT shrMem, cl::sycl::nd_item<1> ndItem);
  void bind(cl::sycl::handler &h);
};
/*!
 @brief Generator/factory for row-based GEMV trees.

 make_Gemv_Row(
    lhs_t &lhs_,
     matrix_t &matrix_,
    vector_t &vector_,
    typename vector_t::index_t nWG_row_,
    typename vector_t::index_t nWG_col_,
    typename vector_t::index_t local_memory_size_
 )
 */
template <int interLoop = 1, bool Lower = true, bool Diag = true,
          bool Upper = true, bool Unit = false, typename lhs_t,
          typename matrix_t, typename vector_t>
GemvRow<interLoop, Lower, Diag, Upper, Unit, lhs_t, matrix_t, vector_t>
make_Gemv_Row(lhs_t &lhs_, matrix_t &matrix_, vector_t &vector_,
              typename vector_t::index_t nWG_row_,
              typename vector_t::index_t nWG_col_,
              typename vector_t::index_t local_memory_size_) {
  return GemvRow<interLoop, Lower, Diag, Upper, Unit, lhs_t, matrix_t,
                 vector_t>(lhs_, matrix_, vector_, nWG_row_, nWG_col_,
                           local_memory_size_);
}

/**** GER BY ROWS M ROWS x N BLOCK USING PROPERLY THE SHARED MEMORY ****/
// template <typename lhs_t,typename rhs_1_t,typename rhs_2_t>
template <bool Single, bool Lower, bool Diag, bool Upper, typename lhs_t,
          typename rhs_1_t, typename rhs_2_t>
struct GerRow {
  using value_t = typename rhs_2_t::value_t;
  using index_t = typename rhs_2_t::index_t;
  lhs_t lhs_;
  rhs_1_t rhs_1_;
  rhs_2_t rhs_2_;
  index_t nWG_row_;
  index_t nWG_col_;
  index_t local_memory_size_;
  value_t scalar_;
  GerRow(lhs_t &_l, value_t _scl, rhs_1_t &_r1, rhs_2_t &_r2, index_t &_nWG_row,
         index_t &_nWG_col, index_t &_shrMemSize);
  index_t get_size() const;
  bool valid_thread(cl::sycl::nd_item<1> ndItem) const;
  value_t eval(index_t i);
  value_t eval(cl::sycl::nd_item<1> ndItem);
  template <typename sharedT>
  value_t eval(sharedT shrMem, cl::sycl::nd_item<1> ndItem);
  void bind(cl::sycl::handler &h);
};

template <bool Single = true, bool Lower = true, bool Diag = true,
          bool Upper = true, typename lhs_t, typename rhs_1_t, typename rhs_2_t>
GerRow<Single, Lower, Diag, Upper, lhs_t, rhs_1_t, rhs_2_t> make_Ger_Row(
    lhs_t &lhs_, typename lhs_t::value_t scalar_, rhs_1_t &rhs_1_,
    rhs_2_t &rhs_2_, typename rhs_2_t::index_t nWG_row_,
    typename rhs_2_t::index_t nWG_col_,
    typename rhs_2_t::index_t local_memory_size_) {
  return GerRow<Single, Lower, Diag, Upper, lhs_t, rhs_1_t, rhs_2_t>(
      lhs_, scalar_, rhs_1_, rhs_2_, nWG_row_, nWG_col_, local_memory_size_);
}

/**** GER BY COLUMNS M ROWS x N BLOCK USING PROPERLY THE SHARED MEMORY ****/
// template <typename lhs_t,typename rhs_1_t,typename rhs_2_t>
template <bool Single, bool Lower, bool Diag, bool Upper, typename lhs_t,
          typename rhs_1_t, typename rhs_2_t>
struct GerCol {
  using value_t = typename rhs_2_t::value_t;
  using index_t = typename rhs_2_t::index_t;

  lhs_t lhs_;
  rhs_1_t rhs_1_;
  rhs_2_t rhs_2_;
  index_t nWG_row_;
  index_t nWG_col_;
  index_t local_memory_size_;
  value_t scalar_;

  GerCol(lhs_t &_l, value_t _scl, rhs_1_t &_r1, rhs_2_t &_r2, index_t &_nWG_row,
         index_t &_nWG_col, index_t &_shrMemSize);
  index_t get_size() const;
  bool valid_thread(cl::sycl::nd_item<1> ndItem) const;
  value_t eval(index_t i);
  value_t eval(cl::sycl::nd_item<1> ndItem);
  template <typename sharedT>
  value_t eval(sharedT shrMem, cl::sycl::nd_item<1> ndItem);
  void bind(cl::sycl::handler &h);
};

// template <typename lhs_t,typename rhs_1_t,typename rhs_2_t>
template <bool Single = true, bool Lower = true, bool Diag = true,
          bool Upper = true, typename lhs_t, typename rhs_1_t, typename rhs_2_t>
GerCol<Single, Lower, Diag, Upper, lhs_t, rhs_1_t, rhs_2_t> make_Ger_Col(
    lhs_t &lhs_, typename lhs_t::value_t scalar_, rhs_1_t &rhs_1_,
    rhs_2_t &rhs_2_, typename rhs_2_t::index_t nWG_row_,
    typename rhs_2_t::index_t nWG_col_,
    typename rhs_2_t::index_t local_memory_size_) {
  return GerCol<Single, Lower, Diag, Upper, lhs_t, rhs_1_t, rhs_2_t>(
      lhs_, scalar_, rhs_1_, rhs_2_, nWG_row_, nWG_col_, local_memory_size_);
}
}  // namespace blas
#endif  // BLAS2_TREES_H
