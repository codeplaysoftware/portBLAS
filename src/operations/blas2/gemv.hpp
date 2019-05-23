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
 *  @filename gemv.hpp
 *
 **************************************************************************/

#ifndef GEMV_HPP
#define GEMV_HPP

#include "../blas_operators.hpp"
#include "operations/blas2_trees.h"
#include "views/view_sycl.hpp"
#include <stdexcept>
#include <vector>

namespace blas {

/**
 * @struct AddSetColumns
 * @brief Tree node representing a column sum (reduction?) - i.e. summing a row,
 * with one row per thread
 */
template <typename rhs_t>
AddSetColumns<rhs_t>::AddSetColumns(rhs_t &_r) : rhs_(_r) {}

template <typename rhs_t>
SYCL_BLAS_INLINE typename AddSetColumns<rhs_t>::index_t
AddSetColumns<rhs_t>::get_size() const {
  return rhs_.get_size_row();
}
template <typename rhs_t>
SYCL_BLAS_INLINE bool AddSetColumns<rhs_t>::valid_thread(
    cl::sycl::nd_item<1> ndItem) const {
  return ((ndItem.get_global_id(0) < get_size()));
}
template <typename rhs_t>
SYCL_BLAS_INLINE typename AddSetColumns<rhs_t>::value_t
AddSetColumns<rhs_t>::eval(typename AddSetColumns<rhs_t>::index_t i) {
  auto dimR = rhs_.get_size_row();
  auto dimC = rhs_.get_size_col();

  auto val = AdditionIdentity::eval(rhs_.eval(0));
  if (i < dimR) {
    for (typename AddSetColumns<rhs_t>::index_t j = 0; j < dimC; j++) {
      val += rhs_.eval(i, j);
    }
  }
  return val;
}

template <typename rhs_t>
SYCL_BLAS_INLINE typename AddSetColumns<rhs_t>::value_t
AddSetColumns<rhs_t>::eval(cl::sycl::nd_item<1> ndItem) {
  return eval(ndItem.get_global_id(0));
}
template <typename rhs_t>
SYCL_BLAS_INLINE void AddSetColumns<rhs_t>::bind(cl::sycl::handler &h) {
  rhs_.bind(h);
}

/**** GEMV BY ROWS M ROWS x N BLOCK ****/
/**
 * @struct GemvRow
 * @brief Tree node representing a row-based/row-parallel generalised matrix_
 * vector_ multiplication.
 */
template <int interLoop, bool Lower, bool Diag, bool Upper, bool Unit,
          typename lhs_t, typename matrix_t, typename vector_t>
SYCL_BLAS_INLINE
GemvRow<interLoop, Lower, Diag, Upper, Unit, lhs_t, matrix_t, vector_t>::
    GemvRow(lhs_t &_l, matrix_t &_matrix, vector_t &_vector,
            typename GemvRow<interLoop, Lower, Diag, Upper, Unit, lhs_t,
                             matrix_t, vector_t>::index_t &_nWG_row,
            typename GemvRow<interLoop, Lower, Diag, Upper, Unit, lhs_t,
                             matrix_t, vector_t>::index_t &_nWG_col,
            typename GemvRow<interLoop, Lower, Diag, Upper, Unit, lhs_t,
                             matrix_t, vector_t>::index_t &_shrMemSize)
    : lhs_(_l),
      matrix_(_matrix),
      vector_(_vector),
      nWG_row_(_nWG_row),
      nWG_col_(_nWG_col),
      local_memory_size_(_shrMemSize) {}

template <int interLoop, bool Lower, bool Diag, bool Upper, bool Unit,
          typename lhs_t, typename matrix_t, typename vector_t>
SYCL_BLAS_INLINE typename GemvRow<interLoop, Lower, Diag, Upper, Unit, lhs_t,
                                  matrix_t, vector_t>::index_t
GemvRow<interLoop, Lower, Diag, Upper, Unit, lhs_t, matrix_t,
        vector_t>::get_size() const {
  return matrix_.get_size();
}
template <int interLoop, bool Lower, bool Diag, bool Upper, bool Unit,
          typename lhs_t, typename matrix_t, typename vector_t>
SYCL_BLAS_INLINE bool
GemvRow<interLoop, Lower, Diag, Upper, Unit, lhs_t, matrix_t,
        vector_t>::valid_thread(cl::sycl::nd_item<1> ndItem) const {
  return true;
}

// TODO (@JOSE) If this function is extra and it is not required please remove
// it.
template <int interLoop, bool Lower, bool Diag, bool Upper, bool Unit,
          typename lhs_t, typename matrix_t, typename vector_t>
SYCL_BLAS_INLINE typename GemvRow<interLoop, Lower, Diag, Upper, Unit, lhs_t,
                                  matrix_t, vector_t>::value_t
GemvRow<interLoop, Lower, Diag, Upper, Unit, lhs_t, matrix_t, vector_t>::eval(
    typename GemvRow<interLoop, Lower, Diag, Upper, Unit, lhs_t, matrix_t,
                     vector_t>::index_t i) {  // NOT VERIFIED
  auto dim = vector_.get_size();

  auto val = AdditionIdentity::eval(vector_.eval(0));
  for (typename GemvRow<interLoop, Lower, Diag, Upper, Unit, lhs_t, matrix_t,
                        vector_t>::index_t j = 0;
       j < dim; j++) {
    auto prod = ProductOperator::eval(matrix_.eval(i, j), vector_.eval(j));
    val = AddOperator::eval(val, prod);
  }
  return lhs_.eval(i) = val;
}

template <int interLoop, bool Lower, bool Diag, bool Upper, bool Unit,
          typename lhs_t, typename matrix_t, typename vector_t>
SYCL_BLAS_INLINE typename GemvRow<interLoop, Lower, Diag, Upper, Unit, lhs_t,
                                  matrix_t, vector_t>::value_t
GemvRow<interLoop, Lower, Diag, Upper, Unit, lhs_t, matrix_t, vector_t>::eval(
    cl::sycl::nd_item<1> ndItem) {
  using index_t = typename GemvRow<interLoop, Lower, Diag, Upper, Unit, lhs_t,
                                   matrix_t, vector_t>::index_t;
  index_t localid = ndItem.get_local_id(0);
  index_t localSz = ndItem.get_local_range(0);
  index_t groupid = ndItem.get_group(0);

  // Get the number of rows of the matrix_
  index_t dimR = matrix_.get_size_row();
  //
  index_t dimC = matrix_.get_size_col();

  index_t rowSz = (dimR + nWG_row_ - 1) / nWG_row_;

  index_t idWFR = groupid / nWG_col_;  // row bloq id of the current workgroup
  index_t idWFC = groupid % nWG_col_;  // col blq id of the current workgroup

  index_t dimWFC =
      ((dimC + (localSz * nWG_col_) - 1) / (localSz * nWG_col_)) * localSz;

  index_t frs_row = idWFR * rowSz;
  index_t lst_row = std::min(dimR, frs_row + rowSz);

  index_t frs_col = idWFC * dimWFC + interLoop * localid;
  index_t lst_col = std::min(dimC, frs_col + dimWFC);

  index_t id_col_thr = idWFC * localSz + localid;

  typedef typename GemvRow<interLoop, Lower, Diag, Upper, Unit, lhs_t, matrix_t,
                           vector_t>::value_t data_value_t;
  static constexpr data_value_t init_val =
      AddOperator::template init<vector_t>();
  data_value_t val = init_val;

  // PROBLEM IF ONLY SOME THREADS OF A WORKGROUP_OF ARE CANCELED
  // TO SOLVE IT, USE GLOBAL VALUES OF frs_col AND lst_col
  if ((!Upper && (((idWFC * dimWFC) + ((!Diag) ? 1 : 0)) > (lst_row - 1))) ||
      (!Lower &&
       ((frs_row + ((!Diag) ? 1 : 0)) > ((idWFC * dimWFC + dimWFC) - 1)))) {
    for (index_t rowid = frs_row; rowid < lst_row; rowid += localSz) {
      lhs_.eval(rowid, id_col_thr) = val;
    }
  } else {
    if (interLoop == 1) {
      if (id_col_thr < dimC) {
        for (index_t row = 0, id_row = frs_row; (id_row < lst_row);
             row++, id_row++) {
          val = init_val;
          for (index_t id_col = frs_col; id_col < lst_col; id_col += localSz) {
            if (Lower && Upper && Diag && !Unit) {
              auto prod = ProductOperator::eval(matrix_.eval(id_row, id_col),
                                                vector_.eval(id_col));
              val = AddOperator::eval(val, prod);
            } else {
              if ((Lower && ((id_col + ((!Diag || Unit) ? 1 : 0)) <= id_row)) ||
                  (Upper && (id_col >= (id_row + ((!Diag || Unit) ? 1 : 0))))) {
                auto prod = ProductOperator::eval(matrix_.eval(id_row, id_col),
                                                  vector_.eval(id_col));
                val = AddOperator::eval(val, prod);
              }
              if (Diag && Unit && (id_row == id_col)) {
                val = AddOperator::eval(val, matrix_.eval(id_row, id_col));
              }
            }
          }
          lhs_.eval(id_row, id_col_thr) = val;
        }
      }
    } else {
      // There's an implied question mark after each of these comments.
      // They are just attempts to understand the code!
      // Iterate over rows of the matrix_
      for (index_t row = 0, id_row = frs_row; (id_row < lst_row);
           row++, id_row++) {
        // initialise an add node, with vector_, the vector_
        // we need to initialise it, as otherwise the type will change during
        // execution!
        val = init_val;
        // Iterate across blocks of columns, in chunks of localSz * interLoop
        for (index_t id_col = frs_col; id_col < lst_col;
             id_col += localSz * interLoop) {
          // If the row length isn't a multiple of localSz * interLoop
          // we need to go for fewer columns. Pick the min.
          auto lst_k_int = std::min(id_col + interLoop, lst_col);
          // Handle lower diagonal etc
          for (index_t k_int =
                   ((Lower)
                        ? id_col
                        : std::max(row + ((!Diag || Unit) ? 1 : 0), id_col));
               k_int <
               ((Upper) ? lst_k_int
                        : std::min(row + ((!Diag || Unit) ? 0 : 1), lst_k_int));
               k_int++) {
            // calculate the product between the row and the vector_.
            auto prod = ProductOperator::eval(matrix_.eval(id_row, k_int),
                                              vector_.eval(k_int));
            // add that to val?
            // Reassignment!
            val = AddOperator::eval(val, prod);
          }
        }
        // what does this do?
        lhs_.eval(id_row, id_col_thr) = val;
      }
    }
  }
  return val;
}

/*
  Evaluate using shared memory.
*/

template <int interLoop, bool Lower, bool Diag, bool Upper, bool Unit,
          typename lhs_t, typename matrix_t, typename vector_t>
template <typename local_memory_t>
SYCL_BLAS_INLINE typename GemvRow<interLoop, Lower, Diag, Upper, Unit, lhs_t,
                                  matrix_t, vector_t>::value_t
GemvRow<interLoop, Lower, Diag, Upper, Unit, lhs_t, matrix_t, vector_t>::eval(
    local_memory_t shrMem, cl::sycl::nd_item<1> ndItem) {
  using index_t = typename GemvRow<interLoop, Lower, Diag, Upper, Unit, lhs_t,
                                   matrix_t, vector_t>::index_t;
  using value_t = typename GemvRow<interLoop, Lower, Diag, Upper, Unit, lhs_t,
                                   matrix_t, vector_t>::value_t;
  index_t localid = ndItem.get_local_id(0);
  index_t localSz = ndItem.get_local_range(0);
  index_t groupid = ndItem.get_group(0);

  // Get the dimensions of the row and column
  index_t dimR = matrix_.get_size_row();
  index_t dimC = matrix_.get_size_col();

  //
  index_t rowSz = (dimR + nWG_row_ - 1) / nWG_row_;
  index_t shrSz = local_memory_size_ / localSz;

  index_t idWFR = groupid / nWG_col_;  // row bloq id of the current workgroup
  index_t idWFC = groupid % nWG_col_;  // col blq id of the current workgroup
  index_t dimWFC =
      ((dimC + (localSz * nWG_col_) - 1) / (localSz * nWG_col_)) * localSz;

  index_t frs_row = idWFR * rowSz;
  index_t lst_row = std::min(dimR, frs_row + rowSz);

  index_t frs_col = idWFC * dimWFC + interLoop * localid;
  index_t lst_col = std::min(dimC, frs_col + dimWFC);
  static constexpr value_t init_val = AddOperator::template init<vector_t>();
  // PROBLEM IF ONLY SOME THREADS OF A WORKGROUP ARE CANCELED
  // TO SOLVE IT, USE GLOBAL VALUES OF frs_col AND lst_col
  if ((!Upper && (((idWFC * dimWFC) + ((!Diag) ? 1 : 0)) > (lst_row - 1))) ||
      (!Lower &&
       ((frs_row + ((!Diag) ? 1 : 0)) > ((idWFC * dimWFC + dimWFC) - 1)))) {
    if (localid == 0) {
      value_t val = AdditionIdentity::eval(vector_.eval(0));
      for (index_t rowid = frs_row; rowid < lst_row; rowid++) {
        lhs_.eval(rowid, idWFC) = val;
      }
    }
  } else {
    for (index_t rowid = frs_row; rowid < lst_row; rowid += shrSz) {
      value_t val = init_val;
      auto blqSz = std::min(shrSz, lst_row - rowid);
      if (interLoop == 1) {
        for (index_t row = 0, id_row = rowid; row < blqSz; row++, id_row++) {
          val = (Diag && Unit &&
                 ((id_row >= frs_col) && (id_row < lst_col) &&
                  (((id_row - frs_col) % localSz) == 0)))
                    ? vector_.eval(id_row)
                    : init_val;
          for (index_t id_col = frs_col; id_col < lst_col; id_col += localSz) {
            if (Lower && Upper && Diag && !Unit) {
              auto prod = ProductOperator::eval(matrix_.eval(id_row, id_col),
                                                vector_.eval(id_col));
              val = AddOperator::eval(val, prod);
            } else {
              if ((Lower && ((id_col + ((!Diag || Unit) ? 1 : 0)) <= id_row)) ||
                  (Upper && (id_col >= (id_row + ((!Diag || Unit) ? 1 : 0))))) {
                auto prod = ProductOperator::eval(matrix_.eval(id_row, id_col),
                                                  vector_.eval(id_col));
                val = AddOperator::eval(val, prod);
              }
            }
          }
          shrMem[row * localSz + localid] = val;
        }
      } else {
        for (index_t row = 0, id_row = rowid; row < blqSz; row++, id_row++) {
          val = init_val;
          for (index_t id_col = frs_col; id_col < lst_col;
               id_col += localSz * interLoop) {
            for (index_t k_int = id_col;
                 k_int < std::min(id_col + interLoop, lst_col); k_int++) {
              if (Lower && Upper && Diag && !Unit) {
                auto prod = ProductOperator::eval(matrix_.eval(id_row, k_int),
                                                  vector_.eval(k_int));
                val = AddOperator::eval(val, prod);
              } else {
                if ((Lower &&
                     ((id_col + ((!Diag || Unit) ? 1 : 0)) <= id_row)) ||
                    (Upper &&
                     (id_col >= (id_row + ((!Diag || Unit) ? 1 : 0))))) {
                  auto prod = ProductOperator::eval(matrix_.eval(id_row, k_int),
                                                    vector_.eval(k_int));
                  val = AddOperator::eval(val, prod);
                }
                if (Diag && Unit && (id_row == id_col)) {
                  val = AddOperator::eval(val, matrix_.eval(id_row, k_int));
                }
              }
            }
          }
          shrMem[row * localSz + localid] = val;
        }
      }

      // This barrier is mandatory to be sure the data is on the shared memory
      ndItem.barrier(cl::sycl::access::fence_space::local_space);
      // Reduction inside the block
      for (index_t offset = localSz >> 1; offset > 0; offset >>= 1) {
        if (localid < offset) {
          for (index_t row = 0, id_row = rowid; row < blqSz; row++, id_row++) {
            shrMem[row * localSz + localid] =
                AddOperator::eval(shrMem[row * localSz + localid],
                                  shrMem[row * localSz + localid + offset]);
          }
        }
        // This barrier is mandatory to be sure the data are on the shared
        // memory
        ndItem.barrier(cl::sycl::access::fence_space::local_space);
      }
      if (localid == 0) {
        for (index_t row = 0, id_row = rowid; row < blqSz; row++, id_row++) {
          lhs_.eval(id_row, idWFC) = shrMem[row * localSz];
        }
      }
    }
  }

  return init_val;
}
template <int interLoop, bool Lower, bool Diag, bool Upper, bool Unit,
          typename lhs_t, typename matrix_t, typename vector_t>
SYCL_BLAS_INLINE void GemvRow<interLoop, Lower, Diag, Upper, Unit, lhs_t,
                              matrix_t, vector_t>::bind(cl::sycl::handler &h) {
  lhs_.bind(h);
  matrix_.bind(h);
  vector_.bind(h);
}

/**** GEMV BY COLUMNS 1 ROW x M BLOCKS USING PROPERLY THE SHARED MEMORY ****/

/**
 * @struct GemvCol
 * @brief Tree node representing a Gemv, with parallel expressed across columns
 * *
 */
template <bool Lower, bool Diag, bool Upper, bool Unit, typename lhs_t,
          typename matrix_t, typename vector_t>
SYCL_BLAS_INLINE
GemvCol<Lower, Diag, Upper, Unit, lhs_t, matrix_t, vector_t>::GemvCol(
    lhs_t &_l, matrix_t &_matrix, vector_t &_vector,
    typename GemvCol<Lower, Diag, Upper, Unit, lhs_t, matrix_t,
                     vector_t>::index_t &_nWG_row,
    typename GemvCol<Lower, Diag, Upper, Unit, lhs_t, matrix_t,
                     vector_t>::index_t &_nWG_col,
    typename GemvCol<Lower, Diag, Upper, Unit, lhs_t, matrix_t,
                     vector_t>::index_t &_shrMemSize)
    : lhs_(_l),
      matrix_(_matrix),
      vector_(_vector),
      nWG_row_(_nWG_row),
      nWG_col_(_nWG_col),
      local_memory_size_(_shrMemSize){};
template <bool Lower, bool Diag, bool Upper, bool Unit, typename lhs_t,
          typename matrix_t, typename vector_t>
SYCL_BLAS_INLINE typename GemvCol<Lower, Diag, Upper, Unit, lhs_t, matrix_t,
                                  vector_t>::index_t
GemvCol<Lower, Diag, Upper, Unit, lhs_t, matrix_t, vector_t>::get_size() const {
  return matrix_.get_size_row();
}
template <bool Lower, bool Diag, bool Upper, bool Unit, typename lhs_t,
          typename matrix_t, typename vector_t>
SYCL_BLAS_INLINE bool
GemvCol<Lower, Diag, Upper, Unit, lhs_t, matrix_t, vector_t>::valid_thread(
    cl::sycl::nd_item<1> ndItem) const {
  return true;
}

template <bool Lower, bool Diag, bool Upper, bool Unit, typename lhs_t,
          typename matrix_t, typename vector_t>
SYCL_BLAS_INLINE typename GemvCol<Lower, Diag, Upper, Unit, lhs_t, matrix_t,
                                  vector_t>::value_t
GemvCol<Lower, Diag, Upper, Unit, lhs_t, matrix_t, vector_t>::eval(index_t i) {
  auto dim = vector_.get_size();

  auto val = AdditionIdentity::eval(vector_.eval(0));
  for (index_t j = 0; j < dim; j++) {
    auto prod = ProductOperator::eval(matrix_.eval(i, j), vector_.eval(j));
    val = AddOperator::eval(val, prod);
  }
  return lhs_.eval(i) = val;
}

template <bool Lower, bool Diag, bool Upper, bool Unit, typename lhs_t,
          typename matrix_t, typename vector_t>
SYCL_BLAS_INLINE typename GemvCol<Lower, Diag, Upper, Unit, lhs_t, matrix_t,
                                  vector_t>::value_t
GemvCol<Lower, Diag, Upper, Unit, lhs_t, matrix_t, vector_t>::eval(
    cl::sycl::nd_item<1> ndItem) {
  using index_t = typename GemvCol<Lower, Diag, Upper, Unit, lhs_t, matrix_t,
                                   vector_t>::index_t;
  index_t localid = ndItem.get_local_id(0);
  index_t localSz = ndItem.get_local_range(0);
  index_t groupid = ndItem.get_group(0);

  index_t dimR = matrix_.get_size_row();
  index_t dimC = matrix_.get_size_col();
  index_t colSz = (dimC + nWG_col_ - 1) / nWG_col_;

  index_t idWFR = (groupid % nWG_row_);
  index_t idWFC = (groupid / nWG_row_);
  index_t dimWFR =
      (dimR + (localSz * nWG_row_) - 1) / (localSz * nWG_row_) * localSz;

  index_t frs_row = idWFR * dimWFR + localid;
  index_t lst_row = std::min(dimR, frs_row + dimWFR);

  index_t frs_col = idWFC * colSz;
  index_t lst_col = std::min(dimC, frs_col + colSz);
  // PROBLEM IF ONLY SOME THREADS OF A WORKGROUP ARE CANCELED
  // TO SOLVE IT, USE GLOBAL VALUES OF frs_row AND lst_row
  if ((!Upper &&
       ((frs_col + ((!Diag) ? 1 : 0)) > ((idWFR * dimWFR + dimWFR) - 1))) ||
      (!Lower && ((idWFR * dimWFR + ((!Diag) ? 1 : 0)) > (lst_col - 1)))) {
    auto val = AdditionIdentity::eval(vector_.eval(0));
    for (index_t rowid = frs_row; rowid < lst_row; rowid += localSz) {
      lhs_.eval(rowid, idWFC) = val;
    }
  } else {
    // The product is computed
    for (index_t rowid = frs_row; rowid < lst_row; rowid += localSz) {
      // The initial value of val is different for the first iteration
      auto val = (Diag && Unit && ((rowid >= frs_col) && (rowid < lst_col)))
                     ? matrix_.eval(rowid, rowid)
                     : AdditionIdentity::eval(vector_.eval(0));
      for (index_t id_col =
               ((Lower) ? frs_col
                        : std::max(rowid + ((!Diag || Unit) ? 1 : 0), frs_col));
           id_col <
           ((Upper) ? lst_col
                    : std::min(rowid + ((!Diag || Unit) ? 0 : 1), lst_col));
           id_col++) {
        auto prod = ProductOperator::eval(matrix_.eval(rowid, id_col),
                                          vector_.eval(id_col));
        val = AddOperator::eval(val, prod);
      }
      // The result is stored in the correct component
      lhs_.eval(rowid, idWFC) = val;
    }
  }

  return lhs_.eval(frs_row, idWFC);
}

template <bool Lower, bool Diag, bool Upper, bool Unit, typename lhs_t,
          typename matrix_t, typename vector_t>
template <typename local_memory_t>
SYCL_BLAS_INLINE typename GemvCol<Lower, Diag, Upper, Unit, lhs_t, matrix_t,
                                  vector_t>::value_t
GemvCol<Lower, Diag, Upper, Unit, lhs_t, matrix_t, vector_t>::eval(
    local_memory_t shrMem, cl::sycl::nd_item<1> ndItem) {
  using index_t = typename GemvCol<Lower, Diag, Upper, Unit, lhs_t, matrix_t,
                                   vector_t>::index_t;
  index_t localid = ndItem.get_local_id(0);
  index_t localSz = ndItem.get_local_range(0);
  index_t groupid = ndItem.get_group(0);

  index_t dimR = matrix_.get_size_row();
  index_t dimC = matrix_.get_size_col();

  index_t colSz = (dimC + nWG_col_ - 1) / nWG_col_;
  index_t idWFR = (groupid % nWG_row_);
  index_t idWFC = (groupid / nWG_row_);
  index_t dimWFR =
      (dimR + (localSz * nWG_row_) - 1) / (localSz * nWG_row_) * localSz;

  index_t frs_row = idWFR * dimWFR + localid;
  index_t lst_row = std::min(dimR, frs_row + dimWFR);

  index_t frs_col = idWFC * colSz;
  index_t lst_col = std::min(dimC, frs_col + colSz);

  // PROBLEM IF ONLY SOME THREADS OF A WORKGROUP ARE CANCELED
  // TO SOLVE IT, USE GLOBAL VALUES OF frs_row AND lst_row
  if ((!Upper &&
       ((frs_col + ((!Diag) ? 1 : 0)) > ((idWFR * dimWFR + dimWFR) - 1))) ||
      (!Lower && ((idWFR * dimWFR + ((!Diag) ? 1 : 0)) > (lst_col - 1)))) {
    auto val = AdditionIdentity::eval(vector_.eval(0));
    for (index_t rowid = frs_row; rowid < lst_row; rowid += localSz) {
      lhs_.eval(rowid, idWFC) = val;
    }
  } else {
    // The computation are made in blocks of local_memory_size_ elements
    for (index_t colid = frs_col; colid < lst_col;
         colid += local_memory_size_) {
      if (colid > frs_col) {
        // This barrier is mandatory to be sure the data is on the shared
        // memory
        ndItem.barrier(cl::sycl::access::fence_space::local_space);
      }
      auto blqSz = std::min(local_memory_size_, lst_col - colid);
      // Copy a block of elements of vector_ vector_ to the shared memory,
      // executing the expresion tree if it is needed
      for (index_t col = localid; (col < blqSz); col += localSz) {
        shrMem[col] = vector_.eval(colid + col);
      }
      // This barrier is mandatory to be sure the data is on the shared memory
      ndItem.barrier(cl::sycl::access::fence_space::local_space);

      // The product is computed
      for (index_t rowid = frs_row; rowid < lst_row; rowid += localSz) {
        // The initial value of val is different for the first iteration
        auto val =
            ((colid == frs_col) ? AdditionIdentity::eval(vector_.eval(0))
                                : lhs_.eval(rowid, idWFC)) +
            ((Diag && Unit && ((rowid >= colid) && (rowid < colid + blqSz)))
                 ? vector_.eval(rowid)
                 : AdditionIdentity::eval(vector_.eval(0)));
        for (index_t id_col = colid, col = 0; col < blqSz; id_col++, col++) {
          if (Lower && Upper && Diag && !Unit) {
            auto prod =
                ProductOperator::eval(matrix_.eval(rowid, id_col), shrMem[col]);
            val = AddOperator::eval(val, prod);
          } else {
            if ((Lower && ((id_col + ((!Diag || Unit) ? 1 : 0)) <= rowid)) ||
                (Upper && (id_col >= (rowid + ((!Diag || Unit) ? 1 : 0))))) {
              auto prod = ProductOperator::eval(matrix_.eval(rowid, id_col),
                                                shrMem[col]);
              val = AddOperator::eval(val, prod);
            }
          }
        }
        // The result is stored in the correct component
        lhs_.eval(rowid, idWFC) = val;
      }
    }
  }
  return lhs_.eval(frs_row, idWFC);
}
template <bool Lower, bool Diag, bool Upper, bool Unit, typename lhs_t,
          typename matrix_t, typename vector_t>
SYCL_BLAS_INLINE void GemvCol<Lower, Diag, Upper, Unit, lhs_t, matrix_t,
                              vector_t>::bind(cl::sycl::handler &h) {
  lhs_.bind(h);
  matrix_.bind(h);
  vector_.bind(h);
}

}  // namespace blas
#endif
