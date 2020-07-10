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
#include "operations/blas2_trees.h"
#include "operations/blas_operators.hpp"
#include "views/view_sycl.hpp"
#include <stdexcept>
#include <vector>
namespace blas {

/**
 * @struct SumMatrixColumns
 * @brief Tree node representing a column sum with one row per thread
 */
template <typename rhs_t>
SumMatrixColumns<rhs_t>::SumMatrixColumns(rhs_t &_r) : rhs_(_r) {}

template <typename rhs_t>
SYCL_BLAS_INLINE typename SumMatrixColumns<rhs_t>::index_t
SumMatrixColumns<rhs_t>::get_size() const {
  return rhs_.get_size_row();
}

template <typename rhs_t>
SYCL_BLAS_INLINE bool SumMatrixColumns<rhs_t>::valid_thread(
    cl::sycl::nd_item<1> ndItem) const {
  return ((ndItem.get_global_id(0) < get_size()));
}

template <typename rhs_t>
SYCL_BLAS_INLINE typename SumMatrixColumns<rhs_t>::value_t
SumMatrixColumns<rhs_t>::eval(typename SumMatrixColumns<rhs_t>::index_t i) {
  auto dimR = rhs_.get_size_row();
  auto dimC = rhs_.get_size_col();

  auto val = AdditionIdentity::eval(rhs_.eval(0));
  if (i < dimR) {
    for (typename SumMatrixColumns<rhs_t>::index_t j = 0; j < dimC; j++) {
      val += rhs_.eval(i, j);
    }
  }
  return val;
}

template <typename rhs_t>
SYCL_BLAS_INLINE typename SumMatrixColumns<rhs_t>::value_t
SumMatrixColumns<rhs_t>::eval(cl::sycl::nd_item<1> ndItem) {
  return eval(ndItem.get_global_id(0));
}

template <typename rhs_t>
SYCL_BLAS_INLINE void SumMatrixColumns<rhs_t>::bind(cl::sycl::handler &h) {
  rhs_.bind(h);
}

template <typename rhs_t>
SYCL_BLAS_INLINE void SumMatrixColumns<rhs_t>::adjust_access_displacement() {
  rhs_.adjust_access_displacement();
}

/*!
 * @brief Constructor for the Gemv class. See blas2_trees.h for details on the
 * parameters.
 */
template <typename lhs_t, typename matrix_t, typename vector_t,
          uint32_t local_range, bool is_transposed, int cache_line_size,
          int work_per_thread>
SYCL_BLAS_INLINE
Gemv<lhs_t, matrix_t, vector_t, local_range, is_transposed, cache_line_size,
     work_per_thread>::Gemv(lhs_t &_l, matrix_t &_matrix_a, vector_t &_vector_x,
                            typename vector_t::index_t &_wgs_per_nc,
                            typename vector_t::index_t &_wgs_per_c)
    : lhs_(_l),              // Result is stored in this
      matrix_a_(_matrix_a),  // Input matrix a
      vector_x_(_vector_x),  // Input vector x
      wgs_per_nc_(
          _wgs_per_nc),  // number of work groups per non-contracting dimension
      wgs_per_c_(_wgs_per_c)  // number of work groups per contracting dimension
{}

/*!
 * @brief Tells the runtime whether a work item "ndItem" should execute. We
 * handle this in the kernel itself so always return true
 */
template <typename lhs_t, typename matrix_t, typename vector_t,
          uint32_t local_range, bool is_transposed, int cache_line_size,
          int work_per_thread>
SYCL_BLAS_INLINE bool
Gemv<lhs_t, matrix_t, vector_t, local_range, is_transposed, cache_line_size,
     work_per_thread>::valid_thread(cl::sycl::nd_item<1> ndItem) const {
  return true;
}

/*!
 * @brief The no-shared-memory version of the GEMV kernel. Currently this is
 * only executed with one work group.
 *
 * TODO: in the future is to make this kernel more like the shared-memory
 * version and use multiple
 */
template <typename lhs_t, typename matrix_t, typename vector_t,
          uint32_t local_range, bool is_transposed, int cache_line_size,
          int work_per_thread>
SYCL_BLAS_INLINE
    typename Gemv<lhs_t, matrix_t, vector_t, local_range, is_transposed,
                  cache_line_size, work_per_thread>::value_t
    Gemv<lhs_t, matrix_t, vector_t, local_range, is_transposed, cache_line_size,
         work_per_thread>::eval(cl::sycl::nd_item<1> ndItem) {
  const index_t local_id = ndItem.get_local_id(0);
  const index_t group_id = ndItem.get_group(0);
  const index_t group_range = ndItem.get_group_range(0);
  const index_t contract_dim =
      is_transposed ? matrix_a_.get_size_row() : matrix_a_.get_size_col();
  const index_t non_contract_dim =
      is_transposed ? matrix_a_.get_size_col() : matrix_a_.get_size_row();
  const index_t group_stride = group_range * local_range;
  const index_t lda = matrix_a_.getSizeL();

  const index_t thread_id = group_id * local_range + local_id;

  const index_t non_contract_local_thread_stride = is_transposed ? lda : 1;
  const index_t contract_stride = is_transposed ? 1 : lda;
  value_t sum = 0;

  index_t non_contract_dim_index = thread_id * non_contract_local_thread_stride;
  const index_t non_contract_dim_index_stride =
      group_stride * non_contract_local_thread_stride;

  for (index_t row_id = 0; row_id < non_contract_dim; row_id += group_stride) {
    if (thread_id + row_id >= non_contract_dim) {
      return 0;
    }

    sum = 0;
    for (index_t col_id = 0; col_id < contract_dim; ++col_id) {
      sum = cl::sycl::mad(matrix_a_.template eval<true>(non_contract_dim_index),
                          vector_x_.eval(col_id), sum);
      non_contract_dim_index += contract_stride;
    }

    non_contract_dim_index += non_contract_dim_index_stride;

    lhs_.eval(thread_id + row_id) = sum;
  }

  return 0;
}

/*!
 * @brief The shared-memory version of the GEMV kernel.
 */
template <typename lhs_t, typename matrix_t, typename vector_t,
          uint32_t local_range, bool is_transposed, int cache_line_size,
          int work_per_thread>
template <typename local_memory_t>
SYCL_BLAS_INLINE
    typename Gemv<lhs_t, matrix_t, vector_t, local_range, is_transposed,
                  cache_line_size, work_per_thread>::value_t
    Gemv<lhs_t, matrix_t, vector_t, local_range, is_transposed, cache_line_size,
         work_per_thread>::eval(local_memory_t local_mem,
                                cl::sycl::nd_item<1> ndItem) {
  const index_t local_id = ndItem.get_local_id(0);
  const index_t group_id = ndItem.get_group(0);
  const index_t group_range = ndItem.get_group_range(0);
  const index_t lda = matrix_a_.getSizeL();
  const index_t nc_dim =
      is_transposed ? matrix_a_.get_size_col() : matrix_a_.get_size_row();
  const index_t c_dim =
      is_transposed ? matrix_a_.get_size_row() : matrix_a_.get_size_col();

  // ID of the group in the non-contracting dimension of the global grid of WGs
  const index_t nc_group_id = group_id / wgs_per_c_;
  // ID of the group in the contracting dimension in the global grid of WGs
  const index_t c_group_id = group_id - nc_group_id * wgs_per_c_;

  value_t *vector_scratch = local_mem.localAcc.get_pointer();

  // Threads pre-fetch portions of X into local group-shared memory
  const index_t x_vec_index = local_id + c_group_id * local_range;
  vector_scratch[local_id] = x_vec_index < vector_x_.get_size()
                                 ? vector_x_.eval(x_vec_index)
                                 : value_t{0};

  // Barrier to ensure whole portion of vector X is in local memory
  ndItem.barrier(cl::sycl::access::fence_space::local_space);

  // Non-contracting dimension index
  const index_t nc_dim_index = local_id + nc_group_id * local_range;

  // Stores the sum for the partial dot product computation
  value_t sum = 0;

  // In the non transposed case
  if (!is_transposed) {
    // Make sure nc_dim_index is within bounds
    // This closes threads that would be calculating beyond the final row
    if (nc_dim_index >= nc_dim) {
      return 0;
    }

    // Calculate the matrix index
    index_t mat_index = nc_dim_index + c_group_id * local_range * lda;

    const index_t last_c_dim_id =
        std::min(c_dim - c_group_id * local_range, local_range);

    // Computes the partial dot product for a row
    for (index_t c_dim_id = 0; c_dim_id < last_c_dim_id; ++c_dim_id) {
      sum = cl::sycl::mad(matrix_a_.template eval<true>(mat_index),
                          vector_scratch[c_dim_id], sum);
      mat_index += lda;
    }

    const index_t out_index = nc_dim_index + (c_group_id * nc_dim);
    return lhs_.eval(out_index) = sum;
  } else {  // In the transposed case

    constexpr int cl_elems = cache_line_size / sizeof(value_t);

    const int tile_c_loops = local_range / cl_elems;

    // Pointer to scratch space used for loading input matrix and transposing it
    // on-the-fly
    value_t *matrix_scratch = local_mem.localAcc.get_pointer() + local_range;

    for (index_t c_tile_id = 0; c_tile_id < tile_c_loops; ++c_tile_id) {
      // Extract a matrix block from global memory
      extract_input_block(matrix_scratch, local_id, group_id, lda, c_tile_id);

      // Ensure memory synchronization within work group
      ndItem.barrier(cl::sycl::access::fence_space::local_space);

      index_t mat_index = local_id;

#pragma unroll
      for (index_t c_dim_id = 0; c_dim_id < cl_elems; ++c_dim_id) {
        sum = cl::sycl::mad(matrix_scratch[mat_index], *vector_scratch++, sum);

        mat_index += local_range + 1;  // Adding one as bank offset
      }

      // Ensure memory synchronization within work group
      ndItem.barrier(cl::sycl::access::fence_space::local_space);
    }

    const index_t out_index = nc_dim_index + (c_group_id * nc_dim);

    if (nc_dim_index < nc_dim) {
      lhs_.eval(out_index) = sum;
    }
    return sum;
  }
}

/*!
 * @brief Extracts a block from the input matrix and transposes it on the fly,
 * placing the transposed matrix in local scratch memory
 */
template <typename lhs_t, typename matrix_t, typename vector_t,
          uint32_t local_range, bool is_transposed, int cache_line_size,
          int work_per_thread>
template <typename ScratchPointerType>
SYCL_BLAS_INLINE void
Gemv<lhs_t, matrix_t, vector_t, local_range, is_transposed, cache_line_size,
     work_per_thread>::extract_input_block(ScratchPointerType matrix_scratch,
                                           const index_t &local_id,
                                           const index_t &group_id,
                                           const index_t &lda,
                                           index_t c_tile_id) {
  constexpr int cl_elems = cache_line_size / sizeof(value_t);

  const index_t nc_dim =
      is_transposed ? matrix_a_.get_size_col() : matrix_a_.get_size_row();
  const index_t c_dim =
      is_transposed ? matrix_a_.get_size_row() : matrix_a_.get_size_col();

  // Group indices within global grid
  const index_t nc_group_id = group_id / wgs_per_c_;
  const index_t c_group_id = group_id - nc_group_id * wgs_per_c_;

  // Tile dimensions
  constexpr int tile_dim_nc = local_range / cl_elems;
  constexpr int tile_dim_c = cl_elems;

  // In-tile indices
  const index_t tile_nc_index = local_id / tile_dim_c;
  const index_t tile_c_index = local_id % tile_dim_c;

  // Tile-group (local) indices
  index_t local_nc_index = tile_nc_index;
  const index_t local_c_index = tile_c_index + c_tile_id * tile_dim_c;

  // Global (grid) indices
  index_t grid_nc_index = local_nc_index + nc_group_id * local_range;
  const index_t grid_c_index = local_c_index + c_group_id * local_range;

  // Matrix index calculated based on global grid indices
  index_t mat_index = grid_c_index + grid_nc_index * lda;

  // Number of iterations to load all tiles
  // local_range / tile_dim_nc == cl_elems
  constexpr int tile_nc_loops = cl_elems;

  // Transposed index in scratch-space with bank offset which adds 1 word of
  // padding every 'tile_dim_c' elements
  const index_t scratch_index = (local_range + 1) * tile_c_index;

  const bool in_c_range = grid_c_index < c_dim;

#pragma unroll
  for (index_t nc_tile_id = 0; nc_tile_id < tile_nc_loops; ++nc_tile_id) {
    // Populate the scratch memory with matrix values, ensuring we don't fetch
    // beyond bounds
    matrix_scratch[scratch_index + local_nc_index] =
        in_c_range && grid_nc_index < nc_dim
            ? matrix_a_.template eval<true>(mat_index)
            : value_t{0};

    // Move to loading the next tile
    mat_index += tile_dim_nc * lda;
    local_nc_index += tile_dim_nc;
    grid_nc_index += tile_dim_nc;
  }
}

template <typename lhs_t, typename matrix_t, typename vector_t,
          uint32_t local_range, bool is_transposed, int cache_line_size,
          int work_per_thread>
SYCL_BLAS_INLINE void
Gemv<lhs_t, matrix_t, vector_t, local_range, is_transposed, cache_line_size,
     work_per_thread>::bind(cl::sycl::handler &h) {
  lhs_.bind(h);
  matrix_a_.bind(h);
  vector_x_.bind(h);
}

template <typename lhs_t, typename matrix_t, typename vector_t,
          uint32_t local_range, bool is_transposed, int cache_line_size,
          int work_per_thread>
SYCL_BLAS_INLINE void
Gemv<lhs_t, matrix_t, vector_t, local_range, is_transposed, cache_line_size,
     work_per_thread>::adjust_access_displacement() {
  lhs_.adjust_access_displacement();
  matrix_a_.adjust_access_displacement();
  vector_x_.adjust_access_displacement();
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
  // TODO(Peter): Does it hurt performance if this is not constexpr?
  static const value_t init_val = AddOperator::template init<vector_t>();
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
template <int interLoop, bool Lower, bool Diag, bool Upper, bool Unit,
          typename lhs_t, typename matrix_t, typename vector_t>
SYCL_BLAS_INLINE void
GemvRow<interLoop, Lower, Diag, Upper, Unit, lhs_t, matrix_t,
        vector_t>::adjust_access_displacement() {
  lhs_.adjust_access_displacement();
  matrix_.adjust_access_displacement();
  vector_.adjust_access_displacement();
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

template <bool Lower, bool Diag, bool Upper, bool Unit, typename lhs_t,
          typename matrix_t, typename vector_t>
SYCL_BLAS_INLINE void GemvCol<Lower, Diag, Upper, Unit, lhs_t, matrix_t,
                              vector_t>::adjust_access_displacement() {
  lhs_.adjust_access_displacement();
  matrix_.adjust_access_displacement();
  vector_.adjust_access_displacement();
}

}  // namespace blas
#endif
