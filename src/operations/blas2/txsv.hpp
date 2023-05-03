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
 *  @filename txsv.hpp
 *
 **************************************************************************/

#ifndef TXSV_HPP
#define TXSV_HPP
#include "operations/blas2_trees.h"
namespace blas {

/**
 * @struct Txsv
 * @brief Tree node representing a linear system solver for triangular matrices.
 */
template <typename vector_t, typename matrix_t, typename sync_t,
          matrix_format_t matrix_format, uint32_t subgroup_size,
          uint32_t subgroups, bool is_upper, bool is_transposed,
          bool is_unitdiag>
SYCL_BLAS_INLINE Txsv<vector_t, matrix_t, sync_t, matrix_format, subgroup_size,
                      subgroups, is_upper, is_transposed, is_unitdiag>::
    Txsv(vector_t &_l, matrix_t &_matrix,
         typename Txsv<vector_t, matrix_t, sync_t, matrix_format, subgroup_size,
                       subgroups, is_upper, is_transposed, is_unitdiag>::index_t
             &_k,
         sync_t &_sync)
    : lhs_(_l), matrix_(_matrix), k_(_k), sync_(_sync) {}

template <typename vector_t, typename matrix_t, typename sync_t,
          matrix_format_t matrix_format, uint32_t subgroup_size,
          uint32_t subgroups, bool is_upper, bool is_transposed,
          bool is_unitdiag>
SYCL_BLAS_INLINE
    typename Txsv<vector_t, matrix_t, sync_t, matrix_format, subgroup_size,
                  subgroups, is_upper, is_transposed, is_unitdiag>::value_t
    Txsv<vector_t, matrix_t, sync_t, matrix_format, subgroup_size, subgroups,
         is_upper, is_transposed, is_unitdiag>::
        read_matrix(
            const typename Txsv<vector_t, matrix_t, sync_t, matrix_format,
                                subgroup_size, subgroups, is_upper,
                                is_transposed, is_unitdiag>::index_t &row,
            const typename Txsv<vector_t, matrix_t, sync_t, matrix_format,
                                subgroup_size, subgroups, is_upper,
                                is_transposed, is_unitdiag>::index_t &col)
            const {
  const index_t _N = lhs_.get_size();

  if (matrix_format == matrix_format_t::full) {
    // trsv
    const bool read_it = (col < _N) && (row < _N);
    return read_it ? matrix_.eval(row, col) : value_t(0);
  } else if (matrix_format == matrix_format_t::packed) {
    // tpsv
    const bool read_it = is_upper ? ((col >= row) && (row < _N) && (col < _N))
                                  : ((col <= row) && (row < _N));

    const index_t col_offset = is_upper ? ((col * (col + 1)) / 2)
                                        : (col * _N) - ((col * (col + 1)) / 2);

    value_t *val = matrix_.get_pointer() + col_offset + row;
    return read_it ? *val : value_t(0);
  } else if (matrix_format == matrix_format_t::banded) {
    // tbsv
    const index_t row_band = (is_upper) ? k_ + row - col : row - col;
    const bool read_it = (row_band < k_ + 1) && (row_band >= 0) && (col < _N);

    return read_it ? matrix_.eval(row_band, col) : value_t(0);
  }

  return value_t(0);
}
template <typename vector_t, typename matrix_t, typename sync_t,
          matrix_format_t matrix_format, uint32_t subgroup_size,
          uint32_t subgroups, bool is_upper, bool is_transposed,
          bool is_unitdiag>
template <typename local_memory_t>
SYCL_BLAS_INLINE typename Txsv<vector_t, matrix_t, sync_t, matrix_format,
                               subgroup_size, subgroups, is_upper,
                               is_transposed, is_unitdiag>::value_t
Txsv<vector_t, matrix_t, sync_t, matrix_format, subgroup_size, subgroups,
     is_upper, is_transposed, is_unitdiag>::eval(local_memory_t local_mem,
                                                 cl::sycl::nd_item<1> ndItem) {
  value_t ret = 0;
#if SYCL_LANGUAGE_VERSION >= 202000

  constexpr bool is_forward =
      (is_upper && is_transposed) || (!is_upper && !is_transposed);

  // Number of sub-groups per work-group
  constexpr index_t sub_num = subgroups;
  constexpr index_t loc_x_dim = subgroup_size;
  constexpr index_t priv_y_dim = loc_x_dim / sub_num;

  const index_t _N = lhs_.get_size();

  // True if not work-item 0
  const bool not_wi0 = ndItem.get_local_id(0);

  // Local bi-dimensional indexes
  const index_t l_idx = ndItem.get_local_id(0) % loc_x_dim;
  const index_t l_idy = ndItem.get_local_id(0) / loc_x_dim;

  // Private memory
  value_t priv_A[priv_y_dim];
  value_t priv_val = 0;

  // Local memory stride
  const index_t l_lda = loc_x_dim + 1;

  // Pointers to local memory
  value_t *const loc_A = local_mem.localAcc.get_pointer();
  value_t *const sub_A = loc_A + l_lda * priv_y_dim * l_idy + l_idx;
  value_t *const sub_At = loc_A + l_lda * l_idx + priv_y_dim * l_idy;

  value_t *const loc_x = loc_A + l_lda * loc_x_dim;
  value_t *const sub_x = loc_x + priv_y_dim * l_idy;

  value_t *const par_x = loc_x + loc_x_dim + loc_x_dim * l_idy;
  value_t *const loc_recip = loc_x + loc_x_dim;

  auto a = sycl::atomic_ref<int32_t, sycl::memory_order::relaxed,
                            sycl::memory_scope::device,
                            sycl::access::address_space::global_space>(
      sync_.eval(0));

  // Get the wg_id of actual workgroup
  const index_t wg_id =
      group_broadcast(ndItem.get_group(), not_wi0        ? 0
                                          : (is_forward) ? a++
                                                         : a--);

  index_t curr_block;

  if (matrix_format == matrix_format_t::banded) {
    const index_t num_blocks = ((k_ + loc_x_dim - 1) / loc_x_dim);
    curr_block =
        is_forward
            ? ((wg_id - num_blocks < 0) ? 0 : wg_id - num_blocks)
            : (wg_id + num_blocks > (((_N + loc_x_dim - 1) / loc_x_dim) - 1)
                   ? (((_N + loc_x_dim - 1) / loc_x_dim) - 1)
                   : (wg_id + num_blocks));
  } else {
    curr_block = ((is_forward) ? 0 : ((_N + loc_x_dim - 1) / loc_x_dim) - 1);
  }

  // Global memory offsets

  index_t curr_offset = curr_block * loc_x_dim +
                        l_idx;  // < offset of the current block processed
  const index_t g_idx = wg_id * loc_x_dim + l_idx;  // < offset of the solution

  // Read first block
  index_t curr_col =
      (is_transposed ? wg_id : curr_block) * loc_x_dim + priv_y_dim * l_idy;
  index_t curr_row = (is_transposed ? curr_block : wg_id) * loc_x_dim + l_idx;

  {
    value_t *lA = sub_A;
#pragma unroll
    for (index_t i = 0; i < priv_y_dim; ++i) {
      *lA = read_matrix(curr_row, curr_col + i);
      lA += l_lda;
    }
  }

  volatile int32_t *p = &sync_.eval(1);
  int32_t ready_block =
      (l_idy == 0)
          ? sycl::group_broadcast(ndItem.get_sub_group(), not_wi0 ? 0 : *p)
          : 0;

  const index_t steps =
      is_forward ? (wg_id - curr_block) : (curr_block - wg_id);

  for (index_t s = 0; s < steps; ++s) {
    // Read next block
    const index_t next_offset =
        curr_offset + (is_forward ? loc_x_dim : -loc_x_dim);
    const index_t next_block = curr_block + (is_forward ? 1 : -1);
    const index_t next_col =
        curr_col + (is_transposed ? 0 : (is_forward ? loc_x_dim : -loc_x_dim));
    const index_t next_row =
        curr_row + (is_transposed ? (is_forward ? loc_x_dim : -loc_x_dim) : 0);

#pragma unroll
    for (index_t i = 0; i < priv_y_dim; ++i) {
      priv_A[i] = read_matrix(next_row, next_col + i);
    }

    if (l_idy == 0) {
      while (!((is_forward && (curr_block < ready_block)) ||
               (!is_forward && (curr_block > ready_block))))
        ready_block =
            sycl::group_broadcast(ndItem.get_sub_group(), not_wi0 ? 0 : *p);

      loc_x[l_idx] = (curr_offset < _N) ? lhs_.eval(curr_offset) : value_t(0);
    }

    curr_offset = next_offset;
    curr_block = next_block;
    curr_col = next_col;
    curr_row = next_row;

    ndItem.barrier(cl::sycl::access::fence_space::local_space);

    // Multiply current block
    {
      value_t *lx = sub_x;
      value_t *lA = is_transposed ? sub_At : sub_A;
#pragma unroll
      for (index_t i = 0; i < priv_y_dim; ++i) {
        priv_val += *lA * *(lx++);
        lA += is_transposed ? 1 : l_lda;
      }
    }

    ndItem.barrier(cl::sycl::access::fence_space::local_space);

    // Copy next block to local memory
    {
      value_t *lA = sub_A;
#pragma unroll
      for (index_t i = 0; i < priv_y_dim; ++i) {
        *lA = priv_A[i];
        lA += l_lda;
      }
    }
  }

  // Store partial values
  if (l_idy != 0) par_x[l_idx] = priv_val;

  // Pre-compute diagonal recip
  if (!is_unitdiag && (l_idx >= priv_y_dim * l_idy) &&
      (l_idx < priv_y_dim * (l_idy + 1)))
    loc_recip[l_idx] = value_t(1) / (loc_A[l_lda * l_idx + l_idx]);

  ndItem.barrier(cl::sycl::access::fence_space::local_space);

  if (l_idy == 0) {
// Accumulate partial values
#pragma unroll
    for (index_t y = 1; y < sub_num; ++y)
      priv_val += par_x[loc_x_dim * y + l_idx];

    // Solve diagonal block
    value_t r_x = g_idx < _N ? (lhs_.eval(g_idx) - priv_val) : value_t(0);
    const value_t A_diag_recip =
        (!is_unitdiag && g_idx < _N) ? loc_recip[l_idx] : value_t(0);
    value_t _A, r_diag;

#pragma unroll
    for (index_t _it = 0; _it < loc_x_dim; ++_it) {
      const index_t l_diag = is_forward ? _it : (loc_x_dim - 1 - _it);

      r_diag =
          sycl::group_broadcast(ndItem.get_sub_group(),
                                is_unitdiag ? r_x : r_x * A_diag_recip, l_diag);
      _A = (is_transposed) ? loc_A[l_lda * l_idx + l_diag]
                           : loc_A[l_lda * l_diag + l_idx];
      r_x -= _A * r_diag;

      if (l_idx == l_diag) loc_x[l_idx] = r_diag;
    }

    volatile value_t *lhs_p = lhs_.get_pointer() + lhs_.get_stride() * g_idx;
    if (g_idx < _N) *lhs_p = ret = loc_x[l_idx];
  }

  sycl::atomic_fence(sycl::memory_order::seq_cst, sycl::memory_scope::device);

  volatile int32_t *sync = sync_.get_pointer() + 1;
  if (!not_wi0) *sync = wg_id + (is_forward ? 1 : -1);

  sycl::atomic_fence(sycl::memory_order::seq_cst, sycl::memory_scope::device);

#endif
  return ret;
}

template <typename vector_t, typename matrix_t, typename sync_t,
          matrix_format_t matrix_format, uint32_t subgroup_size,
          uint32_t subgroups, bool is_upper, bool is_transposed,
          bool is_unitdiag>
SYCL_BLAS_INLINE void
Txsv<vector_t, matrix_t, sync_t, matrix_format, subgroup_size, subgroups,
     is_upper, is_transposed, is_unitdiag>::bind(cl::sycl::handler &h) {
  lhs_.bind(h);
  matrix_.bind(h);
  sync_.bind(h);
}
template <typename vector_t, typename matrix_t, typename sync_t,
          matrix_format_t matrix_format, uint32_t subgroup_size,
          uint32_t subgroups, bool is_upper, bool is_transposed,
          bool is_unitdiag>
SYCL_BLAS_INLINE void
Txsv<vector_t, matrix_t, sync_t, matrix_format, subgroup_size, subgroups,
     is_upper, is_transposed, is_unitdiag>::adjust_access_displacement() {
  lhs_.adjust_access_displacement();
  matrix_.adjust_access_displacement();
  sync_.adjust_access_displacement();
}

}  // namespace blas
#endif
