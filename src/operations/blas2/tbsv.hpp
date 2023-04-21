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
 *  @filename tbsv.hpp
 *
 **************************************************************************/

#ifndef TBSV_HPP
#define TBSV_HPP
#include "operations/blas2_trees.h"
namespace blas {

/**
 * @struct Tbsv
 * @brief Tree node representing a triangular band matrix_ lhs_
 * multiplication.
 */
template <typename vector_t, typename matrix_t, typename sync_t,
          uint32_t x_range, uint32_t subgroups, bool is_upper,
          bool is_transposed, bool is_unitdiag>
SYCL_BLAS_INLINE
Tbsv<vector_t, matrix_t, sync_t, x_range, subgroups, is_upper, is_transposed,
     is_unitdiag>::Tbsv(vector_t &_l, matrix_t &_matrix,
                        typename Tbsv<vector_t, matrix_t, sync_t, x_range,
                                      subgroups, is_upper, is_transposed,
                                      is_unitdiag>::index_t &_k,
                        sync_t &_sync)
    : lhs_(_l), matrix_(_matrix), k_(_k), sync_(_sync) {}

template <typename vector_t, typename matrix_t, typename sync_t,
          uint32_t x_range, uint32_t subgroups, bool is_upper,
          bool is_transposed, bool is_unitdiag>
SYCL_BLAS_INLINE typename Tbsv<vector_t, matrix_t, sync_t, x_range, subgroups,
                               is_upper, is_transposed, is_unitdiag>::index_t
Tbsv<vector_t, matrix_t, sync_t, x_range, subgroups, is_upper, is_transposed,
     is_unitdiag>::get_size() const {
  return matrix_.get_size();
}
template <typename vector_t, typename matrix_t, typename sync_t,
          uint32_t x_range, uint32_t subgroups, bool is_upper,
          bool is_transposed, bool is_unitdiag>
SYCL_BLAS_INLINE bool
Tbsv<vector_t, matrix_t, sync_t, x_range, subgroups, is_upper, is_transposed,
     is_unitdiag>::valid_thread(cl::sycl::nd_item<1> ndItem) const {
  // Valid threads are established by ::eval.
  return true;
}

template <typename vector_t, typename matrix_t, typename sync_t,
          uint32_t x_range, uint32_t subgroups, bool is_upper,
          bool is_transposed, bool is_unitdiag>
template <typename local_memory_t>
SYCL_BLAS_INLINE typename Tbsv<vector_t, matrix_t, sync_t, x_range, subgroups,
                               is_upper, is_transposed, is_unitdiag>::value_t
Tbsv<vector_t, matrix_t, sync_t, x_range, subgroups, is_upper, is_transposed,
     is_unitdiag>::eval(local_memory_t local_mem, cl::sycl::nd_item<1> ndItem) {
#ifndef __COMPUTECPP__

  constexpr bool is_forward =
      (is_upper && is_transposed) || (!is_upper && !is_transposed);

  // Number of sub-groups per work-group
  constexpr index_t sub_num = subgroups;
  constexpr index_t y_range = x_range / sub_num;

  const index_t _N = lhs_.get_size();

  // True if not work-item 0
  const bool not_wi0 = ndItem.get_local_id(0);

  // Local bi-dimensional indexes
  const index_t _idx = ndItem.get_local_id(0) % x_range;
  const index_t _idy = ndItem.get_local_id(0) / x_range;

  // Private memory
  value_t priv_A[y_range];
  value_t priv_val = 0;

  // Local memory stride
  const index_t _llda = x_range + 1;

  // Pointers to local memory
  value_t *const loc_A = local_mem.localAcc.get_pointer();
  value_t *const sub_A = loc_A + _llda * y_range * _idy + _idx;
  value_t *const sub_At = loc_A + _llda * _idx + y_range * _idy;

  value_t *const loc_x = loc_A + _llda * x_range;
  value_t *const sub_x = loc_x + y_range * _idy;

  value_t *const par_x = loc_x + x_range + x_range * _idy;
  value_t *const loc_recip = loc_x + x_range;

  auto a = sycl::atomic_ref<index_t, sycl::memory_order::relaxed,
                            sycl::memory_scope::device,
                            sycl::access::address_space::global_space>(
      sync_.eval(0));

  // Get the wg_id of actual workgroup
  const index_t wg_id =
      group_broadcast(ndItem.get_group(), not_wi0        ? 0
                                          : (is_forward) ? a++
                                                         : a--);

  const index_t num_blocks = ((k_ + x_range - 1) / x_range);
  // Actual extra-diagonal block processed
  index_t curr_block =
      is_forward ? ((wg_id - num_blocks < 0) ? 0 : wg_id - num_blocks)
                 : (wg_id + num_blocks > (((_N + x_range - 1) / x_range) - 1)
                        ? (((_N + x_range - 1) / x_range) - 1)
                        : (wg_id + num_blocks));
  index_t curr_offset = curr_block * x_range + _idx;

  // Global memory offsets
  const index_t g_idx = wg_id * x_range + _idx;

  // Read first block
  {
    value_t *lA = sub_A;

#pragma unroll
    for (index_t i = 0; i < y_range; ++i) {
      const index_t col =
          ((is_transposed ? wg_id : curr_block) * x_range) + y_range * _idy + i;
      const index_t row_full =
          (is_transposed ? curr_block : wg_id) * x_range + _idx;
      const index_t row = (is_upper) ? k_ + row_full - col : row_full - col;

      const bool read_it = (row < k_ + 1) && (row >= 0) && (col < _N);
      *lA = read_it ? matrix_.eval(row, col) : value_t(0);
      lA += _llda;
    }
  }

  // Solve extra-diagonal blocks

  volatile int *p = &sync_.eval(1);
  index_t ready_block =
      (_idy == 0)
          ? sycl::group_broadcast(ndItem.get_sub_group(), not_wi0 ? 0 : *p)
          : 0;

  const index_t steps =
      is_forward ? (wg_id - curr_block) : (curr_block - wg_id);
  for (index_t s = 0; s < steps; ++s) {
    const index_t next_offset = curr_offset + (is_forward ? x_range : -x_range);
    const index_t next_block = curr_block + (is_forward ? 1 : -1);

    // Read next block
    {
#pragma unroll
      for (index_t i = 0; i < y_range; ++i) {
        const index_t col = ((is_transposed ? wg_id : next_block) * x_range) +
                            y_range * _idy + i;
        const index_t row_full =
            (is_transposed ? next_block : wg_id) * x_range + _idx;
        const index_t row = (is_upper) ? k_ + row_full - col : row_full - col;

        const bool read_it = (row < k_ + 1) && (row >= 0) && (col < _N);
        priv_A[i] = read_it ? matrix_.eval(row, col) : value_t(0);
      }
    }

    if (_idy == 0) {
      while (!((is_forward && (curr_block < ready_block)) ||
               (!is_forward && (curr_block > ready_block))))
        ready_block =
            sycl::group_broadcast(ndItem.get_sub_group(), not_wi0 ? 0 : *p);

      loc_x[_idx] = (curr_offset < _N) ? lhs_.eval(curr_offset) : value_t(0);
    }

    curr_offset = next_offset;
    curr_block = next_block;

    ndItem.barrier(cl::sycl::access::fence_space::local_space);

    // Multiply current block
    {
      value_t *lx = sub_x;
      value_t *lA = is_transposed ? sub_At : sub_A;
#pragma unroll
      for (index_t i = 0; i < y_range; ++i) {
        priv_val += *lA * *(lx++);
        lA += is_transposed ? 1 : _llda;
      }
    }

    if (is_transposed)
      ndItem.barrier(cl::sycl::access::fence_space::local_space);

    // Copy next block to local memory
    {
      value_t *lA = sub_A;
#pragma unroll
      for (index_t i = 0; i < y_range; ++i) {
        *lA = priv_A[i];
        lA += _llda;
      }
    }
  }

  // Store partial values
  if (_idy != 0) par_x[_idx] = priv_val;

  // Pre-compute diagonal recip
  if (!is_unitdiag && (_idx >= y_range * _idy) && (_idx < y_range * (_idy + 1)))
    loc_recip[_idx] = value_t(1) / (loc_A[_llda * _idx + _idx]);

  ndItem.barrier(cl::sycl::access::fence_space::local_space);

  if (_idy == 0) {
// Accumulate partial values
#pragma unroll
    for (index_t y = 1; y < sub_num; ++y) priv_val += par_x[x_range * y + _idx];

    // Solve diagonal block
    value_t r_x = g_idx < _N ? (lhs_.eval(g_idx) - priv_val) : value_t(0);
    const value_t A_diag_recip =
        (!is_unitdiag && g_idx < _N) ? loc_recip[_idx] : value_t(0);
    value_t _A, r_diag;

#pragma unroll
    for (index_t _it = 0; _it < x_range; ++_it) {
      const index_t l_diag = is_forward ? _it : (x_range - 1 - _it);

      r_diag =
          sycl::group_broadcast(ndItem.get_sub_group(),
                                is_unitdiag ? r_x : r_x * A_diag_recip, l_diag);
      _A = (is_transposed) ? loc_A[_llda * _idx + l_diag]
                           : loc_A[_llda * l_diag + _idx];
      r_x -= _A * r_diag;

      if (_idx == l_diag) loc_x[_idx] = r_diag;
    }

    volatile value_t *lhs_p = lhs_.get_pointer() + lhs_.get_stride() * g_idx;
    if (g_idx < _N) *lhs_p = loc_x[_idx];
  }

  sycl::atomic_fence(sycl::memory_order::seq_cst, sycl::memory_scope::device);

  volatile int *sync = sync_.get_pointer() + 1;
  if (!not_wi0) *sync = wg_id + (is_forward ? 1 : -1);

  sycl::atomic_fence(sycl::memory_order::seq_cst, sycl::memory_scope::device);

#endif
  return 0;
}

template <typename vector_t, typename matrix_t, typename sync_t,
          uint32_t x_range, uint32_t subgroups, bool is_upper,
          bool is_transposed, bool is_unitdiag>
SYCL_BLAS_INLINE void
Tbsv<vector_t, matrix_t, sync_t, x_range, subgroups, is_upper, is_transposed,
     is_unitdiag>::bind(cl::sycl::handler &h) {
  lhs_.bind(h);
  matrix_.bind(h);
  sync_.bind(h);
}
template <typename vector_t, typename matrix_t, typename sync_t,
          uint32_t x_range, uint32_t subgroups, bool is_upper,
          bool is_transposed, bool is_unitdiag>
SYCL_BLAS_INLINE void
Tbsv<vector_t, matrix_t, sync_t, x_range, subgroups, is_upper, is_transposed,
     is_unitdiag>::adjust_access_displacement() {
  lhs_.adjust_access_displacement();
  matrix_.adjust_access_displacement();
  sync_.adjust_access_displacement();
}

}  // namespace blas
#endif
