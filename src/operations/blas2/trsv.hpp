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
 *  @filename trsv.hpp
 *
 **************************************************************************/

#ifndef TRSV_HPP
#define TRSV_HPP
#include "operations/blas2_trees.h"
namespace blas {

/**
 * @struct Trsv
 * @brief Tree node representing a triangular band matrix_ vector_
 * multiplication.
 */
template <typename lhs_t, typename matrix_t, typename vector_t,
          uint32_t local_range, bool is_upper, bool is_transposed,
          bool is_unitdiag>
SYCL_BLAS_INLINE
Trsv<lhs_t, matrix_t, vector_t, local_range, is_upper, is_transposed,
     is_unitdiag>::Trsv(lhs_t &_l, matrix_t &_matrix,
                        typename Trsv<lhs_t, matrix_t, vector_t, local_range,
                                      is_upper, is_transposed,
                                      is_unitdiag>::index_t &_blk_id,
                        vector_t &_vector)
    : lhs_(_l), matrix_(_matrix), vector_(_vector), blk_id_(_blk_id) {}

template <typename lhs_t, typename matrix_t, typename vector_t,
          uint32_t local_range, bool is_upper, bool is_transposed,
          bool is_unitdiag>
SYCL_BLAS_INLINE typename Trsv<lhs_t, matrix_t, vector_t, local_range, is_upper,
                               is_transposed, is_unitdiag>::index_t
Trsv<lhs_t, matrix_t, vector_t, local_range, is_upper, is_transposed,
     is_unitdiag>::get_size() const {
  return matrix_.get_size();
}
template <typename lhs_t, typename matrix_t, typename vector_t,
          uint32_t local_range, bool is_upper, bool is_transposed,
          bool is_unitdiag>
SYCL_BLAS_INLINE bool
Trsv<lhs_t, matrix_t, vector_t, local_range, is_upper, is_transposed,
     is_unitdiag>::valid_thread(cl::sycl::nd_item<1> ndItem) const {
  // Valid threads are established by ::eval.
  return true;
}

template <typename lhs_t, typename matrix_t, typename vector_t,
          uint32_t local_range, bool is_upper, bool is_transposed,
          bool is_unitdiag>
template <typename local_memory_t>
SYCL_BLAS_INLINE typename Trsv<lhs_t, matrix_t, vector_t, local_range, is_upper,
                               is_transposed, is_unitdiag>::value_t
Trsv<lhs_t, matrix_t, vector_t, local_range, is_upper, is_transposed,
     is_unitdiag>::eval(local_memory_t local_mem, cl::sycl::nd_item<1> ndItem) {
  const index_t _offset = blk_id_ * local_range;

  const index_t g_idx = _offset + ndItem.get_local_id(0);
  const index_t l_idx = ndItem.get_local_id(0);

  const index_t _N = lhs_.get_size();
  auto l_x = local_mem.localAcc;

  // copy lhs_ local memory + sync thread
  if (g_idx < _N) l_x[l_idx] = lhs_.eval(g_idx);

  ndItem.barrier(cl::sycl::access::fence_space::global_and_local);

  constexpr bool is_forward =
      (is_upper && is_transposed) || (!is_upper && !is_transposed);

  const index_t n_it =
      (_offset + local_range < _N) ? local_range : _N - _offset;
  for (index_t _it = 0; _it < n_it; ++_it) {
    const index_t l_diag = (is_forward) ? _it : n_it - 1 - _it;
    const index_t g_diag = _offset + l_diag;

    if (!is_unitdiag && !l_idx) l_x[l_diag] /= matrix_.eval(g_diag, g_diag);

    ndItem.barrier(cl::sycl::access::fence_space::local_space);

    if (((g_idx > g_diag) && (g_idx < _N) && is_forward) ||
        ((g_idx < g_diag) && !is_forward)) {
      const value_t val = (is_transposed) ? matrix_.eval(g_diag, g_idx)
                                          : matrix_.eval(g_idx, g_diag);
      l_x[l_idx] -= val * l_x[l_diag];
    }

    ndItem.barrier(cl::sycl::access::fence_space::local_space);
  }

  if (g_idx < _N) lhs_.eval(g_idx) = l_x[l_idx];

  return 0;
}

template <typename lhs_t, typename matrix_t, typename vector_t,
          uint32_t local_range, bool is_upper, bool is_transposed,
          bool is_unitdiag>
SYCL_BLAS_INLINE void
Trsv<lhs_t, matrix_t, vector_t, local_range, is_upper, is_transposed,
     is_unitdiag>::bind(cl::sycl::handler &h) {
  lhs_.bind(h);
  matrix_.bind(h);
  vector_.bind(h);
}
template <typename lhs_t, typename matrix_t, typename vector_t,
          uint32_t local_range, bool is_upper, bool is_transposed,
          bool is_unitdiag>
SYCL_BLAS_INLINE void
Trsv<lhs_t, matrix_t, vector_t, local_range, is_upper, is_transposed,
     is_unitdiag>::adjust_access_displacement() {
  lhs_.adjust_access_displacement();
  matrix_.adjust_access_displacement();
  vector_.adjust_access_displacement();
}

}  // namespace blas
#endif
