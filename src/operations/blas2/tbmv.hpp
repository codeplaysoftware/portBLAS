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
 *  portBLAS: BLAS implementation using SYCL
 *
 *  @filename tbmv.hpp
 *
 **************************************************************************/

#ifndef TBMV_HPP
#define TBMV_HPP
#include "operations/blas2_trees.h"
#include "operations/blas_operators.hpp"
#include "views/view_sycl.hpp"
#include <stdexcept>
#include <vector>
namespace blas {

/**
 * @struct Tbmv
 * @brief Tree node representing a triangular band matrix_ vector_
 * multiplication.
 */
template <typename lhs_t, typename matrix_t, typename vector_t,
          uint32_t local_range, bool is_upper, bool is_transposed,
          bool is_unitdiag>
PORTBLAS_INLINE
Tbmv<lhs_t, matrix_t, vector_t, local_range, is_upper, is_transposed,
     is_unitdiag>::Tbmv(lhs_t &_l, matrix_t &_matrix,
                        typename Tbmv<lhs_t, matrix_t, vector_t, local_range,
                                      is_upper, is_transposed,
                                      is_unitdiag>::index_t &_k,
                        vector_t &_vector)
    : lhs_(_l), matrix_(_matrix), vector_(_vector), k_(_k) {}

template <typename lhs_t, typename matrix_t, typename vector_t,
          uint32_t local_range, bool is_upper, bool is_transposed,
          bool is_unitdiag>
PORTBLAS_INLINE typename Tbmv<lhs_t, matrix_t, vector_t, local_range, is_upper,
                               is_transposed, is_unitdiag>::index_t
Tbmv<lhs_t, matrix_t, vector_t, local_range, is_upper, is_transposed,
     is_unitdiag>::get_size() const {
  return matrix_.get_size();
}
template <typename lhs_t, typename matrix_t, typename vector_t,
          uint32_t local_range, bool is_upper, bool is_transposed,
          bool is_unitdiag>
PORTBLAS_INLINE bool
Tbmv<lhs_t, matrix_t, vector_t, local_range, is_upper, is_transposed,
     is_unitdiag>::valid_thread(sycl::nd_item<1> ndItem) const {
  // Valid threads are established by ::eval.
  return true;
}

template <typename lhs_t, typename matrix_t, typename vector_t,
          uint32_t local_range, bool is_upper, bool is_transposed,
          bool is_unitdiag>
PORTBLAS_INLINE typename Tbmv<lhs_t, matrix_t, vector_t, local_range, is_upper,
                              is_transposed, is_unitdiag>::value_t
Tbmv<lhs_t, matrix_t, vector_t, local_range, is_upper, is_transposed,
     is_unitdiag>::eval(sycl::nd_item<1> ndItem) {
  const index_t lhs_idx = ndItem.get_global_id(0);

  value_t val = 0;

  if (lhs_idx < lhs_.get_size()) {
    const index_t kl_ = is_upper ? 0 : k_;
    const index_t ku_ = is_upper ? k_ : 0;

    const index_t k_lower = is_transposed ? ku_ : kl_;
    const index_t k_upper = is_transposed ? kl_ : ku_;

    const index_t k_beg = sycl::max(index_t(0), lhs_idx - k_lower);
    const index_t k_end = sycl::min(vector_.get_size(), lhs_idx + k_upper + 1);
    const index_t k_off = ku_ + (is_transposed ? -lhs_idx : lhs_idx);

    for (index_t s_idx = k_beg; s_idx < k_end; ++s_idx) {
      const index_t K = k_off + (is_transposed ? s_idx : -s_idx);
      const index_t J = is_transposed ? lhs_idx : s_idx;
      val = AddOperator::eval(
          val, ProductOperator::eval(is_unitdiag && ((is_upper && (K == k_)) ||
                                                     (!is_upper && (K == 0)))
                                         ? value_t(1)
                                         : matrix_.eval(K, J),
                                     vector_.eval(s_idx)));
    }

    lhs_.eval(lhs_idx) = val;
  }
  return val;
}

template <typename lhs_t, typename matrix_t, typename vector_t,
          uint32_t local_range, bool is_upper, bool is_transposed,
          bool is_unitdiag>
PORTBLAS_INLINE void Tbmv<lhs_t, matrix_t, vector_t, local_range, is_upper,
                          is_transposed, is_unitdiag>::bind(sycl::handler &h) {
  lhs_.bind(h);
  matrix_.bind(h);
  vector_.bind(h);
}
template <typename lhs_t, typename matrix_t, typename vector_t,
          uint32_t local_range, bool is_upper, bool is_transposed,
          bool is_unitdiag>
PORTBLAS_INLINE void
Tbmv<lhs_t, matrix_t, vector_t, local_range, is_upper, is_transposed,
     is_unitdiag>::adjust_access_displacement() {
  lhs_.adjust_access_displacement();
  matrix_.adjust_access_displacement();
  vector_.adjust_access_displacement();
}

}  // namespace blas
#endif
