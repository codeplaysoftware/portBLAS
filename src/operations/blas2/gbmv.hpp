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
 *  @filename gbmv.hpp
 *
 **************************************************************************/

#ifndef GBMV_HPP
#define GBMV_HPP
#include "operations/blas2_trees.h"
#include "operations/blas_operators.hpp"
#include "views/view_sycl.hpp"
#include <stdexcept>
#include <vector>
namespace blas {

/**
 * @struct Gbmv
 * @brief Tree node representing a band matrix_ vector_ multiplication.
 */
template <typename lhs_t, typename matrix_t, typename vector_t,
          uint32_t local_range, bool is_transposed>
SYCL_BLAS_INLINE
Gbmv<lhs_t, matrix_t, vector_t, local_range, is_transposed>::Gbmv(
    lhs_t &_l, matrix_t &_matrix,
    typename Gbmv<lhs_t, matrix_t, vector_t, local_range,
                  is_transposed>::index_t &_kl,
    typename Gbmv<lhs_t, matrix_t, vector_t, local_range,
                  is_transposed>::index_t &_ku,
    vector_t &_vector)
    : lhs_(_l), matrix_(_matrix), vector_(_vector), kl_(_kl), ku_(_ku) {}

template <typename lhs_t, typename matrix_t, typename vector_t,
          uint32_t local_range, bool is_transposed>
SYCL_BLAS_INLINE typename Gbmv<lhs_t, matrix_t, vector_t, local_range,
                               is_transposed>::index_t
Gbmv<lhs_t, matrix_t, vector_t, local_range, is_transposed>::get_size() const {
  return matrix_.get_size();
}
template <typename lhs_t, typename matrix_t, typename vector_t,
          uint32_t local_range, bool is_transposed>
SYCL_BLAS_INLINE bool
Gbmv<lhs_t, matrix_t, vector_t, local_range, is_transposed>::valid_thread(
    cl::sycl::nd_item<1> ndItem) const {
  // Valid threads are established by ::eval.
  return true;
}

template <typename lhs_t, typename matrix_t, typename vector_t,
          uint32_t local_range, bool is_transposed>
SYCL_BLAS_INLINE typename Gbmv<lhs_t, matrix_t, vector_t, local_range,
                               is_transposed>::value_t
Gbmv<lhs_t, matrix_t, vector_t, local_range, is_transposed>::eval(
    cl::sycl::nd_item<1> ndItem) {
  const index_t lhs_idx =
      ndItem.get_group(0) * local_range + ndItem.get_local_id(0);
  value_t val = 0;

  if (lhs_idx < lhs_.get_size_row()) {
    const index_t k_lower = is_transposed ? ku_ : kl_;
    const index_t k_upper = is_transposed ? kl_ : ku_;

    const index_t k_beg = cl::sycl::max(index_t(0), lhs_idx - k_lower);
    const index_t k_end =
        cl::sycl::min(vector_.get_size(), lhs_idx + k_upper + 1);
    const index_t k_off = ku_ + (is_transposed ? -lhs_idx : lhs_idx);

    for (index_t s_idx = k_beg; s_idx < k_end; ++s_idx) {
      const index_t K = k_off + (is_transposed ? s_idx : -s_idx);
      const index_t J = is_transposed ? lhs_idx : s_idx;
      val = AddOperator::eval(
          val, ProductOperator::eval(matrix_.eval(K, J), vector_.eval(s_idx)));
    }

    lhs_.eval(lhs_idx, index_t(0)) = val;
  }
  return val;
}

template <typename lhs_t, typename matrix_t, typename vector_t,
          uint32_t local_range, bool is_transposed>
SYCL_BLAS_INLINE void Gbmv<lhs_t, matrix_t, vector_t, local_range,
                           is_transposed>::bind(cl::sycl::handler &h) {
  lhs_.bind(h);
  matrix_.bind(h);
  vector_.bind(h);
}
template <typename lhs_t, typename matrix_t, typename vector_t,
          uint32_t local_range, bool is_transposed>
SYCL_BLAS_INLINE void Gbmv<lhs_t, matrix_t, vector_t, local_range,
                           is_transposed>::adjust_access_displacement() {
  lhs_.adjust_access_displacement();
  matrix_.adjust_access_displacement();
  vector_.adjust_access_displacement();
}

}  // namespace blas
#endif
