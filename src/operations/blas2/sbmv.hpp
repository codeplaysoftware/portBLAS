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
 *  @filename sbmv.hpp
 *
 **************************************************************************/

#ifndef SBMV_HPP
#define SBMV_HPP
#include "operations/blas2_trees.h"
#include "operations/blas_operators.hpp"
#include "views/view_sycl.hpp"
#include <stdexcept>
#include <vector>
namespace blas {

/**
 * @struct Sbmv
 * @brief Tree node representing a symmetric band matrix_ vector_
 * multiplication.
 */
template <typename lhs_t, typename matrix_t, typename vector_t,
          uint32_t local_range, bool is_upper>
SYCL_BLAS_INLINE Sbmv<lhs_t, matrix_t, vector_t, local_range, is_upper>::Sbmv(
    lhs_t &_l, matrix_t &_matrix,
    typename Sbmv<lhs_t, matrix_t, vector_t, local_range, is_upper>::index_t
        &_k,
    vector_t &_vector,
    typename Sbmv<lhs_t, matrix_t, vector_t, local_range, is_upper>::value_t
        _alpha,
    typename Sbmv<lhs_t, matrix_t, vector_t, local_range, is_upper>::value_t
        _beta)
    : lhs_(_l),
      matrix_(_matrix),
      vector_(_vector),
      k_(_k),
      alpha_(_alpha),
      beta_(_beta) {}

template <typename lhs_t, typename matrix_t, typename vector_t,
          uint32_t local_range, bool is_upper>
SYCL_BLAS_INLINE
    typename Sbmv<lhs_t, matrix_t, vector_t, local_range, is_upper>::index_t
    Sbmv<lhs_t, matrix_t, vector_t, local_range, is_upper>::get_size() const {
  return matrix_.get_size();
}
template <typename lhs_t, typename matrix_t, typename vector_t,
          uint32_t local_range, bool is_upper>
SYCL_BLAS_INLINE bool
Sbmv<lhs_t, matrix_t, vector_t, local_range, is_upper>::valid_thread(
    cl::sycl::nd_item<1> ndItem) const {
  // Valid threads are established by ::eval.
  return true;
}

template <typename lhs_t, typename matrix_t, typename vector_t,
          uint32_t local_range, bool is_upper>
SYCL_BLAS_INLINE
    typename Sbmv<lhs_t, matrix_t, vector_t, local_range, is_upper>::value_t
    Sbmv<lhs_t, matrix_t, vector_t, local_range, is_upper>::eval(
        cl::sycl::nd_item<1> ndItem) {
  const index_t lhs_idx =
      ndItem.get_group(0) * local_range + ndItem.get_local_id(0);
  value_t val = 0;

  if (lhs_idx < lhs_.get_size()) {
    const index_t k_beg = cl::sycl::max(index_t(0), lhs_idx - k_);
    const index_t k_end = cl::sycl::min(vector_.get_size(), lhs_idx + k_ + 1);

    for (index_t s_idx = k_beg; s_idx < k_end; ++s_idx) {
      index_t K, J;

      if (is_upper) {
        K = k_ + ((s_idx < lhs_idx) ? (s_idx - lhs_idx) : (lhs_idx - s_idx));
        J = (s_idx < lhs_idx) ? lhs_idx : s_idx;
      } else {
        K = (s_idx < lhs_idx) ? lhs_idx - s_idx : s_idx - lhs_idx;
        J = (s_idx < lhs_idx) ? s_idx : lhs_idx;
      }

      val = AddOperator::eval(
          val, ProductOperator::eval(matrix_.eval(K, J), vector_.eval(s_idx)));
    }

    lhs_.eval(lhs_idx) =
        AddOperator::eval(ProductOperator::eval(alpha_, val),
                          ProductOperator::eval(beta_, lhs_.eval(lhs_idx)));
  }
  return val;
}

template <typename lhs_t, typename matrix_t, typename vector_t,
          uint32_t local_range, bool is_upper>
SYCL_BLAS_INLINE void Sbmv<lhs_t, matrix_t, vector_t, local_range,
                           is_upper>::bind(cl::sycl::handler &h) {
  lhs_.bind(h);
  matrix_.bind(h);
  vector_.bind(h);
}
template <typename lhs_t, typename matrix_t, typename vector_t,
          uint32_t local_range, bool is_upper>
SYCL_BLAS_INLINE void Sbmv<lhs_t, matrix_t, vector_t, local_range,
                           is_upper>::adjust_access_displacement() {
  lhs_.adjust_access_displacement();
  matrix_.adjust_access_displacement();
  vector_.adjust_access_displacement();
}

}  // namespace blas
#endif
