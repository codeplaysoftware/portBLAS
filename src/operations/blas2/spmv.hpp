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
 *  @filename spmv.hpp
 *
 **************************************************************************/

#ifndef SPMV_HPP
#define SPMV_HPP
#include "operations/blas2_trees.h"
#include "operations/blas_operators.hpp"
#include "views/view_sycl.hpp"
namespace blas {

/**
 * @struct Spmv
 * @brief Tree node representing a symmetric band matrix_ vector_
 * multiplication.
 */
template <typename lhs_t, typename matrix_t, typename vector_t,
          uint32_t local_range_x, uint32_t local_range_y, bool is_upper>
SYCL_BLAS_INLINE
Spmv<lhs_t, matrix_t, vector_t, local_range_x, local_range_y, is_upper>::Spmv(
    lhs_t &_l, matrix_t &_matrix, vector_t &_vector,
    typename Spmv<lhs_t, matrix_t, vector_t, local_range_x, local_range_y,
                  is_upper>::value_t _alpha,
    typename Spmv<lhs_t, matrix_t, vector_t, local_range_x, local_range_y,
                  is_upper>::value_t _beta)
    : lhs_(_l),
      matrix_(_matrix),
      vector_(_vector),
      alpha_(_alpha),
      beta_(_beta) {}

template <typename lhs_t, typename matrix_t, typename vector_t,
          uint32_t local_range_x, uint32_t local_range_y, bool is_upper>
SYCL_BLAS_INLINE typename Spmv<lhs_t, matrix_t, vector_t, local_range_x,
                               local_range_y, is_upper>::index_t
Spmv<lhs_t, matrix_t, vector_t, local_range_x, local_range_y,
     is_upper>::get_size() const {
  return matrix_.get_size();
}
template <typename lhs_t, typename matrix_t, typename vector_t,
          uint32_t local_range_x, uint32_t local_range_y, bool is_upper>
SYCL_BLAS_INLINE bool
Spmv<lhs_t, matrix_t, vector_t, local_range_x, local_range_y,
     is_upper>::valid_thread(cl::sycl::nd_item<1> ndItem) const {
  // Valid threads are established by ::eval.
  return true;
}

template <typename lhs_t, typename matrix_t, typename vector_t,
          uint32_t local_range_x, uint32_t local_range_y, bool is_upper>
template <typename sharedT>
SYCL_BLAS_INLINE typename Spmv<lhs_t, matrix_t, vector_t, local_range_x,
                               local_range_y, is_upper>::value_t
Spmv<lhs_t, matrix_t, vector_t, local_range_x, local_range_y, is_upper>::eval(
    sharedT shrMem, cl::sycl::nd_item<1> ndItem) {
  const index_t gid = ndItem.get_group(0);

  constexpr index_t x_range = local_range_x;
  constexpr index_t y_range = local_range_y;
  constexpr index_t y_chunck = x_range / y_range;
  constexpr index_t loc_lda = x_range + 1;

  const index_t l_idx = ndItem.get_local_id(0) % x_range;
  const index_t l_idy = ndItem.get_local_id(0) / x_range;
  const index_t l_y_offset = y_chunck * l_idy;

  const index_t N = lhs_.get_size();
  const index_t nblock = (N + x_range - 1) / x_range;

  value_t *const loc_x = shrMem.localAcc.get_pointer();
  value_t *const loc_A = loc_x + x_range;

  // ------------------------------------------------------------------------ //

  auto _mat_J_offset = [&N](const index_t &_J) {
    return is_upper ? ((_J * (_J + 1)) / 2) : (_J * N) - ((_J * (_J + 1)) / 2);
  };

  auto _mat_initial_stride = [&N](const index_t &_J) {
    return is_upper ? _J + 1 : N - _J - 1;
  };

  auto _mat_next_stride = [](index_t &_stride) {
    return is_upper ? _stride++ : _stride--;
  };

  // ------------------------------------------------------------------------ //

  // HORIZONTAL

  value_t priv_res = value_t(0);

  {
    const index_t I_offset = gid * x_range + l_idx;
    value_t *const A_I_offset = matrix_.get_pointer() + I_offset;

    value_t priv_A[y_chunck];

    // it doesn't need local memory for storing the matrix
    for (index_t b = (is_upper ? gid + 1 : 0); b < (is_upper ? nblock : gid);
         ++b) {
      if (!l_idy) {
        const index_t x_idx = b * x_range + l_idx;
        const bool read_it = is_upper ? (x_idx < N) : true;
        loc_x[l_idx] = read_it ? vector_.eval(x_idx) : value_t(0);
      }

      const index_t J = b * x_range + l_y_offset;
      value_t *A = A_I_offset + _mat_J_offset(J);
      index_t stride = _mat_initial_stride(J);

      // load A to registers
#pragma unroll
      for (index_t _j = 0; _j < y_chunck; ++_j) {
        const bool read_it = is_upper ? (J + _j < N) : (I_offset < N);
        priv_A[_j] = read_it ? *A : value_t(0);
        A += _mat_next_stride(stride);
      }

      // wait for x
      ndItem.barrier(cl::sycl::access::fence_space::local_space);

      // compute
#pragma unroll
      for (index_t _j = 0; _j < y_chunck; ++_j)
        priv_res += priv_A[_j] * loc_x[l_y_offset + _j];

      ndItem.barrier(cl::sycl::access::fence_space::local_space);
    }
  }

  // ------------------------------------------------------------------------ //

  const index_t J = gid * x_range + l_y_offset;
  value_t *const A_J_offset = matrix_.get_pointer() + _mat_J_offset(J) + l_idx;

  // ------------------------------------------------------------------------ //

  // CENTER
  {
    if (!l_idy) {
      const index_t x_idx = gid * x_range + l_idx;
      const bool read_it = x_idx < N;
      loc_x[l_idx] = read_it ? vector_.eval(x_idx) : value_t(0);
    }

    const index_t I_offset = gid * x_range;
    value_t *A = A_J_offset + I_offset;
    index_t stride = _mat_initial_stride(J);

#pragma unroll
    for (index_t _j = 0; _j < y_chunck; ++_j) {
      const index_t j = l_y_offset + _j;
      const bool read_it = is_upper ? (J + _j < N) : (I_offset + l_idx < N);

      if ((!is_upper && l_idx > j) || (is_upper && l_idx < j))
        loc_A[loc_lda * l_idx + j] = loc_A[loc_lda * j + l_idx] =
            read_it ? *A : value_t(0);

      if (l_idx == j) loc_A[loc_lda * j + l_idx] = read_it ? *A : value_t(0);

      A += _mat_next_stride(stride);
    }
  }

  // ------------------------------------------------------------------------ //

  // VERTICAL

  {
    value_t priv_A[y_chunck];

    // this stores the matrix in local memory
    for (index_t b = (is_upper ? 0 : gid + 1); b < (is_upper ? gid : nblock);
         ++b) {
      ndItem.barrier(cl::sycl::access::fence_space::local_space);
#pragma unroll
      for (index_t _j = 0; _j < y_chunck; ++_j) {
        const index_t j = l_y_offset + _j;
        priv_res += loc_A[loc_lda * j + l_idx] * loc_x[j];
      }

      const index_t I_offset = b * x_range;
      value_t *A = A_J_offset + I_offset;
      index_t stride = _mat_initial_stride(J);

      // row full-blocks
#pragma unroll
      for (index_t _j = 0; _j < y_chunck; ++_j) {
        const index_t j = l_y_offset + _j;
        const bool read_it = is_upper ? (J + _j < N) : (I_offset + l_idx < N);
        priv_A[_j] = read_it ? *A : value_t(0);
        A += _mat_next_stride(stride);
      }

      ndItem.barrier(cl::sycl::access::fence_space::local_space);

      if (!l_idy) {
        const index_t x_idx = b * x_range + l_idx;
        const bool read_it = is_upper ? true : (x_idx < N);
        loc_x[l_idx] = read_it ? vector_.eval(x_idx) : value_t(0);
      }

#pragma unroll
      for (index_t _j = 0; _j < y_chunck; ++_j) {
        const index_t j = l_y_offset + _j;
        loc_A[loc_lda * l_idx + j] = priv_A[_j];
      }
    }
  }

  ndItem.barrier(cl::sycl::access::fence_space::local_space);

#pragma unroll
  for (index_t _j = 0; _j < y_chunck; ++_j) {
    const index_t j = l_y_offset + _j;
    priv_res += loc_A[loc_lda * j + l_idx] * loc_x[j];
  }

  ndItem.barrier(cl::sycl::access::fence_space::local_space);

  loc_A[loc_lda * l_idy + l_idx] = priv_res;

  ndItem.barrier(cl::sycl::access::fence_space::local_space);

  if (!l_idy) {
    value_t res = value_t(0);
#pragma unroll
    for (index_t _j = 0; _j < y_range; ++_j) res += loc_A[loc_lda * _j + l_idx];

    const index_t lhs_idx = gid * x_range + l_idx;
    if (lhs_idx < N)
      lhs_.eval(lhs_idx) =
          AddOperator::eval(ProductOperator::eval(alpha_, res),
                            ProductOperator::eval(beta_, lhs_.eval(lhs_idx)));
  }

  return 0;
}

template <typename lhs_t, typename matrix_t, typename vector_t,
          uint32_t local_range_x, uint32_t local_range_y, bool is_upper>
SYCL_BLAS_INLINE void
Spmv<lhs_t, matrix_t, vector_t, local_range_x, local_range_y, is_upper>::bind(
    cl::sycl::handler &h) {
  lhs_.bind(h);
  matrix_.bind(h);
  vector_.bind(h);
}
template <typename lhs_t, typename matrix_t, typename vector_t,
          uint32_t local_range_x, uint32_t local_range_y, bool is_upper>
SYCL_BLAS_INLINE void
Spmv<lhs_t, matrix_t, vector_t, local_range_x, local_range_y,
     is_upper>::adjust_access_displacement() {
  lhs_.adjust_access_displacement();
  matrix_.adjust_access_displacement();
  vector_.adjust_access_displacement();
}

}  // namespace blas
#endif
