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
 *  @filename spr.hpp
 *
 **************************************************************************/

#ifndef SPR_HPP
#define SPR_HPP

#include <operations/blas2_trees.h>

namespace blas {

template <bool Single, bool isUpper, typename lhs_t, typename rhs_1_t,
          typename rhs_2_t>
PORTBLAS_INLINE Spr<Single, isUpper, lhs_t, rhs_1_t, rhs_2_t>::Spr(
    lhs_t& _l, typename rhs_1_t::index_t _N, value_t _alpha, rhs_1_t& _r1,
    rhs_2_t& _r2)
    : lhs_(_l), N_(_N), alpha_(_alpha), rhs_1_(_r1), rhs_2_(_r2) {}

/*!
 * @brief Compute the integer square root of an integer value by means of a
 * fixed-point iteration method.
 */
template <bool Single, bool isUpper, typename lhs_t, typename rhs_1_t,
          typename rhs_2_t>
PORTBLAS_ALWAYS_INLINE typename rhs_1_t::index_t
Spr<Single, isUpper, lhs_t, rhs_1_t, rhs_2_t>::int_sqrt(int64_t val) {
  using index_t = typename rhs_1_t::index_t;

  if (val < 2) return val;

  // Compute x0 as 2^(floor(log2(val)/2) + 1)
  index_t p = 0;
  int64_t tmp = val;
  while (tmp) {
    ++p;
    tmp >>= 1;
  }
  index_t x0 = 2 << (p / 2);
  index_t x1 = (x0 + val / x0) / 2;

#pragma unroll 5
  while (x1 < x0) {
    x0 = x1;
    x1 = (x0 + val / x0) / 2;
  }
  return x0;
}

/*!
 * @brief Map a global work-item index to triangular matrix coordinates.
 */
template <bool Single, bool isUpper, typename lhs_t, typename rhs_1_t,
          typename rhs_2_t>
PORTBLAS_ALWAYS_INLINE void
Spr<Single, isUpper, lhs_t, rhs_1_t, rhs_2_t>::compute_row_col(
    const int64_t id, const typename rhs_1_t::index_t size,
    typename rhs_1_t::index_t& row, typename rhs_1_t::index_t& col) {
  using index_t = typename rhs_1_t::index_t;
  if constexpr (isUpper) {
    const index_t i = (int_sqrt(8L * id + 1L) - 1) / 2;
    col = i;
    row = id - (i * (i + 1)) / 2;
  } else {
    const index_t rid = size * (size + 1) / 2 - id - 1;
    const index_t i = (int_sqrt(8L * rid + 1L) - 1) / 2;
    col = size - 1 - i;
    row = size - 1 - (rid - i * (i + 1) / 2);
  }
}

template <bool Single, bool isUpper, typename lhs_t, typename rhs_1_t,
          typename rhs_2_t>
typename rhs_1_t::value_t Spr<Single, isUpper, lhs_t, rhs_1_t, rhs_2_t>::eval(
    cl::sycl::nd_item<1> ndItem) {
  const index_t id = ndItem.get_local_linear_id();
  const index_t group_id = ndItem.get_group(0);
  const index_t local_range = static_cast<index_t>(ndItem.get_local_range(0));
  const int64_t global_idx = group_id * local_range + id;
  const int64_t lhs_size = N_ * (N_ + 1) / 2;

  index_t row = 0, col = 0;

  if (!id) {
    Spr<Single, isUpper, lhs_t, rhs_1_t, rhs_2_t>::compute_row_col(
        global_idx, N_, row, col);
  }

  row = cl::sycl::group_broadcast(ndItem.get_group(), row);
  col = cl::sycl::group_broadcast(ndItem.get_group(), col);

  if (global_idx < lhs_size) {
    if constexpr (isUpper) {
      if (id) {
        row += id;
        while (row > col) {
          ++col;
          row -= col;
        }
      }
    } else {
      if (id) {
        row += id;
        while (row >= N_) {
          ++col;
          row = row - N_ + col;
        }
      }
    }

    value_t lhs_val = lhs_.eval(global_idx);
    value_t rhs_1_val = rhs_1_.eval(row);
    value_t rhs_2_val = rhs_2_.eval(col);
    if constexpr (!Single) {
      value_t rhs_1_val_second = rhs_1_.eval(col);
      value_t rhs_2_val_second = rhs_2_.eval(row);
      lhs_.eval(global_idx) = rhs_1_val * rhs_2_val * alpha_ +
            rhs_1_val_second * rhs_2_val_second * alpha_ + lhs_val;
    } else
      lhs_.eval(global_idx) = rhs_1_val * rhs_2_val * alpha_ + lhs_val;
  }
  return lhs_.eval(global_idx);
}
template <bool Single, bool isUpper, typename lhs_t, typename rhs_1_t,
          typename rhs_2_t>
PORTBLAS_INLINE void Spr<Single, isUpper, lhs_t, rhs_1_t, rhs_2_t>::bind(
    cl::sycl::handler& h) {
  lhs_.bind(h);
  rhs_1_.bind(h);
  rhs_2_.bind(h);
}

template <bool Single, bool isUpper, typename lhs_t, typename rhs_1_t,
          typename rhs_2_t>
PORTBLAS_INLINE void
Spr<Single, isUpper, lhs_t, rhs_1_t, rhs_2_t>::adjust_access_displacement() {
  lhs_.adjust_access_displacement();
  rhs_1_.adjust_access_displacement();
  rhs_2_.adjust_access_displacement();
}

template <bool Single, bool isUpper, typename lhs_t, typename rhs_1_t,
          typename rhs_2_t>
PORTBLAS_INLINE
    typename Spr<Single, isUpper, lhs_t, rhs_1_t, rhs_2_t>::index_t
    Spr<Single, isUpper, lhs_t, rhs_1_t, rhs_2_t>::get_size() const {
  return rhs_1_.get_size();
}
template <bool Single, bool isUpper, typename lhs_t, typename rhs_1_t,
          typename rhs_2_t>
PORTBLAS_INLINE bool
Spr<Single, isUpper, lhs_t, rhs_1_t, rhs_2_t>::valid_thread(
    cl::sycl::nd_item<1> ndItem) const {
  return true;
}

}  // namespace blas

#endif
