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
 *  @filename gerp.hpp
 *
 **************************************************************************/

#ifndef GERP_HPP
#define GERP_HPP

#include <operations/blas2_trees.h>

namespace blas {

// Row-Col index calculation for Lower Packed Matrix
template <bool Single, bool isUpper, typename lhs_t, typename rhs_1_t,
          typename rhs_2_t>
template <int N>
struct Gerp<Single, isUpper, lhs_t, rhs_1_t, rhs_2_t>::compute_row_col<N,
                                                                       false> {
  using index_t = Gerp<Single, isUpper, lhs_t, rhs_1_t, rhs_2_t>::index_t;
  using value_t = Gerp<Single, isUpper, lhs_t, rhs_1_t, rhs_2_t>::value_t;
  SYCL_BLAS_ALWAYS_INLINE void operator()(const int64_t id, const index_t size,
                                          index_t& row, index_t& col) {
    index_t temp = 2 * size + 1;
    int64_t internal = temp * temp - 8 * id;
    float val = internal * 1.f;
    float sqrt = 0.f;
    float divisor = internal >= 1048576 ? 2 * size * 1.f : 1.f;
    val = internal / (divisor * divisor);
    sqrt = cl::sycl::sqrt(val) * divisor;
    col = static_cast<index_t>((temp - sqrt) / 2);
    row = id - (col * (temp - col)) / 2 + col;
    // adjust row-col if out of bounds
    if (row < 0 || col < 0 || row >= size || col >= size || row < col) {
      int diff = id < size || row < col ? -1 : row >= size ? 1 : 0;
      col += diff;
      row = id - (col * (temp - col)) / 2 + col;
    }
  }
};

// Row-Col index calculation for Upper Packed Matrix
template <bool Single, bool isUpper, typename lhs_t, typename rhs_1_t,
          typename rhs_2_t>
template <int N>
struct Gerp<Single, isUpper, lhs_t, rhs_1_t, rhs_2_t>::compute_row_col<N,
                                                                       true> {
  using index_t = Gerp<Single, isUpper, lhs_t, rhs_1_t, rhs_2_t>::index_t;
  using value_t = Gerp<Single, isUpper, lhs_t, rhs_1_t, rhs_2_t>::value_t;
  SYCL_BLAS_ALWAYS_INLINE void operator()(const int64_t id, const index_t size,
                                          index_t& row, index_t& col) {
    int64_t internal = 1 + 8 * id;
    float val = internal * 1.f;
    float sqrt = 0.f;
    float divisor = id >= 1048576 ? size * 1.f : 1.f;
    val = internal / (divisor * divisor);
    sqrt = cl::sycl::sqrt(val) * divisor;
    col = static_cast<index_t>((-1 + sqrt) / 2);
    row = id - col * (col + 1) / 2;
    // adjust the row/col if out of bounds
    if (row > col) {
      int diff = row - col;
      col += diff;
      row -= col;
    } else if (row < 0) {
      col--;
      row = id - col * (col + 1) / 2;
    }
  }
};

/**** GERP N COLS x (N + 1)/2 ROWS FOR PACKED MATRIX ****/

template <bool Single, bool isUpper, typename lhs_t, typename rhs_1_t,
          typename rhs_2_t>
SYCL_BLAS_INLINE Gerp<Single, isUpper, lhs_t, rhs_1_t, rhs_2_t>::Gerp(
    lhs_t& _l, typename rhs_1_t::index_t _N, value_t _alpha, rhs_1_t& _r1,
    typename rhs_1_t::index_t _incX_1, rhs_2_t& _r2,
    typename rhs_1_t::index_t _incX_2)
    : lhs_(_l),
      N_(_N),
      alpha_(_alpha),
      rhs_1_(_r1),
      incX_1_(_incX_1),
      rhs_2_(_r2),
      incX_2_(_incX_2) {}

template <bool Single, bool isUpper, typename lhs_t, typename rhs_1_t,
          typename rhs_2_t>
void Gerp<Single, isUpper, lhs_t, rhs_1_t, rhs_2_t>::eval(
    cl::sycl::nd_item<1> ndItem) {
  const index_t id = ndItem.get_local_linear_id();
  const index_t group_id = ndItem.get_group(0);
  const index_t local_range = static_cast<index_t>(ndItem.get_local_range(0));
  const int64_t global_idx = group_id * local_range + id;
  const int64_t lhs_size = N_ * (N_ + 1) / 2;

  index_t row = 0, col = 0;

  auto lhs_ptr = lhs_.get_pointer();
  auto rhs_1_ptr = rhs_1_.get_pointer();
  auto rhs_2_ptr = rhs_2_.get_pointer();

  if (global_idx < lhs_size) {
    value_t lhs_val = *(lhs_ptr + global_idx);

    compute_row_col<0, isUpper> idx_compute_obj;
    idx_compute_obj(global_idx, N_, row, col);

    const index_t rhs_1_idx =
        (incX_1_ > 0 ? row : N_ - row - 1) * cl::sycl::abs(incX_1_);
    const index_t rhs_2_idx =
        (incX_2_ > 0 ? col : N_ - col - 1) * cl::sycl::abs(incX_2_);

    value_t rhs_1_val = *(rhs_1_ptr + rhs_1_idx);
    value_t rhs_2_val = *(rhs_2_ptr + rhs_2_idx);

    value_t out = rhs_1_val * rhs_2_val * alpha_ + lhs_val;

    *(lhs_ptr + global_idx) = out;
  }
}
template <bool Single, bool isUpper, typename lhs_t, typename rhs_1_t,
          typename rhs_2_t>
SYCL_BLAS_INLINE void Gerp<Single, isUpper, lhs_t, rhs_1_t, rhs_2_t>::bind(
    cl::sycl::handler& h) {
  lhs_.bind(h);
  rhs_1_.bind(h);
  rhs_2_.bind(h);
}

template <bool Single, bool isUpper, typename lhs_t, typename rhs_1_t,
          typename rhs_2_t>
SYCL_BLAS_INLINE void
Gerp<Single, isUpper, lhs_t, rhs_1_t, rhs_2_t>::adjust_access_displacement() {
  lhs_.adjust_access_displacement();
  rhs_1_.adjust_access_displacement();
  rhs_2_.adjust_access_displacement();
}

template <bool Single, bool isUpper, typename lhs_t, typename rhs_1_t,
          typename rhs_2_t>
SYCL_BLAS_INLINE
    typename Gerp<Single, isUpper, lhs_t, rhs_1_t, rhs_2_t>::index_t
    Gerp<Single, isUpper, lhs_t, rhs_1_t, rhs_2_t>::get_size() const {
  return rhs_1_.get_size();
}
template <bool Single, bool isUpper, typename lhs_t, typename rhs_1_t,
          typename rhs_2_t>
SYCL_BLAS_INLINE bool
Gerp<Single, isUpper, lhs_t, rhs_1_t, rhs_2_t>::valid_thread(
    cl::sycl::nd_item<1> ndItem) const {
  return true;
}

}  // namespace blas

#endif
