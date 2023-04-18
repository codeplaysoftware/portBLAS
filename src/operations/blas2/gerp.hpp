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
typename rhs_1_t::value_t Gerp<Single, isUpper, lhs_t, rhs_1_t, rhs_2_t>::eval(
    cl::sycl::nd_item<1> ndItem) {
  const index_t id = ndItem.get_local_linear_id();
  const index_t group_id = ndItem.get_group(0);
  const index_t local_range = static_cast<index_t>(ndItem.get_local_range(0));
  const int64_t global_idx = group_id * local_range + id;
  const int64_t lhs_size = N_ * (N_ + 1) / 2;

  index_t row = 0, col = 0;
  value_t out{0};

  if (global_idx < lhs_size) {
    value_t lhs_val = lhs_.eval(global_idx);

    Gerp<Single, isUpper, lhs_t, rhs_1_t, rhs_2_t>::compute_row_col<isUpper>(
        global_idx, N_, row, col);

    value_t rhs_1_val = rhs_1_.eval(row);
    value_t rhs_2_val = rhs_2_.eval(col);
    if constexpr (!Single) {
      value_t rhs_1_val_second = rhs_1_.eval(col);
      value_t rhs_2_val_second = rhs_2_.eval(row);
      out = rhs_1_val * rhs_2_val * alpha_ +
            rhs_1_val_second * rhs_2_val_second * alpha_ + lhs_val;
    } else
      out = rhs_1_val * rhs_2_val * alpha_ + lhs_val;

    lhs_.eval(global_idx) = out;
  }
  return out;
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
