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
#include <operations/blas_operators.hpp>
#include <stdexcept>
#include <vector>
#include <views/view_sycl.hpp>

namespace blas {

/**** GERP BY COLUMNS M ROWS x N BLOCK FOR PACKED MATRIX ****/

template <bool Single, bool Lower, bool Diag, bool Upper, typename lhs_t,
          typename rhs_1_t, typename rhs_2_t>
SYCL_BLAS_INLINE GerPCol<Single, Lower, Diag, Upper, lhs_t, rhs_1_t,
                         rhs_2_t>::GerPCol(lhs_t &_l, value_t _scl,
                                           rhs_1_t &_r1, rhs_2_t &_r2,
                                           index_t &_nWG_col,
                                           index_t &_shrMemSize)
    : lhs_(_l),
      scalar_(_scl),
      rhs_1_(_r1),
      rhs_2_(_r2),
      nWG_col_(_nWG_col),
      local_memory_size_(_shrMemSize) {}

template <bool Single, bool Lower, bool Diag, bool Upper, typename lhs_t,
          typename rhs_1_t, typename rhs_2_t>
SYCL_BLAS_INLINE typename GerPCol<Single, Lower, Diag, Upper, lhs_t, rhs_1_t,
                                  rhs_2_t>::index_t
GerPCol<Single, Lower, Diag, Upper, lhs_t, rhs_1_t, rhs_2_t>::get_size() const {
  return rhs_1_.get_size();
}
template <bool Single, bool Lower, bool Diag, bool Upper, typename lhs_t,
          typename rhs_1_t, typename rhs_2_t>
SYCL_BLAS_INLINE bool
GerPCol<Single, Lower, Diag, Upper, lhs_t, rhs_1_t, rhs_2_t>::valid_thread(
    cl::sycl::nd_item<1> ndItem) const {
  return true;
}

template <bool Single, bool Lower, bool Diag, bool Upper, typename lhs_t,
          typename rhs_1_t, typename rhs_2_t>
SYCL_BLAS_INLINE typename GerPCol<Single, Lower, Diag, Upper, lhs_t, rhs_1_t,
                                  rhs_2_t>::value_t
GerPCol<Single, Lower, Diag, Upper, lhs_t, rhs_1_t, rhs_2_t>::eval(
    cl::sycl::nd_item<1> ndItem) {
  using index_t = typename GerPCol<Single, Lower, Diag, Upper, lhs_t, rhs_1_t,
                                   rhs_2_t>::index_t;
  const index_t id = ndItem.get_global_linear_id();
  const index_t lhs_size = lhs_.get_size();

  index_t row = 0;
  index_t col = 0;
  index_t start = -1;

  auto lhs_ptr = lhs_.get_pointer();
  auto rhs_1_ptr = rhs_1_.get_pointer();
  auto rhs_2_ptr = rhs_2_.get_pointer();

  if (id < lhs_size) {
    auto lhs_val = *(lhs_ptr + id);

    index_t val = 1;
    for (index_t i = 1; id - val > 0; i++, val += i, col++) {
      start = val;
    }

    --start;
    row = (id - start) % col;

    auto rhs_1_val = *(rhs_1_ptr + row);
    auto rhs_2_val = *(rhs_2_ptr + col);

    auto out = rhs_1_val * rhs_2_val * scalar_ + lhs_val;

    *(lhs_ptr + id) = out;

    return out;
  }
}
template <bool Single, bool Lower, bool Diag, bool Upper, typename lhs_t,
          typename rhs_1_t, typename rhs_2_t>
template <typename sharedT>
SYCL_BLAS_INLINE typename GerPCol<Single, Lower, Diag, Upper, lhs_t, rhs_1_t,
                                  rhs_2_t>::value_t
GerPCol<Single, Lower, Diag, Upper, lhs_t, rhs_1_t, rhs_2_t>::eval(
    sharedT shrMem, cl::sycl::nd_item<1> ndItem) {
  using index_t = typename GerPCol<Single, Lower, Diag, Upper, lhs_t, rhs_1_t,
                                   rhs_2_t>::index_t;
  const index_t id = ndItem.get_global_linear_id();
  const index_t lhs_size = lhs_.get_size();

  index_t row = 0;
  index_t col = 0;
  index_t start = -1;

  auto lhs_ptr = lhs_.get_pointer();
  auto rhs_1_ptr = rhs_1_.get_pointer();
  auto rhs_2_ptr = rhs_2_.get_pointer();

  if (id < lhs_size) {
    auto lhs_val = *(lhs_ptr + id);

    index_t val = 1;
    for (index_t i = 1; id - val > 0; i++, val += i, col++) {
      start = val;
    }

    --start;
    row = (id - start) % col;

    auto rhs_1_val = *(rhs_1_ptr + row);
    auto rhs_2_val = *(rhs_2_ptr + col);

    auto out = rhs_1_val * rhs_2_val * scalar_ + lhs_val;

    *(lhs_ptr + id) = out;

    return out;
  }
}

template <bool Single, bool Lower, bool Diag, bool Upper, typename lhs_t,
          typename rhs_1_t, typename rhs_2_t>
SYCL_BLAS_INLINE void GerPCol<Single, Lower, Diag, Upper, lhs_t, rhs_1_t,
                              rhs_2_t>::bind(cl::sycl::handler &h) {
  lhs_.bind(h);
  rhs_1_.bind(h);
  rhs_2_.bind(h);
}

template <bool Single, bool Lower, bool Diag, bool Upper, typename lhs_t,
          typename rhs_1_t, typename rhs_2_t>
SYCL_BLAS_INLINE void GerPCol<Single, Lower, Diag, Upper, lhs_t, rhs_1_t,
                              rhs_2_t>::adjust_access_displacement() {
  lhs_.adjust_access_displacement();
  rhs_1_.adjust_access_displacement();
  rhs_2_.adjust_access_displacement();
}

}  // namespace blas

#endif
