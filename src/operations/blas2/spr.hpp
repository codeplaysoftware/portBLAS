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

// Initial index calculation for Row Major Lower Packed Matrix
template <bool Single, bool isColMajor, bool isUpper, typename lhs_t,
          typename rhs_t>
template <int N>
struct Gerp<Single, isColMajor, isUpper, lhs_t, rhs_t>::get_init_idx<N, false,
                                                                     false> {
  using index_t = Gerp<Single, isColMajor, isUpper, lhs_t, rhs_t>::index_t;
  void operator()(index_t& init_idx, const index_t) { init_idx = 1; }
};

// Initial index calculation for Row Major Upper Packed Matrix
template <bool Single, bool isColMajor, bool isUpper, typename lhs_t,
          typename rhs_t>
template <int N>
struct Gerp<Single, isColMajor, isUpper, lhs_t, rhs_t>::get_init_idx<N, false,
                                                                     true> {
  using index_t = Gerp<Single, isColMajor, isUpper, lhs_t, rhs_t>::index_t;
  void operator()(index_t& init_idx, const index_t num) { init_idx = num; }
};

// Initial index calculation for Col Major Upper Packed Matrix
template <bool Single, bool isColMajor, bool isUpper, typename lhs_t,
          typename rhs_t>
template <int N>
struct Gerp<Single, isColMajor, isUpper, lhs_t, rhs_t>::get_init_idx<N, true,
                                                                     true> {
  using index_t = Gerp<Single, isColMajor, isUpper, lhs_t, rhs_t>::index_t;
  void operator()(index_t& init_idx, const index_t num) { init_idx = 1; }
};

// Initial index calculation for Col Major Lower Packed Matrix
template <bool Single, bool isColMajor, bool isUpper, typename lhs_t,
          typename rhs_t>
template <int N>
struct Gerp<Single, isColMajor, isUpper, lhs_t, rhs_t>::get_init_idx<N, true,
                                                                     false> {
  using index_t = Gerp<Single, isColMajor, isUpper, lhs_t, rhs_t>::index_t;
  void operator()(index_t& init_idx, const index_t num) { init_idx = num; }
};

// Row-Col index calculation for Row Major Lower Packed Matrix
template <bool Single, bool isColMajor, bool isUpper, typename lhs_t,
          typename rhs_t>
template <int N>
struct Gerp<Single, isColMajor, isUpper, lhs_t,
            rhs_t>::compute_row_col<N, false, false> {
  using index_t = Gerp<Single, isColMajor, isUpper, lhs_t, rhs_t>::index_t;
  void operator()(const index_t id, const index_t init_idx, index_t& row,
                  index_t& col) {
    index_t curr = init_idx;
    index_t prev = 0;

    for (index_t i = curr + 1; id - curr >= 0; i++, row++) {
      prev = curr;
      curr += i;
    }

    if (row > 0) {
      col = id - prev;
    }
  }
};

// Row-Col index calculation for Row Major Upper Packed Matrix
template <bool Single, bool isColMajor, bool isUpper, typename lhs_t,
          typename rhs_t>
template <int N>
struct Gerp<Single, isColMajor, isUpper, lhs_t,
            rhs_t>::compute_row_col<N, false, true> {
  using index_t = Gerp<Single, isColMajor, isUpper, lhs_t, rhs_t>::index_t;
  void operator()(const index_t id, const index_t init_idx, index_t& row,
                  index_t& col) {
    index_t curr = init_idx;
    index_t prev = 0;

    for (index_t i = curr - 1, j = 1; id - curr >= 0; i--, row++, j++) {
      prev = curr - j;
      curr += i;
    }

    if (row > 0) {
      col = id - prev;
    } else {
      col = id;
    }
  }
};

// Row-Col index calculation for Col Major Lower Packed Matrix
template <bool Single, bool isColMajor, bool isUpper, typename lhs_t,
          typename rhs_t>
template <int N>
struct Gerp<Single, isColMajor, isUpper, lhs_t, rhs_t>::compute_row_col<N, true,
                                                                        false> {
  using index_t = Gerp<Single, isColMajor, isUpper, lhs_t, rhs_t>::index_t;
  void operator()(const index_t id, const index_t init_idx, index_t& row,
                  index_t& col) {
    index_t curr = init_idx;
    index_t prev = 0;

    for (index_t i = curr - 1, j = 1; id - curr >= 0; i--, col++, j++) {
      prev = curr - j;
      curr += i;
    }

    if (col > 0) {
      row = id - prev;
    } else {
      row = id;
    }
  }
};

// Row-Col index calculation for Col Major Upper Packed Matrix
template <bool Single, bool isColMajor, bool isUpper, typename lhs_t,
          typename rhs_t>
template <int N>
struct Gerp<Single, isColMajor, isUpper, lhs_t, rhs_t>::compute_row_col<N, true,
                                                                        true> {
  using index_t = Gerp<Single, isColMajor, isUpper, lhs_t, rhs_t>::index_t;
  void operator()(const index_t id, const index_t init_idx, index_t& row,
                  index_t& col) {
    index_t curr = init_idx;
    index_t prev = 0;

    for (index_t i = curr + 1; id - curr >= 0; i++, col++) {
      prev = curr;
      curr += i;
    }

    if (col > 0) {
      row = id - prev;
    }
  }
};

/**** GERP N COLS x (N + 1)/2 ROWS FOR PACKED MATRIX ****/

template <bool Single, bool isColMajor, bool isUpper, typename lhs_t,
          typename rhs_t>
SYCL_BLAS_INLINE Gerp<Single, isColMajor, isUpper, lhs_t, rhs_t>::Gerp(
    lhs_t& _l, value_t _scl, rhs_t& _r)
    : lhs_(_l), scalar_(_scl), rhs_(_r) {}

template <bool Single, bool isColMajor, bool isUpper, typename lhs_t,
          typename rhs_t>
template <typename sharedT>
void Gerp<Single, isColMajor, isUpper, lhs_t, rhs_t>::eval(
    sharedT scratch_acc, cl::sycl::nd_item<1> ndItem) {
  const index_t id = ndItem.get_local_linear_id();
  const index_t group_id = ndItem.get_group(0);
  const index_t local_range = static_cast<index_t>(ndItem.get_local_range(0));
  const index_t global_idx = group_id * local_range + id;
  const index_t lhs_size = lhs_.get_size();
  const index_t rhs_size = rhs_.get_size();

  index_t row = 0;
  index_t col = 0;
  index_t start = 0;

  auto lhs_ptr = lhs_.get_pointer();
  auto rhs_ptr = rhs_.get_pointer();
  auto scratch = scratch_acc.localAcc.get_pointer();

  if (rhs_size < local_range) {
    if (id < rhs_size) {
      *(scratch + id) = *(rhs_ptr + id);
    }
  } else {
    index_t idx = id;
    for (index_t i = 0; i < rhs_size / local_range; i++) {
      *(scratch + id) = *(rhs_ptr + idx);
      idx += local_range;
      if (idx >= rhs_size) break;
    }
  }
  ndItem.barrier(cl::sycl::access::fence_space::local_space);

  if (global_idx < lhs_size) {
    auto lhs_val = *(lhs_ptr + global_idx);

    index_t init_idx = 0;
    get_init_idx<0, isColMajor, isUpper> init_obj;
    init_obj(init_idx, rhs_size);
    compute_row_col<0, isColMajor, isUpper> idx_compute_obj;
    idx_compute_obj(global_idx, init_idx, row, col);

    auto rhs_1_val = *(scratch + row);
    auto rhs_2_val = *(scratch + col);

    auto out = rhs_1_val * rhs_2_val * scalar_ + lhs_val;

    *(lhs_ptr + global_idx) = out;
  }
}
template <bool Single, bool isColMajor, bool isUpper, typename lhs_t,
          typename rhs_t>
SYCL_BLAS_INLINE void Gerp<Single, isColMajor, isUpper, lhs_t, rhs_t>::bind(
    cl::sycl::handler& h) {
  lhs_.bind(h);
  rhs_.bind(h);
}

template <bool Single, bool isColMajor, bool isUpper, typename lhs_t,
          typename rhs_t>
SYCL_BLAS_INLINE void
Gerp<Single, isColMajor, isUpper, lhs_t, rhs_t>::adjust_access_displacement() {
  lhs_.adjust_access_displacement();
  rhs_.adjust_access_displacement();
}

template <bool Single, bool isColMajor, bool isUpper, typename lhs_t,
          typename rhs_t>
SYCL_BLAS_INLINE
    typename Gerp<Single, isColMajor, isUpper, lhs_t, rhs_t>::index_t
    Gerp<Single, isColMajor, isUpper, lhs_t, rhs_t>::get_size() const {
  return rhs_.get_size();
}
template <bool Single, bool isColMajor, bool isUpper, typename lhs_t,
          typename rhs_t>
SYCL_BLAS_INLINE bool
Gerp<Single, isColMajor, isUpper, lhs_t, rhs_t>::valid_thread(
    cl::sycl::nd_item<1> ndItem) const {
  return true;
}

}  // namespace blas

#endif
