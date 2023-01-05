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
 *  @filename spr.hpp
 *
 **************************************************************************/

#ifndef SPR_HPP
#define SPR_HPP

#include <operations/blas2_trees.h>
#include <operations/blas_operators.hpp>
#include <stdexcept>
#include <vector>
#include <views/view_sycl.hpp>

namespace blas {

/**** SPR COL MAJOR N COLS x (N + 1)/2 ROWS FOR PACKED MATRIX ****/

template <bool Single, typename lhs_t, typename rhs_t>
struct SprCol<Single, false, true, true, lhs_t, rhs_t> {
  using value_t = typename rhs_t::value_t;
  using index_t = typename rhs_t::index_t;

  lhs_t lhs_;
  rhs_t rhs_;
  value_t scalar_;

  SYCL_BLAS_INLINE SprCol(lhs_t &_l, value_t _scl, rhs_t &_r)
      : lhs_(_l), scalar_(_scl), rhs_(_r) {}

  template <typename sharedT>
  value_t eval(sharedT scratch_acc, cl::sycl::nd_item<1> ndItem) {
    const index_t id = ndItem.get_global_linear_id();
    const index_t local_range = static_cast<index_t>(ndItem.get_local_range(0));
    const index_t lhs_size = lhs_.get_size();
    const index_t rhs_size = rhs_.get_size();

    index_t row = 0;
    index_t col = 0;
    index_t start = 0;

    auto lhs_ptr = lhs_.get_pointer();
    auto scratch = scratch_acc.localAcc.get_pointer();
    auto rhs_ptr = rhs_.get_pointer();

    if (rhs_size < local_range) {
      if (id < rhs_size) {
        *(scratch + id) = *(rhs_ptr + id);
      }
    } else {
      index_t idx = id;
      for (index_t i = 0; i < rhs_size / local_range; i++) {
        *(scratch + idx) = *(rhs_ptr + idx);
        idx += local_range;
        if (idx >= rhs_size) break;
      }
    }
    ndItem.barrier(cl::sycl::access::fence_space::local_space);

    if (id < lhs_size) {
      auto lhs_val = *(lhs_ptr + id);

      index_t val = 1;
      for (index_t i = val + 1; id - val >= 0; i++, col++) {
        start = val;
        val += i;
      }

      if (col > 0) {
        row = id - start;
      }

      auto rhs_1_val = *(scratch + row);
      auto rhs_2_val = *(scratch + col);

      auto out = rhs_1_val * rhs_2_val * scalar_ + lhs_val;

      return *(lhs_ptr + id) = out;
    } else {
      return static_cast<value_t>(0);
    }
  }
  SYCL_BLAS_INLINE void bind(cl::sycl::handler &h) {
    lhs_.bind(h);
    rhs_.bind(h);
  }

  SYCL_BLAS_INLINE void adjust_access_displacement() {
    lhs_.adjust_access_displacement();
    rhs_.adjust_access_displacement();
  }

  SYCL_BLAS_INLINE index_t get_size() const { return rhs_.get_size(); }
  SYCL_BLAS_INLINE bool valid_thread(cl::sycl::nd_item<1> ndItem) const {
    return true;
  }
};

template <bool Single, typename lhs_t, typename rhs_t>
struct SprCol<Single, true, true, false, lhs_t, rhs_t> {
  using value_t = typename rhs_t::value_t;
  using index_t = typename rhs_t::index_t;

  lhs_t lhs_;
  rhs_t rhs_;
  value_t scalar_;

  SYCL_BLAS_INLINE SprCol(lhs_t &_l, value_t _scl, rhs_t &_r)
      : lhs_(_l), scalar_(_scl), rhs_(_r) {}

  template <typename sharedT>
  value_t eval(sharedT scratch_acc, cl::sycl::nd_item<1> ndItem) {
    const index_t id = ndItem.get_global_linear_id();
    const index_t local_range = static_cast<index_t>(ndItem.get_local_range(0));
    const index_t lhs_size = lhs_.get_size();
    const index_t rhs_size = rhs_.get_size();

    index_t row = 0;
    index_t col = 0;
    index_t start = 0;

    auto lhs_ptr = lhs_.get_pointer();
    auto scratch = scratch_acc.localAcc.get_pointer();
    auto rhs_ptr = rhs_.get_pointer();

    if (rhs_size < local_range) {
      if (id < rhs_size) {
        *(scratch + id) = *(rhs_ptr + id);
      }
    } else {
      index_t idx = id;
      for (index_t i = 0; i < rhs_size / local_range; i++) {
        *(scratch + idx) = *(rhs_ptr + idx);
        idx += local_range;
        if (idx >= rhs_size) break;
      }
    }
    ndItem.barrier(cl::sycl::access::fence_space::local_space);

    if (id < lhs_size) {
      auto lhs_val = *(lhs_ptr + id);

      index_t val = rhs_size;
      for (index_t i = val - 1, j = 1; id - val >= 0; i--, col++, j++) {
        start = val - j;
        val += i;
      }

      if (col > 0) {
        row = id - start;
      } else {
        row = id;
      }

      auto rhs_1_val = *(scratch + row);
      auto rhs_2_val = *(scratch + col);

      auto out = rhs_1_val * rhs_2_val * scalar_ + lhs_val;

      return *(lhs_ptr + id) = out;
    } else {
      return static_cast<value_t>(0);
    }
  }

  SYCL_BLAS_INLINE void bind(cl::sycl::handler &h) {
    lhs_.bind(h);
    rhs_.bind(h);
  }

  SYCL_BLAS_INLINE void adjust_access_displacement() {
    lhs_.adjust_access_displacement();
    rhs_.adjust_access_displacement();
  }

  SYCL_BLAS_INLINE index_t get_size() const { return rhs_.get_size(); }
  SYCL_BLAS_INLINE bool valid_thread(cl::sycl::nd_item<1> ndItem) const {
    return true;
  }
};

}  // namespace blas

#endif
