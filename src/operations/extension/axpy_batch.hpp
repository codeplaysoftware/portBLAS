/***************************************************************************
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
 *  @filename axpy_batch.hpp
 *
 **************************************************************************/

#ifndef PORTBLAS_EXTENSION_AXPY_BATCH_HPP
#define PORTBLAS_EXTENSION_AXPY_BATCH_HPP

#include "blas_meta.h"
#include "operations/extension/axpy_batch.h"

namespace blas {

template <bool sameSign, int localSize, int maxBlockPerBatch, typename lhs_t,
          typename rhs_t>
Axpy_batch<sameSign, localSize, maxBlockPerBatch, lhs_t, rhs_t>::Axpy_batch(
    lhs_t _lhs, rhs_t _rhs, typename lhs_t::value_t _alpha,
    typename rhs_t::index_t _N, typename rhs_t::index_t _inc_l,
    typename rhs_t::index_t _lhs_stride, typename rhs_t::index_t _inc_r,
    typename rhs_t::index_t _rhs_stride, typename rhs_t::index_t _batch_size)
    : lhs_(_lhs),
      rhs_(_rhs),
      alpha_(_alpha),
      n_(_N),
      inc_l(_inc_l),
      lhs_stride_(_lhs_stride),
      inc_r(_inc_r),
      rhs_stride_(_rhs_stride),
      batch_size_(_batch_size),
      n_block_per_loop(std::min((n_ + localSize - 1) / localSize,
                                static_cast<index_t>(maxBlockPerBatch))){};

template <bool sameSign, int localSize, int maxBlockPerBatch, typename lhs_t,
          typename rhs_t>
PORTBLAS_INLINE typename lhs_t::value_t
Axpy_batch<sameSign, localSize, maxBlockPerBatch, lhs_t, rhs_t>::eval(
    cl::sycl::nd_item<1> ndItem) {
  const index_t n{n_};
  const value_t alpha{alpha_};
  const auto vx = rhs_.get_data();
  const auto vy = lhs_.get_data();
  const auto nbl{n_block_per_loop};

  const index_t block_id = ndItem.get_group(0) % nbl;
  const index_t l_id =
      static_cast<index_t>(ndItem.get_local_range(0)) * block_id +
      ndItem.get_local_id(0);
  const index_t group_id = static_cast<index_t>(ndItem.get_group(0) / nbl);

  const index_t size_compute_rateo =
      (n > nbl * localSize) ? n / (nbl * localSize) : batch_size_;
  const index_t jump_value{sycl::min(batch_size_, size_compute_rateo)};

  if (group_id >= jump_value || l_id > n) return {};

  const index_t stride_x = ndItem.get_local_range(0) * nbl * inc_r;
  const index_t stride_y = ndItem.get_local_range(0) * nbl * inc_l;
  index_t x_index{};
  index_t y_index{};
  int j{};

  if constexpr (sameSign) {
    for (auto out_loop = group_id; out_loop < batch_size_;
         out_loop += jump_value) {
      x_index = out_loop * rhs_stride_ + l_id * inc_r;
      y_index = out_loop * lhs_stride_ + l_id * inc_l;
      j = y_index;
      for (auto i = x_index; i < (out_loop * rhs_stride_) + n * inc_r;
           i += stride_x, j += stride_y) {
        vy[j] += alpha * vx[i];
      }
    }

  } else {
    for (auto out_loop = group_id; out_loop < batch_size_;
         out_loop += jump_value) {
      x_index = out_loop * rhs_stride_ + inc_r + n * (-inc_r) + l_id * inc_r;
      y_index = out_loop * lhs_stride_ + l_id * inc_l;
      j = y_index;
      for (auto i = x_index; i >= (out_loop * rhs_stride_);
           i += stride_x, j += stride_y) {
        vy[j] += alpha * vx[i];
      }
    }
  }

  return {};
}

template <bool sameSign, int localSize, int maxBlockPerBatch, typename lhs_t,
          typename rhs_t>
PORTBLAS_INLINE void Axpy_batch<sameSign, localSize, maxBlockPerBatch, lhs_t,
                                rhs_t>::bind(cl::sycl::handler& h) {
  lhs_.bind(h);
  rhs_.bind(h);
}

template <bool sameSign, int localSize, int maxBlockPerBatch, typename lhs_t,
          typename rhs_t>
PORTBLAS_INLINE void Axpy_batch<sameSign, localSize, maxBlockPerBatch, lhs_t,
                                rhs_t>::adjust_access_displacement() {
  lhs_.adjust_access_displacement();
  rhs_.adjust_access_displacement();
}

template <bool sameSign, int localSize, int maxBlockPerBatch, typename lhs_t,
          typename rhs_t>
PORTBLAS_INLINE typename rhs_t::index_t Axpy_batch<
    sameSign, localSize, maxBlockPerBatch, lhs_t, rhs_t>::get_size() const {
  return n_ * batch_size_;
}

template <bool sameSign, int localSize, int maxBlockPerBatch, typename lhs_t,
          typename rhs_t>
PORTBLAS_INLINE bool
Axpy_batch<sameSign, localSize, maxBlockPerBatch, lhs_t, rhs_t>::valid_thread(
    cl::sycl::nd_item<1> ndItem) const {
  return true;
}
}  // namespace blas

#endif
