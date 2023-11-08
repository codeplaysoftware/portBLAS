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

template <bool same_sign, typename lhs_t, typename rhs_t>
Axpy_batch<same_sign, lhs_t, rhs_t>::Axpy_batch(
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
      batch_size_(_batch_size){};

template <bool same_sign, typename lhs_t, typename rhs_t>
PORTBLAS_INLINE typename lhs_t::value_t
Axpy_batch<same_sign, lhs_t, rhs_t>::eval(index_t i) {}

template <bool same_sign, typename lhs_t, typename rhs_t>
PORTBLAS_INLINE typename lhs_t::value_t
Axpy_batch<same_sign, lhs_t, rhs_t>::eval(cl::sycl::nd_item<1> ndItem) {
  const index_t n{n_};
  const value_t alpha{alpha_};

  const index_t l_id = static_cast<index_t>(ndItem.get_global_linear_id() % n);
  const index_t group_id =
      static_cast<index_t>(ndItem.get_global_linear_id() / n);

  if (group_id >= batch_size_) return {};

  if constexpr (same_sign) {
    const index_t x_index = group_id * rhs_stride_ + l_id * inc_r;
    const index_t y_index = group_id * lhs_stride_ + l_id * inc_l;

    const value_t ax = alpha * rhs_.get_data()[x_index];
    lhs_.get_data()[y_index] += ax;

  } else {
    const index_t x_index =
        group_id * rhs_stride_ + inc_r + n * sycl::abs(inc_r) + l_id * inc_r;
    const index_t y_index = group_id * lhs_stride_ + l_id * inc_l;

    const value_t ax = alpha * rhs_.get_data()[x_index];
    lhs_.get_data()[y_index] += ax;
  }

  return {};
}

template <bool same_sign, typename lhs_t, typename rhs_t>
template <typename sharedT>
PORTBLAS_INLINE typename lhs_t::value_t
Axpy_batch<same_sign, lhs_t, rhs_t>::eval(sharedT shMem,
                                          sycl::nd_item<1> ndItem){};

template <bool same_sign, typename lhs_t, typename rhs_t>
PORTBLAS_INLINE void Axpy_batch<same_sign, lhs_t, rhs_t>::bind(
    cl::sycl::handler& h) {
  lhs_.bind(h);
  rhs_.bind(h);
}

template <bool same_sign, typename lhs_t, typename rhs_t>
PORTBLAS_INLINE void
Axpy_batch<same_sign, lhs_t, rhs_t>::adjust_access_displacement() {
  lhs_.adjust_access_displacement();
  rhs_.adjust_access_displacement();
}

template <bool same_sign, typename lhs_t, typename rhs_t>
PORTBLAS_INLINE typename rhs_t::index_t
Axpy_batch<same_sign, lhs_t, rhs_t>::get_size() const {
  return n_ * batch_size_;
}

template <bool same_sign, typename lhs_t, typename rhs_t>
PORTBLAS_INLINE bool Axpy_batch<same_sign, lhs_t, rhs_t>::valid_thread(
    cl::sycl::nd_item<1> ndItem) const {
  return true;
}
}  // namespace blas

#endif
