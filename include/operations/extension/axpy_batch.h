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
 *  @filename axpy_batch.h
 *
 **************************************************************************/

#ifndef PORTBLAS_EXTENSION_AXPY_BATCH_H
#define PORTBLAS_EXTENSION_AXPY_BATCH_H

namespace blas {

template <bool same_sign, typename lhs_t, typename rhs_t>
struct Axpy_batch {
  using value_t = typename lhs_t::value_t;
  using index_t = typename rhs_t::index_t;

  lhs_t lhs_;
  rhs_t rhs_;
  value_t alpha_;
  index_t n_, inc_r, inc_l, lhs_stride_, rhs_stride_, batch_size_;

  Axpy_batch(lhs_t _lhs, rhs_t _rhs_1, value_t _alpha, index_t _N,
             index_t _inc_l, index_t _lhs_stride, index_t _inc_r,
             index_t _rhs_stride, index_t _batch_size);
  index_t get_size() const;
  bool valid_thread(cl::sycl::nd_item<1> ndItem) const;
  value_t eval(index_t i);
  value_t eval(cl::sycl::nd_item<1> ndItem);
  template <typename sharedT>
  value_t eval(sharedT shMem, cl::sycl::nd_item<1> ndItem);
  void bind(cl::sycl::handler &h);
  void adjust_access_displacement();
};

template <bool same_sign, typename lhs_t, typename rhs_t>
Axpy_batch<same_sign, lhs_t, rhs_t> make_axpy_batch(
    lhs_t _lhs, rhs_t _rhs_1, typename rhs_t::value_t _alpha,
    typename rhs_t::index_t _N, typename rhs_t::index_t _inc_l,
    typename rhs_t::index_t _lhs_stride, typename rhs_t::index_t _inc_r,
    typename rhs_t::index_t _rhs_stride, typename rhs_t::index_t _batch_size) {
  return Axpy_batch<same_sign, lhs_t, rhs_t>(_lhs, _rhs_1, _alpha, _N, _inc_l,
                                             _lhs_stride, _inc_r, _rhs_stride,
                                             _batch_size);
}

}  // namespace blas

#endif  // PORTBLAS_EXTENSION_AXPY_BATCH_H
