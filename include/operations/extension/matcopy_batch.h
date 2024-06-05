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
 *  @filename Matcopy_batch.h
 *
 **************************************************************************/

#ifndef PORTBLAS_EXTENSION_MATCOPY_BATCH_H
#define PORTBLAS_EXTENSION_MATCOPY_BATCH_H

namespace blas {

template <bool is_add, int TileSize, int TilePerWG, typename lhs_t,
          typename rhs_t, typename rhs_2_t>
struct Matcopy_batch {
 public:
  using value_t = typename lhs_t::value_t;
  using index_t = typename rhs_t::index_t;

  lhs_t lhs_;
  rhs_t rhs_1_;
  rhs_2_t rhs_2_;
  value_t alpha_, beta_;
  index_t m_, n_, lhs_ld_, rhs_1_ld_, rhs_2_ld_, lhs_stride_, rhs_1_stride_,
      rhs_2_stride_, batch_size_;

  Matcopy_batch(lhs_t lhs, rhs_t rhs_1, rhs_2_t rhs_2, value_t alpha,
                value_t beta, index_t m, index_t n, index_t lhs_ld,
                index_t rhs_ld, index_t rhs_2_ld, index_t lhs_stride,
                index_t rhs_stride, index_t rhs_2_stride, index_t batch_size);
  index_t get_size() const;
  bool valid_thread(sycl::nd_item<1> ndItem) const;
  value_t eval(index_t i);
  value_t eval(sycl::nd_item<1> ndItem);
  template <typename sharedT>
  value_t eval(sharedT shMem, sycl::nd_item<1> ndItem);
  void bind(sycl::handler &h);
  void adjust_access_displacement();
  void compute_matcopy_batch(const index_t wg_batch_id, const index_t wg_row,
                             const index_t wg_col, const index_t item_id);
  void compute_omatadd_batch(const index_t wg_batch_id, const index_t wg_row,
                             const index_t wg_col, const index_t item_id);
};

template <bool is_add, int TileSize, int TilePerWG, typename lhs_t,
          typename rhs_t, typename rhs_2_t>
Matcopy_batch<is_add, TileSize, TilePerWG, lhs_t, rhs_t, rhs_2_t>
make_matcopy_batch(
    lhs_t lhs, rhs_t rhs_1, rhs_2_t rhs_2, typename rhs_t::value_t alpha,
    typename rhs_t::value_t beta, typename rhs_t::index_t m,
    typename rhs_t::index_t n, typename rhs_t::index_t lhs_ld,
    typename rhs_t::index_t rhs_ld, typename rhs_t::index_t rhs_2_ld,
    typename rhs_t::index_t lhs_stride, typename rhs_t::index_t rhs_stride,
    typename rhs_t::index_t rhs_2_stride, typename rhs_t::index_t batch_size) {
  return Matcopy_batch<is_add, TileSize, TilePerWG, lhs_t, rhs_t, rhs_2_t>(
      lhs, rhs_1, rhs_2, alpha, beta, m, n, lhs_ld, rhs_ld, rhs_2_ld,
      lhs_stride, rhs_stride, rhs_2_stride, batch_size);
}

}  // namespace blas

#endif  // PORTBLAS_EXTENSION_MATCOPY_BATCH_H
