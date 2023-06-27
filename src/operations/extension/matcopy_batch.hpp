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
 *  @filename matcopy_batch.hpp
 *
 **************************************************************************/

#ifndef SYCL_BLAS_EXTENSION_MATCOPY_BATCH_HPP
#define SYCL_BLAS_EXTENSION_MATCOPY_BATCH_HPP

#include "operations/extension/matcopy_batch.h"

namespace blas {

template <matcopy_op op, int TileSize, int TilePerWG, typename lhs_t,
          typename rhs_t>
Matcopy_batch<op, TileSize, TilePerWG, lhs_t, rhs_t>::Matcopy_batch(
    lhs_t lhs, rhs_t rhs_1, rhs_t rhs_2, typename lhs_t::value_t alpha,
    typename lhs_t::value_t beta, typename rhs_t::index_t m,
    typename rhs_t::index_t n, typename rhs_t::index_t lhs_ld,
    typename rhs_t::index_t rhs_ld, typename rhs_t::index_t rhs_2_ld,
    typename rhs_t::index_t lhs_stride, typename rhs_t::index_t rhs_stride,
    typename rhs_t::index_t rhs_2_stride, typename rhs_t::index_t batch_size)
    : lhs_(lhs),
      rhs_1_(rhs_1),
      rhs_2_(rhs_2),
      alpha_(alpha),
      beta_(beta),
      m_(m),
      n_(n),
      lhs_ld_(lhs_ld),
      rhs_1_ld_(rhs_ld),
      rhs_2_ld_(rhs_2_ld),
      lhs_stride_(lhs_stride),
      rhs_1_stride_(rhs_stride),
      rhs_2_stride_(rhs_2_stride),
      batch_size_(batch_size) {}

template <matcopy_op op, int TileSize, int TilePerWG, typename lhs_t,
          typename rhs_t>
typename lhs_t::value_t
Matcopy_batch<op, TileSize, TilePerWG, lhs_t, rhs_t>::eval(index_t i) {}

template <matcopy_op op, int TileSize, int TilePerWG, typename lhs_t,
          typename rhs_t>
typename lhs_t::value_t
Matcopy_batch<op, TileSize, TilePerWG, lhs_t, rhs_t>::eval(
    cl::sycl::nd_item<1> ndItem) {
  const index_t m{m_};
  const index_t n{n_};

  const index_t required_tile =
      (((m - 1) / TileSize) + 1) * (((n - 1) / TileSize) + 1);

  const index_t tile_for_matrix = ((required_tile - 1) / TilePerWG) + 1;

  const index_t wg_batch_id =
      (ndItem.get_group(0)) / ((required_tile - 1) / TilePerWG + 1);

  const index_t l_lhs_stride = lhs_stride_;
  const index_t l_rhs_stride = rhs_1_stride_;

  const index_t number_of_block_per_row = ((m_ - 1) / TileSize) + 1;

  const index_t wg_id = ndItem.get_local_id(0) / TileSize +
                        ((ndItem.get_group(0) % tile_for_matrix) * TilePerWG);

  /* row tile id  per work group */
  const index_t tile_id_row = wg_id % number_of_block_per_row;
  /* column tile id per work group */
  const index_t tile_id_col = wg_id / number_of_block_per_row;
  /* the start position of the tile-row per work group */
  const index_t wg_row = tile_id_row * TileSize;
  /* the start position of the tile-column per work group */
  const index_t wg_col = tile_id_col * TileSize;

  const index_t item_id = ndItem.get_local_id(0) % TileSize;

  auto orig_lhs = lhs_.get_pointer() + (wg_batch_id * l_lhs_stride);
  auto orig_rhs = rhs_1_.get_pointer() + (wg_batch_id * l_rhs_stride);

  orig_lhs = orig_lhs + wg_row + wg_col * lhs_ld_ + item_id;
  orig_rhs = orig_rhs + wg_row + wg_col * rhs_1_ld_ + item_id;

  value_t reg_rhs[TileSize];
  const index_t alpha = alpha_;

  const bool is_internal_block =
      (m - wg_row >= TileSize) && (n - wg_col >= TileSize);

  // check for short&large
  const bool valid_index =
      (item_id > m || (item_id >= (m - wg_row))) ? false : true;
  if (!valid_index) return 0;

  if (is_internal_block) {
    auto A = orig_rhs;
    auto B = orig_lhs;

#pragma unroll
    for (int i = 0; i < TileSize; ++i) {
      reg_rhs[i] = A[i * rhs_1_ld_];
    }
#pragma unroll
    for (int i = 0; i < TileSize; ++i) {
      B[i * lhs_ld_] = alpha * reg_rhs[i];
    }

  } else {
    const auto limit_m = m - wg_row;
    const auto limit_n = n - wg_col;
    auto A = orig_rhs;
    auto B = orig_lhs;

    for (int i = 0; i < TileSize; ++i) {
      if (i >= limit_n) break;
      reg_rhs[i] = A[i * rhs_1_ld_];
    }
    for (int i = 0; i < TileSize; ++i) {
      if (i >= limit_n) break;
      B[i * lhs_ld_] = alpha * reg_rhs[i];
    }
  }

  return 0;
}

template <matcopy_op op, int TileSize, int TilePerWG, typename lhs_t,
          typename rhs_t>
SYCL_BLAS_INLINE void Matcopy_batch<op, TileSize, TilePerWG, lhs_t,
                                    rhs_t>::bind(cl::sycl::handler& h) {
  lhs_.bind(h);
  rhs_1_.bind(h);
}

template <matcopy_op op, int TileSize, int TilePerWG, typename lhs_t,
          typename rhs_t>
SYCL_BLAS_INLINE void Matcopy_batch<op, TileSize, TilePerWG, lhs_t,
                                    rhs_t>::adjust_access_displacement() {
  lhs_.adjust_access_displacement();
  rhs_1_.adjust_access_displacement();
}

template <matcopy_op op, int TileSize, int TilePerWG, typename lhs_t,
          typename rhs_t>
SYCL_BLAS_INLINE typename rhs_t::index_t
Matcopy_batch<op, TileSize, TilePerWG, lhs_t, rhs_t>::get_size() const {
  return m_ * n_;
}

template <matcopy_op op, int TileSize, int TilePerWG, typename lhs_t,
          typename rhs_t>
SYCL_BLAS_INLINE bool
Matcopy_batch<op, TileSize, TilePerWG, lhs_t, rhs_t>::valid_thread(
    cl::sycl::nd_item<1> ndItem) const {
  return true;
}
}  // namespace blas

#endif  // SYCL_BLAS_EXTENSION_MATCOPY_BATCH_HPP
