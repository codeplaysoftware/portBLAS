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
 *  @filename transpose_launcher.hpp
 *
 **************************************************************************/

#ifndef SYCL_BLAS_TRANSPOSE_LAUNCHER_HPP
#define SYCL_BLAS_TRANSPOSE_LAUNCHER_HPP

#include "interface/transpose_launcher.h"
#include "views/view.h"

namespace blas {
namespace extension {

/*!
 * @brief Wrapper around Transpose. Creates the views, then makes and launches
 * Transpose
 */
template <int Tile_size, int wg_size, int cl_size, bool local_memory>
template <typename sb_handle_t, typename container_0_t, typename container_1_t,
          typename element_t, typename index_t>
typename sb_handle_t::event_t
Transpose_Launcher<Tile_size, wg_size, cl_size, local_memory>::
    _select_transpose_outplace(sb_handle_t& sb_handle, index_t _M, index_t _N,
                               element_t _alpha, container_0_t in_,
                               index_t _ld_in, index_t _inc_in,
                               container_1_t out_, index_t _ld_out,
                               index_t _inc_out) {
  constexpr const index_t num_cache_line_elems = cl_size / sizeof(element_t);
  constexpr const index_t num_tiles_per_cache_line =
      num_cache_line_elems / Tile_size;

  // Matrix Views
  auto in_view = make_matrix_view<col_major>(in_, _M, _N, _ld_in, index_t(1));
  auto out_view =
      make_matrix_view<col_major>(out_, _M, _N, _ld_out, index_t(1));

  // Work items & groups sizes
  index_t n_wg = ((_M - 1) / Tile_size + 1) * ((_N - 1) / Tile_size + 1);
  index_t global_size = n_wg * wg_size;

  // Transpose expression Tree
  auto trans_scale_tree =
      make_transpose<false, Tile_size, wg_size, cl_size, local_memory>(
          in_view, _inc_in, out_view, _inc_out, _alpha);

  if constexpr (local_memory) {
    index_t local_mem =
        static_cast<index_t>((num_cache_line_elems + 1) * num_cache_line_elems /
                             num_tiles_per_cache_line);
    return sb_handle.execute(trans_scale_tree, wg_size, global_size, local_mem);
  } else {
    return sb_handle.execute(trans_scale_tree, wg_size, global_size);
  }
}

template <bool both_trans, int Tile_size, bool local_memory>
template <typename sb_handle_t, typename container_0_t, typename container_1_t,
          typename container_2_t, typename element_t, typename index_t>
typename sb_handle_t::event_t
TransposeAdd_Launcher<both_trans, Tile_size, local_memory>::
    _select_transpose_add(sb_handle_t& sb_handle, index_t _M, index_t _N,
                          element_t _alpha, container_0_t a_, index_t _lda,
                          index_t _nrows_a, index_t _ncols_a, element_t _beta,
                          container_1_t b_, index_t _ldb, index_t _nrows_b,
                          index_t _ncols_b, container_2_t c_, index_t _ldc) {
  // Matrix Views
  auto A_view =
      make_matrix_view<col_major>(a_, _nrows_a, _ncols_a, _lda, (index_t)1);
  auto B_view =
      make_matrix_view<col_major>(b_, _nrows_b, _ncols_b, _ldb, (index_t)1);

  auto C_view = make_matrix_view<col_major>(c_, _M, _N, _ldc, (index_t)1);

  // Work items & groups sizes
  index_t local_size = static_cast<index_t>(Tile_size * Tile_size);
  index_t n_wg = ((_M - 1) / Tile_size + 1) * ((_N - 1) / Tile_size + 1);
  index_t global_size = n_wg * local_size;

  // Transpose Add expression Tree
  auto trans_scale_tree =
      make_transpose_add<both_trans, Tile_size, local_memory>(
          A_view, B_view, C_view, _alpha, _beta);

  if constexpr (local_memory) {
    index_t shared_mem = static_cast<index_t>((Tile_size + 1) * Tile_size) *
                         ((index_t)local_memory);
    return sb_handle.execute(trans_scale_tree, local_size, global_size,
                             shared_mem);
  } else {
    return sb_handle.execute(trans_scale_tree, local_size, global_size);
  }
}

}  // namespace extension
}  // namespace blas

#endif  // SYCL_BLAS_TRANSPOSE_LAUNCHER_HPP
