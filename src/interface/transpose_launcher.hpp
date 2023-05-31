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
namespace internal {

/*!
 * @brief Wrapper around Transpose. Creates the views, then makes and launches
 * Transpose
 */
template <int Tile_size, bool local_memory>
template <typename sb_handle_t, typename container_0_t, typename container_1_t,
          typename element_t, typename index_t>
typename sb_handle_t::event_t
Transpose_Launcher<Tile_size, local_memory>::_select_transpose_outplace(
    sb_handle_t& sb_handle, index_t _M, index_t _N, element_t _alpha,
    container_0_t in_, index_t _ld_in, index_t _inc_in, container_1_t out_,
    index_t _ld_out, index_t _inc_out) {
  // Matrix Views
  auto in_view = make_matrix_view<col_major>(in_, _M, _N, _ld_in, _inc_in);
  auto out_view = make_matrix_view<col_major>(out_, _M, _N, _ld_out, _inc_out);

  // Work items & groups sizes
  index_t local_size = static_cast<index_t>(Tile_size * Tile_size);
  index_t n_wg = ((_M - 1) / Tile_size + 1) * ((_N - 1) / Tile_size + 1);
  index_t global_size = n_wg * local_size;

  // Transpose expression Tree
  auto trans_scale_tree = make_transpose<false, Tile_size, local_memory>(
      in_view, _inc_in, out_view, _inc_out, _alpha);

  if constexpr (local_memory) {
    index_t shared_mem = static_cast<index_t>((Tile_size + 1) * Tile_size) *
                         ((index_t)local_memory);
    return sb_handle.execute(trans_scale_tree, local_size, global_size,
                             shared_mem);
  } else {
    return sb_handle.execute(trans_scale_tree, local_size, global_size);
  }
}

}  // namespace internal
}  // namespace extension
}  // namespace blas

#endif  // SYCL_BLAS_TRANSPOSE_LAUNCHER_HPP
