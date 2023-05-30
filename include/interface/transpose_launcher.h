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
 *  @filename transpose_launcher.h
 *
 **************************************************************************/

#ifndef SYCL_BLAS_EXTENSION_TRANSPOSE_LAUNCHER_H
#define SYCL_BLAS_EXTENSION_TRANSPOSE_LAUNCHER_H

#include "operations/extension/transpose.h"
#include "sb_handle/sycl_blas_handle.h"

namespace blas {
namespace extension {
namespace internal {
/*!
 * @brief Wrapper around Transpose (in & out place). Creates the views, then
 * makes and launches Transpose
 */
template <int Tile_size, bool local_memory>
struct Transpose_Launcher {
  template <typename sb_handle_t, typename container_0_t,
            typename container_1_t, typename element_t, typename index_t>
  static typename sb_handle_t::event_t _select_transpose_outplace(
      sb_handle_t& sb_handle, index_t _M, index_t _N, element_t _alpha,
      container_0_t in_, index_t _ld_in, index_t _inc_in, container_1_t out_,
      index_t _ld_out, index_t _inc_out);
};

}  // namespace internal
}  // namespace extension
}  // namespace blas

#endif  // SYCL_BLAS_EXTENSION_TRANSPOSE_LAUNCHER_H
