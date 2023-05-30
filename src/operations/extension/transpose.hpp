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
 *  @filename transpose.hpp
 *
 **************************************************************************/

#ifndef SYCL_BLAS_EXTENSION_TRANSPOSE_HPP
#define SYCL_BLAS_EXTENSION_TRANSPOSE_HPP

#include "operations/extension/transpose.h"

namespace blas {

// Transpose
template <bool in_place, int Tile_size, bool local_memory, typename in_t,
          typename out_t, typename element_t>
SYCL_BLAS_INLINE bool
Transpose<in_place, Tile_size, local_memory, in_t, out_t,
          element_t>::valid_thread(cl::sycl::nd_item<1> item) const {
  // Valid threads are established by ::eval()
  return true;
}

template <bool in_place, int Tile_size, bool local_memory, typename in_t,
          typename out_t, typename element_t>
SYCL_BLAS_INLINE void Transpose<in_place, Tile_size, local_memory, in_t, out_t,
                                element_t>::bind(cl::sycl::handler &cgh) {
  A_.bind(cgh);
  At_.bind(cgh);
}

template <bool in_place, int Tile_size, bool local_memory, typename in_t,
          typename out_t, typename element_t>
SYCL_BLAS_INLINE typename in_t::index_t
Transpose<in_place, Tile_size, local_memory, in_t, out_t, element_t>::get_size()
    const {
  // Smallest TileSize square-multiple containing input/output matrices
  return (M_pad_ * N_pad_);
}

template <bool in_place, int Tile_size, bool local_memory, typename in_t,
          typename out_t, typename element_t>
SYCL_BLAS_INLINE void Transpose<in_place, Tile_size, local_memory, in_t, out_t,
                                element_t>::adjust_access_displacement() {
  A_.adjust_access_displacement();
  At_.adjust_access_displacement();
}

template <bool in_place, int Tile_size, bool local_memory, typename in_t,
          typename out_t, typename element_t>
SYCL_BLAS_INLINE void Transpose<in_place, Tile_size, local_memory, in_t, out_t,
                                element_t>::eval(cl::sycl::nd_item<1> id) {
  auto idx = id.get_global_linear_id();

  auto j = idx / M_pad_;
  auto i = idx - j * M_pad_;

  if (i < M_ && j < N_) {
    auto A = A_.get_data().get_pointer();
    auto At = At_.get_data().get_pointer();

    auto in_index = i * inc_a_ + j * lda_;
    auto out_index = i * ldat_ + j * inc_at_;

    At[out_index] = alpha_ * A[in_index];
  }
}

/*!
 *@brief get_indices. This function is used in the local-memory kernel to
 *compute local & global input & output indices.
 *
 * @param id [input] the cl::sycl::nd_item<1> of the current work_item
 * @param in_idx [output] the input global-memory index
 * @param out_idx [output] the output global-memory index
 * @param in_local_idx [output] the input local-memory index
 * @param out_local_idx [output] the output local-memory index
 * @param valid_index_in [output] whether current input global index is within
 *input range
 * @param valid_index_in [output] whether current output global index is within
 *outpu range
 *
 */
template <bool in_place, int Tile_size, bool local_memory, typename in_t,
          typename out_t, typename element_t>
SYCL_BLAS_INLINE void Transpose<in_place, Tile_size, local_memory, in_t, out_t,
                                element_t>::get_indices(cl::sycl::nd_item<1> id,
                                                        index_t &in_idx,
                                                        index_t &in_local_idx,
                                                        index_t &out_idx,
                                                        index_t &out_local_idx,
                                                        bool &valid_index_in,
                                                        bool &valid_index_out) {
  index_t idg = id.get_group(0);
  index_t idc = id.get_local_id(0);

  const index_t jg = idg / tile_count_m_;
  const index_t ig = idg - jg * tile_count_m_;

  const index_t jl = idc / Tile_size;
  const index_t il = idc - jl * Tile_size;

  const index_t i_block_start = ig * Tile_size;
  const index_t j_block_start = jg * Tile_size;

  valid_index_in = (i_block_start + il < M_ && j_block_start + jl < N_);
  valid_index_out = (i_block_start + jl < M_ && j_block_start + il < N_);

  in_idx =
      i_block_start * inc_a_ + j_block_start * lda_ + il * inc_a_ + jl * lda_;
  in_local_idx = jl * (Tile_size + 1) + il;

  out_idx = i_block_start * ldat_ + j_block_start * inc_at_ + il * inc_at_ +
            jl * ldat_;
  out_local_idx = il * (Tile_size + 1) + jl;
}

template <bool in_place, int Tile_size, bool local_memory, typename in_t,
          typename out_t, typename element_t>
template <typename local_memory_t>
SYCL_BLAS_INLINE void Transpose<in_place, Tile_size, local_memory, in_t, out_t,
                                element_t>::eval(local_memory_t local_mem,
                                                 cl::sycl::nd_item<1> id) {
  index_t idx = id.get_global_linear_id();

  if (idx < get_size()) {
    value_t *local = local_mem.localAcc.get_pointer();

    auto A = A_.get_data().get_pointer();
    auto At = At_.get_data().get_pointer();

    index_t in_index, in_local_id, out_index, out_local_id;
    bool valid_index_in, valid_index_out;

    get_indices(id, in_index, in_local_id, out_index, out_local_id,
                valid_index_in, valid_index_out);

    // Copy input to local memory
    if (valid_index_in) {
      local[in_local_id] = alpha_ * A[in_index];
    }

    id.barrier(cl::sycl::access::fence_space::local_space);

    // Copy output from local memory
    if (valid_index_out) {
      At[out_index] = local[out_local_id];
    }
  }
}

}  // namespace blas

#endif  // SYCL_BLAS_EXTENSION_TRANSPOSE_HPP