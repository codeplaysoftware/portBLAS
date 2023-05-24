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
  return (tile_count_m_ * tile_count_n_ * Tile_size * Tile_size);
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

  if (idx < get_size()) {
    auto A = A_.get_data().get_pointer();
    auto At = At_.get_data().get_pointer();

    auto j = idx / M_;
    auto i = idx - j * M_;

    auto in_index = i + j * lda_;
    auto out_index = i * ldat_ + j;

    At[out_index] = alpha_ * A[in_index];
  }
}

template <int Tile_size, typename index_t>
SYCL_BLAS_INLINE void compute_trans_indices(
    const index_t &M, const index_t &N, const index_t &m_tiles,
    cl::sycl::nd_item<1> id, const index_t &lda, const index_t &stride_a,
    const index_t &ldat, const index_t &stride_at, index_t &in_idx,
    index_t &in_idc, index_t &out_idx, index_t &out_idc, bool &valid_index_in,
    bool &valid_index_out) {
  index_t idg = id.get_group(0);
  index_t idc = id.get_local_id();

  const index_t jg = idg / m_tiles;
  const index_t ig = idg - jg * m_tiles;

  const index_t jl = idc / Tile_size;
  const index_t il = idc - jl * Tile_size;

  const index_t i_block_start = ig * Tile_size;
  const index_t j_block_start = jg * Tile_size;

  valid_index_in = (i_block_start + il < M && j_block_start + jl < N);
  valid_index_out = (i_block_start + jl < M && j_block_start + il < N);

  in_idx =
      i_block_start * stride_a + j_block_start * lda + il * stride_a + jl * lda;
  in_idc = jl * (Tile_size + 1) + il;

  out_idx = i_block_start * ldat + j_block_start * stride_at + il * stride_at +
            jl * ldat;
  out_idc = il * (Tile_size + 1) + jl;
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

    compute_trans_indices<Tile_size>(
        M_, N_, tile_count_m_, id, lda_, stridea_, ldat_, strideat_, in_index,
        in_local_id, out_index, out_local_id, valid_index_in, valid_index_out);

    // Copy input to local memory
    if (valid_index_in) {
      local[in_local_id] = alpha_ * A[in_index];
    }

    id.barrier(sycl::access::fence_space::local_space);

    // Copy output from local memory
    if (valid_index_out) {
      At[out_index] = local[out_local_id];
    }
  }
}

}  // namespace blas

#endif  // SYCL_BLAS_EXTENSION_TRANSPOSE_HPP