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
 *  portBLAS: BLAS implementation using SYCL
 *
 *  @filename nvidia_gpu.hpp
 *
 **************************************************************************/
#ifndef PORTBLAS_TRANSPOSE_NVIDIA_GPU_BACKEND_HPP
#define PORTBLAS_TRANSPOSE_NVIDIA_GPU_BACKEND_HPP
#include "interface/extension_interface.h"

namespace blas {
namespace transpose {
namespace backend {

template <typename sb_handle_t, typename container_0_t, typename container_1_t,
          typename element_t, typename index_t>
typename sb_handle_t::event_t _transpose_outplace(
    sb_handle_t& sb_handle, index_t _M, index_t _N, element_t _alpha,
    container_0_t in_, index_t _ld_in, index_t _inc_in, index_t _stride_in,
    container_1_t out_, index_t _ld_out, index_t _inc_out, index_t _stride_out,
    index_t _batch_size, const typename sb_handle_t::event_t& _dependencies) {
  if (_M * _N > (1 << 18)) {
    return blas::internal::_transpose_outplace_impl<32, 512, 128, true>(
        sb_handle, _M, _N, _alpha, in_, _ld_in, _inc_in, _stride_in, out_,
        _ld_out, _inc_out, _stride_out, _batch_size, _dependencies);

  } else {
    return blas::internal::_transpose_outplace_impl<32, 128, 128, true>(
        sb_handle, _M, _N, _alpha, in_, _ld_in, _inc_in, _stride_in, out_,
        _ld_out, _inc_out, _stride_out, _batch_size, _dependencies);
  }
}

template <bool both_trans, typename sb_handle_t, typename container_0_t,
          typename container_1_t, typename container_2_t, typename element_t,
          typename index_t>
typename sb_handle_t::event_t _transpose_add(
    sb_handle_t& sb_handle, index_t _M, index_t _N, element_t _alpha,
    container_0_t a_, index_t _ld_a, index_t _a_rows, index_t _a_cols,
    element_t _beta, container_1_t b_, index_t _ld_b, index_t _b_rows,
    index_t _b_cols, container_2_t c_, index_t _ld_c,
    const typename sb_handle_t::event_t& _dependencies) {
  if (_M * _N > (1 << 18)) {
    return blas::internal::_transpose_add_impl<both_trans, 32, 512, 128, true>(
        sb_handle, _M, _N, _alpha, a_, _ld_a, _a_rows, _a_cols, _beta, b_,
        _ld_b, _b_rows, _b_cols, c_, _ld_c, _dependencies);
  } else {
    return blas::internal::_transpose_add_impl<both_trans, 32, 128, 128, true>(
        sb_handle, _M, _N, _alpha, a_, _ld_a, _a_rows, _a_cols, _beta, b_,
        _ld_b, _b_rows, _b_cols, c_, _ld_c, _dependencies);
  }
}

}  // namespace backend
}  // namespace transpose

namespace matcopy_batch {
namespace backend {
template <bool trans, typename sb_handle_t, typename element_t,
          typename index_t, typename in_t, typename out_t>
typename std::enable_if<!trans, typename sb_handle_t::event_t>::type
_matcopy_batch(sb_handle_t& sb_handle, index_t m, index_t n, element_t alpha,
               in_t in_memory, index_t ld_in, index_t in_stride,
               out_t out_memory, index_t ld_out, index_t out_stride,
               index_t batch_size, const typename  sb_handle_t::event_t& _dependencies) {
  if ((m * n) > (1 << 20)) {
    return blas::internal::_matcopy_batch_impl<32, 4, sb_handle_t, element_t,
                                               index_t, in_t, out_t>(
        sb_handle, m, n, alpha, in_memory, ld_in, in_stride, out_memory, ld_out,
        out_stride, batch_size, _dependencies);
  } else if ((m * n) > (1 << 14)) {
    return blas::internal::_matcopy_batch_impl<8, 16, sb_handle_t, element_t,
                                               index_t, in_t, out_t>(
        sb_handle, m, n, alpha, in_memory, ld_in, in_stride, out_memory, ld_out,
        out_stride, batch_size, _dependencies);
  } else {
    return blas::internal::_matcopy_batch_impl<4, 64, sb_handle_t, element_t,
                                               index_t, in_t, out_t>(
        sb_handle, m, n, alpha, in_memory, ld_in, in_stride, out_memory, ld_out,
        out_stride, batch_size, _dependencies);
  }
}
}  // namespace backend
}  // namespace matcopy_batch
}  // namespace blas
#endif
