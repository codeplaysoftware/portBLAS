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
 *  @filename default_cpu.hpp
 *
 **************************************************************************/
#ifndef SYCL_BLAS_TRANSPOSE_DEFAULT_CPU_BACKEND_HPP
#define SYCL_BLAS_TRANSPOSE_DEFAULT_CPU_BACKEND_HPP
#include "interface/transpose_launcher.h"
#include "interface/extension_interface.h"

namespace blas {
namespace extension {
namespace backend {

template <typename sb_handle_t, typename container_0_t, typename container_1_t,
          typename element_t, typename index_t>
typename sb_handle_t::event_t _transpose_outplace(
    sb_handle_t& sb_handle, index_t _M, index_t _N, element_t _alpha,
    container_0_t in_, index_t _ld_in, index_t _inc_in, index_t _stride_in,
    container_1_t out_, index_t _ld_out, index_t _inc_out, index_t _stride_out,
    index_t _batch_size) {
  return Transpose_Launcher<
      16, 64, 64, false>::template _select_transpose_outplace(sb_handle, _M, _N,
                                                              _alpha, in_,
                                                              _ld_in, _inc_in,
                                                              _stride_in, out_,
                                                              _ld_out, _inc_out,
                                                              _stride_out,
                                                              _batch_size);
}

template <bool both_trans, typename sb_handle_t, typename container_0_t,
          typename container_1_t, typename container_2_t, typename element_t,
          typename index_t>
typename sb_handle_t::event_t _transpose_add(
    sb_handle_t& sb_handle, index_t _M, index_t _N, element_t _alpha,
    container_0_t a_, index_t _ld_a, index_t _a_rows, index_t _a_cols,
    element_t _beta, container_1_t b_, index_t _ld_b, index_t _b_rows,
    index_t _b_cols, container_2_t c_, index_t _ld_c) {
  return TransposeAdd_Launcher<both_trans, 16, 64, 64, false>::
      template _select_transpose_add(sb_handle, _M, _N, _alpha, a_, _ld_a,
                                     _a_rows, _a_cols, _beta, b_, _ld_b,
                                     _b_rows, _b_cols, c_, _ld_c);
}

}  // namespace backend
}  // namespace extension

namespace matcopy_batch {
namespace backend {
template <bool trans, typename sb_handle_t, typename element_t,
          typename index_t, typename in_t, typename out_t>
typename std::enable_if<!trans, typename sb_handle_t::event_t>::type
_matcopy_batch(sb_handle_t& sb_handle, index_t m, index_t n, element_t alpha,
                in_t in_memory, index_t ld_in, index_t in_stride,
                out_t out_memory, index_t ld_out, index_t out_stride,
                index_t batch_size) {
  return blas::extension::internal::_matcopy_batch_impl<
      4, false, sb_handle_t, element_t, index_t, in_t, out_t>(
      sb_handle, m, n, alpha, in_memory, ld_in, in_stride, out_memory, ld_out,
      out_stride, batch_size);
}
}  // namespace backend
}  // namespace omatcopy_batch
}  // namespace blas

#endif
