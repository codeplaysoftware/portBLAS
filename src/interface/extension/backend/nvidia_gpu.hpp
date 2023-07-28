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
 *  @filename nvidia_gpu.hpp
 *
 **************************************************************************/
#ifndef SYCL_BLAS_TRANSPOSE_NVIDIA_GPU_BACKEND_HPP
#define SYCL_BLAS_TRANSPOSE_NVIDIA_GPU_BACKEND_HPP
#include "interface/transpose_launcher.h"

namespace blas {
namespace extension {
namespace backend {

template <typename sb_handle_t, typename container_0_t, typename container_1_t,
          typename element_t, typename index_t>
typename sb_handle_t::event_t _transpose_outplace(
    sb_handle_t& sb_handle, index_t _M, index_t _N, element_t _alpha,
    container_0_t in_, index_t _ld_in, index_t _inc_in, container_1_t out_,
    index_t _ld_out, index_t _inc_out) {
  if (_M * _N > (1 << 18)) {
    return Transpose_Launcher<
        32, 512, 128, true>::template _select_transpose_outplace(sb_handle, _M,
                                                                 _N, _alpha,
                                                                 in_, _ld_in,
                                                                 _inc_in, out_,
                                                                 _ld_out,
                                                                 _inc_out);
  } else {
    return Transpose_Launcher<
        32, 128, 128, true>::template _select_transpose_outplace(sb_handle, _M,
                                                                 _N, _alpha,
                                                                 in_, _ld_in,
                                                                 _inc_in, out_,
                                                                 _ld_out,
                                                                 _inc_out);
  }
}

}  // namespace backend
}  // namespace extension
}  // namespace blas

#endif