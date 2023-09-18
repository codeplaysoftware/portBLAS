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
#ifndef PORTBLAS_ASUM_NVIDIA_GPU_BACKEND_HPP
#define PORTBLAS_ASUM_NVIDIA_GPU_BACKEND_HPP
#include "interface/blas1_interface.h"

namespace blas {
namespace asum {
namespace backend {
template <typename sb_handle_t, typename container_0_t, typename container_1_t,
          typename index_t, typename increment_t>
typename sb_handle_t::event_t _asum(
    sb_handle_t& sb_handle, index_t _N, container_0_t _vx, increment_t _incx,
    container_1_t _rs, const typename sb_handle_t::event_t& _dependencies) {
  if (_N < (1 << 23)) {
    constexpr index_t localSize = 512;
    const index_t number_WG = (_N < (1 << 18))
                                  ? (_N + localSize - 1) / localSize
                                  : static_cast<index_t>(256);
    return blas::internal::_asum_impl<static_cast<index_t>(localSize), 32>(
        sb_handle, _N, _vx, _incx, _rs, number_WG, _dependencies);
  } else {
    constexpr int localSize = 512;
    constexpr index_t number_WG = 1024;
    return blas::internal::_asum_impl<localSize, 32>(
        sb_handle, _N, _vx, _incx, _rs, number_WG, _dependencies);
  }
}
}  // namespace backend
}  // namespace asum

}  // namespace blas

#endif
