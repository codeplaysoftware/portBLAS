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
 *  @filename intel_gpu.hpp
 *
 **************************************************************************/

#ifndef SYCL_BLAS_GEMM_INTEL_GPU_BACKEND_HPP
#define SYCL_BLAS_GEMM_INTEL_GPU_BACKEND_HPP

#include "interface/gemm_launcher.h"

namespace blas {
namespace gemm {
namespace backend {
template <bool _t_a, bool _t_b, bool is_beta_zero, typename executor_t,
          typename container_0_t, typename container_1_t,
          typename container_2_t, typename element_t, typename index_t>
typename executor_t::policy_t::event_t _gemm(
    executor_t& ex, index_t _M, index_t _N, index_t _K, element_t _alpha,
    container_0_t _a, index_t _lda, container_1_t _b, index_t _ldb,
    element_t _beta, container_2_t _c, index_t _ldc, index_t batch_size) {
  if ((_M == 512 && _N == 49 && _K == 512) ||
      (_M == 10 && _N == 1024 && _K == 1024)) {
    return blas::Gemm_Launcher<
        256, true, false, false, 64, Tile<4, 4, 16, 16>, _t_a, _t_b,
        static_cast<int>(Gemm_t::local_memory),
        is_beta_zero>::template _select_gemm(ex, _M, _N, _K, _alpha, _a, _lda,
                                             _b, _ldb, _beta, _c, _ldc,
                                             batch_size);
  } else if (_M == 256 && _N == 784 && _K == 256) {
    return blas::Gemm_Launcher<
        256, false, false, false, 64, Tile<8, 8, 16, 16>, _t_a, _t_b,
        static_cast<int>(Gemm_t::no_local_memory),
        is_beta_zero>::template _select_gemm(ex, _M, _N, _K, _alpha, _a, _lda,
                                             _b, _ldb, _beta, _c, _ldc,
                                             batch_size);
  } else if ((_M <= 128 && _N <= 128 && _K <= 128)) {
    return blas::Gemm_Launcher<
        64, true, false, false, 64, Tile<4, 4, 8, 8>, _t_a, _t_b,
        static_cast<int>(Gemm_t::local_memory),
        is_beta_zero>::template _select_gemm(ex, _M, _N, _K, _alpha, _a, _lda,
                                             _b, _ldb, _beta, _c, _ldc,
                                             batch_size);

  } else if (!_t_a) {
    return blas::Gemm_Launcher<
        64, false, false, false, 64, Tile<8, 8, 8, 8>, _t_a, _t_b,
        static_cast<int>(Gemm_t::no_local_memory),
        is_beta_zero>::template _select_gemm(ex, _M, _N, _K, _alpha, _a, _lda,
                                             _b, _ldb, _beta, _c, _ldc,
                                             batch_size);

  } else {
    return blas::Gemm_Launcher<
        64, true, false, false, 64, Tile<8, 8, 8, 8>, _t_a, _t_b,
        static_cast<int>(Gemm_t::local_memory),
        is_beta_zero>::template _select_gemm(ex, _M, _N, _K, _alpha, _a, _lda,
                                             _b, _ldb, _beta, _c, _ldc,
                                             batch_size);
  }
}
}  // namespace backend
}  // namespace gemm
}  // namespace blas
#endif
