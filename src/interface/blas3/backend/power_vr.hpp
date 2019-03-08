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
 *  @filename power_vr.hpp
 *
 **************************************************************************/
#ifndef SYCL_BLAS_GEMM_POWERVR_BACKEND_HPP
#define SYCL_BLAS_GEMM_POWERVR_BACKEND_HPP
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
  // The following _M, _N ,and _K is used for SSD + Mobilenet v2 (TF version)
  // We computed the best tile combination for each sizes -(4-March-2018)
  // POWER_VR Rogue
  if ((_M == 96 && _K == 16 && _N == 22500) ||
      (_M == 273 && _K == 576 && _N == 100) ||
      (_M == 384 && _K == 64 && _N == 361)) {
    return blas::Gemm_Launcher<
        96, true, false, false, 16, Tile<4, 6, 12, 8>, _t_a, _t_b,
        static_cast<int>(Gemm_t::local_memory),
        is_beta_zero>::template _select_gemm(ex, _M, _N, _K, _alpha, _a, _lda,
                                             _b, _ldb, _beta, _c, _ldc,
                                             batch_size);
  }  // The following _M, _N ,and _K is used for SSD + Mobilenet v2 (TF version)
  // We computed the best tile combination for each sizes -(4-March-2018)
  // POWER_VR Rogue
  else if ((_M == 546 && _K == 512 && _N == 4) ||
           (_M == 24 && _K == 512 && _N == 4) ||
           (_M == 24 && _K == 256 && _N == 1) ||
           (_M == 64 && _K == 256 && _N == 4) ||
           (_M == 24 && _K == 256 && _N == 1) ||
           (_M == 128 && _K == 64 && _N == 1)) {
    return blas::Gemm_Launcher<
        64, false, false, false, 128, Tile<1, 1, 8, 8>, _t_a, _t_b,
        static_cast<int>(Gemm_t::local_memory),
        is_beta_zero>::template _select_gemm(ex, _M, _N, _K, _alpha, _a, _lda,
                                             _b, _ldb, _beta, _c, _ldc,
                                             batch_size);
  }  // The following _M, _N ,and _K is used for SSD + Mobilenet v2 (TF version)
  // We computed the best tile combination for each sizes -(4-March-2018)
  // POWER_VR Rogue
  else if ((_M == 546 && _K == 128 && _N == 1) ||
           (_M == 546 && _K == 256 && _N == 1)) {
    return blas::Gemm_Launcher<
        64, false, false, false, 64, Tile<4, 4, 8, 8>, _t_a, _t_b,
        static_cast<int>(Gemm_t::no_local_memory),
        is_beta_zero>::template _select_gemm(ex, _M, _N, _K, _alpha, _a, _lda,
                                             _b, _ldb, _beta, _c, _ldc,
                                             batch_size);
  }  // The following _M, _N ,and _K is used for SSD + Mobilenet v2 (TF version)
  // We computed the best tile combination for each sizes -(4-March-2018)
  // POWER_VR Rogue
  else if ((_M == 576 && _K == 96 && _N == 361) ||
           (_M == 64 && _K == 384 && _N == 361) ||
           (_M == 160 && _K == 576 && _N == 100) ||
           (_M == 1280 && _K == 320 && _N == 100) ||
           (_M == 256 && _K == 1280 && _N == 100) ||
           (_M == 960 && _K == 160 && _N == 100) ||
           (_M == 192 && _K == 32 && _N == 1444) ||
           (_M > 64 && _K > 64 && _N > 64 && is_power_of_2(_M) &&
            is_power_of_2(_K) && is_power_of_2(_N))) {
    return blas::Gemm_Launcher<
        128, false, false, false, 16, Tile<4, 8, 16, 8>, _t_a, _t_b,
        static_cast<int>(Gemm_t::local_memory),
        is_beta_zero>::template _select_gemm(ex, _M, _N, _K, _alpha, _a, _lda,
                                             _b, _ldb, _beta, _c, _ldc,
                                             batch_size);
  } else {
    return blas::Gemm_Launcher<
        64, false, false, false, 32, Tile<4, 4, 8, 8>, _t_a, _t_b,
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
