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

template <bool _t_a, bool _t_b, bool is_beta_zero, typename Executor,
          typename ContainerT0, typename ContainerT1, typename ContainerT2,
          typename T, typename IndexType>
typename Executor::Policy::event_type _gemm(
    Executor& ex, IndexType _M, IndexType _N, IndexType _K, T _alpha,
    ContainerT0 _A, IndexType _lda, ContainerT1 _B, IndexType _ldb, T _beta,
    ContainerT2 _C, IndexType _ldc, IndexType batch_size) {
  if ((_M == 96 && _K == 16 && _N == 22500) ||
      (_M == 273 && _K == 576 && _N == 100) ||
      (_M == 384 && _K == 64 && _N == 361)) {
    return blas::Gemm_Launcher<
        96, true, false, false, 16, Tile<4, 6, 12, 8>, _t_a, _t_b,
        static_cast<int>(Gemm_t::local_memory),
        is_beta_zero>::template _select_gemm(ex, _M, _N, _K, _alpha, _A, _lda,
                                             _B, _ldb, _beta, _C, _ldc,
                                             batch_size);
  } else if ((_M == 546 && _K == 512 && _N == 4) ||
             (_M == 24 && _K == 512 && _N == 4) ||
             (_M == 24 && _K == 256 && _N == 1) ||
             (_M == 64 && _K == 256 && _N == 4) ||
             (_M == 24 && _K == 256 && _N == 1) ||
             (_M == 128 && _K == 64 && _N == 1)) {
    return blas::Gemm_Launcher<
        64, false, false, false, 128, Tile<1, 1, 8, 8>, _t_a, _t_b,
        static_cast<int>(Gemm_t::local_memory),
        is_beta_zero>::template _select_gemm(ex, _M, _N, _K, _alpha, _A, _lda,
                                             _B, _ldb, _beta, _C, _ldc,
                                             batch_size);
  } else if ((_M == 546 && _K == 128 && _N == 1) ||
             (_M == 546 && _K == 256 && _N == 1)) {
    return blas::Gemm_Launcher<
        64, false, false, false, 64, Tile<4, 4, 8, 8>, _t_a, _t_b,
        static_cast<int>(Gemm_t::no_local_memory),
        is_beta_zero>::template _select_gemm(ex, _M, _N, _K, _alpha, _A, _lda,
                                             _B, _ldb, _beta, _C, _ldc,
                                             batch_size);
  } else if ((_M == 576 && _K == 96 && _N == 361) ||
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
        is_beta_zero>::template _select_gemm(ex, _M, _N, _K, _alpha, _A, _lda,
                                             _B, _ldb, _beta, _C, _ldc,
                                             batch_size);
  } else {
    return blas::Gemm_Launcher<
        64, false, false, false, 32, Tile<4, 4, 8, 8>, _t_a, _t_b,
        static_cast<int>(Gemm_t::local_memory),
        is_beta_zero>::template _select_gemm(ex, _M, _N, _K, _alpha, _A, _lda,
                                             _B, _ldb, _beta, _C, _ldc,
                                             batch_size);
  }
}
}  // namespace backend
}  // namespace gemm
}  // namespace blas
#endif
