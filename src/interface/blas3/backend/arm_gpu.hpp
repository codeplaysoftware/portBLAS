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
 *  @filename arm_gpu.hpp
 *
 **************************************************************************/
#ifndef SYCL_BLAS_GEMM_ARM_GPU_BACKEND_HPP
#define SYCL_BLAS_GEMM_ARM_GPU_BACKEND_HPP
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
  static constexpr int GemmMemoryType =
      static_cast<int>(gemm_memory_t::no_local);
  static constexpr int GemmAlgorithm =
      static_cast<int>(gemm_algorithm_t::standard);
  static constexpr int GemmVectorization =
      static_cast<int>(gemm_vectorization_t::partial);
  static constexpr int ClSize = 64;
  static constexpr int VectorSize = 4;
  static constexpr int MiniBatchSize = 4;
  if (_M == 512 && _N == 49 && _K == 512) {
    return blas::Gemm_Launcher<
        /* WgSize= */ 64, false, false, false, ClSize, Tile<4, 4, 8, 8>, _t_a,
        _t_b, GemmMemoryType, GemmAlgorithm, GemmVectorization, is_beta_zero,
        VectorSize, 1>::template _select_gemm(ex, _M, _N, _K, _alpha, _a, _lda,
                                              _b, _ldb, _beta, _c, _ldc,
                                              batch_size);
  } else if (_t_a) {
    if (batch_size % MiniBatchSize == 0) {
      return blas::Gemm_Launcher<
          /* WgSize= */ 128, false, false, false, ClSize, Tile<4, 8, 16, 8>,
          _t_a, _t_b, GemmMemoryType, GemmAlgorithm, GemmVectorization,
          is_beta_zero, VectorSize,
          MiniBatchSize>::template _select_gemm(ex, _M, _N, _K, _alpha, _a,
                                                _lda, _b, _ldb, _beta, _c, _ldc,
                                                batch_size);
    } else {
      return blas::Gemm_Launcher<
          /* WgSize= */ 128, false, false, false, ClSize, Tile<4, 8, 16, 8>,
          _t_a, _t_b, GemmMemoryType, GemmAlgorithm, GemmVectorization,
          is_beta_zero, VectorSize, 1>::template _select_gemm(ex, _M, _N, _K,
                                                              _alpha, _a, _lda,
                                                              _b, _ldb, _beta,
                                                              _c, _ldc,
                                                              batch_size);
    }
  } else {
    if (batch_size % MiniBatchSize == 0) {
      return blas::Gemm_Launcher<
          /* WgSize= */ 32, false, false, false, ClSize, Tile<8, 4, 4, 8>, _t_a,
          _t_b, GemmMemoryType, GemmAlgorithm, GemmVectorization, is_beta_zero,
          VectorSize, MiniBatchSize>::template _select_gemm(ex, _M, _N, _K,
                                                            _alpha, _a, _lda,
                                                            _b, _ldb, _beta, _c,
                                                            _ldc, batch_size);
    } else {
      return blas::Gemm_Launcher<
          /* WgSize= */ 32, false, false, false, ClSize, Tile<8, 4, 4, 8>, _t_a,
          _t_b, GemmMemoryType, GemmAlgorithm, GemmVectorization, is_beta_zero,
          VectorSize, 1>::template _select_gemm(ex, _M, _N, _K, _alpha, _a,
                                                _lda, _b, _ldb, _beta, _c, _ldc,
                                                batch_size);
    }
  }
}
}  // namespace backend
}  // namespace gemm
}  // namespace blas
#endif
