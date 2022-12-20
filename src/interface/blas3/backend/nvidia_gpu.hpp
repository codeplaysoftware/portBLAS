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
 *  @filename NVIDIA_GPU.hpp
 *
 **************************************************************************/
#ifndef SYCL_BLAS_GEMM_NVIDIA_GPU_BACKEND_HPP
#define SYCL_BLAS_GEMM_NVIDIA_GPU_BACKEND_HPP
#include "interface/gemm_launcher.h"

namespace blas {
namespace gemm {
namespace backend {

template <bool _t_a, bool _t_b, bool is_beta_zero, typename sb_handle_t,
          typename container_0_t, typename container_1_t,
          typename container_2_t, typename element_t, typename index_t>
typename sb_handle_t::event_t _gemm(sb_handle_t& sb_handle, index_t _M,
                                    index_t _N, index_t _K, element_t _alpha,
                                    container_0_t _a, index_t _lda,
                                    container_1_t _b, index_t _ldb,
                                    element_t _beta, container_2_t _c,
                                    index_t _ldc, index_t batch_size,
                                    gemm_batch_type_t batch_type) {
  if (batch_type == gemm_batch_type_t::interleaved) {
    return blas::Gemm_Launcher<
        64, false, false, false, 64,
        Tile<2, 2, 4, 4, 1, 1, 1, 1, 4, 4, 1, 1, 1, float, float>, _t_a, _t_b,
        static_cast<int>(gemm_memory_t::no_local),
        static_cast<int>(gemm_algorithm_t::standard),
        static_cast<int>(gemm_vectorization_t::full), is_beta_zero, 4,
        static_cast<int>(
            gemm_batch_type_t::interleaved)>::template _select_gemm(sb_handle,
                                                                    _M, _N, _K,
                                                                    _alpha, _a,
                                                                    _lda, _b,
                                                                    _ldb, _beta,
                                                                    _c, _ldc,
                                                                    batch_size);
  }

#ifdef SB_ENABLE_JOINT_MATRIX
  const char* en_joint_matrix = std::getenv("SB_ENABLE_JOINT_MATRIX");
  if (en_joint_matrix != NULL && *en_joint_matrix == '1') {
    if (_M > 1024 && _N > 1024) {
      return blas::Gemm_Launcher<
          256, false, true, true, 128,
          Tile<8, 8, 16, 16, 16, 2, 1, 1, 1, 1, 16, 16, 16, cl::sycl::half,
               float>,
          _t_a, _t_b, static_cast<int>(gemm_memory_t::local),
          static_cast<int>(gemm_algorithm_t::standard),
          static_cast<int>(gemm_vectorization_t::none), is_beta_zero, 1,
          static_cast<int>(gemm_batch_type_t::strided),
          true>::template _select_gemm(sb_handle, _M, _N, _K, _alpha, _a, _lda,
                                       _b, _ldb, _beta, _c, _ldc, batch_size);
    } else if (_M > 64 && _N > 64) {
      return blas::Gemm_Launcher<
          128, false, true, true, 128,
          Tile<4, 8, 16, 8, 16, 2, 1, 1, 1, 1, 16, 16, 16, cl::sycl::half,
               float>,
          _t_a, _t_b, static_cast<int>(gemm_memory_t::local),
          static_cast<int>(gemm_algorithm_t::standard),
          static_cast<int>(gemm_vectorization_t::none), is_beta_zero, 1,
          static_cast<int>(gemm_batch_type_t::strided),
          true>::template _select_gemm(sb_handle, _M, _N, _K, _alpha, _a, _lda,
                                       _b, _ldb, _beta, _c, _ldc, batch_size);

    } else {
      return blas::Gemm_Launcher<
          128, false, true, true, 128,
          Tile<2, 4, 16, 8, 16, 2, 1, 1, 1, 1, 16, 16, 16, cl::sycl::half,
               float>,
          _t_a, _t_b, static_cast<int>(gemm_memory_t::local),
          static_cast<int>(gemm_algorithm_t::standard),
          static_cast<int>(gemm_vectorization_t::none), is_beta_zero, 1,
          static_cast<int>(gemm_batch_type_t::strided),
          true>::template _select_gemm(sb_handle, _M, _N, _K, _alpha, _a, _lda,
                                       _b, _ldb, _beta, _c, _ldc, batch_size);
    }
  } else {
    return blas::Gemm_Launcher<
        64, false, false, true, 64,
        Tile<8, 8, 8, 8, 1, 1, 2, 2, 1, 1, 1, 1, 1, float, float>, _t_a, _t_b,
        static_cast<int>(gemm_memory_t::local),
        static_cast<int>(gemm_algorithm_t::standard),
        static_cast<int>(gemm_vectorization_t::full), is_beta_zero, 1,
        static_cast<int>(gemm_batch_type_t::strided),
        false>::template _select_gemm(sb_handle, _M, _N, _K, _alpha, _a, _lda,
                                      _b, _ldb, _beta, _c, _ldc, batch_size);
  }

#else  // SB_ENABLE_JOINT_MATRIX
  else {
    return blas::Gemm_Launcher<
        64, false, false, true, 64,
        Tile<8, 8, 8, 8, 1, 1, 2, 2, 1, 1, 1, 1, 1, float, float>, _t_a, _t_b,
        static_cast<int>(gemm_memory_t::local),
        static_cast<int>(gemm_algorithm_t::standard),
        static_cast<int>(gemm_vectorization_t::full), is_beta_zero, 1,
        static_cast<int>(gemm_batch_type_t::strided),
        false>::template _select_gemm(sb_handle, _M, _N, _K, _alpha, _a, _lda,
                                      _b, _ldb, _beta, _c, _ldc, batch_size);
  }
#endif
}
}  // namespace backend
}  // namespace gemm
}  // namespace blas
#endif
