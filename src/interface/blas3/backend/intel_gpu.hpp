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
        64, false, false, false, 64, Tile<4, 4, 4, 4, 1, 1, 1, 1, 4, 4>, _t_a,
        _t_b, static_cast<int>(gemm_memory_t::no_local),
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
#ifdef GEMM_TALL_SKINNY_SUPPORT
  /* Tall & Skinny matrices. */
  if (batch_size == 1 &&
      ((_K >= 4096 && _M * _N <= 16384) || (_K >= 1024 && _M * _N <= 4096))) {
    if (_M >= 16 && _N <= 4) {
      return blas::Gemm_Launcher<
          32, true, true, true, 64, Tile<2, 1, 8, 4>, _t_a, _t_b,
          static_cast<int>(gemm_memory_t::local),
          static_cast<int>(gemm_algorithm_t::tall_skinny),
          static_cast<int>(gemm_vectorization_t::none), is_beta_zero, 4,
          static_cast<int>(
              gemm_batch_type_t::strided)>::template _select_gemm(sb_handle, _M,
                                                                  _N, _K,
                                                                  _alpha, _a,
                                                                  _lda, _b,
                                                                  _ldb, _beta,
                                                                  _c, _ldc,
                                                                  batch_size);
    } else if (_M <= 4 || _N <= 4) {
      // Need to increase the work group size for cl::sycl::half for the
      // launcher to be instancianted
      constexpr int wg_size = sizeof(element_t) == 2 ? 8 : 4;
      return blas::Gemm_Launcher<
          16, true, false, false, 64, Tile<1, 1, wg_size, wg_size>, _t_a, _t_b,
          static_cast<int>(gemm_memory_t::local),
          static_cast<int>(gemm_algorithm_t::tall_skinny),
          static_cast<int>(gemm_vectorization_t::none), is_beta_zero, 4,
          static_cast<int>(
              gemm_batch_type_t::strided)>::template _select_gemm(sb_handle, _M,
                                                                  _N, _K,
                                                                  _alpha, _a,
                                                                  _lda, _b,
                                                                  _ldb, _beta,
                                                                  _c, _ldc,
                                                                  batch_size);
    } else if (_M >= 16 && _N <= 8) {
      return blas::Gemm_Launcher<
          32, true, true, true, 64, Tile<2, 2, 8, 4>, _t_a, _t_b,
          static_cast<int>(gemm_memory_t::local),
          static_cast<int>(gemm_algorithm_t::tall_skinny),
          static_cast<int>(gemm_vectorization_t::none), is_beta_zero, 4,
          static_cast<int>(
              gemm_batch_type_t::strided)>::template _select_gemm(sb_handle, _M,
                                                                  _N, _K,
                                                                  _alpha, _a,
                                                                  _lda, _b,
                                                                  _ldb, _beta,
                                                                  _c, _ldc,
                                                                  batch_size);
    } else if (_M <= 8 || _N <= 8) {
      // Need to increase the work group size for cl::sycl::half for the
      // launcher to be instancianted
      constexpr int wg_size = sizeof(element_t) == 2 ? 8 : 4;
      return blas::Gemm_Launcher<
          16, true, false, false, 64, Tile<2, 2, wg_size, wg_size>, _t_a, _t_b,
          static_cast<int>(gemm_memory_t::local),
          static_cast<int>(gemm_algorithm_t::tall_skinny),
          static_cast<int>(gemm_vectorization_t::none), is_beta_zero, 4,
          static_cast<int>(
              gemm_batch_type_t::strided)>::template _select_gemm(sb_handle, _M,
                                                                  _N, _K,
                                                                  _alpha, _a,
                                                                  _lda, _b,
                                                                  _ldb, _beta,
                                                                  _c, _ldc,
                                                                  batch_size);
    } else if (_M <= 16 || _N <= 16) {
      return blas::Gemm_Launcher<
          64, true, true, true, 64, Tile<2, 2, 8, 8>, _t_a, _t_b,
          static_cast<int>(gemm_memory_t::local),
          static_cast<int>(gemm_algorithm_t::tall_skinny),
          static_cast<int>(gemm_vectorization_t::none), is_beta_zero, 4,
          static_cast<int>(
              gemm_batch_type_t::strided)>::template _select_gemm(sb_handle, _M,
                                                                  _N, _K,
                                                                  _alpha, _a,
                                                                  _lda, _b,
                                                                  _ldb, _beta,
                                                                  _c, _ldc,
                                                                  batch_size);
    } else if (_M <= 32 || _N <= 32) {
      return blas::Gemm_Launcher<
          64, true, true, true, 64, Tile<4, 4, 8, 8>, _t_a, _t_b,
          static_cast<int>(gemm_memory_t::local),
          static_cast<int>(gemm_algorithm_t::tall_skinny),
          static_cast<int>(gemm_vectorization_t::none), is_beta_zero, 4,
          static_cast<int>(
              gemm_batch_type_t::strided)>::template _select_gemm(sb_handle, _M,
                                                                  _N, _K,
                                                                  _alpha, _a,
                                                                  _lda, _b,
                                                                  _ldb, _beta,
                                                                  _c, _ldc,
                                                                  batch_size);
    } else {
      constexpr int wg_size = sizeof(element_t) == 8 ? 8 : 16;
      return blas::Gemm_Launcher<
          256, true, true, true, 64, Tile<4, 4, wg_size, wg_size>, _t_a, _t_b,
          static_cast<int>(gemm_memory_t::local),
          static_cast<int>(gemm_algorithm_t::tall_skinny),
          static_cast<int>(gemm_vectorization_t::none), is_beta_zero, 4,
          static_cast<int>(
              gemm_batch_type_t::strided)>::template _select_gemm(sb_handle, _M,
                                                                  _N, _K,
                                                                  _alpha, _a,
                                                                  _lda, _b,
                                                                  _ldb, _beta,
                                                                  _c, _ldc,
                                                                  batch_size);
    }
  } else if (batch_size == 1 && (_t_a || (_t_b && _M * _N > 1048576))) {
    if (_M <= 64 || _N <= 64) {
      return blas::Gemm_Launcher<
          64, true, true, true, 64, Tile<4, 4, 8, 8>, _t_a, _t_b,
          static_cast<int>(gemm_memory_t::local),
          static_cast<int>(gemm_algorithm_t::tall_skinny),
          static_cast<int>(gemm_vectorization_t::none), is_beta_zero, 4,
          static_cast<int>(
              gemm_batch_type_t::strided)>::template _select_gemm(sb_handle, _M,
                                                                  _N, _K,
                                                                  _alpha, _a,
                                                                  _lda, _b,
                                                                  _ldb, _beta,
                                                                  _c, _ldc,
                                                                  batch_size);
    } else {
      // Need to increase the work group size for double for the
      // launcher to be instancianted
      constexpr int wg_size = sizeof(element_t) == 8 ? 8 : 16;
      return blas::Gemm_Launcher<
          256, true, true, true, 64, Tile<4, 4, wg_size, wg_size>, _t_a, _t_b,
          static_cast<int>(gemm_memory_t::local),
          static_cast<int>(gemm_algorithm_t::tall_skinny),
          static_cast<int>(gemm_vectorization_t::none), is_beta_zero, 4,
          static_cast<int>(
              gemm_batch_type_t::strided)>::template _select_gemm(sb_handle, _M,
                                                                  _N, _K,
                                                                  _alpha, _a,
                                                                  _lda, _b,
                                                                  _ldb, _beta,
                                                                  _c, _ldc,
                                                                  batch_size);
    }
  }
#endif
  if (_M <= 128 && _N <= 128) {
    return blas::Gemm_Launcher<
        64, true, false, false, 64, Tile<4, 4, 8, 8>, _t_a, _t_b,
        static_cast<int>(gemm_memory_t::local),
        static_cast<int>(gemm_algorithm_t::standard),
        static_cast<int>(gemm_vectorization_t::full), is_beta_zero, 4,
        static_cast<int>(
            gemm_batch_type_t::strided)>::template _select_gemm(sb_handle, _M,
                                                                _N, _K, _alpha,
                                                                _a, _lda, _b,
                                                                _ldb, _beta, _c,
                                                                _ldc,
                                                                batch_size);
  } else if (_t_b && !_t_a) {
    return blas::Gemm_Launcher<
        64, false, false, false, 64, Tile<8, 8, 8, 8>, _t_a, _t_b,
        static_cast<int>(gemm_memory_t::no_local),
        static_cast<int>(gemm_algorithm_t::standard),
        static_cast<int>(gemm_vectorization_t::partial), is_beta_zero, 4,
        static_cast<int>(
            gemm_batch_type_t::strided)>::template _select_gemm(sb_handle, _M,
                                                                _N, _K, _alpha,
                                                                _a, _lda, _b,
                                                                _ldb, _beta, _c,
                                                                _ldc,
                                                                batch_size);
  } else {
    return blas::Gemm_Launcher<
        64, false, false, false, 64, Tile<8, 8, 8, 8>, _t_a, _t_b,
        static_cast<int>(gemm_memory_t::local),
        static_cast<int>(gemm_algorithm_t::standard),
        static_cast<int>(gemm_vectorization_t::full), is_beta_zero, 4,
        static_cast<int>(
            gemm_batch_type_t::strided)>::template _select_gemm(sb_handle, _M,
                                                                _N, _K, _alpha,
                                                                _a, _lda, _b,
                                                                _ldb, _beta, _c,
                                                                _ldc,
                                                                batch_size);
  }
}
}  // namespace backend
}  // namespace gemm
}  // namespace blas
#endif
