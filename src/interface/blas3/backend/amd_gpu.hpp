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
 *  @filename amd_gpu.hpp
 *
 **************************************************************************/
#ifndef PORTBLAS_GEMM_AMD_GPU_BACKEND_HPP
#define PORTBLAS_GEMM_AMD_GPU_BACKEND_HPP
#include "interface/gemm_launcher.h"

namespace blas {
namespace gemm {

namespace backend {
template <bool _t_a, bool _t_b, bool s_a, bool s_b, bool is_beta_zero,
          typename sb_handle_t, typename container_0_t, typename container_1_t,
          typename container_2_t, typename element_t, typename index_t>
typename std::enable_if<is_sycl_scalar<element_t>::value,
                        typename sb_handle_t::event_t>::type
_gemm(sb_handle_t& sb_handle, index_t _M, index_t _N, index_t _K,
      element_t _alpha, container_0_t _a, index_t _lda, index_t _stridea,
      container_1_t _b, index_t _ldb, index_t _strideb, element_t _beta,
      container_2_t _c, index_t _ldc, index_t _stridec, index_t batch_size,
      gemm_batch_type_t batch_type,
      const typename sb_handle_t::event_t& _dependencies) {
  // Unused configuration cases
  if constexpr (s_a && s_b || ((s_a && _t_b) || (s_b && _t_a))) {
    return _dependencies;
  } else {
    // computing arithmetic intensity with combination of input to use it as
    // heuristic numerator is the number of fma, denominator is the number of
    // bytes access.
    const auto n_fma = (static_cast<int64_t>(_M) * static_cast<int64_t>(_K) *
                        static_cast<int64_t>(_N));
    const auto n_bytes_access = (_M * _K + _K * _N + _M * _N);
    const auto arith_intensity = n_fma / n_bytes_access;
    static constexpr int ClSize = 64;
    static constexpr int tileWgSize = ClSize / sizeof(element_t);
    if (batch_type == gemm_batch_type_t::interleaved) {
      return blas::Gemm_Launcher<
          container_0_t, container_1_t, container_2_t, 64, false, false, false,
          64, Tile<4, 4, 4, 4, 1, 1, 1, 1, 4, 4>, _t_a, _t_b, s_a, s_b,
          static_cast<int>(gemm_memory_t::no_local),
          static_cast<int>(gemm_algorithm_t::standard),
          static_cast<int>(gemm_vectorization_t::full), is_beta_zero, 4,
          static_cast<int>(gemm_batch_type_t::interleaved)>::
          template _select_gemm(sb_handle, _M, _N, _K, _alpha, _a, _lda,
                                _stridea, _b, _ldb, _strideb, _beta, _c, _ldc,
                                _stridec, batch_size, _dependencies);
    }

/* Tall & Skinny matrices. */
#ifdef GEMM_TALL_SKINNY_SUPPORT
    if (batch_size == 1 &&
        ((_K > 8192 && _M <= 1024 && _N <= 1024) ||
         (_K > 1024 && _M <= 256 && _N <= 256)) &&
        (!s_a && !s_b)) {
      if (_M <= 16 && _N > 32) {
        return blas::Gemm_Launcher<
            container_0_t, container_1_t, container_2_t, 256, true, true, true,
            ClSize, Tile<1, 4, tileWgSize, tileWgSize>, _t_a, _t_b, s_a, s_b,
            static_cast<int>(gemm_memory_t::local),
            static_cast<int>(gemm_algorithm_t::tall_skinny),
            static_cast<int>(gemm_vectorization_t::none), is_beta_zero, 2,
            static_cast<int>(gemm_batch_type_t::strided)>::
            template _select_gemm(sb_handle, _M, _N, _K, _alpha, _a, _lda,
                                  _stridea, _b, _ldb, _strideb, _beta, _c, _ldc,
                                  _stridec, batch_size, _dependencies);
      } else if (_M > 64 && _N <= 32) {
        return blas::Gemm_Launcher<
            container_0_t, container_1_t, container_2_t, 256, true, true, true,
            ClSize, Tile<4, 1, tileWgSize, tileWgSize>, _t_a, _t_b, s_a, s_b,
            static_cast<int>(gemm_memory_t::local),
            static_cast<int>(gemm_algorithm_t::tall_skinny),
            static_cast<int>(gemm_vectorization_t::none), is_beta_zero, 2,
            static_cast<int>(gemm_batch_type_t::strided)>::
            template _select_gemm(sb_handle, _M, _N, _K, _alpha, _a, _lda,
                                  _stridea, _b, _ldb, _strideb, _beta, _c, _ldc,
                                  _stridec, batch_size, _dependencies);
      } else if (_M <= 16 || _N <= 16) {
        return blas::Gemm_Launcher<
            container_0_t, container_1_t, container_2_t, 256, true, true, true,
            ClSize, Tile<1, 1, tileWgSize, tileWgSize>, _t_a, _t_b, s_a, s_b,
            static_cast<int>(gemm_memory_t::local),
            static_cast<int>(gemm_algorithm_t::tall_skinny),
            static_cast<int>(gemm_vectorization_t::none), is_beta_zero, 2,
            static_cast<int>(gemm_batch_type_t::strided)>::
            template _select_gemm(sb_handle, _M, _N, _K, _alpha, _a, _lda,
                                  _stridea, _b, _ldb, _strideb, _beta, _c, _ldc,
                                  _stridec, batch_size, _dependencies);
      } else if (_M <= 32 || _N <= 32) {
        return blas::Gemm_Launcher<
            container_0_t, container_1_t, container_2_t, 256, true, true, true,
            ClSize, Tile<2, 2, tileWgSize, tileWgSize>, _t_a, _t_b, s_a, s_b,
            static_cast<int>(gemm_memory_t::local),
            static_cast<int>(gemm_algorithm_t::tall_skinny),
            static_cast<int>(gemm_vectorization_t::none), is_beta_zero, 2,
            static_cast<int>(gemm_batch_type_t::strided)>::
            template _select_gemm(sb_handle, _M, _N, _K, _alpha, _a, _lda,
                                  _stridea, _b, _ldb, _strideb, _beta, _c, _ldc,
                                  _stridec, batch_size, _dependencies);
      } else {
        return blas::Gemm_Launcher<
            container_0_t, container_1_t, container_2_t, 256, true, true, true,
            ClSize, Tile<4, 4, tileWgSize, tileWgSize>, _t_a, _t_b, s_a, s_b,
            static_cast<int>(gemm_memory_t::local),
            static_cast<int>(gemm_algorithm_t::tall_skinny),
            static_cast<int>(gemm_vectorization_t::none), is_beta_zero, 2,
            static_cast<int>(gemm_batch_type_t::strided)>::
            template _select_gemm(sb_handle, _M, _N, _K, _alpha, _a, _lda,
                                  _stridea, _b, _ldb, _strideb, _beta, _c, _ldc,
                                  _stridec, batch_size, _dependencies);
      }
    } else
#endif  // GEMM_TALL_SKINNY_SUPPORT
      // Folling configuration are taken using the auto tuner on amd-mi210
      // and divided following their arith_intensity or another ratio between _N
      // and _K input size
      if ((_N >> 4) > _K) {
        if (arith_intensity <= 100) {
          return blas::Gemm_Launcher<
              container_0_t, container_1_t, container_2_t, 256, false, false,
              true, 64, Tile<4, 8, 16, 16>, _t_a, _t_b, s_a, s_b,
              static_cast<int>(gemm_memory_t::local),
              static_cast<int>(gemm_algorithm_t::standard),
              static_cast<int>(gemm_vectorization_t::full), is_beta_zero, 1,
              static_cast<int>(gemm_batch_type_t::strided)>::
              template _select_gemm(sb_handle, _M, _N, _K, _alpha, _a, _lda,
                                    _stridea, _b, _ldb, _strideb, _beta, _c,
                                    _ldc, _stridec, batch_size, _dependencies);
        } else {
          return blas::Gemm_Launcher<
              container_0_t, container_1_t, container_2_t, 256, false, false,
              true, 32, Tile<8, 8, 16, 16>, _t_a, _t_b, s_a, s_b,
              static_cast<int>(gemm_memory_t::local),
              static_cast<int>(gemm_algorithm_t::standard),
              static_cast<int>(gemm_vectorization_t::full), is_beta_zero, 1,
              static_cast<int>(gemm_batch_type_t::strided)>::
              template _select_gemm(sb_handle, _M, _N, _K, _alpha, _a, _lda,
                                    _stridea, _b, _ldb, _strideb, _beta, _c,
                                    _ldc, _stridec, batch_size, _dependencies);
        }
      } else if (arith_intensity >= 360) {
        return blas::Gemm_Launcher<
            container_0_t, container_1_t, container_2_t, 256, false, false,
            true, 32, Tile<8, 8, 16, 16>, _t_a, _t_b, s_a, s_b,
            static_cast<int>(gemm_memory_t::local),
            static_cast<int>(gemm_algorithm_t::standard),
            static_cast<int>(gemm_vectorization_t::full), is_beta_zero, 1,
            static_cast<int>(gemm_batch_type_t::strided)>::
            template _select_gemm(sb_handle, _M, _N, _K, _alpha, _a, _lda,
                                  _stridea, _b, _ldb, _strideb, _beta, _c, _ldc,
                                  _stridec, batch_size, _dependencies);
      } else if (arith_intensity >= 240) {
        return blas::Gemm_Launcher<
            container_0_t, container_1_t, container_2_t, 256, false, false,
            true, 64, Tile<4, 4, 16, 8>, _t_a, _t_b, s_a, s_b,
            static_cast<int>(gemm_memory_t::local),
            static_cast<int>(gemm_algorithm_t::standard),
            static_cast<int>(gemm_vectorization_t::full), is_beta_zero, 1,
            static_cast<int>(gemm_batch_type_t::strided)>::
            template _select_gemm(sb_handle, _M, _N, _K, _alpha, _a, _lda,
                                  _stridea, _b, _ldb, _strideb, _beta, _c, _ldc,
                                  _stridec, batch_size, _dependencies);
      } else if (arith_intensity > 162) {
        return blas::Gemm_Launcher<
            container_0_t, container_1_t, container_2_t, 256, false, false,
            true, 128, Tile<4, 4, 16, 8>, _t_a, _t_b, s_a, s_b,
            static_cast<int>(gemm_memory_t::local),
            static_cast<int>(gemm_algorithm_t::standard),
            static_cast<int>(gemm_vectorization_t::full), is_beta_zero, 1,
            static_cast<int>(gemm_batch_type_t::strided)>::
            template _select_gemm(sb_handle, _M, _N, _K, _alpha, _a, _lda,
                                  _stridea, _b, _ldb, _strideb, _beta, _c, _ldc,
                                  _stridec, batch_size, _dependencies);
      } else if (arith_intensity >= 100 && arith_intensity <= 162) {
        return blas::Gemm_Launcher<
            container_0_t, container_1_t, container_2_t, 256, false, true, true,
            128, Tile<2, 2, 16, 8>, _t_a, _t_b, s_a, s_b,
            static_cast<int>(gemm_memory_t::local),
            static_cast<int>(gemm_algorithm_t::standard),
            static_cast<int>(gemm_vectorization_t::full), is_beta_zero, 1,
            static_cast<int>(gemm_batch_type_t::strided)>::
            template _select_gemm(sb_handle, _M, _N, _K, _alpha, _a, _lda,
                                  _stridea, _b, _ldb, _strideb, _beta, _c, _ldc,
                                  _stridec, batch_size, _dependencies);
      } else if (arith_intensity <= 100) {
        return blas::Gemm_Launcher<
            container_0_t, container_1_t, container_2_t, 256, false, false,
            true, 128, Tile<1, 1, 16, 8>, _t_a, _t_b, s_a, s_b,
            static_cast<int>(gemm_memory_t::local),
            static_cast<int>(gemm_algorithm_t::standard),
            static_cast<int>(gemm_vectorization_t::full), is_beta_zero, 1,
            static_cast<int>(gemm_batch_type_t::strided)>::
            template _select_gemm(sb_handle, _M, _N, _K, _alpha, _a, _lda,
                                  _stridea, _b, _ldb, _strideb, _beta, _c, _ldc,
                                  _stridec, batch_size, _dependencies);
      } else {
        // this branch is a safe net just in case no other branch is taken
        return blas::Gemm_Launcher<
            container_0_t, container_1_t, container_2_t, 256, false, false,
            false, ClSize, Tile<4, 4, tileWgSize, tileWgSize>, _t_a, _t_b, s_a,
            s_b, static_cast<int>(gemm_memory_t::local),
            static_cast<int>(gemm_algorithm_t::standard),
            static_cast<int>(gemm_vectorization_t::full), is_beta_zero, 2,
            static_cast<int>(gemm_batch_type_t::strided)>::
            template _select_gemm(sb_handle, _M, _N, _K, _alpha, _a, _lda,
                                  _stridea, _b, _ldb, _strideb, _beta, _c, _ldc,
                                  _stridec, batch_size, _dependencies);
      }
  }
}

// Complex Configurations
#ifdef BLAS_ENABLE_COMPLEX
template <bool _t_a, bool _t_b, bool s_a, bool s_b, bool is_beta_zero,
          typename sb_handle_t, typename container_0_t, typename container_1_t,
          typename container_2_t, typename element_t, typename index_t>
typename std::enable_if<is_complex_sycl<element_t>::value,
                        typename sb_handle_t::event_t>::type
_gemm(sb_handle_t& sb_handle, index_t _M, index_t _N, index_t _K,
      element_t _alpha, container_0_t _a, index_t _lda, index_t _stridea,
      container_1_t _b, index_t _ldb, index_t _strideb, element_t _beta,
      container_2_t _c, index_t _ldc, index_t _stridec, index_t batch_size,
      gemm_batch_type_t batch_type,
      const typename sb_handle_t::event_t& _dependencies) {
  static constexpr int ClSize = 64;
  static constexpr int tileWgSize = ClSize / sizeof(element_t);
/* Tall & Skinny matrices. */
#ifdef GEMM_TALL_SKINNY_SUPPORT
  if (batch_size == 1 && (_M / _N > 8 || _N / _M > 8)) {
    return blas::Gemm_Launcher<
        container_0_t, container_1_t, container_2_t, 256, true, true, true,
        ClSize, Tile<1, 4, tileWgSize, tileWgSize>, _t_a, _t_b, false, false,
        static_cast<int>(gemm_memory_t::local),
        static_cast<int>(gemm_algorithm_t::tall_skinny),
        static_cast<int>(gemm_vectorization_t::none), is_beta_zero, 1,
        static_cast<int>(gemm_batch_type_t::strided)>::
        template _select_gemm(sb_handle, _M, _N, _K, _alpha, _a, _lda, _stridea,
                              _b, _ldb, _strideb, _beta, _c, _ldc, _stridec,
                              batch_size, _dependencies);
  }
#endif
  if (_M * _N <= 65536) {
    return blas::Gemm_Launcher<
        container_0_t, container_1_t, container_2_t, 256, false, false, false,
        ClSize, Tile<1, 1, tileWgSize, tileWgSize>, _t_a, _t_b, false, false,
        static_cast<int>(gemm_memory_t::local),
        static_cast<int>(gemm_algorithm_t::standard),
        static_cast<int>(gemm_vectorization_t::full), is_beta_zero, 1,
        static_cast<int>(gemm_batch_type_t::strided)>::
        template _select_gemm(sb_handle, _M, _N, _K, _alpha, _a, _lda, _stridea,
                              _b, _ldb, _strideb, _beta, _c, _ldc, _stridec,
                              batch_size, _dependencies);
  } else {
    return blas::Gemm_Launcher<
        container_0_t, container_1_t, container_2_t, 256, false, false, false,
        ClSize, Tile<4, 4, tileWgSize, tileWgSize>, _t_a, _t_b, false, false,
        static_cast<int>(gemm_memory_t::local),
        static_cast<int>(gemm_algorithm_t::standard),
        static_cast<int>(gemm_vectorization_t::full), is_beta_zero, 1,
        static_cast<int>(gemm_batch_type_t::strided)>::
        template _select_gemm(sb_handle, _M, _N, _K, _alpha, _a, _lda, _stridea,
                              _b, _ldb, _strideb, _beta, _c, _ldc, _stridec,
                              batch_size, _dependencies);
  }
}
#endif

}  // namespace backend
}  // namespace gemm
}  // namespace blas
#endif
