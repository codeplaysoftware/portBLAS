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
 *  @filename launch_gemm.hpp
 *
 **************************************************************************/

#include "blas_test.hpp"
#include "interface/gemm_launcher.hpp"
#include "portblas.hpp"
#include <utility>

template <bool _t_a, bool _t_b, bool s_a, bool s_b, int jm_M, int jm_N,
          int jm_K, typename inp_jmT, typename out_jmT, bool is_beta_zero,
          typename sb_handle_t, typename index_t, typename element_t,
          typename container_0_t, typename container_1_t,
          typename container_2_t>
PORTBLAS_ALWAYS_INLINE
    typename std::enable_if<jm_M == 16, typename sb_handle_t::event_t>::type
    launch_gemm(sb_handle_t& sb_handle, index_t _M, index_t _N, index_t _K,
                element_t _alpha, container_0_t _a, index_t _lda,
                index_t _stridea, container_1_t _b, index_t _ldb,
                index_t _strideb, element_t _beta, container_2_t _c,
                index_t _ldc, index_t _stridec, index_t batch_size,
                gemm_batch_type_t batch_type,
                const typename sb_handle_t::event_t& _dependencies) {
  if (_M > 1024 && _N > 1024) {
    return blas::Gemm_Launcher<
        container_0_t, container_1_t, container_2_t, 256, false, true, true,
        128,
        Tile<8, 8, 16, 16, 16, 2, 1, 1, 1, 1, jm_M, jm_N, jm_K, inp_jmT,
             out_jmT>,
        _t_a, _t_b, s_a, s_b, static_cast<int>(gemm_memory_t::local),
        static_cast<int>(gemm_algorithm_t::standard),
        static_cast<int>(gemm_vectorization_t::none), is_beta_zero, 1,
        static_cast<int>(gemm_batch_type_t::strided),
        true>::template _select_gemm(sb_handle, _M, _N, _K, _alpha, _a, _lda,
                                     _stridea, _b, _ldb, _strideb, _beta, _c,
                                     _ldc, _stridec, batch_size, _dependencies);
  } else if (_M > 64 && _N > 64) {
    return blas::Gemm_Launcher<
        container_0_t, container_1_t, container_2_t, 128, false, true, true,
        128,
        Tile<4, 8, 16, 8, 16, 2, 1, 1, 1, 1, jm_M, jm_N, jm_K, inp_jmT,
             out_jmT>,
        _t_a, _t_b, s_a, s_b, static_cast<int>(gemm_memory_t::local),
        static_cast<int>(gemm_algorithm_t::standard),
        static_cast<int>(gemm_vectorization_t::none), is_beta_zero, 1,
        static_cast<int>(gemm_batch_type_t::strided),
        true>::template _select_gemm(sb_handle, _M, _N, _K, _alpha, _a, _lda,
                                     _stridea, _b, _ldb, _strideb, _beta, _c,
                                     _ldc, _stridec, batch_size, _dependencies);

  } else {
    return blas::Gemm_Launcher<
        container_0_t, container_1_t, container_2_t, 128, false, true, true,
        128,
        Tile<2, 4, 16, 8, 16, 2, 1, 1, 1, 1, jm_M, jm_N, jm_K, inp_jmT,
             out_jmT>,
        _t_a, _t_b, s_a, s_b, static_cast<int>(gemm_memory_t::local),
        static_cast<int>(gemm_algorithm_t::standard),
        static_cast<int>(gemm_vectorization_t::none), is_beta_zero, 1,
        static_cast<int>(gemm_batch_type_t::strided),
        true>::template _select_gemm(sb_handle, _M, _N, _K, _alpha, _a, _lda,
                                     _stridea, _b, _ldb, _strideb, _beta, _c,
                                     _ldc, _stridec, batch_size, _dependencies);
  }
}

template <bool _t_a, bool _t_b, bool s_a, bool s_b, int jm_M, int jm_N,
          int jm_K, typename inp_jmT, typename out_jmT, bool is_beta_zero,
          typename sb_handle_t, typename index_t, typename element_t,
          typename container_0_t, typename container_1_t,
          typename container_2_t>
PORTBLAS_ALWAYS_INLINE
    typename std::enable_if<jm_M == 32, typename sb_handle_t::event_t>::type
    launch_gemm(sb_handle_t& sb_handle, index_t _M, index_t _N, index_t _K,
                element_t _alpha, container_0_t _a, index_t _lda,
                index_t _stridea, container_1_t _b, index_t _ldb,
                index_t _strideb, element_t _beta, container_2_t _c,
                index_t _ldc, index_t _stridec, index_t batch_size,
                gemm_batch_type_t batch_type,
                const typename sb_handle_t::event_t& _dependencies) {
  if (_M > 1024 && _N > 1024) {
    return blas::Gemm_Launcher<
        container_0_t, container_1_t, container_2_t, 128, false, true, true,
        128,
        Tile<8, 16, 16, 8, 16, 2, 1, 1, 1, 1, jm_M, jm_N, jm_K, inp_jmT,
             out_jmT>,
        _t_a, _t_b, s_a, s_b, static_cast<int>(gemm_memory_t::local),
        static_cast<int>(gemm_algorithm_t::standard),
        static_cast<int>(gemm_vectorization_t::none), is_beta_zero, 1,
        static_cast<int>(gemm_batch_type_t::strided),
        true>::template _select_gemm(sb_handle, _M, _N, _K, _alpha, _a, _lda,
                                     _stridea, _b, _ldb, _strideb, _beta, _c,
                                     _ldc, _stridec, batch_size, _dependencies);
  } else if (_M > 64 && _N > 64) {
    return blas::Gemm_Launcher<
        container_0_t, container_1_t, container_2_t, 64, false, true, true, 128,
        Tile<8, 8, 8, 8, 16, 2, 1, 1, 1, 1, jm_M, jm_N, jm_K, inp_jmT, out_jmT>,
        _t_a, _t_b, s_a, s_b, static_cast<int>(gemm_memory_t::local),
        static_cast<int>(gemm_algorithm_t::standard),
        static_cast<int>(gemm_vectorization_t::none), is_beta_zero, 1,
        static_cast<int>(gemm_batch_type_t::strided),
        true>::template _select_gemm(sb_handle, _M, _N, _K, _alpha, _a, _lda,
                                     _stridea, _b, _ldb, _strideb, _beta, _c,
                                     _ldc, _stridec, batch_size, _dependencies);

  } else {
    return blas::Gemm_Launcher<
        container_0_t, container_1_t, container_2_t, 128, false, true, true,
        128,
        Tile<2, 4, 16, 8, 16, 2, 1, 1, 1, 1, jm_M, jm_N, jm_K, inp_jmT,
             out_jmT>,
        _t_a, _t_b, s_a, s_b, static_cast<int>(gemm_memory_t::local),
        static_cast<int>(gemm_algorithm_t::standard),
        static_cast<int>(gemm_vectorization_t::none), is_beta_zero, 1,
        static_cast<int>(gemm_batch_type_t::strided),
        true>::template _select_gemm(sb_handle, _M, _N, _K, _alpha, _a, _lda,
                                     _stridea, _b, _ldb, _strideb, _beta, _c,
                                     _ldc, _stridec, batch_size, _dependencies);
  }
}

template <bool _t_a, bool _t_b, bool s_a, bool s_b, int jm_M, int jm_N,
          int jm_K, typename inp_jmT, typename out_jmT, bool is_beta_zero,
          typename sb_handle_t, typename index_t, typename element_t,
          typename container_0_t, typename container_1_t,
          typename container_2_t>
PORTBLAS_ALWAYS_INLINE
    typename std::enable_if<jm_M == 8, typename sb_handle_t::event_t>::type
    launch_gemm(sb_handle_t& sb_handle, index_t _M, index_t _N, index_t _K,
                element_t _alpha, container_0_t _a, index_t _lda,
                index_t _stridea, container_1_t _b, index_t _ldb,
                index_t _strideb, element_t _beta, container_2_t _c,
                index_t _ldc, index_t _stridec, index_t batch_size,
                gemm_batch_type_t batch_type,
                const typename sb_handle_t::event_t& _dependencies) {
  if (_M > 1024 && _N > 1024) {
    return blas::Gemm_Launcher<
        container_0_t, container_1_t, container_2_t, 256, false, true, true,
        128,
        Tile<4, 4, 16, 16, 16, 2, 1, 1, 1, 1, jm_M, jm_N, jm_K, inp_jmT,
             out_jmT>,
        _t_a, _t_b, s_a, s_b, static_cast<int>(gemm_memory_t::local),
        static_cast<int>(gemm_algorithm_t::standard),
        static_cast<int>(gemm_vectorization_t::none), is_beta_zero, 1,
        static_cast<int>(gemm_batch_type_t::strided),
        true>::template _select_gemm(sb_handle, _M, _N, _K, _alpha, _a, _lda,
                                     _stridea, _b, _ldb, _strideb, _beta, _c,
                                     _ldc, _stridec, batch_size, _dependencies);
  } else {
    return blas::Gemm_Launcher<
        container_0_t, container_1_t, container_2_t, 128, false, true, true,
        128,
        Tile<2, 4, 16, 8, 16, 2, 1, 1, 1, 1, jm_M, jm_N, jm_K, inp_jmT,
             out_jmT>,
        _t_a, _t_b, s_a, s_b, static_cast<int>(gemm_memory_t::local),
        static_cast<int>(gemm_algorithm_t::standard),
        static_cast<int>(gemm_vectorization_t::none), is_beta_zero, 1,
        static_cast<int>(gemm_batch_type_t::strided),
        true>::template _select_gemm(sb_handle, _M, _N, _K, _alpha, _a, _lda,
                                     _stridea, _b, _ldb, _strideb, _beta, _c,
                                     _ldc, _stridec, batch_size, _dependencies);
  }
}

template <int jm_M, int jm_N, int jm_K, typename inp_jmT, typename out_jmT,
          bool is_beta_zero, typename sb_handle_t, typename container_0_t,
          typename container_1_t, typename container_2_t, typename element_t,
          typename index_t>
PORTBLAS_ALWAYS_INLINE typename sb_handle_t::event_t launch_gemm_with_transpose(
    sb_handle_t& sb_handle, char _trans_a, char _trans_b, index_t _M,
    index_t _N, index_t _K, element_t _alpha, container_0_t _a, index_t _lda,
    index_t _stridea, container_1_t _b, index_t _ldb, index_t _strideb,
    element_t _beta, container_2_t _c, index_t _ldc, index_t _stridec,
    index_t batch_size, gemm_batch_type_t batch_type,
    const typename sb_handle_t::event_t& _dependencies) {
  typename sb_handle_t::event_t gemm_event;
  if (_trans_a == 't' && _trans_b == 't') {
    gemm_event = launch_gemm<true, true, false, false, jm_M, jm_N, jm_K,
                             inp_jmT, out_jmT, is_beta_zero>(
        sb_handle, _M, _N, _K, _alpha, _a, _lda, _stridea, _b, _ldb, _strideb,
        _beta, _c, _ldc, _stridec, batch_size, batch_type, _dependencies);
  } else if (_trans_a == 'n' && _trans_b == 'n') {
    gemm_event = launch_gemm<false, false, false, false, jm_M, jm_N, jm_K,
                             inp_jmT, out_jmT, is_beta_zero>(
        sb_handle, _M, _N, _K, _alpha, _a, _lda, _stridea, _b, _ldb, _strideb,
        _beta, _c, _ldc, _stridec, batch_size, batch_type, _dependencies);
  } else if (_trans_a == 't' && _trans_b == 'n') {
    gemm_event = launch_gemm<true, false, false, false, jm_M, jm_N, jm_K,
                             inp_jmT, out_jmT, is_beta_zero>(
        sb_handle, _M, _N, _K, _alpha, _a, _lda, _stridea, _b, _ldb, _strideb,
        _beta, _c, _ldc, _stridec, batch_size, batch_type, _dependencies);
  } else if (_trans_a == 'n' && _trans_b == 't') {
    gemm_event = launch_gemm<false, true, false, false, jm_M, jm_N, jm_K,
                             inp_jmT, out_jmT, is_beta_zero>(
        sb_handle, _M, _N, _K, _alpha, _a, _lda, _stridea, _b, _ldb, _strideb,
        _beta, _c, _ldc, _stridec, batch_size, batch_type, _dependencies);
  }
  return gemm_event;
}

template <int jm_M, int jm_N, int jm_K, typename inp_jmT, typename out_jmT,
          typename sb_handle_t, typename container_0_t, typename container_1_t,
          typename container_2_t, typename element_t, typename index_t>
PORTBLAS_ALWAYS_INLINE typename sb_handle_t::event_t launch_gemm_with_beta(
    sb_handle_t& sb_handle, char _trans_a, char _trans_b, index_t _M,
    index_t _N, index_t _K, element_t _alpha, container_0_t _a, index_t _lda,
    index_t _stridea, container_1_t _b, index_t _ldb, index_t _strideb,
    element_t _beta, container_2_t _c, index_t _ldc, index_t _stridec,
    index_t batch_size, gemm_batch_type_t batch_type,
    const typename sb_handle_t::event_t& _dependencies) {
  typename sb_handle_t::event_t gemm_event;
  if (_beta == (element_t)0) {
    gemm_event =
        launch_gemm_with_transpose<jm_M, jm_N, jm_K, inp_jmT, out_jmT, true>(
            sb_handle, _trans_a, _trans_b, _M, _N, _K, _alpha, _a, _lda,
            _stridea, _b, _ldb, _strideb, _beta, _c, _ldc, _stridec, batch_size,
            batch_type, _dependencies);
  } else {
    gemm_event =
        launch_gemm_with_transpose<jm_M, jm_N, jm_K, inp_jmT, out_jmT, false>(
            sb_handle, _trans_a, _trans_b, _M, _N, _K, _alpha, _a, _lda,
            _stridea, _b, _ldb, _strideb, _beta, _c, _ldc, _stridec, batch_size,
            batch_type, _dependencies);
  }
  return gemm_event;
}
