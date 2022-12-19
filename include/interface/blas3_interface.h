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
 *  @filename blas3_interface.h
 *
 **************************************************************************/

#ifndef SYCL_BLAS_BLAS3_INTERFACE_H
#define SYCL_BLAS_BLAS3_INTERFACE_H

#include "operations/blas3_trees.h"

namespace blas {
namespace internal {
/*!
 * @brief This is a top-level wrapper for GemmFactory, which provides a
 *        "standard" BLAS gemm interface.
 *
 * See the netlib blas interface documentation for more details of the hig
 * level interface:
 * http://www.netlib.org/lapack/explore-html/d4/de2/sgemm_8f.html
 */
template <typename sb_handle_t, typename container_0_t, typename container_1_t,
          typename container_2_t, typename element_t, typename index_t>
typename sb_handle_t::event_t _gemm(sb_handle_t& sb_handle, char _TransA,
                                    char _TransB, index_t _M, index_t _N,
                                    index_t _K, element_t _alpha,
                                    container_0_t a_, index_t _lda,
                                    container_1_t b_, index_t _ldb,
                                    element_t _beta, container_2_t _C,
                                    index_t _ldc);

template <typename sb_handle_t, typename container_0_t, typename container_1_t,
          typename container_2_t, typename element_t, typename index_t>
typename sb_handle_t::event_t _gemm_batched(
    sb_handle_t& sb_handle, char _TransA, char _TransB, index_t _M, index_t _N,
    index_t _K, element_t _alpha, container_0_t a_, index_t _lda,
    container_1_t b_, index_t _ldb, element_t _beta, container_2_t _C,
    index_t _ldc, index_t batch_size,
    gemm_batch_type_t batch_type = gemm_batch_type_t::strided);

template <typename sb_handle_t, typename container_0_t, typename container_1_t,
          typename element_t, typename index_t>
typename sb_handle_t::event_t _trsm(sb_handle_t& sb_handle, char side,
                                    char uplo, char trans, char diag, index_t M,
                                    index_t N, element_t alpha, container_0_t A,
                                    index_t lda, container_1_t B, index_t ldb);

}  // namespace internal

template <typename sb_handle_t, typename container_0_t, typename container_1_t,
          typename container_2_t, typename element_t, typename index_t>
typename sb_handle_t::event_t _gemm(sb_handle_t& sb_handle, char _TransA,
                                    char _TransB, index_t _M, index_t _N,
                                    index_t _K, element_t _alpha,
                                    container_0_t a_, index_t _lda,
                                    container_1_t b_, index_t _ldb,
                                    element_t _beta, container_2_t _C,
                                    index_t _ldc) {
  return internal::_gemm(sb_handle, _TransA, _TransB, _M, _N, _K, _alpha, a_,
                         _lda, b_, _ldb, _beta, _C, _ldc);
}

template <typename sb_handle_t, typename container_0_t, typename container_1_t,
          typename container_2_t, typename element_t, typename index_t>
typename sb_handle_t::event_t _gemm_batched(
    sb_handle_t& sb_handle, char _TransA, char _TransB, index_t _M, index_t _N,
    index_t _K, element_t _alpha, container_0_t a_, index_t _lda,
    container_1_t b_, index_t _ldb, element_t _beta, container_2_t _C,
    index_t _ldc, index_t batch_size,
    gemm_batch_type_t batch_type = gemm_batch_type_t::strided) {
  return internal::_gemm_batched(sb_handle, _TransA, _TransB, _M, _N, _K,
                                 _alpha, a_, _lda, b_, _ldb, _beta, _C, _ldc,
                                 batch_size, batch_type);
}

template <typename sb_handle_t, typename container_0_t, typename container_1_t,
          typename element_t, typename index_t>
typename sb_handle_t::event_t inline _trsm(sb_handle_t& sb_handle, char side,
                                           char uplo, char trans, char diag,
                                           index_t M, index_t N,
                                           element_t alpha, container_0_t A,
                                           index_t lda, container_1_t B,
                                           index_t ldb) {
  return internal::_trsm(sb_handle, side, uplo, trans, diag, M, N, alpha, A,
                         lda, B, ldb);
}

}  // namespace blas
#endif  // SYCL_BLAS_BLAS3_INTERFACE
