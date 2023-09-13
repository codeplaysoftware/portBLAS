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
 *  @filename blas3_interface.h
 *
 **************************************************************************/

#ifndef PORTBLAS_BLAS3_INTERFACE_H
#define PORTBLAS_BLAS3_INTERFACE_H

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
typename sb_handle_t::event_t _gemm(
    sb_handle_t& sb_handle, char _TransA, char _TransB, index_t _M, index_t _N,
    index_t _K, element_t _alpha, container_0_t a_, index_t _lda,
    container_1_t b_, index_t _ldb, element_t _beta, container_2_t _C,
    index_t _ldc, const typename sb_handle_t::event_t& _dependencies);

template <typename sb_handle_t, typename container_0_t, typename container_1_t,
          typename container_2_t, typename element_t, typename index_t>
typename sb_handle_t::event_t _gemm_batched(
    sb_handle_t& sb_handle, char _TransA, char _TransB, index_t _M, index_t _N,
    index_t _K, element_t _alpha, container_0_t a_, index_t _lda,
    container_1_t b_, index_t _ldb, element_t _beta, container_2_t _C,
    index_t _ldc, index_t batch_size,
    gemm_batch_type_t batch_type = gemm_batch_type_t::strided,
    const typename sb_handle_t::event_t& _dependencies = {});

template <typename sb_handle_t, typename container_0_t, typename container_1_t,
          typename container_2_t, typename element_t, typename index_t>
typename sb_handle_t::event_t _gemm_strided_batched(
    sb_handle_t& sb_handle, char _TransA, char _TransB, index_t _M, index_t _N,
    index_t _K, element_t _alpha, container_0_t a_, index_t _lda,
    index_t _stridea, container_1_t b_, index_t _ldb, index_t _strideb,
    element_t _beta, container_2_t _C, index_t _ldc, index_t _stridec,
    index_t batch_size, const typename sb_handle_t::event_t& _dependencies);

template <typename sb_handle_t, typename container_0_t, typename container_1_t,
          typename element_t, typename index_t>
typename sb_handle_t::event_t _trsm(
    sb_handle_t& sb_handle, char side, char uplo, char trans, char diag,
    index_t M, index_t N, element_t alpha, container_0_t A, index_t lda,
    container_1_t B, index_t ldb,
    const typename sb_handle_t::event_t& _dependencies);

template <typename sb_handle_t, typename container_0_t, typename container_1_t,
          typename container_2_t, typename element_t, typename index_t>
typename sb_handle_t::event_t _symm(
    sb_handle_t& sb_handle, char _side, char _uplo, index_t _M, index_t _N,
    element_t _alpha, container_0_t a_, index_t _lda, container_1_t b_,
    index_t _ldb, element_t _beta, container_2_t _C, index_t _ldc,
    const typename sb_handle_t::event_t& _dependencies);

}  // namespace internal

template <typename sb_handle_t, typename container_0_t, typename container_1_t,
          typename container_2_t, typename element_t, typename index_t>
typename sb_handle_t::event_t _gemm(
    sb_handle_t& sb_handle, char _TransA, char _TransB, index_t _M, index_t _N,
    index_t _K, element_t _alpha, container_0_t a_, index_t _lda,
    container_1_t b_, index_t _ldb, element_t _beta, container_2_t _C,
    index_t _ldc, const typename sb_handle_t::event_t& _dependencies = {}) {
  return internal::_gemm(sb_handle, _TransA, _TransB, _M, _N, _K, _alpha, a_,
                         _lda, b_, _ldb, _beta, _C, _ldc, _dependencies);
}

template <typename sb_handle_t, typename container_0_t, typename container_1_t,
          typename container_2_t, typename element_t, typename index_t>
typename sb_handle_t::event_t _gemm_batched(
    sb_handle_t& sb_handle, char _TransA, char _TransB, index_t _M, index_t _N,
    index_t _K, element_t _alpha, container_0_t a_, index_t _lda,
    container_1_t b_, index_t _ldb, element_t _beta, container_2_t _C,
    index_t _ldc, index_t batch_size,
    gemm_batch_type_t batch_type = gemm_batch_type_t::strided,
    const typename sb_handle_t::event_t& _dependencies = {}) {
  return internal::_gemm_batched(sb_handle, _TransA, _TransB, _M, _N, _K,
                                 _alpha, a_, _lda, b_, _ldb, _beta, _C, _ldc,
                                 batch_size, batch_type, _dependencies);
}

template <typename sb_handle_t, typename container_0_t, typename container_1_t,
          typename container_2_t, typename element_t, typename index_t>
typename sb_handle_t::event_t _gemm_strided_batched(
    sb_handle_t& sb_handle, char _TransA, char _TransB, index_t _M, index_t _N,
    index_t _K, element_t _alpha, container_0_t a_, index_t _lda,
    index_t _stridea, container_1_t b_, index_t _ldb, index_t _strideb,
    element_t _beta, container_2_t _C, index_t _ldc, index_t _stridec,
    index_t batch_size,
    const typename sb_handle_t::event_t& _dependencies = {}) {
  return internal::_gemm_strided_batched(
      sb_handle, _TransA, _TransB, _M, _N, _K, _alpha, a_, _lda, _stridea, b_,
      _ldb, _strideb, _beta, _C, _ldc, _stridec, batch_size, _dependencies);
}

template <typename sb_handle_t, typename container_0_t, typename container_1_t,
          typename element_t, typename index_t>
typename sb_handle_t::event_t inline _trsm(
    sb_handle_t& sb_handle, char side, char uplo, char trans, char diag,
    index_t M, index_t N, element_t alpha, container_0_t A, index_t lda,
    container_1_t B, index_t ldb,
    const typename sb_handle_t::event_t& _dependencies = {}) {
  return internal::_trsm(sb_handle, side, uplo, trans, diag, M, N, alpha, A,
                         lda, B, ldb, _dependencies);
}

template <typename sb_handle_t, typename container_0_t, typename container_1_t,
          typename container_2_t, typename element_t, typename index_t>
typename sb_handle_t::event_t _symm(
    sb_handle_t& sb_handle, char _side, char _uplo, index_t _M, index_t _N,
    element_t _alpha, container_0_t a_, index_t _lda, container_1_t b_,
    index_t _ldb, element_t _beta, container_2_t _C, index_t _ldc,
    const typename sb_handle_t::event_t& _dependencies = {}) {
  return internal::_symm(sb_handle, _side, _uplo, _M, _N, _alpha, a_, _lda, b_,
                         _ldb, _beta, _C, _ldc, _dependencies);
}

}  // namespace blas
#endif  // PORTBLAS_BLAS3_INTERFACE
