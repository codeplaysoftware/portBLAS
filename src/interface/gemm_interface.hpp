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
 **************************************************************************/

#ifndef SYCL_BLAS_BLAS3_GEMM_INTERFACE_HPP
#define SYCL_BLAS_BLAS3_GEMM_INTERFACE_HPP

#include "blas_meta.h"
#include "interface/blas1_interface.h"
#include "interface/blas3/backend/backend.hpp"
#include "interface/blas3_interface.h"
#include "operations/blas3_trees.h"
#include "sb_handle/sycl_blas_handle.h"
#include "sycl_blas_helper.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <stdexcept>
#include <vector>

namespace blas {

/*!
 * @brief This is a top-level wrapper for GemmFactory, which provides a
 *        "standard" BLAS gemm interface.
 *
 * See netlib.org/blas for details.
 */
namespace internal {

template <bool _t_a, bool _t_b, bool is_beta_zero, typename sb_handle_t,
          typename container_0_t, typename container_1_t,
          typename container_2_t, typename element_t, typename index_t>
typename sb_handle_t::event_t _gemm_platform_specific(
    sb_handle_t& sb_handle, index_t _M, index_t _N, index_t _K,
    element_t _alpha, container_0_t a_, index_t _lda, container_1_t b_,
    index_t _ldb, element_t _beta, container_2_t _C, index_t _ldc,
    index_t batch_size, gemm_batch_type_t batch_type) {
  return blas::gemm::backend::_gemm<_t_a, _t_b, is_beta_zero>(
      sb_handle, _M, _N, _K, _alpha, a_, _lda, b_, _ldb, _beta, _C, _ldc,
      batch_size, batch_type);
}

template <bool _t_a, bool _t_b, typename sb_handle_t, typename container_0_t,
          typename container_1_t, typename container_2_t, typename element_t,
          typename index_t>
typename sb_handle_t::event_t _gemm_is_beta_zero(
    sb_handle_t& sb_handle, index_t _M, index_t _N, index_t _K,
    element_t _alpha, container_0_t a_, index_t _lda, container_1_t b_,
    index_t _ldb, element_t _beta, container_2_t _C, index_t _ldc,
    index_t batch_size, gemm_batch_type_t batch_type) {
  return ((_beta == static_cast<element_t>(0))
              ? _gemm_platform_specific<_t_a, _t_b, true>(
                    sb_handle, _M, _N, _K, _alpha, a_, _lda, b_, _ldb, _beta,
                    _C, _ldc, batch_size, batch_type)
              : _gemm_platform_specific<_t_a, _t_b, false>(
                    sb_handle, _M, _N, _K, _alpha, a_, _lda, b_, _ldb, _beta,
                    _C, _ldc, batch_size, batch_type));
}

template <typename sb_handle_t, typename container_0_t, typename container_1_t,
          typename container_2_t, typename element_t, typename index_t>
typename sb_handle_t::event_t _gemm_backend(
    sb_handle_t& sb_handle, char _TransA, char _TransB, index_t _M, index_t _N,
    index_t _K, element_t _alpha, container_0_t a_, index_t _lda,
    container_1_t b_, index_t _ldb, element_t _beta, container_2_t _C,
    index_t _ldc, index_t batch_size, gemm_batch_type_t batch_type) {
  if (_alpha == element_t{0}) {
    // When alpha = 0, GEMM is equivalent to C = beta * C.
    if (_ldc == _M) {
      // When LDC is M, we can scale the full matrix at once.
      const auto matrix_size = _N * _M * batch_size;
      return ::blas::_scal(sb_handle, matrix_size, _beta, _C, index_t{1});
    } else {
      // When LDC is not M, we must scale each column of C separately.
      typename sb_handle_t::event_t events;
      const auto num_columns = batch_size * _N;
      for (index_t i = 0; i < num_columns; ++i) {
        auto ev =
            ::blas::_scal(sb_handle, _M, _beta, _C + i * _ldc, index_t{1});
        append_vector(events, ev);
      }
      return events;
    }
  }

  _TransA = tolower(_TransA);
  _TransB = tolower(_TransB);

  if (_TransA != 'n' && _TransA != 't' && _TransA != 'c') {
    throw std::invalid_argument("invalid _TransA");
  } else if (_TransB != 'n' && _TransB != 't' && _TransB != 'c') {
    throw std::invalid_argument("invalid _TransB");
  }

  bool _TrA = _TransA != 'n';
  bool _TrB = _TransB != 'n';
  if (_TrA && _TrB) {
    return _gemm_is_beta_zero<true, true>(sb_handle, _M, _N, _K, _alpha, a_,
                                          _lda, b_, _ldb, _beta, _C, _ldc,
                                          batch_size, batch_type);
  } else if (!_TrA && _TrB) {
    return _gemm_is_beta_zero<false, true>(sb_handle, _M, _N, _K, _alpha, a_,
                                           _lda, b_, _ldb, _beta, _C, _ldc,
                                           batch_size, batch_type);
  } else if (_TrA && !_TrB) {
    return _gemm_is_beta_zero<true, false>(sb_handle, _M, _N, _K, _alpha, a_,
                                           _lda, b_, _ldb, _beta, _C, _ldc,
                                           batch_size, batch_type);
  } else {
    return _gemm_is_beta_zero<false, false>(sb_handle, _M, _N, _K, _alpha, a_,
                                            _lda, b_, _ldb, _beta, _C, _ldc,
                                            batch_size, batch_type);
  }
}

template <typename sb_handle_t, typename container_0_t, typename container_1_t,
          typename container_2_t, typename element_t, typename index_t>
typename sb_handle_t::event_t _gemm(sb_handle_t& sb_handle, char _TransA,
                                    char _TransB, index_t _M, index_t _N,
                                    index_t _K, element_t _alpha,
                                    container_0_t a_, index_t _lda,
                                    container_1_t b_, index_t _ldb,
                                    element_t _beta, container_2_t _C,
                                    index_t _ldc) {
  return _gemm_backend(sb_handle, _TransA, _TransB, _M, _N, _K, _alpha, a_,
                       _lda, b_, _ldb, _beta, _C, _ldc, index_t(1),
                       gemm_batch_type_t::strided);
}

template <typename sb_handle_t, typename container_0_t, typename container_1_t,
          typename container_2_t, typename element_t, typename index_t>
typename sb_handle_t::event_t _gemm_batched(
    sb_handle_t& sb_handle, char _TransA, char _TransB, index_t _M, index_t _N,
    index_t _K, element_t _alpha, container_0_t a_, index_t _lda,
    container_1_t b_, index_t _ldb, element_t _beta, container_2_t _C,
    index_t _ldc, index_t batch_size, gemm_batch_type_t batch_type) {
  return _gemm_backend(sb_handle, _TransA, _TransB, _M, _N, _K, _alpha, a_,
                       _lda, b_, _ldb, _beta, _C, _ldc, batch_size, batch_type);
}

}  // namespace internal
}  // namespace blas

#endif  // SYCL_BLAS_BLAS3_GEMM_INTERFACE_HPP
