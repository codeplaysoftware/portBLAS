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
 *  @filename blas3_interface.hpp
 *
 **************************************************************************/

#ifndef SYCL_BLAS_BLAS3_INTERFACE_HPP
#define SYCL_BLAS_BLAS3_INTERFACE_HPP

#include "blas_meta.h"
#include "executors/executor.h"
#include "interface/blas3/backend/backend.hpp"
#include "interface/blas3_interface.h"
#include "operations/blas3_trees.h"
#include "policy/sycl_policy_handler.h"
#include <algorithm>
#include <cctype>
#include <cmath>
#include <iostream>
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

template <bool _t_a, bool _t_b, bool is_beta_zero, typename executor_t,
          typename container_0_t, typename container_1_t,
          typename container_2_t, typename element_t, typename index_t>
typename executor_t::policy_t::event_t _gemm_platform_specific(
    executor_t& ex, index_t _M, index_t _N, index_t _K, element_t _alpha,
    container_0_t a_, index_t _lda, container_1_t b_, index_t _ldb,
    element_t _beta, container_2_t _C, index_t _ldc, index_t batch_size) {
  return blas::gemm::backend::_gemm<_t_a, _t_b, is_beta_zero>(
      ex, _M, _N, _K, _alpha, a_, _lda, b_, _ldb, _beta, _C, _ldc, batch_size);
}

template <bool _t_a, bool _t_b, typename executor_t, typename container_0_t,
          typename container_1_t, typename container_2_t, typename element_t,
          typename index_t>
typename executor_t::policy_t::event_t _gemm_is_beta_zero(
    executor_t& ex, index_t _M, index_t _N, index_t _K, element_t _alpha,
    container_0_t a_, index_t _lda, container_1_t b_, index_t _ldb,
    element_t _beta, container_2_t _C, index_t _ldc, index_t batch_size) {
  return ((_beta == static_cast<element_t>(0))
              ? _gemm_platform_specific<_t_a, _t_b, true>(
                    ex, _M, _N, _K, _alpha, a_, _lda, b_, _ldb, _beta, _C, _ldc,
                    batch_size)
              : _gemm_platform_specific<_t_a, _t_b, false>(
                    ex, _M, _N, _K, _alpha, a_, _lda, b_, _ldb, _beta, _C, _ldc,
                    batch_size));
}

template <typename executor_t, typename container_0_t, typename container_1_t,
          typename container_2_t, typename element_t, typename index_t>
typename executor_t::policy_t::event_t _gemm_backend(
    executor_t& ex, char _TransA, char _TransB, index_t _M, index_t _N,
    index_t _K, element_t _alpha, container_0_t a_, index_t _lda,
    container_1_t b_, index_t _ldb, element_t _beta, container_2_t _C,
    index_t _ldc, index_t batch_size) {
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
    return _gemm_is_beta_zero<true, true>(ex, _M, _N, _K, _alpha, a_, _lda, b_,
                                          _ldb, _beta, _C, _ldc, batch_size);
  } else if (!_TrA && _TrB) {
    return _gemm_is_beta_zero<false, true>(ex, _M, _N, _K, _alpha, a_, _lda, b_,
                                           _ldb, _beta, _C, _ldc, batch_size);
  } else if (_TrA && !_TrB) {
    return _gemm_is_beta_zero<true, false>(ex, _M, _N, _K, _alpha, a_, _lda, b_,
                                           _ldb, _beta, _C, _ldc, batch_size);
  } else {
    return _gemm_is_beta_zero<false, false>(ex, _M, _N, _K, _alpha, a_, _lda,
                                            b_, _ldb, _beta, _C, _ldc,
                                            batch_size);
  }
}

template <typename executor_t, typename container_0_t, typename container_1_t,
          typename container_2_t, typename element_t, typename index_t>
typename executor_t::policy_t::event_t _gemm(executor_t& ex, char _TransA,
                                             char _TransB, index_t _M,
                                             index_t _N, index_t _K,
                                             element_t _alpha, container_0_t a_,
                                             index_t _lda, container_1_t b_,
                                             index_t _ldb, element_t _beta,
                                             container_2_t _C, index_t _ldc) {
  return _gemm_backend(ex, _TransA, _TransB, _M, _N, _K, _alpha, a_, _lda, b_,
                       _ldb, _beta, _C, _ldc, index_t(1));
}

template <typename executor_t, typename container_0_t, typename container_1_t,
          typename container_2_t, typename element_t, typename index_t>
typename executor_t::policy_t::event_t _gemm_batched(
    executor_t& ex, char _TransA, char _TransB, index_t _M, index_t _N,
    index_t _K, element_t _alpha, container_0_t a_, index_t _lda,
    container_1_t b_, index_t _ldb, element_t _beta, container_2_t _C,
    index_t _ldc, index_t batch_size) {
  return _gemm_backend(ex, _TransA, _TransB, _M, _N, _K, _alpha, a_, _lda, b_,
                       _ldb, _beta, _C, _ldc, batch_size);
}

}  // namespace internal

}  // namespace blas

#endif  // BLAS3_INTERFACE_HPP
