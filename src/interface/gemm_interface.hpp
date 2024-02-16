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
 **************************************************************************/

#ifndef PORTBLAS_BLAS3_GEMM_INTERFACE_HPP
#define PORTBLAS_BLAS3_GEMM_INTERFACE_HPP

#include "blas_meta.h"
#include "interface/blas1_interface.h"
#include "interface/blas3/backend/backend.hpp"
#include "interface/blas3_interface.h"
#include "operations/blas3_trees.h"
#include "portblas_helper.h"
#include "sb_handle/portblas_handle.h"

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

// Check whether value is zero (complex & half/float/double)
template <typename T>
inline typename std::enable_if<is_sycl_scalar<T>::value, bool>::type isZero(
    const T& value) {
  return (value == static_cast<T>(0));
}

#ifdef BLAS_ENABLE_COMPLEX
template <typename T>
inline typename std::enable_if<is_complex_sycl<T>::value, bool>::type isZero(
    const T& value) {
  using value_t = typename T::value_type;
  return (value == T(value_t(0), value_t(0)));
}
#endif

template <bool _t_a, bool _t_b, bool s_a, bool s_b, bool is_beta_zero,
          typename sb_handle_t, typename container_0_t, typename container_1_t,
          typename container_2_t, typename element_t, typename index_t>
typename sb_handle_t::event_t _gemm_platform_specific(
    sb_handle_t& sb_handle, index_t _M, index_t _N, index_t _K,
    element_t _alpha, container_0_t a_, index_t _lda, index_t _stridea,
    container_1_t b_, index_t _ldb, index_t _strideb, element_t _beta,
    container_2_t _C, index_t _ldc, index_t _stridec, index_t batch_size,
    gemm_batch_type_t batch_type,
    const typename sb_handle_t::event_t& _dependencies) {
  return blas::gemm::backend::_gemm<_t_a, _t_b, s_a, s_b, is_beta_zero>(
      sb_handle, _M, _N, _K, _alpha, a_, _lda, _stridea, b_, _ldb, _strideb,
      _beta, _C, _ldc, _stridec, batch_size, batch_type, _dependencies);
}

template <bool _t_a, bool _t_b, bool s_a, bool s_b, typename sb_handle_t,
          typename container_0_t, typename container_1_t,
          typename container_2_t, typename element_t, typename index_t>
typename sb_handle_t::event_t _gemm_is_beta_zero(
    sb_handle_t& sb_handle, index_t _M, index_t _N, index_t _K,
    element_t _alpha, container_0_t a_, index_t _lda, index_t _stridea,
    container_1_t b_, index_t _ldb, index_t _strideb, element_t _beta,
    container_2_t _C, index_t _ldc, index_t _stridec, index_t batch_size,
    gemm_batch_type_t batch_type,
    const typename sb_handle_t::event_t& _dependencies) {
  return isZero(_beta) ? _gemm_platform_specific<_t_a, _t_b, s_a, s_b, true>(
                             sb_handle, _M, _N, _K, _alpha, a_, _lda, _stridea,
                             b_, _ldb, _strideb, _beta, _C, _ldc, _stridec,
                             batch_size, batch_type, _dependencies)
                       : _gemm_platform_specific<_t_a, _t_b, s_a, s_b, false>(
                             sb_handle, _M, _N, _K, _alpha, a_, _lda, _stridea,
                             b_, _ldb, _strideb, _beta, _C, _ldc, _stridec,
                             batch_size, batch_type, _dependencies);
}

template <bool symm_A, bool symm_B, typename sb_handle_t,
          typename container_0_t, typename container_1_t,
          typename container_2_t, typename element_t, typename index_t>
typename sb_handle_t::event_t _gemm_backend(
    sb_handle_t& sb_handle, char _TransA, char _TransB, index_t _M, index_t _N,
    index_t _K, element_t _alpha, container_0_t a_, index_t _lda,
    index_t _stridea, container_1_t b_, index_t _ldb, index_t _strideb,
    element_t _beta, container_2_t _C, index_t _ldc, index_t _stridec,
    index_t batch_size, gemm_batch_type_t batch_type,
    const typename sb_handle_t::event_t& _dependencies) {
  if (_alpha == element_t{0}) {
    // When alpha = 0, GEMM is equivalent to C = beta * C.
    if (_ldc == _M) {
      // When LDC is M, we can scale the full matrix at once.
      const auto matrix_size = _N * _M * batch_size;
      return ::blas::_scal(sb_handle, matrix_size, _beta, _C, index_t{1},
                           _dependencies);
    } else {
      // When LDC is not M, we must scale each column of C separately.
      typename sb_handle_t::event_t events;
      const auto num_columns = batch_size * _N;
      for (index_t i = 0; i < num_columns; ++i) {
        auto ev = ::blas::_scal(sb_handle, _M, _beta, _C + i * _ldc, index_t{1},
                                _dependencies);
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

  // Checking Strides conformity when batch_size>1
  index_t a_size = _TrA ? _M * _lda : _K * _lda;
  index_t b_size = _TrB ? _ldb * _K : _N * _ldb;
  index_t c_size = _ldc * _N;

  if (batch_size > index_t(1) && (batch_type == gemm_batch_type_t::strided)) {
    if (_stridec < c_size || _stridec < 0) {
      throw std::invalid_argument("invalid _stridec");
    } else if (_stridea < 0) {
      throw std::invalid_argument("invalid _stridea");
    } else if (_strideb < 0) {
      throw std::invalid_argument("invalid _strideb");
    }
  }

  if (_TrA && _TrB) {
    return _gemm_is_beta_zero<true, true, symm_A, symm_B>(
        sb_handle, _M, _N, _K, _alpha, a_, _lda, _stridea, b_, _ldb, _strideb,
        _beta, _C, _ldc, _stridec, batch_size, batch_type, _dependencies);
  } else if (!_TrA && _TrB) {
    return _gemm_is_beta_zero<false, true, symm_A, symm_B>(
        sb_handle, _M, _N, _K, _alpha, a_, _lda, _stridea, b_, _ldb, _strideb,
        _beta, _C, _ldc, _stridec, batch_size, batch_type, _dependencies);
  } else if (_TrA && !_TrB) {
    return _gemm_is_beta_zero<true, false, symm_A, symm_B>(
        sb_handle, _M, _N, _K, _alpha, a_, _lda, _stridea, b_, _ldb, _strideb,
        _beta, _C, _ldc, _stridec, batch_size, batch_type, _dependencies);
  } else {
    return _gemm_is_beta_zero<false, false, symm_A, symm_B>(
        sb_handle, _M, _N, _K, _alpha, a_, _lda, _stridea, b_, _ldb, _strideb,
        _beta, _C, _ldc, _stridec, batch_size, batch_type, _dependencies);
  }
}

template <typename sb_handle_t, typename container_0_t, typename container_1_t,
          typename container_2_t, typename element_t, typename index_t>
typename sb_handle_t::event_t _gemm(
    sb_handle_t& sb_handle, char _TransA, char _TransB, index_t _M, index_t _N,
    index_t _K, element_t _alpha, container_0_t a_, index_t _lda,
    container_1_t b_, index_t _ldb, element_t _beta, container_2_t _C,
    index_t _ldc, const typename sb_handle_t::event_t& _dependencies) {
  return _gemm_backend<false, false>(
      sb_handle, _TransA, _TransB, _M, _N, _K, _alpha, a_, _lda, index_t(0), b_,
      _ldb, index_t(0), _beta, _C, _ldc, index_t(0), index_t(1),
      gemm_batch_type_t::strided, _dependencies);
}

template <typename sb_handle_t, typename container_0_t, typename container_1_t,
          typename container_2_t, typename element_t, typename index_t>
typename sb_handle_t::event_t _gemm_batched(
    sb_handle_t& sb_handle, char _TransA, char _TransB, index_t _M, index_t _N,
    index_t _K, element_t _alpha, container_0_t a_, index_t _lda,
    container_1_t b_, index_t _ldb, element_t _beta, container_2_t _C,
    index_t _ldc, index_t batch_size, gemm_batch_type_t batch_type,
    const typename sb_handle_t::event_t& _dependencies) {
  bool is_strided = batch_type == gemm_batch_type_t::strided;
  index_t _stridea = 0;
  index_t _strideb = 0;
  index_t _stridec = 0;

  if (is_strided) {
    // By default strides are equal to matrices sizes when using
    // gemm_batch_type_t::strided
    _stridea = (tolower(_TransA) != 'n') ? _M * _lda : _K * _lda;
    _strideb = (tolower(_TransB) != 'n') ? _ldb * _K : _N * _ldb;
    _stridec = _ldc * _N;
  }
  // strides are not used otherwise (gemm_batch_type_t::interleaved)

  return _gemm_backend<false, false>(sb_handle, _TransA, _TransB, _M, _N, _K,
                                     _alpha, a_, _lda, _stridea, b_, _ldb,
                                     _strideb, _beta, _C, _ldc, _stridec,
                                     batch_size, batch_type, _dependencies);
}

template <typename sb_handle_t, typename container_0_t, typename container_1_t,
          typename container_2_t, typename element_t, typename index_t>
typename sb_handle_t::event_t _gemm_strided_batched(
    sb_handle_t& sb_handle, char _TransA, char _TransB, index_t _M, index_t _N,
    index_t _K, element_t _alpha, container_0_t a_, index_t _lda,
    index_t _stridea, container_1_t b_, index_t _ldb, index_t _strideb,
    element_t _beta, container_2_t _C, index_t _ldc, index_t _stridec,
    index_t batch_size, const typename sb_handle_t::event_t& _dependencies) {
  return _gemm_backend<false, false>(
      sb_handle, _TransA, _TransB, _M, _N, _K, _alpha, a_, _lda, _stridea, b_,
      _ldb, _strideb, _beta, _C, _ldc, _stridec, batch_size,
      gemm_batch_type_t::strided, _dependencies);
}

}  // namespace internal
}  // namespace blas

#endif  // PORTBLAS_BLAS3_GEMM_INTERFACE_HPP
