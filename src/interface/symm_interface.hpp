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
 *  @filename symm_interface.hpp
 *
 **************************************************************************/

#ifndef PORTBLAS_SYMM_INTERFACE_HPP
#define PORTBLAS_SYMM_INTERFACE_HPP

#include "interface/gemm_interface.hpp"

namespace blas {
namespace internal {

template <typename sb_handle_t, typename container_0_t, typename container_1_t,
          typename container_2_t, typename element_t, typename index_t>
typename sb_handle_t::event_t _symm(
    sb_handle_t& sb_handle, char _side, char _uplo, index_t _M, index_t _N,
    element_t _alpha, container_0_t a_, index_t _lda, container_1_t b_,
    index_t _ldb, element_t _beta, container_2_t _C, index_t _ldc,
    const typename sb_handle_t::event_t& _dependencies) {
  const char TRANS_NO = 'n';
  const char TRANS_YES = 't';
  const char SIDE_RIGHT = 'r';
  const char SIDE_LEFT = 'l';
  const char UPLO_LOWER = 'l';
  const char UPLO_UPPER = 'u';

  _side = tolower(_side);
  _uplo = tolower(_uplo);

  if (_uplo != UPLO_UPPER && _uplo != UPLO_LOWER) {
    throw std::invalid_argument("invalid _uplo");
  }
  if (_side == SIDE_LEFT) {  // C <- alpha * A * B + beta * C
    char trans_symm = _uplo == UPLO_UPPER ? TRANS_YES : TRANS_NO;
    return _gemm_backend<true, false>(
        sb_handle, trans_symm, TRANS_NO, _M, _N, _M, _alpha, a_, _lda,
        index_t(0), b_, _ldb, index_t(0), _beta, _C, _ldc, index_t(0),
        index_t(1), gemm_batch_type_t::strided, _dependencies);
  } else if (_side == SIDE_RIGHT) {  // C <- alpha * B * A + beta * C
    // if the valid values are in the upper side, transpose the matrix
    // to make gemm to start reading rows on a valid value.
    // This is to reduce the number of modifications on the gemm
    // implementation needed to support symm.
    char trans_symm = _uplo == UPLO_LOWER ? TRANS_YES : TRANS_NO;
    return _gemm_backend<false, true>(
        sb_handle, TRANS_NO, trans_symm, _M, _N, _N, _alpha, b_, _ldb,
        index_t(0), a_, _lda, index_t(0), _beta, _C, _ldc, index_t(0),
        index_t(1), gemm_batch_type_t::strided, _dependencies);
  } else {
    throw std::invalid_argument("invalid _side");
  }
}

}  // namespace internal
}  // namespace blas

#endif  // PORTBLAS_SYMM_INTERFACE_HPP
