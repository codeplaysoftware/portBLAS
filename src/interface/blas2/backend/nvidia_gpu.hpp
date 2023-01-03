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
 *  @filename nvidia_gpu.hpp
 *
 **************************************************************************/
#ifndef SYCL_BLAS_GEMV_NVIDIA_GPU_BACKEND_HPP
#define SYCL_BLAS_GEMV_NVIDIA_GPU_BACKEND_HPP
#include "interface/blas2_interface.h"

namespace blas {
namespace gemv {
namespace backend {
template <transpose_type trn, typename SB_Handle, typename index_t,
          typename element_t, typename container_t0, typename container_t1,
          typename increment_t, typename container_t2>
typename SB_Handle::event_t _gemv(SB_Handle& sb_handle, index_t _M, index_t _N,
                                  element_t _alpha, container_t0 _mA,
                                  index_t _lda, container_t1 _vx,
                                  increment_t _incx, element_t _beta,
                                  container_t2 _vy, increment_t _incy) {
  if (trn == transpose_type::Normal) {
    return blas::internal::_gemv_impl<256, 32, gemv_memory_t::local, trn>(
        sb_handle, _M, _N, _alpha, _mA, _lda, _vx, _incx, _beta, _vy, _incy);
  } else {
    return blas::internal::_gemv_impl<128, 32, gemv_memory_t::local, trn>(
        sb_handle, _M, _N, _alpha, _mA, _lda, _vx, _incx, _beta, _vy, _incy);
  }
}
}  // namespace backend
}  // namespace gemv

namespace gbmv {
namespace backend {
template <transpose_type trn, typename SB_Handle, typename index_t,
          typename element_t, typename container_t0, typename container_t1,
          typename increment_t, typename container_t2>
typename SB_Handle::event_t inline _gbmv(SB_Handle& sb_handle, index_t _M,
                                         index_t _N, index_t _KL, index_t _KU,
                                         element_t _alpha, container_t0 _mA,
                                         index_t _lda, container_t1 _vx,
                                         increment_t _incx, element_t _beta,
                                         container_t2 _vy, increment_t _incy) {
  return blas::internal::_gbmv_impl<64, trn>(sb_handle, _M, _N, _KL, _KU,
                                             _alpha, _mA, _lda, _vx, _incx,
                                             _beta, _vy, _incy);
}
}  // namespace backend
}  // namespace gbmv

namespace sbmv {
namespace backend {
template <uplo_type uplo, typename SB_Handle, typename index_t,
          typename element_t, typename container_t0, typename container_t1,
          typename increment_t, typename container_t2>
typename SB_Handle::event_t inline _sbmv(SB_Handle& sb_handle, index_t _N,
                                         index_t _K, element_t _alpha,
                                         container_t0 _mA, index_t _lda,
                                         container_t1 _vx, increment_t _incx,
                                         element_t _beta, container_t2 _vy,
                                         increment_t _incy) {
  return blas::internal::_sbmv_impl<64, uplo>(
      sb_handle, _N, _K, _alpha, _mA, _lda, _vx, _incx, _beta, _vy, _incy);
}
}  // namespace backend
}  // namespace sbmv
}  // namespace blas
#endif
