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
 *  @filename default.hpp
 *
 **************************************************************************/
#ifndef PORTBLAS_GEMV_DEFAULT_BACKEND_HPP
#define PORTBLAS_GEMV_DEFAULT_BACKEND_HPP
#include "interface/blas2_interface.h"

namespace blas {
namespace gemv {
namespace backend {
template <transpose_type trn, typename SB_Handle, typename index_t,
          typename element_t, typename container_t0, typename container_t1,
          typename increment_t, typename container_t2>
typename SB_Handle::event_t _gemv(
    SB_Handle& sb_handle, index_t _M, index_t _N, element_t _alpha,
    container_t0 _mA, index_t _lda, container_t1 _vx, increment_t _incx,
    element_t _beta, container_t2 _vy, increment_t _incy,
    const typename SB_Handle::event_t& _dependencies) {
  if (trn == transpose_type::Normal) {
    return blas::internal::_gemv_impl<256, 32, gemv_memory_t::local, trn>(
        sb_handle, _M, _N, _alpha, _mA, _lda, _vx, _incx, _beta, _vy, _incy,
        _dependencies);
  } else {
    return blas::internal::_gemv_impl<128, 32, gemv_memory_t::local, trn>(
        sb_handle, _M, _N, _alpha, _mA, _lda, _vx, _incx, _beta, _vy, _incy,
        _dependencies);
  }
}
}  // namespace backend
}  // namespace gemv

namespace gbmv {
namespace backend {
template <transpose_type trn, typename SB_Handle, typename index_t,
          typename element_t, typename container_t0, typename container_t1,
          typename increment_t, typename container_t2>
typename SB_Handle::event_t inline _gbmv(
    SB_Handle& sb_handle, index_t _M, index_t _N, index_t _KL, index_t _KU,
    element_t _alpha, container_t0 _mA, index_t _lda, container_t1 _vx,
    increment_t _incx, element_t _beta, container_t2 _vy, increment_t _incy,
    const typename SB_Handle::event_t& _dependencies) {
  return blas::internal::_gbmv_impl<256, trn>(sb_handle, _M, _N, _KL, _KU,
                                              _alpha, _mA, _lda, _vx, _incx,
                                              _beta, _vy, _incy, _dependencies);
}
}  // namespace backend
}  // namespace gbmv

namespace sbmv {
namespace backend {
template <uplo_type uplo, typename SB_Handle, typename index_t,
          typename element_t, typename container_t0, typename container_t1,
          typename increment_t, typename container_t2>
typename SB_Handle::event_t inline _sbmv(
    SB_Handle& sb_handle, index_t _N, index_t _K, element_t _alpha,
    container_t0 _mA, index_t _lda, container_t1 _vx, increment_t _incx,
    element_t _beta, container_t2 _vy, increment_t _incy,
    const typename SB_Handle::event_t& _dependencies) {
  return blas::internal::_sbmv_impl<256, uplo>(sb_handle, _N, _K, _alpha, _mA,
                                               _lda, _vx, _incx, _beta, _vy,
                                               _incy, _dependencies);
}
}  // namespace backend
}  // namespace sbmv

namespace spmv {
namespace backend {
template <uplo_type uplo, typename SB_Handle, typename index_t,
          typename element_t, typename container_t0, typename container_t1,
          typename increment_t, typename container_t2>
typename SB_Handle::event_t inline _spmv(
    SB_Handle& sb_handle, index_t _N, element_t _alpha, container_t0 _mA,
    container_t1 _vx, increment_t _incx, element_t _beta, container_t2 _vy,
    increment_t _incy, const typename SB_Handle::event_t& _dependencies) {
  return blas::internal::_spmv_impl<4, 4, uplo>(
      sb_handle, _N, _alpha, _mA, _vx, _incx, _beta, _vy, _incy, _dependencies);
}
}  // namespace backend
}  // namespace spmv

namespace tbmv {
namespace backend {
template <uplo_type uplo, transpose_type trn, diag_type diag,
          typename sb_handle_t, typename index_t, typename container_t0,
          typename container_t1, typename increment_t>
typename sb_handle_t::event_t _tbmv(
    sb_handle_t& sb_handle, index_t _N, index_t _K, container_t0 _mA,
    index_t _lda, container_t1 _vx, increment_t _incx,
    typename sb_handle_t::event_t _dependencies) {
  return blas::internal::_tbmv_impl<256, uplo, trn, diag>(
      sb_handle, _N, _K, _mA, _lda, _vx, _incx, _dependencies);
}
}  // namespace backend
}  // namespace tbmv

namespace tpmv {
namespace backend {
template <uplo_type uplo, transpose_type trn, diag_type diag,
          typename sb_handle_t, typename index_t, typename container_t0,
          typename container_t1, typename increment_t>
typename sb_handle_t::event_t _tpmv(
    sb_handle_t& sb_handle, index_t _N, container_t0 _mA, container_t1 _vx,
    increment_t _incx, typename sb_handle_t::event_t _dependencies) {
  return blas::internal::_tpmv_impl<4, 4, uplo, trn, diag>(
      sb_handle, _N, _mA, _vx, _incx, _dependencies);
}
}  // namespace backend
}  // namespace tpmv

namespace trsv {
namespace backend {
template <uplo_type uplo, transpose_type trn, diag_type diag,
          typename sb_handle_t, typename index_t, typename container_t0,
          typename container_t1, typename increment_t>
typename sb_handle_t::event_t _trsv(
    sb_handle_t& sb_handle, index_t _N, container_t0 _mA, index_t _lda,
    container_t1 _vx, increment_t _incx,
    typename sb_handle_t::event_t _dependencies) {
  const auto device = sb_handle.get_queue().get_device();
  if (device.is_gpu()) {
    const std::string vendor =
        device.template get_info<sycl::info::device::vendor>();
    if (vendor.find("Intel") == vendor.npos) {
      return blas::internal::_trsv_impl<32, 4, uplo, trn, diag>(
          sb_handle, _N, _mA, _lda, _vx, _incx, _dependencies);
    } else {
      // This configuration works only for Intel iGPU
      return blas::internal::_trsv_impl<8, 4, uplo, trn, diag>(
          sb_handle, _N, _mA, _lda, _vx, _incx, _dependencies);
    }
  } else {
    return blas::internal::_trsv_impl<4, 2, uplo, trn, diag>(
        sb_handle, _N, _mA, _lda, _vx, _incx, _dependencies);
  }
}
}  // namespace backend
}  // namespace trsv

namespace tbsv {
namespace backend {
template <uplo_type uplo, transpose_type trn, diag_type diag,
          typename sb_handle_t, typename index_t, typename container_t0,
          typename container_t1, typename increment_t>
typename sb_handle_t::event_t _tbsv(
    sb_handle_t& sb_handle, index_t _N, index_t _K, container_t0 _mA,
    index_t _lda, container_t1 _vx, increment_t _incx,
    const typename sb_handle_t::event_t& _dependencies) {
  const auto device = sb_handle.get_queue().get_device();
  if (device.is_gpu()) {
    const std::string vendor =
        device.template get_info<sycl::info::device::vendor>();
    if (vendor.find("Intel") == vendor.npos) {
      return blas::internal::_tbsv_impl<32, 4, uplo, trn, diag>(
          sb_handle, _N, _K, _mA, _lda, _vx, _incx, _dependencies);
    } else {
      // This configuration works only for Intel iGPU
      return blas::internal::_tbsv_impl<8, 4, uplo, trn, diag>(
          sb_handle, _N, _K, _mA, _lda, _vx, _incx, _dependencies);
    }
  } else {
    return blas::internal::_tbsv_impl<4, 2, uplo, trn, diag>(
        sb_handle, _N, _K, _mA, _lda, _vx, _incx, _dependencies);
  }
}
}  // namespace backend
}  // namespace tbsv

namespace tpsv {
namespace backend {
template <uplo_type uplo, transpose_type trn, diag_type diag,
          typename sb_handle_t, typename index_t, typename container_t0,
          typename container_t1, typename increment_t>
typename sb_handle_t::event_t _tpsv(
    sb_handle_t& sb_handle, index_t _N, container_t0 _mA, container_t1 _vx,
    increment_t _incx, const typename sb_handle_t::event_t& _dependencies) {
  const auto device = sb_handle.get_queue().get_device();
  if (device.is_gpu()) {
    const std::string vendor =
        device.template get_info<sycl::info::device::vendor>();
    if (vendor.find("Intel") == vendor.npos) {
      return blas::internal::_tpsv_impl<32, 4, uplo, trn, diag>(
          sb_handle, _N, _mA, _vx, _incx, _dependencies);
    } else {
      // This configuration works only for Intel iGPU
      return blas::internal::_tpsv_impl<8, 4, uplo, trn, diag>(
          sb_handle, _N, _mA, _vx, _incx, _dependencies);
    }
  } else {
    return blas::internal::_tpsv_impl<4, 2, uplo, trn, diag>(
        sb_handle, _N, _mA, _vx, _incx, _dependencies);
  }
}
}  // namespace backend
}  // namespace tpsv
}  // namespace blas
#endif
