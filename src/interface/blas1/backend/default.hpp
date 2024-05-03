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
 *  @filename defaul_cpu.hpp
 *
 **************************************************************************/
#ifndef PORTBLAS_ASUM_DEFAULT_BACKEND_HPP
#define PORTBLAS_ASUM_DEFAULT_BACKEND_HPP
#include "interface/blas1_interface.h"

namespace blas {
namespace asum {
namespace backend {
template <typename sb_handle_t, typename container_0_t, typename container_1_t,
          typename index_t, typename increment_t>
typename sb_handle_t::event_t _asum(
    sb_handle_t& sb_handle, index_t _N, container_0_t _vx, increment_t _incx,
    container_1_t _rs, const typename sb_handle_t::event_t& _dependencies) {
  constexpr int localSize = 8;
  constexpr index_t number_WG = 16;
  return blas::internal::_asum_impl<localSize, 0>(
      sb_handle, _N, _vx, _incx, _rs, number_WG, _dependencies);
}
}  // namespace backend
}  // namespace asum

namespace iamax {
namespace backend {
template <typename sb_handle_t, typename container_0_t, typename container_1_t,
          typename index_t, typename increment_t>
typename sb_handle_t::event_t _iamax(
    sb_handle_t& sb_handle, index_t _N, container_0_t _vx, increment_t _incx,
    container_1_t _rs, const typename sb_handle_t::event_t& _dependencies) {
#ifndef __ADAPTIVECPP__
  constexpr int localSize = 128;
  if (_N < 8192) {
    return blas::internal::_iamax_iamin_impl<localSize, 0, true, true>(
        sb_handle, _N, _vx, _incx, _rs, static_cast<index_t>(1), _dependencies);
  } else {
    const index_t nWG = std::min((_N + localSize - 1) / (localSize * 4),
                                 static_cast<index_t>(512));
    return blas::internal::_iamax_iamin_impl<localSize, 0, true, false>(
        sb_handle, _N, _vx, _incx, _rs, nWG, _dependencies);
  }
#else
  // Temporary work-around to avoid non-local memory implementation of
  // iamin/iamax with AdaptiveCpp.
  constexpr int localSize = 128;
  if (_N < 8192) {
    return blas::internal::_iamax_iamin_impl<localSize, localSize, true, true>(
        sb_handle, _N, _vx, _incx, _rs, static_cast<index_t>(1), _dependencies);
  } else {
    const index_t nWG = std::min((_N + localSize - 1) / (localSize * 4),
                                 static_cast<index_t>(512));
    return blas::internal::_iamax_iamin_impl<localSize, localSize, true, false>(
        sb_handle, _N, _vx, _incx, _rs, nWG, _dependencies);
  }
#endif
}
}  // namespace backend
}  // namespace iamax

namespace iamin {
namespace backend {
template <typename sb_handle_t, typename container_0_t, typename container_1_t,
          typename index_t, typename increment_t>
typename sb_handle_t::event_t _iamin(
    sb_handle_t& sb_handle, index_t _N, container_0_t _vx, increment_t _incx,
    container_1_t _rs, const typename sb_handle_t::event_t& _dependencies) {
#ifndef __ADAPTIVECPP__
  constexpr int localSize = 128;
  if (_N < 8192) {
    return blas::internal::_iamax_iamin_impl<localSize, 0, false, true>(
        sb_handle, _N, _vx, _incx, _rs, static_cast<index_t>(1), _dependencies);
  } else {
    const index_t nWG = std::min((_N + localSize - 1) / (localSize * 4),
                                 static_cast<index_t>(512));
    return blas::internal::_iamax_iamin_impl<localSize, 0, false, false>(
        sb_handle, _N, _vx, _incx, _rs, nWG, _dependencies);
  }
#else
  // Temporary work-around to avoid non-local memory implementation of
  // iamin/iamax with AdaptiveCpp.
  constexpr int localSize = 128;
  if (_N < 8192) {
    return blas::internal::_iamax_iamin_impl<localSize, localSize, false, true>(
        sb_handle, _N, _vx, _incx, _rs, static_cast<index_t>(1), _dependencies);
  } else {
    const index_t nWG = std::min((_N + localSize - 1) / (localSize * 4),
                                 static_cast<index_t>(512));
    return blas::internal::_iamax_iamin_impl<localSize, localSize, false,
                                             false>(sb_handle, _N, _vx, _incx,
                                                    _rs, nWG, _dependencies);
  }
#endif
}
}  // namespace backend
}  // namespace iamin

namespace nrm2 {
namespace backend {
template <typename sb_handle_t, typename container_0_t, typename container_1_t,
          typename index_t, typename increment_t>
typename sb_handle_t::event_t _nrm2(
    sb_handle_t& sb_handle, index_t _N, container_0_t _vx, increment_t _incx,
    container_1_t _rs, const typename sb_handle_t::event_t& _dependencies) {
  constexpr int localSize = 8;
  constexpr index_t number_WG = 16;
  return blas::internal::_nrm2_impl<localSize, 0>(
      sb_handle, _N, _vx, _incx, _rs, number_WG, _dependencies);
}
}  // namespace backend
}  // namespace nrm2

namespace dot {
namespace backend {
template <typename sb_handle_t, typename container_0_t, typename container_1_t,
          typename container_2_t, typename index_t, typename increment_t>
typename sb_handle_t::event_t _dot(
    sb_handle_t& sb_handle, index_t _N, container_0_t _vx, increment_t _incx,
    container_1_t _vy, increment_t _incy, container_2_t _rs,
    const typename sb_handle_t::event_t& _dependencies) {
  constexpr int localSize = 8;
  constexpr index_t number_WG = 16;
  return blas::internal::_dot_impl<localSize, 0>(
      sb_handle, _N, _vx, _incx, _vy, _incy, _rs, number_WG, _dependencies);
}
}  // namespace backend
}  // namespace dot
}  // namespace blas

#endif
