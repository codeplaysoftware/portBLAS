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
 *  @filename amd_gpu.hpp
 *
 **************************************************************************/
#ifndef PORTBLAS_ASUM_AMD_GPU_BACKEND_HPP
#define PORTBLAS_ASUM_AMD_GPU_BACKEND_HPP
#include "interface/blas1_interface.h"
#include "portblas_helper.h"

namespace blas {
namespace asum {
namespace backend {
template <typename sb_handle_t, typename container_0_t, typename container_1_t,
          typename index_t, typename increment_t>
typename sb_handle_t::event_t _asum(
    sb_handle_t& sb_handle, index_t _N, container_0_t _vx, increment_t _incx,
    container_1_t _rs, const typename sb_handle_t::event_t& _dependencies) {
  /**
   * This compile time check is absolutely necessary for AMD GPUs.
   * AMD's atomic operations require a specific combination of hardware that
   * cannot be checked nor enforced. Since the reduction operator kernel
   * implementation uses atomic operations, without that particular hardware
   * combination the reduction may silently fail.
   **/
#ifdef SB_ENABLE_USM
  const bool usm_managed_mem = blas::helper::is_malloc_shared(sb_handle, _rs);
#else
  constexpr bool usm_managed_mem{false};
#endif
  if (usm_managed_mem) {
    if (_N < (1 << 18)) {
      constexpr index_t localSize = 1024;
      const index_t number_WG = (_N + localSize - 1) / localSize;
      return blas::internal::_asum_impl<static_cast<int>(localSize), 32, true>(
          sb_handle, _N, _vx, _incx, _rs, number_WG, _dependencies);
    } else {
      constexpr int localSize = 512;
      constexpr index_t number_WG = 256;
      return blas::internal::_asum_impl<localSize, 32, true>(
          sb_handle, _N, _vx, _incx, _rs, number_WG, _dependencies);
    }
  } else {
    if (_N < (1 << 18)) {
      constexpr index_t localSize = 1024;
      const index_t number_WG = (_N + localSize - 1) / localSize;
      return blas::internal::_asum_impl<static_cast<int>(localSize), 32, false>(
          sb_handle, _N, _vx, _incx, _rs, number_WG, _dependencies);
    } else {
      constexpr int localSize = 512;
      constexpr index_t number_WG = 256;
      return blas::internal::_asum_impl<localSize, 32, false>(
          sb_handle, _N, _vx, _incx, _rs, number_WG, _dependencies);
    }
  }
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
  if (_N < 65536) {
    constexpr int localSize = 1024;
    return blas::internal::_iamax_iamin_impl<localSize, localSize, true, true>(
        sb_handle, _N, _vx, _incx, _rs, static_cast<index_t>(1), _dependencies);
  } else {
    constexpr int localSize = 256;
    const index_t nWG = std::min((_N + localSize - 1) / (localSize * 4),
                                 static_cast<index_t>(512));
    return blas::internal::_iamax_iamin_impl<localSize, localSize, true, false>(
        sb_handle, _N, _vx, _incx, _rs, nWG, _dependencies);
  }
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
  if (_N < 65536) {
    constexpr int localSize = 1024;
    return blas::internal::_iamax_iamin_impl<localSize, localSize, false, true>(
        sb_handle, _N, _vx, _incx, _rs, static_cast<index_t>(1), _dependencies);
  } else {
    constexpr int localSize = 256;
    const index_t nWG = std::min((_N + localSize - 1) / (localSize * 4),
                                 static_cast<index_t>(512));
    return blas::internal::_iamax_iamin_impl<localSize, localSize, false,
                                             false>(sb_handle, _N, _vx, _incx,
                                                    _rs, nWG, _dependencies);
  }
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
  /**
   * This compile time check is absolutely necessary for AMD GPUs.
   * AMD's atomic operations require a specific combination of hardware that
   * cannot be checked nor enforced. Since the reduction operator kernel
   * implementation uses atomic operations, without that particular hardware
   * combination the reduction may silently fail.
   **/
#ifdef SB_ENABLE_USM
  const bool usm_managed_mem = blas::helper::is_malloc_shared(sb_handle, _rs);
#else
  constexpr bool usm_managed_mem{false};
#endif
  if (usm_managed_mem) {
    if (_N < (1 << 18)) {
      constexpr index_t localSize = 1024;
      const index_t number_WG = (_N + localSize - 1) / localSize;
      return blas::internal::_nrm2_impl<static_cast<int>(localSize), 32, true>(
          sb_handle, _N, _vx, _incx, _rs, number_WG, _dependencies);
    } else {
      constexpr int localSize = 512;
      constexpr index_t number_WG = 512;
      return blas::internal::_nrm2_impl<localSize, 32, true>(
          sb_handle, _N, _vx, _incx, _rs, number_WG, _dependencies);
    }
  } else {
    if (_N < (1 << 18)) {
      constexpr index_t localSize = 1024;
      const index_t number_WG = (_N + localSize - 1) / localSize;
      return blas::internal::_nrm2_impl<static_cast<int>(localSize), 32, false>(
          sb_handle, _N, _vx, _incx, _rs, number_WG, _dependencies);
    } else {
      constexpr int localSize = 512;
      constexpr index_t number_WG = 512;
      return blas::internal::_nrm2_impl<localSize, 32, false>(
          sb_handle, _N, _vx, _incx, _rs, number_WG, _dependencies);
    }
  }
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
  /**
   * This compile time check is absolutely necessary for AMD GPUs.
   * AMD's atomic operations require a specific combination of hardware that
   * cannot be checked nor enforced. Since the reduction operator kernel
   * implementation uses atomic operations, without that particular hardware
   * combination the reduction may silently fail.
   **/
#ifdef SB_ENABLE_USM
  const bool usm_managed_mem = blas::helper::is_malloc_shared(sb_handle, _rs);
#else
  constexpr bool usm_managed_mem{false};
#endif
  if (usm_managed_mem) {
    if (_N < (1 << 18)) {
      constexpr index_t localSize = 1024;
      const index_t number_WG = (_N + localSize - 1) / localSize;
      return blas::internal::_dot_impl<static_cast<int>(localSize), 32, true>(
          sb_handle, _N, _vx, _incx, _vy, _incy, _rs, number_WG, _dependencies);
    } else {
      constexpr int localSize = 512;
      constexpr index_t number_WG = 512;
      return blas::internal::_dot_impl<localSize, 32, true>(
          sb_handle, _N, _vx, _incx, _vy, _incy, _rs, number_WG, _dependencies);
    }
  } else {
    if (_N < (1 << 18)) {
      constexpr index_t localSize = 1024;
      const index_t number_WG = (_N + localSize - 1) / localSize;
      return blas::internal::_dot_impl<static_cast<int>(localSize), 32, false>(
          sb_handle, _N, _vx, _incx, _vy, _incy, _rs, number_WG, _dependencies);
    } else {
      constexpr int localSize = 512;
      constexpr index_t number_WG = 512;
      return blas::internal::_dot_impl<localSize, 32, false>(
          sb_handle, _N, _vx, _incx, _vy, _incy, _rs, number_WG, _dependencies);
    }
  }
}
}  // namespace backend
}  // namespace dot
}  // namespace blas

#endif
