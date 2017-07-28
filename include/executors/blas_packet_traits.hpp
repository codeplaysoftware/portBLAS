/***************************************************************************
 *
 *  @license
 *  Copyright (C) 2016 Codeplay Software Limited
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
 *  @filename blas_packet_traits.hpp
 *
 **************************************************************************/

#include <executors/blas_device.hpp>

#ifndef BLAS_PACKET_TRAITS_HPP
#define BLAS_PACKET_TRAITS_HPP

namespace blas {

/*!
 * default_packet_traits.
 * @brief Base class for packet traits, providing default support indicators.
 * @tparam T Type for which the support is described.
 * @tparam Device Device for which the support is described.
 */
template <typename T, typename Device>
struct default_packet_traits {
  using packet_type = T;
  enum {
    Size = 1,
    Supported = 1,
    has_abs = 1,
    has_sqrt = 1,
    has_sin = 1,
    has_cos = 1,
    has_add = 1,
    has_sub = 1,
    has_mul = 1,
    has_div = 1,
    has_mad = 0,
    has_dot = 0,
    has_length = 0,
    has_min = 1,
    has_max = 1
  };
};

/*!
 * Packet_traits.
 * @brief Contains information avoid type/device support for a certain
 * operation.
 */
template <typename T, typename Device>
struct Packet_traits : default_packet_traits<T, Device> {};

template <typename T, typename Device>
using packet_type = typename Packet_traits<T, Device>::packet_type;

}  // namespace BLAS

#endif
