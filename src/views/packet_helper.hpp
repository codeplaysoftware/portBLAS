/*
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
 *  @filename packetize.hpp
 *
 */

#ifndef SYCL_BLAS_PACKET_HELPER_HPP
#define SYCL_BLAS_PACKET_HELPER_HPP

#include "blas_meta.h"
#include <CL/sycl.hpp>

namespace blas {

/**
 * @brief Helper class that contain methods to load and store data
 * using vectorization.
 *
 */
template <int vector_size, typename value_t, typename index_t>
struct PacketHelper {
  using packet_t = cl::sycl::vec<value_t, vector_size>;
  static constexpr int packet_size = vector_size;

  /**
   * @brief Performs a vectorised load using sycl::vec::load when stride is 1.
   * In the case stride != 1 the loads will be element wise.
   *
   * @param unit_stride A boolean value indicating whether stride is equal to 1
   * @param SrcPointerType A pointer type for the source data
   * @param stride The stride between consecutive elements in the source data
   *
   * @return The loaded packet
   */
  template <bool unit_stride, typename SrcPointerType>
  static SYCL_BLAS_INLINE packet_t load(SrcPointerType src, index_t stride) {
    packet_t packet{};
    if constexpr (unit_stride) {
      using address_t = cl::sycl::access::address_space;
      packet.template load<address_t::global_space>(
          0, cl::sycl::multi_ptr<const value_t, address_t::global_space>(src));
    } else {
      for (index_t i = 0; i < packet_size; i++) {
        reinterpret_cast<value_t *>(&packet)[i] = *(src + i * stride);
      }
    }
    return packet;
  }

  /**
   * @brief Performs a non-vectorized load with boundary checking.
   *
   * @param unit_stride A boolean value indicating whether stride is equal to 1
   * @param SrcPointerType A pointer type for the source data
   * @param stride The stride between consecutive elements in the source data
   * @param EdgePredicate A predicate that returns true if an element is within
   * range
   *
   * @return The loaded packet
   */
  template <bool unit_stride, typename SrcPointerType, typename EdgePredicate>
  static SYCL_BLAS_INLINE packet_t load(SrcPointerType src, index_t stride,
                                        EdgePredicate edge_in_range) {
    packet_t packet{};
    if constexpr (unit_stride) {
      for (index_t i = 0; i < packet_size; i++) {
        reinterpret_cast<value_t *>(&packet)[i] =
            edge_in_range(i) ? *(src + i) : value_t{0};
      }
    } else {
      for (index_t i = 0; i < packet_size; i++) {
        reinterpret_cast<value_t *>(&packet)[i] =
            edge_in_range(i * stride) ? *(src + i * stride) : value_t{0};
      }
    }
    return packet;
  }

  /**
   * @brief Store a vector packet into global memory.
   * If stride == 1, store is vectorised using sycl::vec::store.
   *
   * @param unit_stride A boolean value indicating whether stride is equal to 1
   * @param DestPointerType A pointer type for the destination data
   * @param packet The packet to store
   * @param stride The stride between consecutive elements in the destination
   * data
   */
  template <bool unit_stride, typename DestPointerType>
  static SYCL_BLAS_INLINE void store(packet_t &packet, DestPointerType dest,
                                     index_t stride) {
    if constexpr (unit_stride) {
      using address_t = cl::sycl::access::address_space;
      packet.template store<address_t::global_space>(
          0, cl::sycl::multi_ptr<value_t, address_t::global_space>(dest));
    } else {
      for (index_t i = 0; i < packet_size; i++) {
        *(dest + i * stride) = reinterpret_cast<value_t *>(&packet)[i];
      }
    }
  }

  /**
   * @brief Store elements of a vector packet into global memory.
   * This will check if the element is within a range before storing.
   *
   * @param unit_stride A boolean value indicating whether stride is equal to 1
   * @param DestPointerType A pointer type for the destination data
   * @param packet The packet to store
   * @param stride The stride between consecutive elements in the destination
   * data
   * @param EdgePredicate A predicate that returns true if an element is within
   * range
   */
  template <bool unit_stride, typename DestPointerType, typename EdgePredicate>
  static SYCL_BLAS_INLINE void store(packet_t &packet, DestPointerType dest,
                                     index_t stride,
                                     EdgePredicate edge_in_range) {
    if constexpr (unit_stride) {
      for (index_t i = 0; i < packet_size; i++) {
        if (edge_in_range(i)) {
          *(dest + i) = reinterpret_cast<value_t *>(&packet)[i];
        }
      }
    } else {
      for (index_t i = 0; i < packet_size; i++) {
        if (edge_in_range(i * stride)) {
          *(dest + i * stride) = reinterpret_cast<value_t *>(&packet)[i];
        }
      }
    }
  }
};

}  // namespace blas

#endif  // SYCL_BLAS_PACKET_HELPER_HPP
