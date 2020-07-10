/***************************************************************************
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
 *  @filename gemm_load_store.hpp
 *
 **************************************************************************/

#ifndef SYCL_BLAS_BLAS3_GEMM_LOAD_STORE_HPP
#define SYCL_BLAS_BLAS3_GEMM_LOAD_STORE_HPP

namespace blas {

/*! @brief Contains static methods for loading and storing vector packets
from/to non-vectorized memory as well as some constants for the vector type and
packet size. SFINAE is used to select the appropriate method when called.
* @tparam vector_size The desired vector size to be used. If
GEMM_VECTORIZATION_SUPPORT is not enabled in CMake a vector_size of 1 will be
used no matter what value is passed here.
* @tparam value_t The type of the matrix data (typically float or double, if
supported).
*/
template <size_t vector_size, typename value_t, typename index_t>
struct Packetize {
#ifdef GEMM_VECTORIZATION_SUPPORT
  using PacketType = cl::sycl::vec<value_t, vector_size>;
  static constexpr size_t packet_size = vector_size;
  template <index_t dimension>
  SYCL_BLAS_INLINE static constexpr bool check_size() {
    return dimension == packet_size;
  }
#else
  // In the case where vectorization is not enabled, always set to 1
  using PacketType = cl::sycl::vec<value_t, 1>;
  static constexpr size_t packet_size = 1;
  template <index_t dimension>
  SYCL_BLAS_INLINE static constexpr bool check_size() {
    return true;
  }
#endif

  /*! @brief Performs a coalesced non-vectorized load when the current block is
   * not internal.
   * @tparam trans Whether the source matrix is transposed or not.
   * @tparam internal True if the current block is internal and no bounds
   * checking is required.
   * @tparam ld The leading dimension of the destination memory.
   */

  template <bool trans, bool internal, int ld, typename SrcPointerType,
            typename DestPointerType, typename EdgePredicate>
  static SYCL_BLAS_INLINE typename std::enable_if<!internal>::type load(
      const bool in_range, SrcPointerType src, DestPointerType dest,
      EdgePredicate) {
    *(dest) = in_range ? *(src) : value_t{0};
  }
  /*! @brief Performs a vectorised load using sycl::vec::load when the current
   * block is internal. In the case where k < the
   * number of elements being loaded then edge loads will be element wise with
   * additional bounds checking.
   * @tparam trans Whether the source matrix is transposed or not.
   * @tparam internal True if the current block is internal and no bounds
   * checking is required.
   * @tparam ld The leading dimension of the destination memory. */
  template <bool trans, bool internal, index_t ld, typename SrcPointerType,
            typename DestPointerType, typename EdgePredicate>
  static SYCL_BLAS_INLINE typename std::enable_if<internal>::type load(
      const bool in_range, SrcPointerType src, DestPointerType dest,
      EdgePredicate edge_in_range) {
    PacketType packet{0};

    if (in_range) {
      using address_t = cl::sycl::access::address_space;
      packet.template load<address_t::global_space>(0, src);
    } else {
#pragma unroll
      for (index_t i = 0; i < packet_size; i++) {
        reinterpret_cast<value_t *>(&packet)[i] =
            edge_in_range(i) ? *(src + i) : value_t{0};
      }
    }
    store<trans, ld>(packet, dest);
  }
  /*! @brief Store a vector packet into local memory when the source is
   * transposed. This will untranspose the elements individually when storing so
   * the data in local memory is always consistent.
   * @tparam trans Whether the source matrix is transposed or not.
   * @tparam ld The leading dimension of the destination memory.*/
  template <bool trans, index_t ld, typename DestPointerType>
  static SYCL_BLAS_INLINE typename std::enable_if<trans>::type store(
      PacketType &packet, DestPointerType dest) {
#pragma unroll
    for (index_t i = 0; i < packet_size; i++) {
      *(dest + ld * i) = reinterpret_cast<value_t *>(&packet)[i];
    }
  }

  /*! @brief Store a vector packet into local memory when the source is not
   * transposed. This will use sycl::vec::store function.
   * @tparam trans Whether the source matrix is transposed or not.
   * @tparam ld The leading dimension of the destination memory.*/
  template <bool trans, int ld, typename DestPointerType>
  static SYCL_BLAS_INLINE typename std::enable_if<!trans>::type store(
      PacketType &packet, DestPointerType dest) {
    using address_t = cl::sycl::access::address_space;
    packet.template store<address_t::local_space>(0, dest);
  }
};

}  // namespace blas
#endif  // SYCL_BLAS_BLAS3_GEMM_LOAD_STORE_HPP