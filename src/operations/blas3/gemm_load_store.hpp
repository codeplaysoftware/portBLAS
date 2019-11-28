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
from/to non-vectorised memory as well as some constants for the vector type and
packet size. SFINAE is used to select the appropriate method when called.
* @tparam aligned If both matrix's memory is aligned then this will control the
* use of reinterpret_cast instead of vload/vstore.
* @tparam vector_size The desired vector size to be used. If
GEMM_VECTORISATION_SUPPORT is not enabled in CMake a vector_size of 1 will be
used no matter what value is passed here.
* @tparam The type of the matrix data (typically float or double, if supported).
*/
template <bool aligned, size_t vector_size, typename value_t>
struct Packetize {
#ifdef GEMM_VECTORISATION_SUPPORT
  using PacketType = cl::sycl::vec<value_t, vector_size>;
  static constexpr size_t packet_size = vector_size;
#else
  // In the case where vectorization is not enabled, always set to 1
  using PacketType = cl::sycl::vec<value_t, 1>;
  static constexpr size_t packet_size = 1;
#endif

  /*! @brief Performs a coalesced non-vectorised load when the current block is
   * not internal.
   * @tparam trans Whether the source matrix is transposed or not.
   * @tparam internal True if the current block is internal and no bounds
   * checking is required.
   * @tparam ld The leading dimension of the destination memory.
   */
  template <bool trans, bool internal, int ld, typename packet_t = PacketType,
            typename SrcPointerType, typename DestPointerType, typename index_t,
            typename EdgePredicate, bool align = aligned>
  static SYCL_BLAS_INLINE typename std::enable_if<(!internal)>::type load(
      const bool in_range, SrcPointerType src, const index_t &src_offset,
      DestPointerType dest, const index_t &dest_offset, EdgePredicate) {
    *(dest + dest_offset) = in_range ? *(src + src_offset) : value_t{0};
  }
  /*! @brief Performs a vectorised load using sycl::vec::load when the current
   * block is internal and the memory is not aligned. In the case where k < the
   * number of elements being loaded then edge loads will be element wise with
   * additional bounds checking.
   * @tparam trans Whether the source matrix is transposed or not.
   * @tparam internal True if the current block is internal and no bounds
   * checking is required.
   * @tparam ld The leading dimension of the destination memory. */
  template <bool trans, bool internal, int ld, typename packet_t = PacketType,
            typename SrcPointerType, typename DestPointerType, typename index_t,
            typename EdgePredicate, bool align = aligned>
  static SYCL_BLAS_INLINE typename std::enable_if<(internal && !align)>::type
  load(const bool in_range, SrcPointerType src, const index_t &src_offset,
       DestPointerType dest, const index_t &dest_offset,
       EdgePredicate edge_in_range) {
    packet_t packet{0};
    if (in_range) {
      using address_t = cl::sycl::access::address_space;
      packet.template load<address_t::global_space>(0, src + src_offset);
    } else {
#pragma unroll
      for (index_t i = 0; i < packet_size; i++) {
        reinterpret_cast<value_t *>(&packet)[i] =
            edge_in_range(i) ? *(src + src_offset + i) : 0;
      }
    }
    store<trans, ld>(packet, dest, dest_offset);
  }
  /*! @brief Store a vector packet into local memory when the source is
   * transposed. This will untranspose the elements individually when storing so
   * the data in local memory is always consistent.
   * @tparam trans Whether the source matrix is transposed or not.
   * @tparam ld The leading dimension of the destination memory.*/
  template <bool trans, int ld, typename packet_t = PacketType,
            typename DestPointerType, typename index_t>
  static SYCL_BLAS_INLINE typename std::enable_if<trans>::type store(
      packet_t &packet, DestPointerType dest, const index_t &dest_offset) {
#pragma unroll
    for (index_t i = 0; i < packet_size; i++) {
      *(dest + dest_offset + ld * i) = reinterpret_cast<value_t *>(&packet)[i];
    }
  }

  /*! @brief Store a vector packet into local memory when the source is not
   * transposed. This will use sycl::vec::store function.
   * @tparam trans Whether the source matrix is transposed or not.
   * @tparam ld The leading dimension of the destination memory.*/
  template <bool trans, int ld, typename packet_t = PacketType,
            typename DestPointerType, typename index_t>
  static SYCL_BLAS_INLINE typename std::enable_if<!trans>::type store(
      packet_t &packet, DestPointerType dest, const index_t &dest_offset) {
    using address_t = cl::sycl::access::address_space;
    packet.template store<address_t::local_space>(0, dest + dest_offset);
  }

  // Aligned versions of functions

  /*! @brief Performs a vectorised load using reinterpret_cast when the current
   * block is internal, the memory is aligned and the source is not transposed.
   * In the case where k < the number of elements being loaded then edge loads
   * will be element wise with additional bounds checking.
   * @tparam trans Whether the source matrix is transposed or not.
   * @tparam internal True if the current block is internal and no bounds
   * checking is required.
   * @tparam ld The leading dimension of the destination memory. */
  template <bool trans, bool internal, int ld, typename packet_t = PacketType,
            typename SrcPointerType, typename DestPointerType, typename index_t,
            typename EdgePredicate, bool align = aligned>
  static SYCL_BLAS_INLINE
      typename std::enable_if<(internal && !trans && align)>::type
      load(const bool in_range, SrcPointerType src, const index_t &src_offset,
           DestPointerType dest, const index_t &dest_offset,
           EdgePredicate edge_in_range) {
    if (in_range) {
      *reinterpret_cast<packet_t *>(
          static_cast<value_t *>(dest + dest_offset)) =
          *reinterpret_cast<packet_t *>(
              static_cast<value_t *>(src + src_offset));
    } else {
#pragma unroll
      for (index_t i = 0; i < packet_size; i++) {
        *(dest + dest_offset) =
            edge_in_range(i) ? *(src + src_offset + i) : value_t{0};
      }
    }
  }
  /*! @brief Performs a vectorised load using sycl::vec load when the current
   * block is internal, the memory is aligned and the source is transposed.
   * In the case where k < the number of elements being loaded then edge loads
   * will be element wise with additional bounds checking.
   * @tparam trans Whether the source matrix is transposed or not.
   * @tparam internal True if the current block is internal and no bounds
   * checking is required.
   * @tparam ld The leading dimension of the destination memory. */
  template <bool trans, bool internal, int ld, typename packet_t = PacketType,
            typename SrcPointerType, typename DestPointerType, typename index_t,
            typename EdgePredicate, bool align = aligned>
  static SYCL_BLAS_INLINE
      typename std::enable_if<(internal && trans && align)>::type
      load(const bool in_range, SrcPointerType src, const index_t &src_offset,
           DestPointerType dest, const index_t &dest_offset,
           EdgePredicate edge_in_range) {
    packet_t packet{0};
    if (in_range) {
      packet = *reinterpret_cast<packet_t *>(
          static_cast<value_t *>(src + src_offset));
      store<trans, ld>(packet, dest, dest_offset);
    } else {
#pragma unroll
      for (index_t i = 0; i < packet_size; i++) {
        reinterpret_cast<value_t *>(&packet)[i] =
            edge_in_range(i) ? *(src + src_offset + i) : 0;
      }
      store<trans, ld>(packet, dest, dest_offset);
    }
  }
};

}  // namespace blas
#endif  // SYCL_BLAS_BLAS3_GEMM_LOAD_STORE_HPP