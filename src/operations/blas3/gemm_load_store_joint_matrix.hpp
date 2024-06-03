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
 *  portBLAS: BLAS implementation using SYCL
 *
 *  @filename gemm_load_store_joint_matrix.hpp
 *
 **************************************************************************/

#ifndef PORTBLAS_BLAS3_GEMM_LOAD_STORE_JOINT_MATRIX_HPP
#define PORTBLAS_BLAS3_GEMM_LOAD_STORE_JOINT_MATRIX_HPP

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
template <int vector_size, typename value_t, typename index_t>
struct PacketizeJointMatrix {
#ifdef GEMM_VECTORIZATION_SUPPORT
  using PacketType = sycl::vec<value_t, vector_size>;
  static constexpr int packet_size = vector_size;
  template <index_t dimension>
  PORTBLAS_INLINE static constexpr bool check_size() {
    return packet_size == 1 || dimension == packet_size;
  }
#else
  // In the case where vectorization is not enabled, always set to 1
  using PacketType = sycl::vec<value_t, 1>;
  static constexpr int packet_size = 1;
  template <index_t dimension>
  PORTBLAS_INLINE static constexpr bool check_size() {
    return true;
  }
#endif

  /*! @brief Performs a coalesced non-vectorized load when the current block is
   * not internal.
   * @tparam internal True if the current block is internal and no bounds
   * checking is required.
   */

  template <bool internal, typename SrcPointerType, typename DestPointerType,
            typename EdgePredicate>
  static PORTBLAS_INLINE typename std::enable_if<!internal>::type load(
      const bool in_range, SrcPointerType src, DestPointerType dest,
      EdgePredicate) {
    value_t val = in_range ? *src : value_t{0};
    using address_t = sycl::access::address_space;
    if constexpr (std::is_same<sycl::multi_ptr<sycl::half,
                                                   address_t::local_space>,
                               DestPointerType>::value) {
      using dtype = sycl::half;
      *dest = static_cast<dtype>(val);
    } else if constexpr (std::is_same<sycl::multi_ptr<
                                          sycl::ext::oneapi::bfloat16,
                                          address_t::local_space>,
                                      DestPointerType>::value) {
      using namespace sycl::ext::oneapi;
      *dest = bfloat16(val);
    } else {
      using namespace sycl::ext::oneapi::experimental::matrix;
      *dest = round_to_tf32(val);
    }
  }

  /*! @brief Performs a vectorised load using sycl::vec::load when the current
   * block is internal. In the case where k < the
   * number of elements being loaded then edge loads will be element wise with
   * additional bounds checking.
   * @tparam internal True if the current block is internal and no bounds
   * checking is required.
   */
  template <bool internal, typename SrcPointerType, typename DestPointerType,
            typename EdgePredicate>
  static PORTBLAS_INLINE typename std::enable_if<internal>::type load(
      const bool in_range, SrcPointerType src, DestPointerType dest,
      EdgePredicate edge_in_range) {
    PacketType packet{};

    using address_t = sycl::access::address_space;
    if (in_range) {
      packet.template load<address_t::global_space>(
          0, sycl::multi_ptr<const value_t, address_t::global_space>(src));
      store(packet, dest);
    } else {
      // avoid writing to variable, instead directly write to
      // shared local memory to avoid race condition experienced
      // with release compiler.
#pragma unroll
      for (index_t i = 0; i < packet_size; i++, dest++, src++) {
        if constexpr (std::is_same<sycl::multi_ptr<sycl::half,
                                                       address_t::local_space>,
                                   DestPointerType>::value) {
          using dtype = sycl::half;
          *dest = static_cast<dtype>(edge_in_range(i) ? *src : 0);
        } else if constexpr (std::is_same<sycl::multi_ptr<
                                              sycl::ext::oneapi::bfloat16,
                                              address_t::local_space>,
                                          DestPointerType>::value) {
          using namespace sycl::ext::oneapi;
          *dest = bfloat16(edge_in_range(i) ? *src : 0.f);
        } else {
          using namespace sycl::ext::oneapi::experimental::matrix;
          *dest = edge_in_range(i) ? round_to_tf32(*src) : 0.f;
        }
      }
    }
  }

  /*! @brief Store a vector packet into local memory. This will use
   *  sycl::vec::store function.
   */
  template <typename DestPointerType>
  static PORTBLAS_INLINE void store(PacketType &packet, DestPointerType dest) {
    using address_t = sycl::access::address_space;
    if constexpr (std::is_same<sycl::multi_ptr<sycl::half,
                                                   address_t::local_space>,
                               DestPointerType>::value) {
      using dtype = sycl::half;
      sycl::vec<dtype, vector_size> new_vec{};
      for (index_t i = 0; i < packet_size; i++) {
        reinterpret_cast<dtype *>(&new_vec)[i] =
            static_cast<dtype>(reinterpret_cast<value_t *>(&packet)[i]);
      }
      new_vec.template store<address_t::local_space>(
          0, sycl::multi_ptr<dtype, address_t::local_space>(dest));
    } else if constexpr (std::is_same<sycl::multi_ptr<
                                          sycl::ext::oneapi::bfloat16,
                                          address_t::local_space>,
                                      DestPointerType>::value) {
      // sycl::vec doesn't accept bfloat16 as a valid input type
      // so we need to write the packet elements individually to
      // the shared memory.
      using namespace sycl::ext::oneapi;
      for (index_t i = 0; i < packet_size; i++, dest++) {
        *dest = bfloat16(reinterpret_cast<value_t *>(&packet)[i]);
      }
    } else {
      using namespace sycl::ext::oneapi::experimental::matrix;
      using dtype = float;
      sycl::vec<dtype, vector_size> new_vec;
      for (index_t i = 0; i < packet_size; i++) {
        reinterpret_cast<dtype *>(&new_vec)[i] =
            round_to_tf32(reinterpret_cast<value_t *>(&packet)[i]);
      }
      new_vec.template store<address_t::local_space>(
          0, sycl::multi_ptr<dtype, address_t::local_space>(dest));
    }
  }
};

}  // namespace blas
#endif  // PORTBLAS_BLAS3_GEMM_LOAD_STORE_JOINT_MATRIX_HPP
