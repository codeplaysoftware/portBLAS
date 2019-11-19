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
 *  @filename gemm_packetize.hpp
 *
 **************************************************************************/

#ifndef SYCL_BLAS_BLAS3_GEMM_PACKETIZE_HPP
#define SYCL_BLAS_BLAS3_GEMM_PACKETIZE_HPP

namespace blas {
template <typename T, size_t Size>
struct VectorizationParams {
#ifdef GEMM_VECTORISATION_SUPPORT
  using vectorised_t = cl::sycl::vec<T, Size>;
  static constexpr size_t packet_size = Size;
#else
  // In the case where vectorization is not enabled, always set to 1
  using vectorised_t = cl::sycl::vec<T, 1>;
  static constexpr size_t packet_size = 1;
#endif
};

template <bool aligned, bool trans, size_t packet_size, typename value_t>
struct Packetize {
  using packet_t = VectorizationParams<value_t, packet_size>;
  template <bool internal, int ld, bool check_row = false,
            typename packet_t = cl::sycl::vec<value_t, packet_size>,
            typename SrcPointerType, typename DestPointerType, typename index_t,
            typename EdgePredicate>
  static SYCL_BLAS_INLINE typename std::enable_if<(internal == false)>::type
  load(const bool in_range, SrcPointerType src, const index_t &src_offset,
       DestPointerType dest, const index_t &dest_offset, EdgePredicate) {
    *(dest + dest_offset) = in_range ? *(src + src_offset) : value_t{0};
  }

  template <bool internal, int ld,
            typename packet_t = cl::sycl::vec<value_t, packet_size>,
            typename SrcPointerType, typename DestPointerType, typename index_t,
            typename EdgePredicate>
  static SYCL_BLAS_INLINE typename std::enable_if<(internal == true)>::type
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

  template <bool tran, int ld,
            typename packet_t = cl::sycl::vec<value_t, packet_size>,
            typename DestPointerType, typename index_t>
  static SYCL_BLAS_INLINE typename std::enable_if<tran>::type store(
      packet_t &packet, DestPointerType dest, const index_t &dest_offset) {
#pragma unroll
    for (index_t i = 0; i < packet_size; i++) {
      *(dest + dest_offset + ld * i) = reinterpret_cast<value_t *>(&packet)[i];
    }
  }
  template <bool tran, int ld,
            typename packet_t = cl::sycl::vec<value_t, packet_size>,
            typename DestPointerType, typename index_t>
  static SYCL_BLAS_INLINE typename std::enable_if<!tran>::type store(
      packet_t &packet, DestPointerType dest, const index_t &dest_offset) {
    using address_t = cl::sycl::access::address_space;
    packet.template store<address_t::local_space>(0, dest + dest_offset);
  }
};

// template <bool trans, size_t packet_size, typename value_t>
// struct Packetize<true, trans, packet_size, value_t> {
//   template <bool internal, int ld, bool check_row = false,
//             typename packet_t = cl::sycl::vec<value_t, packet_size>,
//             typename SrcPointerType, typename DestPointerType, typename
//             index_t, typename EdgePredicate>
//   static SYCL_BLAS_INLINE typename std::enable_if<(!internal)>::type load(
//       const bool in_range, SrcPointerType src, const index_t &src_offset,
//       DestPointerType dest, const index_t &dest_offset, EdgePredicate) {
//     *(dest + dest_offset) = in_range ? *(src + src_offset) : value_t{0};
//   }

//   template <bool internal, int ld, bool tran = trans,
//             typename packet_t = cl::sycl::vec<value_t, packet_size>,
//             typename SrcPointerType, typename DestPointerType, typename
//             index_t, typename EdgePredicate>
//   static SYCL_BLAS_INLINE typename std::enable_if<(internal && !tran)>::type
//   load(const bool in_range, SrcPointerType src, const index_t &src_offset,
//        DestPointerType dest, const index_t &dest_offset,
//        EdgePredicate edge_in_range) {
//     if (in_range) {
//       *reinterpret_cast<packet_t *>(
//           static_cast<value_t *>(dest + dest_offset)) =
//           *reinterpret_cast<packet_t *>(
//               static_cast<value_t *>(src + src_offset));
//     } else {
// #pragma unroll
//       for (index_t i = 0; i < packet_size; i++) {
//         *(dest + dest_offset) =
//             edge_in_range(i) ? *(src + src_offset + i) : value_t{0};
//       }
//     }
//   }

//   template <bool internal, int ld, bool tran = trans,
//             typename packet_t = cl::sycl::vec<value_t, packet_size>,
//             typename SrcPointerType, typename DestPointerType, typename
//             index_t, typename EdgePredicate>
//   static SYCL_BLAS_INLINE typename std::enable_if<(internal && tran)>::type
//   load(const bool in_range, SrcPointerType src, const index_t &src_offset,
//        DestPointerType dest, const index_t &dest_offset,
//        EdgePredicate edge_in_range) {
//     packet_t packet{0};
//     if (in_range) {
//       *reinterpret_cast<packet_t *>(&packet) = *reinterpret_cast<packet_t *>(
//           static_cast<value_t *>(src + src_offset));
//     } else {
// #pragma unroll
//       for (index_t i = 0; i < packet_size; i++) {
//         reinterpret_cast<value_t *>(&packet)[i] =
//             edge_in_range(i) ? *(src + src_offset + i) : 0;
//       }
//     }
//     store<ld>(packet, dest, dest_offset);
//   }

//   template <int ld, typename packet_t = cl::sycl::vec<value_t, packet_size>,
//             typename DestPointerType, typename index_t>
//   static SYCL_BLAS_INLINE void store(packet_t &packet, DestPointerType dest,
//                                      const index_t &dest_offset) {
// #pragma unroll
//     for (index_t i = 0; i < packet_size; i++) {
//       *(dest + dest_offset + ld * i) = reinterpret_cast<value_t
//       *>(&packet)[i];
//     }
//   }
// };

}  // namespace blas
#endif  // SYCL_BLAS_BLAS3_GEMM_PACKETIZE_HPP