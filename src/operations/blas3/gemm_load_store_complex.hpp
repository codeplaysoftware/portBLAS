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
 *  @filename gemm_load_store_complex.hpp
 *
 **************************************************************************/

#ifndef PORTBLAS_BLAS3_GEMM_LOAD_STORE_CPLX_HPP
#define PORTBLAS_BLAS3_GEMM_LOAD_STORE_CPLX_HPP

namespace blas {
#ifdef BLAS_ENABLE_COMPLEX
/*! @brief vec_complex is an intermediate wrapper of sycl::complex used in
 * Packetize. It serves as a temporary workaround to the upcoming
 * sycl::vec<syc::complex> container
 * github.com/intel/llvm/blob/sycl/sycl/doc/extensions/experimental/sycl_ext_oneapi_complex.asciidoc
 * and only supports size = 1.
 * @tparam DataT Complex type of the vector's data
 * @tparam NumElements Elements count of the vector (only 1 is supported)
 */
template <typename DataT, int NumElements = 1>
class vec_complex {
  static_assert(NumElements == 1,
                "Vector wrapper arround sycl::complex of size>1 unsupported.");
  using address_t = sycl::access::address_space;
  using decorated_t = sycl::access::decorated;
  using DataType = DataT;
  static constexpr int getNumElements() { return NumElements; }
  size_t size() const noexcept { return NumElements; }

 private:
  DataType m_Data;

 public:
  vec_complex() = default;

  constexpr vec_complex(const vec_complex &rhs) = default;
  constexpr vec_complex(vec_complex &&rhs) = default;
  constexpr vec_complex &operator=(const vec_complex &rhs) = default;

  vec_complex(const DataType &rhs_data) : m_Data{rhs_data} {}

  // Conversion operator (valid with NumElements==1)
  operator DataT() const { return m_Data; }

  // Subscript operators
  DataT &operator[](int i) {
    assert(i < NumElements);
    return (m_Data);
  }
  const DataT &operator[](int i) const {
    assert(i < NumElements);
    return (m_Data);
  }

  // Binary Ops
  // Multiply
  vec_complex operator*(const vec_complex &rhs) {
    return (vec_complex{m_Data * static_cast<DataT>(rhs)});
  }

  vec_complex operator*(const DataType &rhs) {
    return (vec_complex{m_Data * rhs});
  }

  // Compound Multiply
  vec_complex &operator*=(const DataType &rhs) {
    this->m_Data = this->m_Data * rhs;
    return (*this);
  }

  vec_complex &operator*=(const vec_complex &rhs) {
    this->m_Data = this->m_Data * static_cast<DataT>(rhs);
    return (*this);
  }

  // Add
  vec_complex operator+(const vec_complex &rhs) {
    return (vec_complex{m_Data + static_cast<DataT>(rhs)});
  }

  vec_complex operator+(const DataType &rhs) {
    return (vec_complex{m_Data + rhs});
  }

  // Compound Add
  vec_complex &operator+=(const DataType &rhs) {
    this->m_Data = this->m_Data * rhs;
    return (*this);
  }

  vec_complex &operator+=(const vec_complex &rhs) {
    this->m_Data = this->m_Data + static_cast<DataT>(rhs);
    return (*this);
  }

  // Load
  template <address_t Space, decorated_t DecorateAddress>
  void load(size_t Offset,
            sycl::multi_ptr<const DataT, Space, DecorateAddress> Ptr) {
    m_Data = *(Ptr + Offset * NumElements);
  }

  // Store
  template <address_t Space, decorated_t DecorateAddress>
  void store(size_t Offset,
             sycl::multi_ptr<DataT, Space, DecorateAddress> Ptr) const {
    *(Ptr + Offset * NumElements) = m_Data;
  }
};

/*! @brief Partial specialization of the Packetize class dedicated to
sycl::complex types. It contains static methods for loading and storing size=1
complex packets from/to memory.
* @tparam vector_size The desired vector size to be used. Only size = 1 is
supported so far.
* @tparam value_t The complex type of the matrix data.
*/
template <int vector_size, typename T, typename index_t>
struct Packetize<vector_size, complex_sycl<T>, index_t> {
  // Vectorization is not enabled for complex, always set to 1
  using value_t = complex_sycl<T>;
  using PacketType = vec_complex<value_t, 1>;
  static constexpr int packet_size = 1;
  template <index_t dimension>
  static PORTBLAS_INLINE constexpr bool check_size() {
    return true;
  }

  /*! @brief Performs a non-vectorised load of sycl::complex data element while
   * whether block is internal or not since vectorization is not enabled for
   * complex types yet.
   * @tparam trans Whether the source matrix is transposed or not.
   * @tparam internal True if the current block is internal and no bounds
   * checking is required.
   * @tparam ld The leading dimension of the destination memory. */
  template <bool trans, bool internal, index_t ld, typename SrcPointerType,
            typename DestPointerType, typename EdgePredicate>
  static PORTBLAS_INLINE void load(const bool in_range, SrcPointerType src,
                                   DestPointerType dest,
                                   EdgePredicate edge_in_range) {
    *(dest) = in_range ? *(src) : value_t{(T)0, (T)0};
  }

  /*! @brief Store a size = 1 vector packet of sycl::complex data into local
   * memory (whether source is transposed or not since it's only 1 element).
   * @tparam trans Whether the source matrix is transposed or not.
   * @tparam ld The leading dimension of the destination memory.*/
  template <bool trans, index_t ld, typename DestPointerType>
  static PORTBLAS_INLINE void store(PacketType &packet, DestPointerType dest) {
    *dest = packet[0];
  }
};
#endif
}  // namespace blas

#endif  // PORTBLAS_BLAS3_GEMM_LOAD_STORE_CPLX_HPP
