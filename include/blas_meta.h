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
 *  SYCL-BLAS: BLAS implementation using SYCL
 *
 *  @filename blas_meta.h
 *
 **************************************************************************/

#ifndef SYCL_BLAS_META_H
#define SYCL_BLAS_META_H

#include <CL/sycl.hpp>
#include <type_traits>
#include <utility>

namespace blas {

/**
 * @class Access
 * @brief A wrapper type for Layout, providing common functionality and a safer
 * interface.
 */
enum class access_layout { row_major, col_major };

struct row_major {
  static constexpr bool is_col_major() { return false; }
};
struct col_major {
  static constexpr bool is_col_major() { return true; }
};

template <access_layout layout>
struct Layout;

template <>
struct Layout<access_layout::row_major> {
  using type = row_major;
};

template <>
struct Layout<access_layout::col_major> {
  using type = col_major;
};
/**
 * @enum Trans
 * @brief The possible transposition options for a matrix, expressed
 * algebraically.
 */
enum class transpose_type : char {
  Normal = 'n',
  Transposed = 't',
  Conjugate = 'c'
};

/**
 * @enum Uplo
 * @brief Specifies whether the lower or upper triangle needs to be accessed.
 */
enum class uplo_type : char { Upper = 'u', Lower = 'l' };

/**
 * @enum Diag
 * @brief Specifies the values on the diagonal of a triangular matrix.
 */
enum class diag_type : char { Nonunit = 'n', Unit = 'u' };

// choosing value at compile-time
template <bool Conds, typename val_t, val_t value_one_t, val_t value_two_t>
struct Choose {
  static constexpr auto type = value_one_t;
};

template <typename val_t, val_t value_one_t, val_t value_two_t>
struct Choose<false, val_t, value_one_t, value_two_t> {
  static constexpr auto type = value_two_t;
};
/// \struct RemoveAll
/// \brief These methods are used to remove all the & const and * from  a type.
/// template parameters
/// \tparam element_t : the type we are interested in
template <typename element_t>
struct RemoveAll {
  using Type = typename std::remove_reference<
      typename std::remove_cv<element_t>::type>::type;
};

template <typename container_t>
struct ValueType {
  using type = typename RemoveAll<container_t>::Type;
};

template <typename element_t, typename container_t>
struct RebindType {
  using type = RemoveAll<element_t> *;
};

template <typename index_t>
inline bool is_power_of_2(index_t ind) {
  return ind > 0 && !(ind & (ind - 1));
}

// This function returns the nearest power of 2
// if roundup is true returns result>=wgsize
// else it return result <= wgsize
template <typename index_t>
static inline index_t get_power_of_two(index_t wGSize, bool rounUp) {
  if (rounUp) --wGSize;
  wGSize |= (wGSize >> 1);
  wGSize |= (wGSize >> 2);
  wGSize |= (wGSize >> 4);
  wGSize |= (wGSize >> 8);
  wGSize |= (wGSize >> 16);
#if defined(__x86_64__) || defined(_M_X64) || defined(__amd64) || \
    defined(__aarch64__) || defined(_WIN64)
  wGSize |= (wGSize >> 32);
#endif
  return ((!rounUp) ? (wGSize - (wGSize >> 1)) : ++wGSize);
}

#ifdef __SYCL_DEVICE_ONLY__
#define SYCL_BLAS_ALWAYS_INLINE \
  __attribute__((flatten)) __attribute__((always_inline))
#else
#define SYCL_BLAS_ALWAYS_INLINE
#endif

#define SYCL_BLAS_INLINE SYCL_BLAS_ALWAYS_INLINE inline

template <typename index_t>
static SYCL_BLAS_INLINE index_t roundUp(index_t x, index_t y) {
  return ((x + y - 1) / y) * y;
}

template <typename index_t, typename vector_t>
index_t vec_total_size(index_t &vector_size, vector_t &&current_vector) {
  vector_size += static_cast<index_t>(current_vector.size());
  return 0;
}

template <typename vector_t>
int append_vector(vector_t &lhs_vector, vector_t const &rhs_vector) {
  lhs_vector.insert(lhs_vector.end(), rhs_vector.begin(), rhs_vector.end());
  return 0;
}

template <typename first_vector_t, typename... other_vector_t>
first_vector_t concatenate_vectors(first_vector_t first_vector,
                                   other_vector_t &&... other_vectors) {
  int first_Vector_size = static_cast<int>(first_vector.size());
  int s[] = {vec_total_size(first_Vector_size, other_vectors)..., 0};
  first_vector.reserve(first_Vector_size);
  int val[] = {append_vector(first_vector,
                             std::forward<other_vector_t>(other_vectors))...,
               0};
  return std::move(first_vector);
}

/**
 * @brief Defines if a type is a scalar within the context of sycl.
 * @tparam type The type to be tested.
 */
template <typename type>
struct is_sycl_scalar
    : std::conditional<std::is_scalar<type>::value, std::true_type,
                       std::false_type>::type {};

template <>
struct is_sycl_scalar<cl::sycl::half> : std::true_type {};

}  // namespace blas

#endif  // BLAS_META_H
