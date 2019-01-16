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
#include <vector>
namespace blas {

// choosing value at compile-time
template <bool Conds, int T1, int T2>
struct Choose {
  static const int type = T1;
};

template <int T1, int T2>
struct Choose<false, T1, T2> {
  static const int type = T2;
};
/// \struct remove_all
/// \brief These methods are used to remove all the & const and * from  a type.
/// template parameters
/// \tparam T : the type we are interested in
template <typename T>
struct remove_all {
  using Type =
      typename std::remove_reference<typename std::remove_cv<T>::type>::type;
};

template <typename ContainerT>
struct scalar_type {
  using type = typename remove_all<ContainerT>::Type;
};

template <typename T, typename ContainerT>
struct rebind_type {
  using type = remove_all<T> *;
};

// This function returns the nearest power of 2
// if roundup is ture returns result>=wgsize
// else it return result <= wgsize
template <typename Index>
static inline Index get_power_of_two(Index wGSize, bool rounUp) {
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

#define sycl_blas_inline inline __attribute__((always_inline))

template <typename IndexType, typename VectorType>
IndexType vec_total_size(IndexType &vector_size, VectorType &&current_vector) {
  vector_size += current_vector.size();
  return 0;
}

template <typename VectorType>
int append_vector(VectorType &lhs_vector, VectorType const &rhs_vector) {
  lhs_vector.insert(lhs_vector.end(), rhs_vector.begin(), rhs_vector.end());
  return 0;
}

template <typename FirstVector, typename... OtherVectors>
FirstVector concatenate_vectors(FirstVector first_vector,
                                OtherVectors &&... other_vectors) {
  auto first_Vector_size = first_vector.size();
  int s[] = {vec_total_size(first_Vector_size, other_vectors)..., 0};
  first_vector.reserve(first_Vector_size);
  int val[] = {
      append_vector(first_vector, std::forward<OtherVectors>(other_vectors))...,
      0};
  return std::move(first_vector);
}
}  // namespace blas

#endif  // BLAS_META_H
