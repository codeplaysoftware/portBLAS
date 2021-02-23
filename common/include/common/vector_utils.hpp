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
 *  @filename vector_utils.hpp
 *
 **************************************************************************/

#ifndef BLAS_COMMON_UTILS_VECTOR_UTILS
#define BLAS_COMMON_UTILS_VECTOR_UTILS

#include <utility>

namespace blas {
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

}  // namespace blas

#endif  // BLAS_COMMON_UTILS_VECTOR_UTILS
