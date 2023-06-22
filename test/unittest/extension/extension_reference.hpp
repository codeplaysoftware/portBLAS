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
 *  @filename extension_reference.hpp
 *
 **************************************************************************/
#ifndef SYCL_BLAS_EXTENSION_REFERENCE_IMPLEMENTATION_HPP
#define SYCL_BLAS_EXTENSION_REFERENCE_IMPLEMENTATION_HPP

#include <vector>

namespace reference_blas {
/*!
 * @brief Host-baseline implementation for omatcopy used as reference in test
 * and benchmark
 * @param trans tranpose input matrix or not
 * @param m,n matrices dimensions
 * @param alpha scalar factor for input matrix
 * @param A input matrix
 * @param lda leading dimension of input matrix
 * @param B output matrix
 * @param ldb leading dimension of output matrix
 */
template <typename index_t, typename scalar_t>
void omatcopy_ref(char trans, const index_t m, const index_t n,
                  const scalar_t alpha, std::vector<scalar_t>& A,
                  const index_t lda, std::vector<scalar_t>& B, index_t ldb) {
  if (trans != 't') {
    for (index_t j = 0; j < n; j++) {
      for (index_t i = 0; i < m; i++) {
        B[j * ldb + i] = alpha * A[j * lda + i];
      }
    }
  } else {
    for (index_t j = 0; j < n; j++) {
      for (index_t i = 0; i < m; i++) {
        B[i * ldb + j] = alpha * A[j * lda + i];
      }
    }
  }
}

/*!
 * @brief Host-baseline implementation of omatcopy2 used as reference.
 */
template <typename scalar_t, typename index_t>
void omatcopy2_ref(const char& t, const index_t& m, const index_t& n,
                   const scalar_t& alpha, std::vector<scalar_t>& in_matrix,
                   const index_t& ld_in, const index_t& inc_in,
                   std::vector<scalar_t>& out_matrix, const index_t& ld_out,
                   const index_t inc_out) {
  if (t == 't') {
    for (int i = 0; i < m; ++i) {
      for (int j = 0, c = 0; j < n; ++j, ++c) {
        {
          out_matrix[j * inc_out + i * ld_out] =
              alpha * in_matrix[i * inc_in + j * ld_in];
        }
      }
    }
  } else {
    for (int i = 0; i < n; ++i) {
      for (int j = 0, c = 0; j < m; ++j, ++c) {
        {
          out_matrix[j * inc_out + i * ld_out] =
              alpha * in_matrix[j * inc_in + i * ld_in];
        }
      }
    }
  }
  return;
}

}  // namespace reference_blas

#endif
