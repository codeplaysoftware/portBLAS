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
 *  portBLAS: BLAS implementation using SYCL
 *
 *  @filename extension_reference.hpp
 *
 **************************************************************************/
#ifndef PORTBLAS_EXTENSION_REFERENCE_IMPLEMENTATION_HPP
#define PORTBLAS_EXTENSION_REFERENCE_IMPLEMENTATION_HPP

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
template <bool onemkl_index = true, typename index_t, typename scalar_t>
void ext_omatcopy(char trans, const index_t m, const index_t n,
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
        // oneMKL and cuBLAS follow different convention on size and
        // leading dimension. Choose the correct with template parameter,
        // by default it follows oneMKL specifications.
        if constexpr (onemkl_index) {
          B[i * ldb + j] = alpha * A[j * lda + i];
        } else {
          B[j * ldb + i] = alpha * A[i * lda + j];
        }
      }
    }
  }
}

/*!
 * @brief Host-baseline implementation of omatcopy2 used as reference.
 */
template <typename scalar_t, typename index_t>
void ext_omatcopy2(const char& t, const index_t& m, const index_t& n,
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
/**
 * @brief Reference out-of-place transpose host-implementation.
 *
 * @param in Input matrix pointer
 * @param ld_in Input matrix leading dimension
 * @param out Output matrix pointer
 * @param ld_in Output matrix leading dimension
 * @param M Number of rows in input matrix (Columns in output )
 * @param N Number of columns in input matrix (Rows in out output)
 */

template <typename T, typename index_t>
void Transpose(const T* in, const index_t& ld_in, T* out, const index_t& ld_out,
               const index_t& M, const index_t& N) {
  for (index_t i = 0; i < M; i++) {
    for (index_t j = 0; j < N; j++) {
      out[j + i * ld_out] = in[i + j * ld_in];
    }
  }
}

/**
 * @brief Reference omatadd implementation using reference omatcopy.
 *
 * @param trans_a (char) 'n' or 't' corresponding to non-transposed or
 * transposed matrix A respectively.
 * @param trans_b (char) 'n' or 't' corresponding to non-transposed or
 * transposed matrix B respectively.
 * @param m Number of rows in output matrix C
 * @param n Number of columns in output matrix C
 * @param alpha Scaling factor of matrix A
 * @param A (vector) Input matrix A
 * @param lda_m Matrix A leading dimension multiplier. (lda = lda_m * A_rows)
 * @param beta scaling factor of matrix B
 * @param B (vector) Input matrix B
 * @param ldb_m Matrix B leading dimension multiplier. (ldb = ldb_m * B_rows)
 * @param C (vector) Output matrix C
 * @param ldc_m Matrix C leading dimension multiplier. (ldc = ldc_m * C_rows)
 */
template <typename index_t, typename scalar_t>
void ext_omatadd(char transa, char transb, index_t m, index_t n, scalar_t alpha,
                 std::vector<scalar_t>& A, index_t lda, scalar_t beta,
                 std::vector<scalar_t>& B, index_t ldb,
                 std::vector<scalar_t>& C, index_t ldc) {
  for (index_t j = 0; j < n; j++) {
    for (index_t i = 0; i < m; i++) {
      C[j * ldc + i] = alpha * A[(transa != 't') ? j * lda + i : i * lda + j] +
                       beta * B[(transb != 't') ? j * ldb + i : i * ldb + j];
    }
  }
}

}  // namespace reference_blas

#endif
