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
 *  @filename omatadd_test.cpp
 *
 **************************************************************************/

#include "blas_test.hpp"

using index_t = int;
namespace reference_blas {

/**
 * @brief Reference omat-add implementation using reference omatcopy.
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
template <typename scalar_t>
void omatadd(const char trans_a, const char trans_b, const index_t m,
             const index_t n, const scalar_t alpha, std::vector<scalar_t> &A,
             const index_t lda_m, const scalar_t beta, std::vector<scalar_t> &B,
             const index_t ldb_m, std::vector<scalar_t> &C,
             const index_t ldc_m) {
  const index_t a_rows = trans_a == 't' ? n : m;
  const index_t a_cols = trans_a == 't' ? m : n;
  const index_t b_rows = trans_b == 't' ? n : m;
  const index_t b_cols = trans_b == 't' ? m : n;

  index_t ldc = ldc_m * m;

  // Temp Matrix 1 for computing A -> alpha * op(A)
  std::vector<scalar_t> TempMatrix1(ldc * n, 0);
  omatcopy(trans_a, a_rows, a_cols, alpha, A.data(), lda_m * a_rows,
           TempMatrix1.data(), ldc);
  // Temp Matrix 2 for computing B -> beta * op(B)
  std::vector<scalar_t> TempMatrix2(ldc * n, 0);
  omatcopy(trans_b, b_rows, b_cols, beta, B.data(), ldb_m * b_rows,
           TempMatrix2.data(), ldc);

  // Compute Sum of Temp matrices -> C
  for (index_t j = 0; j < n; j++) {
    for (index_t i = 0; i < m; i++) {
      C.at(i + j * ldc) =
          TempMatrix1.at(i + j * ldc) + TempMatrix2.at(i + j * ldc);
    }
  }
}

}  // namespace reference_blas

template <typename scalar_t>
using combination_t =
    std::tuple<char, char, int, int, scalar_t, scalar_t, int, int, int>;

template <typename scalar_t>
void run_test(const combination_t<scalar_t> combi) {
  char trans_a, trans_b;
  index_t m, n, ld_a_mul, ld_b_mul, ld_c_mul;
  scalar_t alpha, beta;

  std::tie(trans_a, trans_b, m, n, alpha, beta, ld_a_mul, ld_b_mul, ld_c_mul) =
      combi;

  auto q = make_queue();
  blas::SB_Handle sb_handle(q);

  index_t base_size = m * n;

  std::vector<scalar_t> A(base_size * ld_a_mul);
  std::vector<scalar_t> B(base_size * ld_b_mul);
  std::vector<scalar_t> C(base_size * ld_c_mul, (scalar_t)0);

  fill_random(A);
  fill_random(B);

  std::vector<scalar_t> C_ref = C;

  const index_t lda = (trans_a == 'n') ? m * ld_a_mul : n * ld_a_mul;
  const index_t ldb = (trans_b == 'n') ? m * ld_b_mul : n * ld_b_mul;
  const index_t ldc = m * ld_c_mul;

  // Reference implementation
  reference_blas::omatadd(trans_a, trans_b, m, n, alpha, A, ld_a_mul, beta, B,
                          ld_b_mul, C_ref, ld_c_mul);

  auto m_a_gpu =
      blas::make_sycl_iterator_buffer<scalar_t>(A, base_size * ld_a_mul);
  auto m_b_gpu =
      blas::make_sycl_iterator_buffer<scalar_t>(B, base_size * ld_b_mul);
  auto m_c_gpu =
      blas::make_sycl_iterator_buffer<scalar_t>(C, base_size * ld_c_mul);

  blas::extension::_omatadd(sb_handle, trans_a, trans_b, m, n, alpha, m_a_gpu,
                            lda, beta, m_b_gpu, ldb, m_c_gpu, ldc);

  auto event = blas::helper::copy_to_host<scalar_t>(
      sb_handle.get_queue(), m_c_gpu, C.data(), base_size * ld_c_mul);
  sb_handle.wait(event);

  // Validate the result
  const bool isAlmostEqual = utils::compare_vectors(C, C_ref);
  ASSERT_TRUE(isAlmostEqual);
}

template <typename scalar_t>
const auto combi =
    ::testing::Combine(::testing::Values<char>('n', 't'),         // trans_a
                       ::testing::Values<char>('n', 't'),         // trans_b
                       ::testing::Values<index_t>(64, 129, 255),  // m
                       ::testing::Values<index_t>(64, 129, 255),  // n
                       ::testing::Values<scalar_t>(0, 1, 2),      // alpha
                       ::testing::Values<scalar_t>(0, 1, 2),      // beta
                       ::testing::Values<index_t>(1, 2),          // lda_mul
                       ::testing::Values<index_t>(1, 2),          // ldb_mul
                       ::testing::Values<index_t>(1, 2, 3));      // ldc_mul

template <class T>
static std::string generate_name(
    const ::testing::TestParamInfo<combination_t<T>> &info) {
  char trans_a, trans_b;
  index_t m, n, lda_mul, ldb_mul, ldc_mul;
  T alpha, beta;
  BLAS_GENERATE_NAME(info.param, trans_a, trans_b, m, n, alpha, beta, lda_mul,
                     ldb_mul, ldc_mul);
}

BLAS_REGISTER_TEST_ALL(OmatAdd, combination_t, combi, generate_name);
