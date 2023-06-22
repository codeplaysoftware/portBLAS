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
#include "extension_reference.hpp"

template <typename scalar_t>
using combination_t =
    std::tuple<char, char, index_t, index_t, scalar_t, scalar_t, index_t, index_t, index_t>;

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
  reference_blas::omatadd_ref(trans_a, trans_b, m, n, alpha, A, lda, beta, B,
                              ldb, C_ref, ldc);

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
