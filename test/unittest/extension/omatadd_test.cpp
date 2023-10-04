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
 *  @filename omatadd_test.cpp
 *
 **************************************************************************/

#include "blas_test.hpp"
#include "extension_reference.hpp"

template <typename scalar_t>
using combination_t = std::tuple<std::string, char, char, index_t, index_t, scalar_t,
                                 scalar_t, index_t, index_t, index_t>;

template <typename scalar_t, helper::AllocType mem_alloc>
void run_test(const combination_t<scalar_t> combi) {
  std::string alloc;
  char trans_a, trans_b;
  index_t m, n, ld_a_mul, ld_b_mul, ld_c_mul;
  scalar_t alpha, beta;

  std::tie(alloc, trans_a, trans_b, m, n, alpha, beta, ld_a_mul, ld_b_mul, ld_c_mul) =
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
  reference_blas::ext_omatadd(trans_a, trans_b, m, n, alpha, A.data(), lda,
                              beta, B.data(), ldb, C_ref.data(), ldc);

  const auto size_m_a = base_size * ld_a_mul;
  const auto size_m_b = base_size * ld_b_mul;
  const auto size_m_c = base_size * ld_c_mul;

  auto m_a_gpu = helper::allocate<mem_alloc, scalar_t>(size_m_a, q);
  auto m_b_gpu = helper::allocate<mem_alloc, scalar_t>(size_m_b, q);
  auto m_c_gpu = helper::allocate<mem_alloc, scalar_t>(size_m_c, q);

  auto copy_m_a = helper::copy_to_device<scalar_t>(q, A.data(), m_a_gpu, size_m_a);
  auto copy_m_b = helper::copy_to_device<scalar_t>(q, B.data(), m_b_gpu, size_m_b);
  auto copy_m_c = helper::copy_to_device<scalar_t>(q, C.data(), m_c_gpu, size_m_c);

  auto omatadd_event = blas::_omatadd(sb_handle, trans_a, trans_b, m, n, alpha, m_a_gpu, lda, beta,
                 m_b_gpu, ldb, m_c_gpu, ldc, {copy_m_a, copy_m_b, copy_m_c});
  sb_handle.wait(omatadd_event);

  auto event = blas::helper::copy_to_host<scalar_t>(
      sb_handle.get_queue(), m_c_gpu, C.data(), size_m_c);
  sb_handle.wait(event);

  // Validate the result
  const bool isAlmostEqual = utils::compare_vectors(C, C_ref);
  ASSERT_TRUE(isAlmostEqual);

  helper::deallocate<mem_alloc>(m_a_gpu, q);
  helper::deallocate<mem_alloc>(m_b_gpu, q);
  helper::deallocate<mem_alloc>(m_c_gpu, q);
}

template <typename scalar_t>
void run_test(const combination_t<scalar_t> combi) {
  std::string alloc;
  char trans_a, trans_b;
  index_t m, n, ld_a_mul, ld_b_mul, ld_c_mul;
  scalar_t alpha, beta;

  std::tie(alloc, trans_a, trans_b, m, n, alpha, beta, ld_a_mul, ld_b_mul, ld_c_mul) =
      combi;

  if (alloc == "usm") {
#ifdef SB_ENABLE_USM
    run_test<scalar_t, helper::AllocType::usm>(combi);
#else
    GTEST_SKIP();
#endif
  } else {
    run_test<scalar_t, helper::AllocType::buffer>(combi);
  }
}

#ifdef STRESS_TESTING
template <typename scalar_t>
const auto combi =
    ::testing::Combine(::testing::Values<char>('n', 't'),  // trans_a
                       ::testing::Values<char>('n', 't'),  // trans_b
                       ::testing::Values<index_t>(1024, 4050, 16380),  // m
                       ::testing::Values<index_t>(1024, 4050, 16380),  // n
                       ::testing::Values<scalar_t>(0, 1.05, 2.01),     // alpha
                       ::testing::Values<scalar_t>(0, 1.05, 2.01),     // beta
                       ::testing::Values<index_t>(1, 2),      // lda_mul
                       ::testing::Values<index_t>(1, 2),      // ldb_mul
                       ::testing::Values<index_t>(1, 2, 3),   // ldc_mul
#else
template <typename scalar_t>
const auto combi =
    ::testing::Combine(::testing::Values("usm", "buf"),        // allocation type
                       ::testing::Values<char>('n', 't'),         // trans_a
                       ::testing::Values<char>('n', 't'),         // trans_b
                       ::testing::Values<index_t>(64, 129, 255),  // m
                       ::testing::Values<index_t>(64, 129, 255),  // n
                       ::testing::Values<scalar_t>(0, 1, 2),      // alpha
                       ::testing::Values<scalar_t>(0, 1, 2),      // beta
                       ::testing::Values<index_t>(1, 2),          // lda_mul
                       ::testing::Values<index_t>(1, 2),          // ldb_mul
                       ::testing::Values<index_t>(1, 2, 3));      // ldc_mul
#endif

template <class T>
static std::string generate_name(
    const ::testing::TestParamInfo<combination_t<T>> &info) {
  std::string alloc;
  char trans_a, trans_b;
  index_t m, n, lda_mul, ldb_mul, ldc_mul;
  T alpha, beta;
  BLAS_GENERATE_NAME(info.param, alloc, trans_a, trans_b, m, n, alpha, beta, lda_mul,
                     ldb_mul, ldc_mul);
}

BLAS_REGISTER_TEST_ALL(OmatAdd, combination_t, combi, generate_name);
