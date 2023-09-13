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
 *  @filename blas3_symm_test.cpp
 *
 **************************************************************************/

#include <utility>

#include "blas_test.hpp"

template <typename T>
using symm_arguments_t =
    std::tuple<std::string, int, int, char, char, T, T, int, int, int>;

template <typename scalar_t, helper::AllocType mem_alloc>
inline void verify_symm(const symm_arguments_t<scalar_t> arguments) {
  std::string alloc;
  index_t m;
  index_t n;
  char side;
  char uplo;
  scalar_t alpha;
  scalar_t beta;
  index_t lda_mul;
  index_t ldb_mul;
  index_t ldc_mul;
  std::tie(alloc, m, n, side, uplo, alpha, beta, lda_mul, ldb_mul, ldc_mul) =
      arguments;

  auto q = make_queue();
  blas::SB_Handle sb_handle(q);

  // l -> A MxM B MxN
  // r -> B MxN A NxN
  const index_t k = side == 'l' ? m : n;

  const index_t lda = k * lda_mul;
  const index_t ldb = m * ldb_mul;
  const index_t ldc = m * ldc_mul;

  const index_t size_a = k * k * lda_mul;
  const index_t size_b = m * n * ldb_mul;
  const index_t size_c = m * n * ldc_mul;

  std::vector<scalar_t> a_m(size_a);
  std::vector<scalar_t> b_m(size_b);
  std::vector<scalar_t> c_m_gpu(size_c);

  fill_random(a_m);
  fill_random(b_m);
  fill_random(c_m_gpu);
  std::vector<scalar_t> c_m_cpu = c_m_gpu;

  // Use system blas to create a reference output
  const char side_str[2] = {side, '\0'};
  const char uplo_str[2] = {uplo, '\0'};
  reference_blas::symm(side_str, uplo_str, m, n, alpha, a_m.data(), lda,
                       b_m.data(), ldb, beta, c_m_cpu.data(), ldc);

  auto m_a_gpu = blas::helper::allocate<mem_alloc, scalar_t>(size_a, q);
  auto m_b_gpu = blas::helper::allocate<mem_alloc, scalar_t>(size_b, q);
  auto m_c_gpu = blas::helper::allocate<mem_alloc, scalar_t>(size_c, q);

  auto copy_a = blas::helper::copy_to_device(q, a_m.data(), m_a_gpu, size_a);
  auto copy_b = blas::helper::copy_to_device(q, b_m.data(), m_b_gpu, size_b);
  auto copy_c =
      blas::helper::copy_to_device(q, c_m_gpu.data(), m_c_gpu, size_c);

  // portBLAS SYMM implementation
  auto symm_event =
      _symm(sb_handle, side, uplo, m, n, alpha, m_a_gpu, lda, m_b_gpu, ldb,
            beta, m_c_gpu, ldc, {copy_a, copy_b, copy_c});

  sb_handle.wait(symm_event);

  auto event = blas::helper::copy_to_host(q, m_c_gpu, c_m_gpu.data(), size_c);
  sb_handle.wait(event);

  const bool isAlmostEqual = utils::compare_vectors(c_m_gpu, c_m_cpu);
  ASSERT_TRUE(isAlmostEqual);

  helper::deallocate<mem_alloc>(m_a_gpu, q);
  helper::deallocate<mem_alloc>(m_b_gpu, q);
  helper::deallocate<mem_alloc>(m_c_gpu, q);
}

template <typename scalar_t>
inline void verify_symm(const symm_arguments_t<scalar_t> arguments) {
  std::string alloc;
  index_t m;
  index_t n;
  char side;
  char uplo;
  scalar_t alpha;
  scalar_t beta;
  index_t lda_mul;
  index_t ldb_mul;
  index_t ldc_mul;
  std::tie(alloc, m, n, side, uplo, alpha, beta, lda_mul, ldb_mul, ldc_mul) =
      arguments;

  if (alloc == "usm") {
#ifdef SB_ENABLE_USM
    verify_symm<scalar_t, helper::AllocType::usm>(arguments);
#else
    GTEST_SKIP();
#endif
  } else {
    verify_symm<scalar_t, helper::AllocType::buffer>(arguments);
  }
}

template <class T>
static std::string generate_name(
    const ::testing::TestParamInfo<symm_arguments_t<T>>& info) {
  std::string alloc;
  int m, n, ldaMul, ldbMul, ldcMul;
  char side, uplo;
  T alpha, beta;
  BLAS_GENERATE_NAME(info.param, alloc, m, n, side, uplo, alpha, beta, ldaMul,
                     ldbMul, ldcMul);
}

/** Registers SYMM test for all supported data types
 * @param test_suite Name of the test suite
 * @param combination Combinations object
 * @see BLAS_REGISTER_TEST_CUSTOM_NAME
 */
#define GENERATE_SYMM_TEST(test_suite, combination)                          \
  BLAS_REGISTER_TEST_CUSTOM_NAME(test_suite, test_suite##combination,        \
                                 verify_symm, symm_arguments_t, combination, \
                                 generate_name);

template <typename scalar_t>
const auto SmallBetaNonZeroLDMatch =
    ::testing::Combine(::testing::Values("usm", "buf"),   // allocation type
                       ::testing::Values(11, 16, 32),     // m
                       ::testing::Values(11, 16, 32),     // n
                       ::testing::Values('l', 'r'),       // side
                       ::testing::Values('l', 'u'),       // uplo
                       ::testing::Values<scalar_t>(1.5),  // alpha
                       ::testing::Values<scalar_t>(0.5),  // beta
                       ::testing::Values(1),              // lda_mul
                       ::testing::Values(1),              // ldb_mul
                       ::testing::Values(1)               // ldc_mul
    );
GENERATE_SYMM_TEST(Symm, SmallBetaNonZeroLDMatch);

template <typename scalar_t>
const auto AlphaZero =
    ::testing::Combine(::testing::Values("usm", "buf"),   // allocation type
                       ::testing::Values(16),             // m
                       ::testing::Values(16),             // n
                       ::testing::Values('l', 'r'),       // side
                       ::testing::Values('l', 'u'),       // uplo
                       ::testing::Values<scalar_t>(0.0),  // alpha
                       ::testing::Values<scalar_t>(0.0, 1.0),  // beta
                       ::testing::Values(1, 2),                // lda_mul
                       ::testing::Values(1, 2),                // ldb_mul
                       ::testing::Values(1, 2)                 // ldc_mul
    );
GENERATE_SYMM_TEST(Symm, AlphaZero);

template <typename scalar_t>
const auto OffsetNonZero =
    ::testing::Combine(::testing::Values("usm", "buf"),   // allocation type
                       ::testing::Values(16, 63),         // m
                       ::testing::Values(16, 63),         // n
                       ::testing::Values('l', 'r'),       // side
                       ::testing::Values('l', 'u'),       // uplo
                       ::testing::Values<scalar_t>(1.0),  // alpha
                       ::testing::Values<scalar_t>(1.0),  // beta
                       ::testing::Values(1, 2),           // lda_mul
                       ::testing::Values(1, 2),           // ldb_mul
                       ::testing::Values(1, 2)            // ldc_mul
    );
GENERATE_SYMM_TEST(Symm, OffsetNonZero);

template <typename scalar_t>
const auto LargeBetaNonZeroLDMatch =
    ::testing::Combine(::testing::Values("usm", "buf"),   // allocation type
                       ::testing::Values(253, 511),       // m
                       ::testing::Values(257, 511),       // n
                       ::testing::Values('l', 'r'),       // side
                       ::testing::Values('l', 'u'),       // uplo
                       ::testing::Values<scalar_t>(1.0),  // alpha
                       ::testing::Values<scalar_t>(1.0),  // beta
                       ::testing::Values(1),              // lda_mul
                       ::testing::Values(1),              // ldb_mul
                       ::testing::Values(1)               // ldc_mul
    );
GENERATE_SYMM_TEST(Symm, LargeBetaNonZeroLDMatch);
