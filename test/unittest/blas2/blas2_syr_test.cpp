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
 *  @filename blas2_syr_test.cpp
 *
 **************************************************************************/

#include "blas_test.hpp"

template <typename scalar_t>
using combination_t = std::tuple<char, int, scalar_t, int, int>;

template <typename scalar_t>
void run_test(const combination_t<scalar_t> combi) {
  int n;
  int lda_mul;
  int incX;
  char uplo;
  scalar_t alpha;
  std::tie(uplo, n, alpha, incX, lda_mul) = combi;
  int lda = n * lda_mul;

  using data_t = utils::data_storage_t<scalar_t>;

  // Input vector
  std::vector<data_t> x_v(n * incX);
  fill_random(x_v);

  // Output matrix
  std::vector<data_t> a_m(n * lda, 7.0);
  std::vector<data_t> a_cpu_m(n * lda, 7.0);

  // SYSTEM SYR
  reference_blas::syr(&uplo, n, static_cast<data_t>(alpha), x_v.data(), incX,
                      a_cpu_m.data(), lda);

  auto q = make_queue();
  test_executor_t ex(q);
  auto x_v_gpu = utils::make_quantized_buffer<scalar_t>(ex, x_v);
  auto a_m_gpu = utils::make_quantized_buffer<scalar_t>(ex, a_m);

  // SYCLsyr
  _syr(ex, uplo, n, alpha, x_v_gpu, incX, a_m_gpu, lda);

  auto event = utils::quantized_copy_to_host<scalar_t>(ex, a_m_gpu, a_m);
  ex.get_policy_handler().wait(event);

  const bool isAlmostEqual =
      utils::compare_vectors<data_t, scalar_t>(a_m, a_cpu_m);
  ASSERT_TRUE(isAlmostEqual);
}

#ifdef STRESS_TESTING
const auto combi =
    ::testing::Combine(::testing::Values('u', 'l'),                 // UPLO
                       ::testing::Values(14, 63, 257, 1010, 2025),  // n
                       ::testing::Values(0.0, 1.0, 1.5),            // alpha
                       ::testing::Values(1, 2),                     // incX
                       ::testing::Values(1, 2)                      // lda_mul
    );
#else
// For the purpose of travis and other slower platforms, we need a faster test
const auto combi = ::testing::Combine(::testing::Values('u', 'l'),  // UPLO
                                      ::testing::Values(14, 1010),  // n
                                      ::testing::Values(0.0, 1.5),  // alpha
                                      ::testing::Values(2),         // incX
                                      ::testing::Values(2)          // lda_mul
);
#endif

BLAS_REGISTER_TEST(Syr, combination_t, combi);
