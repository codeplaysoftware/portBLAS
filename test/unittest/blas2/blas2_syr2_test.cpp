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
 *  @filename blas2_syr2_test.cpp
 *
 **************************************************************************/

#include "blas_test.hpp"

template <typename scalar_t>
using combination_t = std::tuple<char, int, scalar_t, int, int, int>;

template <typename scalar_t>
void run_test(const combination_t<scalar_t> combi) {
  using type_t = blas_test_args<scalar_t, void>;
  using blas_test_t = BLAS_Test<type_t>;
  using executor_t = typename type_t::executor_t;

  int n;
  int lda_mul;
  int incX;
  int incY;
  char uplo;
  scalar_t alpha;
  std::tie(uplo, n, alpha, incX, incY, lda_mul) = combi;
  int lda = n * lda_mul;

  // Input vector
  std::vector<scalar_t> x_v(n * incX);
  std::vector<scalar_t> y_v(n * incY);
  blas_test_t::set_rand(x_v, n * incX);
  blas_test_t::set_rand(y_v, n * incY);

  // Output matrix
  std::vector<scalar_t> a_m(n * lda, 7.0);
  std::vector<scalar_t> a_cpu_m(n * lda, 7.0);

  // SYSTEM SYR
  reference_blas::syr2(&uplo, n, alpha, x_v.data(), incX, y_v.data(), incY,
                       a_cpu_m.data(), lda);

  SYCL_DEVICE_SELECTOR d;
  auto q = blas_test_t::make_queue(d);
  Executor<executor_t> ex(q);
  auto x_v_gpu = blas::make_sycl_iterator_buffer<scalar_t>(x_v, n * incX);
  auto y_v_gpu = blas::make_sycl_iterator_buffer<scalar_t>(y_v, n * incY);
  auto a_m_gpu = blas::make_sycl_iterator_buffer<scalar_t>(a_m, lda * n);

  // SYCLsyr2
  _syr2(ex, uplo, n, alpha, x_v_gpu, incX, y_v_gpu, incY, a_m_gpu, lda);

  auto event =
      ex.get_policy_handler().copy_to_host(a_m_gpu, a_m.data(), n * lda);
  ex.get_policy_handler().wait(event);

  for (int i = 0; i < n * lda; ++i) {
    ASSERT_T_EQUAL(scalar_t, a_m[i], a_cpu_m[i]);
  }
}

#ifdef STRESS_TESTING
const auto combi =
    ::testing::Combine(::testing::Values('u', 'l'),                 // UPLO
                       ::testing::Values(14, 63, 257, 1010, 2025),  // n
                       ::testing::Values(0.0, 1.0, 1.5),            // alpha
                       ::testing::Values(1, 2),                     // incX
                       ::testing::Values(1, 2),                     // incY
                       ::testing::Values(1, 2)                      // lda_mul
    );
#else
// For the purpose of travis and other slower platforms, we need a faster test
const auto combi = ::testing::Combine(::testing::Values('u', 'l'),  // UPLO
                                      ::testing::Values(2025),      // n
                                      ::testing::Values(0.0, 1.5),  // alpha
                                      ::testing::Values(2),         // incX
                                      ::testing::Values(2),         // incY
                                      ::testing::Values(2)          // lda_mul
);
#endif

class Syr2Float : public ::testing::TestWithParam<combination_t<float>> {};
TEST_P(Syr2Float, test) { run_test<float>(GetParam()); };
INSTANTIATE_TEST_SUITE_P(syr2, Syr2Float, combi);

#if DOUBLE_SUPPORT
class Syr2Double : public ::testing::TestWithParam<combination_t<double>> {};
TEST_P(Syr2Double, test) { run_test<double>(GetParam()); };
INSTANTIATE_TEST_SUITE_P(syr2, Syr2Double, combi);
#endif
