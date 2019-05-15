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
 *  @filename blas2_symv_test.cpp
 *
 **************************************************************************/

#include "blas_test.hpp"

template <typename scalar_t>
using combination_t = std::tuple<char, int, scalar_t, int, int, scalar_t, int>;

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
  scalar_t beta;
  std::tie(uplo, n, alpha, lda_mul, incX, beta, incY) = combi;
  int lda = n * lda_mul;

  // Input matrix
  std::vector<scalar_t> a_m(lda * n);
  blas_test_t::set_rand(a_m, lda * n);

  // Input vector
  std::vector<scalar_t> x_v(n * incX);
  blas_test_t::set_rand(x_v, n * incX);

  // Output Vector
  std::vector<scalar_t> y_v(n * incY, 1.0);
  std::vector<scalar_t> y_cpu_v(n * incY, 1.0);

  // SYSTEM symv
  reference_blas::symv(&uplo, n, alpha, a_m.data(), lda, x_v.data(), incX, beta,
                       y_cpu_v.data(), incY);

  auto q = make_queue();
  Executor<executor_t> ex(q);
  auto a_m_gpu = blas::make_sycl_iterator_buffer<scalar_t>(a_m, lda * n);
  auto x_v_gpu = blas::make_sycl_iterator_buffer<scalar_t>(x_v, n * incX);
  auto y_v_gpu = blas::make_sycl_iterator_buffer<scalar_t>(y_v, n * incY);

  // SYCLsymv
  _symv(ex, uplo, n, alpha, a_m_gpu, lda, x_v_gpu, incX, beta, y_v_gpu, incY);

  auto event =
      ex.get_policy_handler().copy_to_host(y_v_gpu, y_v.data(), n * incY);
  ex.get_policy_handler().wait(event);

  for (int i = 0; i < n * incY; ++i) {
    ASSERT_T_EQUAL(scalar_t, y_v[i], y_cpu_v[i]);
  }
}

#ifdef STRESS_TESTING
const auto combi =
    ::testing::Combine(::testing::Values('u', 'l'),                 // UPLO
                       ::testing::Values(14, 63, 257, 1010, 2025),  // n
                       ::testing::Values(0.0, 1.0, 1.5),            // alpha
                       ::testing::Values(1, 2),                     // lda_mul
                       ::testing::Values(1, 2),                     // incX
                       ::testing::Values(0.0, 1.0, 1.5),            // beta
                       ::testing::Values(1, 3)                      // incY
    );
#else
// For the purpose of travis and other slower platforms, we need a faster test
const auto combi = ::testing::Combine(::testing::Values('u', 'l'),  // UPLO
                                      ::testing::Values(2025),      // n
                                      ::testing::Values(0.0, 1.5),  // alpha
                                      ::testing::Values(2),         // lda_mul
                                      ::testing::Values(2),         // incX
                                      ::testing::Values(0.0, 1.5),  // beta
                                      ::testing::Values(3)          // incY
);
#endif

class SymvFloat : public ::testing::TestWithParam<combination_t<float>> {};
TEST_P(SymvFloat, test) { run_test<float>(GetParam()); };
INSTANTIATE_TEST_SUITE_P(symv, SymvFloat, combi);

#if DOUBLE_SUPPORT
class SymvDouble : public ::testing::TestWithParam<combination_t<double>> {};
TEST_P(SymvDouble, test) { run_test<double>(GetParam()); };
INSTANTIATE_TEST_SUITE_P(symv, SymvDouble, combi);
#endif
