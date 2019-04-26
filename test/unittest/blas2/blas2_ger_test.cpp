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
 *  @filename blas2_ger_test.cpp
 *
 **************************************************************************/

#include "blas_test.hpp"

template <typename scalar_t>
using combination_t = std::tuple<int, int, scalar_t, int, int, int>;

template <typename scalar_t>
void run_test(const combination_t<scalar_t> combi) {
  using type_t = blas_test_args<scalar_t, void>;
  using blas_test_t = BLAS_Test<type_t>;
  using executor_t = typename type_t::executor_t;

  int m;
  int n;
  int lda_mul;
  int incX;
  int incY;
  scalar_t alpha;
  std::tie(m, n, alpha, incX, incY, lda_mul) = combi;
  int lda = m * lda_mul;

  // Input matrix
  std::vector<scalar_t> a_v(m * incX);
  // Input Vector
  std::vector<scalar_t> b_v(n * incY);
  // output Vector
  std::vector<scalar_t> c_m_gpu_result(lda * n, scalar_t(10));
  // output system vector
  std::vector<scalar_t> c_m_cpu(lda * n, scalar_t(10));
  blas_test_t::set_rand(a_v, m * incX);
  blas_test_t::set_rand(b_v, n * incY);

  // SYSTEM GER
  reference_blas::ger(m, n, alpha, a_v.data(), incX, b_v.data(), incY,
                      c_m_cpu.data(), lda);

  SYCL_DEVICE_SELECTOR d;
  auto q = blas_test_t::make_queue(d);
  Executor<executor_t> ex(q);
  auto v_a_gpu = blas::make_sycl_iterator_buffer<scalar_t>(a_v, m * incX);
  auto v_b_gpu = blas::make_sycl_iterator_buffer<scalar_t>(b_v, n * incY);
  auto m_c_gpu =
      blas::make_sycl_iterator_buffer<scalar_t>(c_m_gpu_result, lda * n);

  // SYCLger
  _ger(ex, m, n, alpha, v_a_gpu, incX, v_b_gpu, incY, m_c_gpu, lda);

  auto event = ex.get_policy_handler().copy_to_host(
      m_c_gpu, c_m_gpu_result.data(), lda * n);
  ex.get_policy_handler().wait(event);

  for (int i = 0; i < lda * n; ++i) {
    ASSERT_T_EQUAL(scalar_t, c_m_gpu_result[i], c_m_cpu[i]);
  }
}

#ifdef STRESS_TESTING
const auto combi =
    ::testing::Combine(::testing::Values(11, 65, 255, 1023, 1024 * 1024),  // m
                       ::testing::Values(14, 63, 257, 1010, 1024 * 1024),  // n
                       ::testing::Values(0.0, 1.0, 1.5),  // alpha
                       ::testing::Values(1, 2),           // incX
                       ::testing::Values(1, 3),           // incY
                       ::testing::Values(1, 2)            // lda_mul
    );
#else
// For the purpose of travis and other slower platforms, we need a faster test
const auto combi = ::testing::Combine(::testing::Values(11, 1023),  // m
                                      ::testing::Values(14, 1010),  // n
                                      ::testing::Values(0.0, 1.5),  // alpha
                                      ::testing::Values(2),         // incX
                                      ::testing::Values(3),         // incY
                                      ::testing::Values(2)          // lda_mul
);
#endif

class GerFloat : public ::testing::TestWithParam<combination_t<float>> {};
TEST_P(GerFloat, test) { run_test<float>(GetParam()); };
INSTANTIATE_TEST_SUITE_P(gemv, GerFloat, combi);

#if DOUBLE_SUPPORT
class GerDouble : public ::testing::TestWithParam<combination_t<double>> {};
TEST_P(GerDouble, test) { run_test<double>(GetParam()); };
INSTANTIATE_TEST_SUITE_P(gemv, GerDouble, combi);
#endif
