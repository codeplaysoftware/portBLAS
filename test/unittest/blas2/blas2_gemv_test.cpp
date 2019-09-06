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
 *  @filename blas2_gemv_test.cpp
 *
 **************************************************************************/

#include "blas_test.hpp"

template <typename T>
using combination_t = std::tuple<int, int, T, T, bool, int, int, int>;

template <typename scalar_t>
void run_test(const combination_t<scalar_t> combi) {
  int m;
  int n;
  bool trans;
  scalar_t alpha;
  scalar_t beta;
  int incX;
  int incY;
  int lda_mul;
  std::tie(m, n, alpha, beta, trans, incX, incY, lda_mul) = combi;

  const char *t_str = trans ? "t" : "n";

  int a_size = m * n * lda_mul;
  int x_size = trans ? (1 + (m - 1) * incX) : (1 + (n - 1) * incX);
  int y_size = trans ? (1 + (n - 1) * incY) : (1 + (m - 1) * incY);

  // Input matrix
  std::vector<scalar_t> a_m(a_size, 1.0);
  // Input Vector
  std::vector<scalar_t> x_v(x_size, 1.0);
  // output Vector
  std::vector<scalar_t> y_v_gpu_result(y_size, scalar_t(10.0));
  // output system vector
  std::vector<scalar_t> y_v_cpu(y_size, scalar_t(10.0));

  fill_random(a_m);
  fill_random(x_v);

  // SYSTEM GEMMV
  reference_blas::gemv(t_str, m, n, alpha, a_m.data(), lda_mul * m, x_v.data(),
                       incX, beta, y_v_cpu.data(), incY);

  auto q = make_queue();
  test_executor_t ex(q);
  auto m_a_gpu = blas::make_sycl_iterator_buffer<scalar_t>(a_m, a_size);
  auto v_x_gpu = blas::make_sycl_iterator_buffer<scalar_t>(x_v, x_size);
  auto v_y_gpu =
      blas::make_sycl_iterator_buffer<scalar_t>(y_v_gpu_result, y_size);

  // SYCLGEMV
  _gemv(ex, *t_str, m, n, alpha, m_a_gpu, lda_mul * m, v_x_gpu, incX, beta,
        v_y_gpu, incY);
  auto event = ex.get_policy_handler().copy_to_host(
      v_y_gpu, y_v_gpu_result.data(), y_size);
  ex.get_policy_handler().wait(event);

  ASSERT_TRUE(utils::compare_vectors(y_v_gpu_result, y_v_cpu));
}

#ifdef STRESS_TESTING
const auto combi =
    ::testing::Combine(::testing::Values(11, 65, 255, 1023),  // m
                       ::testing::Values(14, 63, 257, 1010),  // n
                       ::testing::Values(0.0, 1.0, 1.5),      // alpha
                       ::testing::Values(0.0, 1.0, 1.5),      // beta
                       ::testing::Values(true, false),        // trans
                       ::testing::Values(1, 2),               // incX
                       ::testing::Values(1, 3),               // incY
                       ::testing::Values(1, 2)                // lda_mul
    );
#else
// For the purpose of travis and other slower platforms, we need a faster test
// (the stress_test above takes about ~5 minutes)
const auto combi = ::testing::Combine(::testing::Values(11, 1023),     // m
                                      ::testing::Values(14, 1010),     // n
                                      ::testing::Values(1.5),          // alpha
                                      ::testing::Values(0.0, 1.5),     // beta
                                      ::testing::Values(false, true),  // trans
                                      ::testing::Values(2),            // incX
                                      ::testing::Values(3),            // incY
                                      ::testing::Values(2)  // lda_mul
);
#endif

class GemvFloat : public ::testing::TestWithParam<combination_t<float>> {};
TEST_P(GemvFloat, test) { run_test<float>(GetParam()); };
INSTANTIATE_TEST_SUITE_P(gemv, GemvFloat, combi);

#if DOUBLE_SUPPORT
class GemvDouble : public ::testing::TestWithParam<combination_t<double>> {};
TEST_P(GemvDouble, test) { run_test<double>(GetParam()); };
INSTANTIATE_TEST_SUITE_P(gemv, GemvDouble, combi);
#endif
