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
using combination_t = std::tuple<int, int, bool, T, T, int, int, int>;

template <typename scalar_t>
void run_test(const combination_t<scalar_t> combi) {
  using type_t = blas_test_args<scalar_t, void>;
  using blas_test_t = BLAS_Test<type_t>;
  using executor_t = typename type_t::executor_t;

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

  int lda = m * lda_mul;
  int x = trans ? m : n;
  int y = trans ? n : m;

  // Input matrix
  std::vector<scalar_t> a_m(lda * n);
  // Input Vector
  std::vector<scalar_t> b_v(x * incX);
  // output Vector
  std::vector<scalar_t> c_v_gpu_result(y * incY, scalar_t(10.0));
  // output system vector
  std::vector<scalar_t> c_v_cpu(y * incY, scalar_t(10.0));
  blas_test_t::set_rand(a_m, lda * n);
  blas_test_t::set_rand(b_v, x * incX);

  // SYSTEM GEMMV
  reference_blas::gemv(t_str, m, n, alpha, a_m.data(), lda, b_v.data(), incX,
                       beta, c_v_cpu.data(), incY);

  SYCL_DEVICE_SELECTOR d;
  auto q = blas_test_t::make_queue(d);
  Executor<executor_t> ex(q);
  auto m_a_gpu = ex.get_policy_handler().template allocate<scalar_t>(lda * n);
  auto v_b_gpu = ex.get_policy_handler().template allocate<scalar_t>(x * incX);
  auto v_c_gpu = ex.get_policy_handler().template allocate<scalar_t>(y * incY);
  ex.get_policy_handler().copy_to_device(a_m.data(), m_a_gpu, lda * n);
  ex.get_policy_handler().copy_to_device(b_v.data(), v_b_gpu, x * incX);
  ex.get_policy_handler().copy_to_device(c_v_gpu_result.data(), v_c_gpu,
                                         y * incY);
  // SYCLGEMV
  _gemv(ex, *t_str, m, n, alpha, m_a_gpu, lda, v_b_gpu, incX, beta, v_c_gpu,
        incY);
  auto event = ex.get_policy_handler().copy_to_host(
      v_c_gpu, c_v_gpu_result.data(), y * incY);
  ex.get_policy_handler().wait(event);

  for (int i = 0; i < y * incY; ++i) {
    assert_t_equal<scalar_t>(c_v_gpu_result[i], c_v_cpu[i]);
  }

  ex.get_policy_handler().template deallocate<scalar_t>(m_a_gpu);
  ex.get_policy_handler().template deallocate<scalar_t>(v_b_gpu);
  ex.get_policy_handler().template deallocate<scalar_t>(v_c_gpu);
}


#ifdef STRESS_TESTING
const auto combi =
    ::testing::Combine(::testing::Values(11, 65, 255, 1023),  // m
                       ::testing::Values(14, 63, 257, 1010),  // n
                       ::testing::Values(true, false),        // trans
                       ::testing::Values(0.0, 1.0, 1.5),      // alpha
                       ::testing::Values(0.0, 1.0, 1.5),      // beta
                       ::testing::Values(1, 2),               // incX
                       ::testing::Values(1, 3),               // incY
                       ::testing::Values(1, 2)                // lda_mul
    );
#else
// For the purpose of travis and other slower platforms, we need a faster test
// (the stress_test above takes about ~5 minutes)
const auto combi =
    ::testing::Combine(::testing::Values(11, 65, 255, 1023),  // m
                       ::testing::Values(14, 63, 257, 1010),  // n
                       ::testing::Values(true, false),        // trans
                       ::testing::Values(1.5),                // alpha
                       ::testing::Values(0.0, 1.5),           // beta
                       ::testing::Values(2),                  // incX
                       ::testing::Values(3),                  // incY
                       ::testing::Values(2)                   // lda_mul
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
