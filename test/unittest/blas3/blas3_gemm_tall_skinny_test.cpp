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
 *  @filename blas3_gemm_tall_skinny_test.cpp
 *
 **************************************************************************/

#include "blas_test.hpp"

template <typename scalar_t>
using combination_t =
    std::tuple<int, int, int, char, char, scalar_t, scalar_t, int, int, int>;

const auto combi = ::testing::Combine(::testing::Values(7, 65),      // m
                                      ::testing::Values(9, 126),     // n
                                      ::testing::Values(5678),       // k
                                      ::testing::Values('n', 't'),   // transa
                                      ::testing::Values('n', 't'),   // transb
                                      ::testing::Values(1.5),        // alpha
                                      ::testing::Values(0.0, 0.5),   // beta
                                      ::testing::Values(3),          // lda_mul
                                      ::testing::Values(2),          // ldb_mul
                                      ::testing::Values(3)           // ldc_mul
);

template <typename scalar_t>
void run_test(const combination_t<scalar_t> combi) {
  int m, n, k;
  char transa, transb;
  scalar_t alpha, beta;
  int lda_mul, ldb_mul, ldc_mul;
  std::tie(m, n, k, transa, transb, alpha, beta, lda_mul, ldb_mul, ldc_mul) =
      combi;

  const char ta_str[2] = {transa, '\0'};
  const char tb_str[2] = {transb, '\0'};

  auto q = make_queue();
  test_executor_t ex(q);

  auto policy_handler = ex.get_policy_handler();

  int lda = ((transa != 'n') ? k : m) * lda_mul;
  int ldb = ((transb != 'n') ? n : k) * ldb_mul;
  int ldc = m * ldc_mul;

  std::vector<scalar_t> a_m(m * k * lda_mul);
  std::vector<scalar_t> b_m(k * n * ldb_mul);
  std::vector<scalar_t> c_m_gpu(m * n * ldc_mul);
  std::vector<scalar_t> c_m_cpu(m * n * ldc_mul);

  fill_random(a_m);
  fill_random(b_m);
  fill_random(c_m_gpu);
  std::copy(c_m_gpu.begin(), c_m_gpu.end(), c_m_cpu.begin());

  reference_blas::gemm(ta_str, tb_str, m, n, k, (scalar_t)alpha, a_m.data(), lda,
                       b_m.data(), ldb, (scalar_t)beta, c_m_cpu.data(), ldc);

  {
    auto m_a_gpu =
        blas::make_sycl_iterator_buffer<scalar_t>(a_m, m * k * lda_mul);
    auto m_b_gpu =
        blas::make_sycl_iterator_buffer<scalar_t>(b_m, k * n * ldb_mul);
    auto m_c_gpu =
        blas::make_sycl_iterator_buffer<scalar_t>(c_m_gpu, m * n * ldc_mul);
    try {
      _gemm(ex, transa, transb, m, n, k, alpha, m_a_gpu, lda, m_b_gpu, ldb, beta,
            m_c_gpu, ldc);
    } catch (cl::sycl::exception &e) {
      std::cerr << "Exception occured:" << std::endl;
      std::cerr << e.what() << std::endl;
    }
  }

  ASSERT_TRUE(utils::compare_vectors(c_m_gpu, c_m_cpu));
}

class GemmTallSkinnyFloat : public ::testing::TestWithParam<combination_t<float>> {};
TEST_P(GemmTallSkinnyFloat, test) { run_test<float>(GetParam()); };
INSTANTIATE_TEST_SUITE_P(tsgemm, GemmTallSkinnyFloat, combi);

#if DOUBLE_SUPPORT
class GemmTallSkinnyDouble : public ::testing::TestWithParam<combination_t<double>> {};
TEST_P(GemmTallSkinnyDouble, test) { run_test<double>(GetParam()); };
INSTANTIATE_TEST_SUITE_P(tsgemm, GemmTallSkinnyDouble, combi);
#endif
