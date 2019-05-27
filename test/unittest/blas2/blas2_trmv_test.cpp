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
 *  @filename blas2_trmv_test.cpp
 *
 **************************************************************************/

#include "blas_test.hpp"

// Please note:
// TRMV is broken in OpenBLAS 0.2.18
// If you are seeing fails from this, it could be that
// Seems to have been fixed in modern OpenBLASes

using combination_t = std::tuple<char, char, char, int, int, int>;

template <typename scalar_t>
void run_test(const combination_t combi) {
  int n;
  int lda_mul;
  int incX;
  char uplo;
  char trans;
  char diag;
  std::tie(uplo, trans, diag, n, incX, lda_mul) = combi;
  int lda = n * lda_mul;

  // Input matrix
  std::vector<scalar_t> a_m(lda * n);
  fill_random(a_m);

  // Output Vector
  std::vector<scalar_t> x_v(n * incX, 7.0);
  std::vector<scalar_t> x_cpu_v(n * incX, 7.0);

  // If this is a unit triangle, we should set the diagonal
  if (diag == 'u' || diag == 'U') {
    for (int i = 0; i < n; i++) {
      // a_m[i][i], basically
      a_m[(i * lda) + i] = 1.0;
    }
  }

  // SYSTEM GER
  reference_blas::trmv(&uplo, &trans, &diag, n, a_m.data(), lda, x_cpu_v.data(),
                       incX);

  auto q = make_queue();
  test_executor_t ex(q);
  auto a_m_gpu = blas::make_sycl_iterator_buffer<scalar_t>(a_m, lda * n);
  auto x_v_gpu = blas::make_sycl_iterator_buffer<scalar_t>(x_v, n * incX);

  // SYCLtrmv
  _trmv(ex, uplo, trans, diag, n, a_m_gpu, lda, x_v_gpu, incX);

  auto event =
      ex.get_policy_handler().copy_to_host(x_v_gpu, x_v.data(), n * incX);
  ex.get_policy_handler().wait(event);

  ASSERT_TRUE(utils::compare_vectors(x_v, x_cpu_v));
}

#ifdef STRESS_TESTING
// For the purpose of travis and other slower platforms, we need a faster test
const auto combi =
    ::testing::Combine(::testing::Values('u', 'l'),                     // UPLO
                       ::testing::Values('n', 't'),                     // TRANS
                       ::testing::Values('u', 'n'),                     // DIAG
                       ::testing::Values(14, 63, 257, 1010, 1024 * 5),  // n
                       ::testing::Values(1, 2),                         // incX
                       ::testing::Values(1, 2)  // lda_mul
    );
#else
// For the purpose of travis and other slower platforms, we need a faster test
const auto combi = ::testing::Combine(::testing::Values('u', 'l'),  // UPLO
                                      ::testing::Values('n', 't'),  // TRANS
                                      ::testing::Values('u', 'n'),  // DIAG
                                      ::testing::Values(2025),      // n
                                      ::testing::Values(2),         // incX
                                      ::testing::Values(2)          // lda_mul
);
#endif

class TrmvFloat : public ::testing::TestWithParam<combination_t> {};
TEST_P(TrmvFloat, test) { run_test<float>(GetParam()); };
INSTANTIATE_TEST_SUITE_P(trmv, TrmvFloat, combi);

#if DOUBLE_SUPPORT
class TrmvDouble : public ::testing::TestWithParam<combination_t> {};
TEST_P(TrmvDouble, test) { run_test<double>(GetParam()); };
INSTANTIATE_TEST_SUITE_P(trmv, TrmvDouble, combi);
#endif
