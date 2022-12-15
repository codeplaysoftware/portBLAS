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

template <typename scalar_t>
using combination_t = std::tuple<char, char, char, int, int, int>;

template <typename scalar_t>
void run_test(const combination_t<scalar_t> combi) {
  index_t n;
  index_t lda_mul;
  index_t incX;
  char uplo;
  char trans;
  char diag;
  std::tie(uplo, trans, diag, n, incX, lda_mul) = combi;
  index_t lda = n * lda_mul;

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
  blas::SB_Handle sb_handle(q);
  auto a_m_gpu = blas::make_sycl_iterator_buffer<scalar_t>(a_m, lda * n);
  auto x_v_gpu = blas::make_sycl_iterator_buffer<scalar_t>(x_v, n * incX);

  // SYCLtrmv
  _trmv(sb_handle, uplo, trans, diag, n, a_m_gpu, lda, x_v_gpu, incX);

  auto event = blas::helper::copy_to_host(sb_handle.get_queue(), x_v_gpu,
                                          x_v.data(), n * incX);
  sb_handle.wait(event);

  const bool isAlmostEqual = utils::compare_vectors(x_v, x_cpu_v);
  ASSERT_TRUE(isAlmostEqual);
}

#ifdef STRESS_TESTING
// For the purpose of travis and other slower platforms, we need a faster test
template <typename scalar_t>
const auto combi = ::testing::Combine(::testing::Values('u', 'l'),  // UPLO
                                      ::testing::Values('n', 't'),  // TRANS
                                      ::testing::Values('u', 'n'),  // DIAG
                                      ::testing::Values(14, 63, 257, 1010,
                                                        1024 * 5),  // n
                                      ::testing::Values(1, 2),      // incX
                                      ::testing::Values(1, 2)       // lda_mul
);
#else
// For the purpose of travis and other slower platforms, we need a faster test
template <typename scalar_t>
const auto combi = ::testing::Combine(::testing::Values('u', 'l'),  // UPLO
                                      ::testing::Values('n', 't'),  // TRANS
                                      ::testing::Values('u', 'n'),  // DIAG
                                      ::testing::Values(2025),      // n
                                      ::testing::Values(2),         // incX
                                      ::testing::Values(2)          // lda_mul
);
#endif

template <class T>
static std::string generate_name(
    const ::testing::TestParamInfo<combination_t<T>>& info) {
  char upl0, trans, diag;
  int n, incX, ldaMul;
  BLAS_GENERATE_NAME(info.param, upl0, trans, diag, n, incX, ldaMul);
}

BLAS_REGISTER_TEST_ALL(Trmv, combination_t, combi, generate_name);
