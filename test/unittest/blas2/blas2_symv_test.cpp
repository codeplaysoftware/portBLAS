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
 *  @filename blas2_symv_test.cpp
 *
 **************************************************************************/

#include "blas_test.hpp"

template <typename scalar_t>
using combination_t =
    std::tuple<std::string, char, int, scalar_t, int, int, scalar_t, int>;

template <typename scalar_t, helper::AllocType mem_alloc>
void run_test(const combination_t<scalar_t> combi) {
  std::string alloc;
  index_t n;
  index_t lda_mul;
  index_t incX;
  index_t incY;
  char uplo;
  scalar_t alpha;
  scalar_t beta;
  std::tie(alloc, uplo, n, alpha, lda_mul, incX, beta, incY) = combi;
  index_t lda = n * lda_mul;

  // Input matrix
  std::vector<scalar_t> a_m(lda * n);
  fill_random(a_m);

  // Input vector
  std::vector<scalar_t> x_v(n * incX);
  fill_random(x_v);

  // Output Vector
  std::vector<scalar_t> y_v(n * incY, 1.0);
  std::vector<scalar_t> y_cpu_v(n * incY, 1.0);

  // SYSTEM symv
  reference_blas::symv(&uplo, n, alpha, a_m.data(), lda, x_v.data(), incX, beta,
                       y_cpu_v.data(), incY);

  auto q = make_queue();
  blas::SB_Handle sb_handle(q);
  auto a_m_gpu = helper::allocate<mem_alloc, scalar_t>(lda * n, q);
  auto x_v_gpu = helper::allocate<mem_alloc, scalar_t>(n * incX, q);
  auto y_v_gpu = helper::allocate<mem_alloc, scalar_t>(n * incY, q);

  auto copy_a =
      helper::copy_to_device<scalar_t>(q, a_m.data(), a_m_gpu, lda * n);
  auto copy_x =
      helper::copy_to_device<scalar_t>(q, x_v.data(), x_v_gpu, n * incX);
  auto copy_y =
      helper::copy_to_device<scalar_t>(q, y_v.data(), y_v_gpu, n * incY);

  sb_handle.wait({copy_a, copy_x, copy_y});

  // SYCLsymv
  auto symv_event = _symv(sb_handle, uplo, n, alpha, a_m_gpu, lda, x_v_gpu,
                          incX, beta, y_v_gpu, incY);

  sb_handle.wait(symv_event);

  auto event = blas::helper::copy_to_host(q, y_v_gpu, y_v.data(), n * incY);
  sb_handle.wait(event);

  const bool isAlmostEqual = utils::compare_vectors(y_v, y_cpu_v);
  ASSERT_TRUE(isAlmostEqual);

  helper::deallocate<mem_alloc>(a_m_gpu, q);
  helper::deallocate<mem_alloc>(x_v_gpu, q);
  helper::deallocate<mem_alloc>(y_v_gpu, q);
}

template <typename scalar_t>
void run_test(const combination_t<scalar_t> combi) {
  std::string alloc;
  index_t n;
  index_t lda_mul;
  index_t incX;
  index_t incY;
  char uplo;
  scalar_t alpha;
  scalar_t beta;
  std::tie(alloc, uplo, n, alpha, lda_mul, incX, beta, incY) = combi;
  index_t lda = n * lda_mul;

  if (alloc == "usm") {
#ifdef SB_ENABLE_USM
    run_test<scalar_t, helper::AllocType::usm>(combi);
#else
    GTEST_SKIP();
#endif
  } else {
    run_test<scalar_t, helper::AllocType::buffer>(combi);
  }
}

#ifdef STRESS_TESTING
template <typename scalar_t>
const auto combi =
    ::testing::Combine(::testing::Values("usm", "buf"),  // allocation type
                       ::testing::Values('u', 'l'),      // UPLO
                       ::testing::Values(14, 63, 257, 1010, 2025),  // n
                       ::testing::Values<scalar_t>(0.0, 1.0, 1.5),  // alpha
                       ::testing::Values(1, 2),                     // lda_mul
                       ::testing::Values(1, 2),                     // incX
                       ::testing::Values<scalar_t>(0.0, 1.0, 1.5),  // beta
                       ::testing::Values(1, 3)                      // incY
    );
#else
// For the purpose of travis and other slower platforms, we need a faster test
template <typename scalar_t>
const auto combi =
    ::testing::Combine(::testing::Values("usm", "buf"),  // allocation type
                       ::testing::Values('u', 'l'),      // UPLO
                       ::testing::Values(2025),          // n
                       ::testing::Values<scalar_t>(0.0, 1.5),  // alpha
                       ::testing::Values(2),                   // lda_mul
                       ::testing::Values(2),                   // incX
                       ::testing::Values<scalar_t>(0.0, 1.5),  // beta
                       ::testing::Values(3)                    // incY
    );
#endif

template <class T>
static std::string generate_name(
    const ::testing::TestParamInfo<combination_t<T>>& info) {
  std::string alloc;
  char upl0;
  int n, ldaMul, incX, incY;
  T alpha, beta;
  BLAS_GENERATE_NAME(info.param, alloc, upl0, n, alpha, ldaMul, incX, beta,
                     incY);
}

BLAS_REGISTER_TEST_ALL(Symv, combination_t, combi, generate_name);
