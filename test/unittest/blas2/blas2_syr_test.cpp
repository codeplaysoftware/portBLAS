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
 *  @filename blas2_syr_test.cpp
 *
 **************************************************************************/

#include "blas_test.hpp"

template <typename scalar_t>
using combination_t = std::tuple<std::string, char, int, scalar_t, int, int>;

template <typename scalar_t, helper::AllocType mem_alloc>
void run_test(const combination_t<scalar_t> combi) {
  std::string alloc;
  index_t n;
  index_t lda_mul;
  index_t incX;
  char uplo;
  scalar_t alpha;
  std::tie(alloc, uplo, n, alpha, incX, lda_mul) = combi;
  index_t lda = n * lda_mul;

  // Input vector
  std::vector<scalar_t> x_v(n * incX);
  fill_random(x_v);

  // Output matrix
  std::vector<scalar_t> a_m(n * lda, 7.0);
  std::vector<scalar_t> a_cpu_m(n * lda, 7.0);

  // SYSTEM SYR
  reference_blas::syr(&uplo, n, alpha, x_v.data(), incX, a_cpu_m.data(), lda);

  auto q = make_queue();
  blas::SB_Handle sb_handle(q);
  auto x_v_gpu = helper::allocate<mem_alloc, scalar_t>(n * incX, q);
  auto a_m_gpu = helper::allocate<mem_alloc, scalar_t>(lda * n, q);

  auto copy_x =
      helper::copy_to_device<scalar_t>(q, x_v.data(), x_v_gpu, n * incX);
  auto copy_a =
      helper::copy_to_device<scalar_t>(q, a_m.data(), a_m_gpu, lda * n);

  sb_handle.wait({copy_x, copy_a});

  // SYCLsyr
  auto syr_event = _syr(sb_handle, uplo, n, alpha, x_v_gpu, incX, a_m_gpu, lda);

  sb_handle.wait(syr_event);

  auto event = blas::helper::copy_to_host(q, a_m_gpu, a_m.data(), n * lda);
  sb_handle.wait(event);

  const bool isAlmostEqual = utils::compare_vectors(a_m, a_cpu_m);
  ASSERT_TRUE(isAlmostEqual);

  helper::deallocate<mem_alloc>(x_v_gpu, q);
  helper::deallocate<mem_alloc>(a_m_gpu, q);
}

template <typename scalar_t>
void run_test(const combination_t<scalar_t> combi) {
  std::string alloc;
  index_t n;
  index_t lda_mul;
  index_t incX;
  char uplo;
  scalar_t alpha;
  std::tie(alloc, uplo, n, alpha, incX, lda_mul) = combi;

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
                       ::testing::Values(1, 2),                     // incX
                       ::testing::Values(1, 2)                      // lda_mul
    );
#else
// For the purpose of travis and other slower platforms, we need a faster test
template <typename scalar_t>
const auto combi =
    ::testing::Combine(::testing::Values("usm", "buf"),  // allocation type
                       ::testing::Values('u', 'l'),      // UPLO
                       ::testing::Values(14, 1010),      // n
                       ::testing::Values<scalar_t>(0.0, 1.5),  // alpha
                       ::testing::Values(2),                   // incX
                       ::testing::Values(2)                    // lda_mul
    );
#endif

template <class T>
static std::string generate_name(
    const ::testing::TestParamInfo<combination_t<T>>& info) {
  std::string alloc;
  char upl0;
  int n, incX, ldaMul;
  T alpha;
  BLAS_GENERATE_NAME(info.param, alloc, upl0, n, alpha, incX, ldaMul);
}

BLAS_REGISTER_TEST_ALL(Syr, combination_t, combi, generate_name);
