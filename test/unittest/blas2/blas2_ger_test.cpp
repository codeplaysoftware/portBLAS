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
 *  @filename blas2_ger_test.cpp
 *
 **************************************************************************/

#include "blas_test.hpp"

template <typename scalar_t>
using combination_t =
    std::tuple<std::string, int, int, scalar_t, int, int, int>;

template <typename scalar_t, helper::AllocType mem_alloc>
void run_test(const combination_t<scalar_t> combi) {
  std::string alloc;
  index_t m;
  index_t n;
  index_t lda_mul;
  index_t incX;
  index_t incY;
  scalar_t alpha;
  std::tie(alloc, m, n, alpha, incX, incY, lda_mul) = combi;
  index_t lda = m * lda_mul;

  // Input matrix
  std::vector<scalar_t> a_v(m * incX);
  // Input Vector
  std::vector<scalar_t> b_v(n * incY);
  // output Vector
  std::vector<scalar_t> c_m_gpu_result(lda * n, scalar_t(10));
  // output system vector
  std::vector<scalar_t> c_m_cpu(lda * n, scalar_t(10));
  fill_random(a_v);
  fill_random(b_v);

  // SYSTEM GER
  reference_blas::ger(m, n, alpha, a_v.data(), incX, b_v.data(), incY,
                      c_m_cpu.data(), lda);

  auto q = make_queue();
  blas::SB_Handle sb_handle(q);
  auto v_a_gpu = helper::allocate<mem_alloc, scalar_t>(m * incX, q);
  auto v_b_gpu = helper::allocate<mem_alloc, scalar_t>(n * incY, q);
  auto m_c_gpu = helper::allocate<mem_alloc, scalar_t>(lda * n, q);

  auto copy_a =
      helper::copy_to_device<scalar_t>(q, a_v.data(), v_a_gpu, m * incX);
  auto copy_b =
      helper::copy_to_device<scalar_t>(q, b_v.data(), v_b_gpu, n * incY);
  auto copy_c = helper::copy_to_device<scalar_t>(q, c_m_gpu_result.data(),
                                                 m_c_gpu, lda * n);

  sb_handle.wait({copy_a, copy_b, copy_c});

  // SYCLger
  auto ger_event =
      _ger(sb_handle, m, n, alpha, v_a_gpu, incX, v_b_gpu, incY, m_c_gpu, lda);

  sb_handle.wait(ger_event);

  auto event =
      blas::helper::copy_to_host(q, m_c_gpu, c_m_gpu_result.data(), lda * n);
  sb_handle.wait(event);

  const bool isAlmostEqual = utils::compare_vectors(c_m_gpu_result, c_m_cpu);
  ASSERT_TRUE(isAlmostEqual);

  helper::deallocate<mem_alloc>(v_a_gpu, q);
  helper::deallocate<mem_alloc>(v_b_gpu, q);
  helper::deallocate<mem_alloc>(m_c_gpu, q);
}

template <typename scalar_t>
void run_test(const combination_t<scalar_t> combi) {
  std::string alloc;
  index_t m;
  index_t n;
  index_t lda_mul;
  index_t incX;
  index_t incY;
  scalar_t alpha;
  std::tie(alloc, m, n, alpha, incX, incY, lda_mul) = combi;

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
                       ::testing::Values(11, 65, 255, 1023, 1024 * 1024),  // m
                       ::testing::Values(14, 63, 257, 1010, 1024 * 1024),  // n
                       ::testing::Values<scalar_t>(0.0, 1.0, 1.5),  // alpha
                       ::testing::Values(1, 2),                     // incX
                       ::testing::Values(1, 3),                     // incY
                       ::testing::Values(1, 2)                      // lda_mul
    );
#else
// For the purpose of travis and other slower platforms, we need a faster test
template <typename scalar_t>
const auto combi =
    ::testing::Combine(::testing::Values("usm", "buf"),  // allocation type
                       ::testing::Values(11, 1023, 8888, 10968),      // m
                       ::testing::Values(14, 1010, 9999),      // n
                       ::testing::Values<scalar_t>(0, 1.5),  // alpha
                       ::testing::Values(2),                   // incX
                       ::testing::Values(3),                   // incY
                       ::testing::Values(2)                    // lda_mul
    );
#endif

template <class T>
static std::string generate_name(
    const ::testing::TestParamInfo<combination_t<T>>& info) {
  std::string alloc;
  int m, n, incX, incY, ldaMul;
  T alpha;
  BLAS_GENERATE_NAME(info.param, alloc, m, n, alpha, incX, incY, ldaMul);
}

BLAS_REGISTER_TEST_ALL(Ger, combination_t, combi, generate_name);
