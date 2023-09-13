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
 *  @filename blas2_gemv_test.cpp
 *
 **************************************************************************/

#include "blas_test.hpp"

template <typename T>
using combination_t =
    std::tuple<std::string, int, int, T, T, bool, int, int, int>;

template <typename scalar_t, helper::AllocType mem_alloc>
void run_test(const combination_t<scalar_t> combi) {
  std::string alloc;
  index_t m;
  index_t n;
  bool trans;
  scalar_t alpha;
  scalar_t beta;
  index_t incX;
  index_t incY;
  index_t lda_mul;
  std::tie(alloc, m, n, alpha, beta, trans, incX, incY, lda_mul) = combi;

  const char* t_str = trans ? "t" : "n";

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

  // SYSTEM GEMV
  reference_blas::gemv(t_str, m, n, alpha, a_m.data(), lda_mul * m, x_v.data(),
                       incX, beta, y_v_cpu.data(), incY);

  auto q = make_queue();
  blas::SB_Handle sb_handle(q);
  auto m_a_gpu = helper::allocate<mem_alloc, scalar_t>(a_size, q);
  auto v_x_gpu = helper::allocate<mem_alloc, scalar_t>(x_size, q);
  auto v_y_gpu = helper::allocate<mem_alloc, scalar_t>(y_size, q);

  auto copy_m =
      helper::copy_to_device<scalar_t>(q, a_m.data(), m_a_gpu, a_size);
  auto copy_x =
      helper::copy_to_device<scalar_t>(q, x_v.data(), v_x_gpu, x_size);
  auto copy_y = helper::copy_to_device<scalar_t>(q, y_v_gpu_result.data(),
                                                 v_y_gpu, y_size);

  sb_handle.wait({copy_m, copy_x, copy_y});

  // SYCLGEMV
  auto gemv_event = _gemv(sb_handle, *t_str, m, n, alpha, m_a_gpu, lda_mul * m,
                          v_x_gpu, incX, beta, v_y_gpu, incY);
  sb_handle.wait(gemv_event);

  auto event =
      blas::helper::copy_to_host(q, v_y_gpu, y_v_gpu_result.data(), y_size);
  sb_handle.wait(event);

  const bool isAlmostEqual = utils::compare_vectors(y_v_gpu_result, y_v_cpu);
  ASSERT_TRUE(isAlmostEqual);

  helper::deallocate<mem_alloc>(m_a_gpu, q);
  helper::deallocate<mem_alloc>(v_x_gpu, q);
  helper::deallocate<mem_alloc>(v_y_gpu, q);
}

template <typename scalar_t>
void run_test(const combination_t<scalar_t> combi) {
  std::string alloc;
  index_t m;
  index_t n;
  bool trans;
  scalar_t alpha;
  scalar_t beta;
  index_t incX;
  index_t incY;
  index_t lda_mul;
  std::tie(alloc, m, n, alpha, beta, trans, incX, incY, lda_mul) = combi;

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
    ::testing::Combine(::testing::Values("usm", "buf"),       // allocation type
                       ::testing::Values(11, 65, 255, 1023),  // m
                       ::testing::Values(14, 63, 257, 1010),  // n
                       ::testing::Values<scalar_t>(0.0, 1.0, 1.5),  // alpha
                       ::testing::Values<scalar_t>(0.0, 1.0, 1.5),  // beta
                       ::testing::Values(true, false),              // trans
                       ::testing::Values(1, 2),                     // incX
                       ::testing::Values(1, 3),                     // incY
                       ::testing::Values(1, 2)                      // lda_mul
    );
#else
// For the purpose of travis and other slower platforms, we need a faster test
// (the stress_test above takes about ~5 minutes)
template <typename scalar_t>
const auto combi =
    ::testing::Combine(::testing::Values("usm", "buf"),   // allocation type
                       ::testing::Values(11, 1023),       // m
                       ::testing::Values(14, 1010),       // n
                       ::testing::Values<scalar_t>(1.5),  // alpha
                       ::testing::Values<scalar_t>(0.0, 1.5),  // beta
                       ::testing::Values(false, true),         // trans
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
  T alpha, beta;
  bool trans;
  BLAS_GENERATE_NAME(info.param, alloc, m, n, alpha, beta, trans, incX, incY,
                     ldaMul);
}

BLAS_REGISTER_TEST_ALL(Gemv, combination_t, combi, generate_name);
