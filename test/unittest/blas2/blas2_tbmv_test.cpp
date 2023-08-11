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
 *  @filename blas2_tbmv_test.cpp
 *
 **************************************************************************/

#include "blas_test.hpp"

template <typename T>
using combination_t =
    std::tuple<std::string, int, int, bool, bool, bool, int, int>;

template <typename scalar_t, helper::AllocType mem_alloc>
void run_test(const combination_t<scalar_t> combi) {
  std::string alloc;
  index_t n;
  index_t k;
  bool trans;
  bool is_upper;
  bool is_unit;
  index_t incX;
  index_t lda_mul;
  std::tie(alloc, n, k, is_upper, trans, is_unit, incX, lda_mul) = combi;

  const char* t_str = trans ? "t" : "n";
  const char* uplo_str = is_upper ? "u" : "l";
  const char* diag_str = is_unit ? "u" : "n";

  int a_size = (k + 1) * n * lda_mul;
  int x_size = 1 + (n - 1) * incX;

  // Input matrix
  std::vector<scalar_t> a_m(a_size, 10.0);
  // Input/output vector
  std::vector<scalar_t> x_v(x_size, 10.0);
  // Input/output system vector
  std::vector<scalar_t> x_v_cpu(x_size, scalar_t(10.0));

  fill_random(a_m);
  fill_random(x_v);

  x_v_cpu = x_v;

  // SYSTEM TBMV
  reference_blas::tbmv(uplo_str, t_str, diag_str, n, k, a_m.data(),
                       (k + 1) * lda_mul, x_v_cpu.data(), incX);

  auto q = make_queue();
  blas::SB_Handle sb_handle(q);
  auto m_a_gpu = helper::allocate<mem_alloc, scalar_t>(a_size, q);
  auto v_x_gpu = helper::allocate<mem_alloc, scalar_t>(x_size, q);

  auto copy_a =
      helper::copy_to_device<scalar_t>(q, a_m.data(), m_a_gpu, a_size);
  auto copy_x =
      helper::copy_to_device<scalar_t>(q, x_v.data(), v_x_gpu, x_size);

  sb_handle.wait({copy_a, copy_x});

  // SYCL TBMV
  auto tbmv_event = _tbmv(sb_handle, *uplo_str, *t_str, *diag_str, n, k,
                          m_a_gpu, (k + 1) * lda_mul, v_x_gpu, incX);

  sb_handle.wait(tbmv_event);

  auto event = blas::helper::copy_to_host(q, v_x_gpu, x_v.data(), x_size);
  sb_handle.wait(event);

  const bool isAlmostEqual = utils::compare_vectors(x_v, x_v_cpu);
  ASSERT_TRUE(isAlmostEqual);

  helper::deallocate<mem_alloc>(m_a_gpu, q);
  helper::deallocate<mem_alloc>(v_x_gpu, q);
}

template <typename scalar_t>
void run_test(const combination_t<scalar_t> combi) {
  std::string alloc;
  index_t n;
  index_t k;
  bool trans;
  bool is_upper;
  bool is_unit;
  index_t incX;
  index_t lda_mul;
  std::tie(alloc, n, k, is_upper, trans, is_unit, incX, lda_mul) = combi;

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
                       ::testing::Values(14, 63, 257, 1010),  // n
                       ::testing::Values(3, 4, 9),            // k
                       ::testing::Values(true, false),        // is_upper
                       ::testing::Values(true, false),        // trans
                       ::testing::Values(true, false),        // is_unit
                       ::testing::Values(1, 2),               // incX
                       ::testing::Values(1, 2)                // lda_mul
    );
#else
// For the purpose of travis and other slower platforms, we need a faster test
// (the stress_test above takes about ~5 minutes)
template <typename scalar_t>
const auto combi =
    ::testing::Combine(::testing::Values("usm", "buf"),  // allocation type
                       ::testing::Values(14, 1010),      // n
                       ::testing::Values(3, 4),          // k
                       ::testing::Values(true, false),   // is_upper
                       ::testing::Values(true, false),   // trans
                       ::testing::Values(true, false),   // is_unit
                       ::testing::Values(2),             // incX
                       ::testing::Values(2)              // lda_mul
    );
#endif

template <class T>
static std::string generate_name(
    const ::testing::TestParamInfo<combination_t<T>>& info) {
  std::string alloc;
  int n, k, incX, ldaMul;
  bool is_upper;
  bool trans;
  bool is_unit;
  BLAS_GENERATE_NAME(info.param, alloc, n, k, is_upper, trans, is_unit, incX,
                     ldaMul);
}

BLAS_REGISTER_TEST_ALL(Tbmv, combination_t, combi, generate_name);
