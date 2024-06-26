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
 *  @filename blas2_trsv_test.cpp
 *
 **************************************************************************/

#include "blas_test.hpp"

template <typename T>
using combination_t =
    std::tuple<std::string, index_t, bool, bool, bool, index_t, index_t, T>;

template <typename scalar_t, blas::helper::AllocType mem_alloc>
void run_test(const combination_t<scalar_t> combi) {
  std::string alloc;
  index_t n;
  bool trans;
  bool is_upper;
  bool is_unit;
  index_t incX;
  index_t lda_mul;
  scalar_t unused; /* Work around dpcpp compiler bug
                      (https://github.com/intel/llvm/issues/7075) */
  std::tie(alloc, n, is_upper, trans, is_unit, incX, lda_mul, unused) = combi;

  const char* t_str = trans ? "t" : "n";
  const char* uplo_str = is_upper ? "u" : "l";
  const char* diag_str = is_unit ? "u" : "n";

  index_t a_size = n * n * lda_mul;
  index_t x_size = 1 + (n - 1) * incX;

  // Input matrix
  std::vector<scalar_t> a_m(a_size);
  // Input/output vector
  std::vector<scalar_t> x_v(x_size);
  // Input/output system vector
  std::vector<scalar_t> x_v_cpu(x_size);

  // Control the magnitude of extra-diagonal elements
  for (index_t i = 0; i < n; ++i)
    for (index_t j = 0; j < n; ++j)
      a_m[(j * n * lda_mul) + i] =
          ((!is_upper && (i > j)) || (is_upper && (i < j)))
              ? random_scalar(scalar_t(-10), scalar_t(10)) / scalar_t(n)
              : NAN;

  if (!is_unit) {
    // Populate main diagonal with dominant elements
    for (index_t i = 0; i < n; ++i)
      a_m[(i * n * lda_mul) + i] = random_scalar(scalar_t(9), scalar_t(11));
  }

  fill_random(x_v);
  x_v_cpu = x_v;

  // SYSTEM TRSV
  reference_blas::trsv(uplo_str, t_str, diag_str, n, a_m.data(), n * lda_mul,
                       x_v_cpu.data(), incX);

  auto q = make_queue();
  blas::SB_Handle sb_handle(q);
  auto m_a_gpu = blas::helper::allocate<mem_alloc, scalar_t>(a_size, q);
  auto v_x_gpu = blas::helper::allocate<mem_alloc, scalar_t>(x_size, q);

  auto copy_m =
      blas::helper::copy_to_device<scalar_t>(q, a_m.data(), m_a_gpu, a_size);
  auto copy_v =
      blas::helper::copy_to_device<scalar_t>(q, x_v.data(), v_x_gpu, x_size);

  try {
    // SYCL TRSV
    auto trsv_event = _trsv(sb_handle, *uplo_str, *t_str, *diag_str, n, m_a_gpu,
                            n * lda_mul, v_x_gpu, incX, {copy_m, copy_v});
    sb_handle.wait(trsv_event);
  } catch (const blas::unimplemented_exception& ue) {
    GTEST_SKIP();
  }

  auto event = blas::helper::copy_to_host(sb_handle.get_queue(), v_x_gpu,
                                          x_v.data(), x_size);
  sb_handle.wait(event);

  const bool isAlmostEqual = utils::compare_vectors(x_v, x_v_cpu);
  ASSERT_TRUE(isAlmostEqual);

  blas::helper::deallocate<mem_alloc>(m_a_gpu, q);
  blas::helper::deallocate<mem_alloc>(v_x_gpu, q);
}

template <typename scalar_t>
void run_test(const combination_t<scalar_t> combi) {
  std::string alloc;
  index_t n;
  bool trans;
  bool is_upper;
  bool is_unit;
  index_t incX;
  index_t lda_mul;
  scalar_t wa;
  std::tie(alloc, n, is_upper, trans, is_unit, incX, lda_mul, wa) = combi;

  if (alloc == "usm") {
#ifdef SB_ENABLE_USM
    run_test<scalar_t, blas::helper::AllocType::usm>(combi);
#else
    GTEST_SKIP();
#endif
  } else {
    run_test<scalar_t, blas::helper::AllocType::buffer>(combi);
  }
}

#ifdef STRESS_TESTING
template <typename scalar_t>
const auto combi =
    ::testing::Combine(::testing::Values("usm", "buffer"),  // allocation type
                       ::testing::Values(32, 64, 128, 512, 14, 127, 504, 780,
                                         1010, 1140, 2300, 8192),  // n
                       ::testing::Values(true, false),  // is_upper
                       ::testing::Values(true, false),  // trans
                       ::testing::Values(true, false),  // is_unit
                       ::testing::Values(1, 2),         // incX
                       ::testing::Values(1, 2),         // lda_mul
                       ::testing::Values(0)             // unused
    );
#else
// For the purpose of travis and other slower platforms, we need a faster test
// (the stress_test above takes about ~5 minutes)
template <typename scalar_t>
const auto combi =
    ::testing::Combine(::testing::Values("usm", "buffer"),  // allocation type
                       ::testing::Values(14, 64, 33, 515, 1024, 1200, 3000),  // n
                       ::testing::Values(true, false),             // is_upper
                       ::testing::Values(true, false),             // trans
                       ::testing::Values(true, false),             // is_unit
                       ::testing::Values(4),                       // incX
                       ::testing::Values(3),                       // lda_mul
                       ::testing::Values(0)                        // unused
    );
#endif

template <class T>
static std::string generate_name(
    const ::testing::TestParamInfo<combination_t<T>>& info) {
  std::string alloc;
  index_t n, incX, ldaMul;
  bool is_upper;
  bool trans;
  bool is_unit;
  T unused;
  BLAS_GENERATE_NAME(info.param, alloc, n, is_upper, trans, is_unit, incX, ldaMul,
                     unused);
}

BLAS_REGISTER_TEST_ALL(Trsv, combination_t, combi, generate_name);
