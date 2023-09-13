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
 *  @filename blas2_spr_test.cpp
 *
 **************************************************************************/

#include "blas_test.hpp"

template <typename scalar_t>
using combination_t =
    std::tuple<std::string, char, char, index_t, scalar_t, index_t>;

template <typename scalar_t, helper::AllocType mem_alloc>
void run_test(const combination_t<scalar_t> combi) {
  std::string alloc;
  index_t n;
  index_t lda_mul;
  index_t incX;
  char layout, uplo;
  scalar_t alpha;
  std::tie(alloc, layout, uplo, n, alpha, incX) = combi;
  index_t mA_size = n * n;
  index_t x_size = 1 + (n - 1) * std::abs(incX);

  // Input vector
  std::vector<scalar_t> x_v(x_size);
  fill_random(x_v);

  // Output matrix
  std::vector<scalar_t> a_mp(mA_size, 7.0);
  std::vector<scalar_t> a_cpu_mp(mA_size, 7.0);

  uplo = (uplo == 'u' && layout == 'c') || (uplo == 'l' && layout == 'r') ? 'u'
                                                                          : 'l';

  // SYSTEM SPR
  reference_blas::spr<scalar_t>(&uplo, n, alpha, x_v.data(), incX,
                                a_cpu_mp.data());

  auto q = make_queue();
  blas::SB_Handle sb_handle(q);
  auto x_v_gpu = helper::allocate<mem_alloc, scalar_t>(x_size, q);
  auto a_mp_gpu = helper::allocate<mem_alloc, scalar_t>(mA_size, q);

  auto copy_x =
      helper::copy_to_device<scalar_t>(q, x_v.data(), x_v_gpu, x_size);
  auto copy_a =
      helper::copy_to_device<scalar_t>(q, a_mp.data(), a_mp_gpu, mA_size);

  sb_handle.wait({copy_x, copy_a});

  // SYCLspr
  auto spr_event = _spr<blas::SB_Handle, index_t, scalar_t, decltype(x_v_gpu),
                        index_t, decltype(a_mp_gpu)>(sb_handle, uplo, n, alpha,
                                                     x_v_gpu, incX, a_mp_gpu);

  sb_handle.wait(spr_event);

  auto event = blas::helper::copy_to_host(q, a_mp_gpu, a_mp.data(), mA_size);
  sb_handle.wait(event);

  const bool isAlmostEqual = utils::compare_vectors(a_mp, a_cpu_mp);
  ASSERT_TRUE(isAlmostEqual);

  helper::deallocate<mem_alloc>(x_v_gpu, q);
  helper::deallocate<mem_alloc>(a_mp_gpu, q);
}

template <typename scalar_t>
void run_test(const combination_t<scalar_t> combi) {
  std::string alloc;
  index_t n;
  index_t lda_mul;
  index_t incX;
  char layout, uplo;
  scalar_t alpha;
  std::tie(alloc, layout, uplo, n, alpha, incX) = combi;

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
                       ::testing::Values('r', 'c'),      // matrix layout
                       ::testing::Values('u', 'l'),      // UPLO
                       ::testing::Values(1024, 2048, 4096, 8192, 16384),  // n
                       ::testing::Values<scalar_t>(0.0, 1.0, 1.5),  // alpha
                       ::testing::Values(1, 2)                      // incX
    );
#else
// For the purpose of travis and other slower platforms, we need a faster test
template <typename scalar_t>
const auto combi =
    ::testing::Combine(::testing::Values("usm", "buf"),       // allocation type
                       ::testing::Values('r', 'c'),           // matrix layout
                       ::testing::Values('u', 'l'),           // UPLO
                       ::testing::Values(14, 63, 257, 1010),  // n
                       ::testing::Values<scalar_t>(1.0),      // alpha
                       ::testing::Values(1, 2)                // incX
    );
#endif

template <class T>
static std::string generate_name(
    const ::testing::TestParamInfo<combination_t<T>>& info) {
  std::string alloc;
  char layout, uplo;
  int n, incX;
  T alpha;
  BLAS_GENERATE_NAME(info.param, alloc, layout, uplo, n, alpha, incX);
}

BLAS_REGISTER_TEST_ALL(Spr, combination_t, combi, generate_name);
