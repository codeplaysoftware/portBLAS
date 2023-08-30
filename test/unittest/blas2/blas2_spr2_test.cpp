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
 *  @filename blas2_spr2_test.cpp
 *
 **************************************************************************/

#include "blas_test.hpp"

template <typename scalar_t>
using combination_t =
    std::tuple<std::string, char, char, index_t, scalar_t, index_t, index_t>;

template <typename scalar_t, helper::AllocType mem_alloc>
void run_test(const combination_t<scalar_t> combi) {
  std::string alloc;
  index_t n;
  scalar_t alpha;
  char layout, uplo;
  index_t incX, incY;
  std::tie(alloc, layout, uplo, n, alpha, incX, incY) = combi;

  const size_t x_size = 1 + (n - 1) * incX;
  const size_t y_size = 1 + (n - 1) * incY;
  const size_t m_size = n * n;

  // Input vector
  std::vector<scalar_t> vx_cpu(x_size);
  fill_random(vx_cpu);

  std::vector<scalar_t> vy_cpu(y_size);
  fill_random(vy_cpu);

  // Output matrix
  std::vector<scalar_t> a_mp(m_size, 7.0);
  std::vector<scalar_t> a_cpu_mp(m_size, 7.0);

  uplo = (uplo == 'u' && layout == 'c') || (uplo == 'l' && layout == 'r') ? 'u'
                                                                          : 'l';
  // SYSTEM SPR2
  reference_blas::spr2<scalar_t>(&uplo, n, alpha, vx_cpu.data(), incX,
                                 vy_cpu.data(), incY, a_cpu_mp.data());

  auto q = make_queue();
  SB_Handle sb_handle(q);

  auto vx_gpu = helper::allocate<mem_alloc, scalar_t>(x_size, q);
  auto vy_gpu = helper::allocate<mem_alloc, scalar_t>(y_size, q);

  auto a_mp_gpu = helper::allocate<mem_alloc, scalar_t>(m_size, q);

  auto copy_x =
      helper::copy_to_device<scalar_t>(q, vx_cpu.data(), vx_gpu, x_size);
  auto copy_y =
      helper::copy_to_device<scalar_t>(q, vy_cpu.data(), vy_gpu, y_size);
  auto copy_a =
      helper::copy_to_device<scalar_t>(q, a_mp.data(), a_mp_gpu, m_size);

  auto spr2_event = _spr2(sb_handle, uplo, n, alpha, vx_gpu, incX, vy_gpu, incY,
                          a_mp_gpu, {copy_x, copy_y, copy_a});

  sb_handle.wait(spr2_event);

  auto event = helper::copy_to_host(sb_handle.get_queue(), a_mp_gpu,
                                    a_mp.data(), m_size);

  sb_handle.wait(event);

  const bool isAlmostEqual = utils::compare_vectors(a_mp, a_cpu_mp);
  ASSERT_TRUE(isAlmostEqual);

  helper::deallocate<mem_alloc>(vx_gpu, q);
  helper::deallocate<mem_alloc>(vy_gpu, q);
  helper::deallocate<mem_alloc>(a_mp_gpu, q);
}

template <typename scalar_t>
void run_test(const combination_t<scalar_t> combi) {
  std::string alloc;
  index_t n;
  scalar_t alpha;
  char layout, uplo;
  index_t incX, incY;
  std::tie(alloc, layout, uplo, n, alpha, incX, incY) = combi;

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
                       ::testing::Values(1, 2),                     // incX
                       ::testing::Values(1, 2)                      // incY
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
                       ::testing::Values(1, 2),               // incX
                       ::testing::Values(1, 2)                // incY
    );
#endif

template <class T>
static std::string generate_name(
    const ::testing::TestParamInfo<combination_t<T>>& info) {
  std::string alloc;
  char layout, uplo;
  index_t n, incX, incY;
  T alpha;
  BLAS_GENERATE_NAME(info.param, alloc, layout, uplo, n, alpha, incX, incY);
}

BLAS_REGISTER_TEST_ALL(Spr2, combination_t, combi, generate_name);
