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
 *  @filename blas2_spr2_test.cpp
 *
 **************************************************************************/

#include "blas_test.hpp"

template <typename scalar_t>
using combination_t =
    std::tuple<char, char, index_t, scalar_t, index_t, index_t>;

template <typename scalar_t>
void run_test(const combination_t<scalar_t> combi) {
  int n;
  scalar_t alpha;
  char layout, uplo;
  int incX, incY;
  std::tie(layout, uplo, n, alpha, incX, incY) = combi;

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

  auto vx_gpu = blas::make_sycl_iterator_buffer(vx_cpu, x_size);
  auto vy_gpu = blas::make_sycl_iterator_buffer(vy_cpu, y_size);

  auto a_mp_gpu = blas::make_sycl_iterator_buffer(a_mp, m_size);

  _spr2(sb_handle, uplo, n, alpha, vx_gpu, incX, vy_gpu, incY, a_mp_gpu);

  auto event = helper::copy_to_host(sb_handle.get_queue(), a_mp_gpu,
                                    a_mp.data(), m_size);

  sb_handle.wait(event);

  const bool isAlmostEqual = utils::compare_vectors(a_mp, a_cpu_mp);
  ASSERT_TRUE(isAlmostEqual);
}

#ifdef STRESS_TESTING
template <typename scalar_t>
const auto combi =
    ::testing::Combine(::testing::Values('r', 'c'),  // matrix layout
                       ::testing::Values('u', 'l'),  // UPLO
                       ::testing::Values(1024, 2048, 4096, 8192, 16384),  // n
                       ::testing::Values<scalar_t>(0.0, 1.0, 1.5),  // alpha
                       ::testing::Values(1, 2),                     // incX
                       ::testing::Values(1, 2)                      // incY
    );
#else
// For the purpose of travis and other slower platforms, we need a faster test
template <typename scalar_t>
const auto combi =
    ::testing::Combine(::testing::Values('r', 'c'),           // matrix layout
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
  char layout, uplo;
  int n, incX, incY;
  T alpha;
  BLAS_GENERATE_NAME(info.param, layout, uplo, n, alpha, incX, incY);
}

BLAS_REGISTER_TEST_ALL(Spr2, combination_t, combi, generate_name);
