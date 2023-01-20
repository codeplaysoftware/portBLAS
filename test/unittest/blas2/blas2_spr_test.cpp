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
 *  @filename blas2_spr_test.cpp
 *
 **************************************************************************/

#include "blas_test.hpp"

template <typename scalar_t>
using combination_t = std::tuple<char, char, index_t, scalar_t, index_t>;

template <typename scalar_t>
void run_test(const combination_t<scalar_t> combi) {
  index_t n;
  index_t lda_mul;
  index_t incX;
  char layout, uplo;
  scalar_t alpha;
  std::tie(layout, uplo, n, alpha, incX) = combi;
  index_t mA_size = n * n;
  index_t x_size = 1 + (n - 1) * std::abs(incX);

  // Input vector
  std::vector<scalar_t> x_v(x_size);
  fill_random(x_v);

  // Output matrix
  std::vector<scalar_t> a_mp(mA_size, 7.0);
  std::vector<scalar_t> a_cpu_mp(mA_size, 7.0);

  // SYSTEM SPR
  if (layout == 'c') {
    reference_blas::spr<scalar_t, true>(&uplo, n, alpha, x_v.data(), incX,
                                        a_cpu_mp.data());
  } else {
    reference_blas::spr<scalar_t, false>(&uplo, n, alpha, x_v.data(), incX,
                                         a_cpu_mp.data());
  }

  auto q = make_queue();
  blas::SB_Handle sb_handle(q);
  auto x_v_gpu = blas::make_sycl_iterator_buffer<scalar_t>(x_v, x_size);
  auto a_mp_gpu = blas::make_sycl_iterator_buffer<scalar_t>(a_mp, mA_size);

  // SYCLspr
  if (layout == 'c') {
    _spr<blas::SB_Handle, index_t, scalar_t, decltype(x_v_gpu), index_t,
         decltype(a_mp_gpu), col_major>(sb_handle, uplo, n, alpha, x_v_gpu,
                                        incX, a_mp_gpu);
  } else {
    _spr<blas::SB_Handle, index_t, scalar_t, decltype(x_v_gpu), index_t,
         decltype(a_mp_gpu), row_major>(sb_handle, uplo, n, alpha, x_v_gpu,
                                        incX, a_mp_gpu);
  }

  auto event = blas::helper::copy_to_host(sb_handle.get_queue(), a_mp_gpu,
                                          a_mp.data(), mA_size);
  sb_handle.wait(event);

  const bool isAlmostEqual = utils::compare_vectors(a_mp, a_cpu_mp);
  ASSERT_TRUE(isAlmostEqual);
}

#ifdef STRESS_TESTING
template <typename scalar_t>
const auto combi =
    ::testing::Combine(::testing::Values('r', 'c'),  // matrix layout
                       ::testing::Values('u', 'l'),  // UPLO
                       ::testing::Values(14, 63, 257, 1010, 2025),  // n
                       ::testing::Values<scalar_t>(0.0, 1.0, 1.5),  // alpha
                       ::testing::Values(1, 2)                      // incX
    );
#else
// For the purpose of travis and other slower platforms, we need a faster test
template <typename scalar_t>
const auto combi =
    ::testing::Combine(::testing::Values('r', 'c'),           // matrix layout
                       ::testing::Values('u', 'l'),           // UPLO
                       ::testing::Values(14, 63, 257, 1010),  // n
                       ::testing::Values<scalar_t>(1.0),      // alpha
                       ::testing::Values(1, 2)                // incX
    );
#endif

template <class T>
static std::string generate_name(
    const ::testing::TestParamInfo<combination_t<T>>& info) {
  char layout, upl0;
  int n, incX;
  T alpha;
  BLAS_GENERATE_NAME(info.param, layout, upl0, n, alpha, incX);
}

BLAS_REGISTER_TEST_ALL(Spr, combination_t, combi, generate_name);
