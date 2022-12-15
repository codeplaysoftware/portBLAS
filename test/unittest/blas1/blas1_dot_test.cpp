/***************************************************************************
 *
 *  @license
 *  Dotright (C) Codeplay Software Limited
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a dot of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  For your convenience, a dot of the License has been included in this
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
 *  @filename blas1_dot_test.cpp
 *
 **************************************************************************/

#include "blas_test.hpp"

template <typename scalar_t>
using combination_t = std::tuple<int, int, int>;

template <typename scalar_t>
void run_test(const combination_t<scalar_t> combi) {
  index_t size;
  index_t incX;
  index_t incY;
  std::tie(size, incX, incY) = combi;

  // Input vectors
  std::vector<scalar_t> x_v(size * incX);
  fill_random(x_v);
  std::vector<scalar_t> y_v(size * incY);
  fill_random(y_v);

  // Output vector
  std::vector<scalar_t> out_s(1, 10.0);

  // Reference implementation
  auto out_cpu_s =
      reference_blas::dot(size, x_v.data(), incX, y_v.data(), incY);

  // SYCL implementation
  auto q = make_queue();
  blas::SB_Handle sb_handle(q);

  // Iterators
  auto gpu_x_v = blas::make_sycl_iterator_buffer<scalar_t>(x_v, size * incX);
  auto gpu_y_v = blas::make_sycl_iterator_buffer<scalar_t>(y_v, size * incY);
  auto gpu_out_s = blas::make_sycl_iterator_buffer<scalar_t>(out_s, 1);

  _dot(sb_handle, size, gpu_x_v, incX, gpu_y_v, incY, gpu_out_s);
  auto event = blas::helper::copy_to_host(sb_handle.get_queue(), gpu_out_s,
                                          out_s.data(), 1);
  sb_handle.wait(event);

  // Validate the result
  const bool isAlmostEqual = utils::almost_equal(out_s[0], out_cpu_s);
  ASSERT_TRUE(isAlmostEqual);
}

#ifdef STRESS_TESTING
template <typename scalar_t>
const auto combi = ::testing::Combine(::testing::Values(11, 65, 1002,
                                                        1002400),  // size
                                      ::testing::Values(1, 4),     // incX
                                      ::testing::Values(1, 3)      // incY
);
#else
template <typename scalar_t>
const auto combi = ::testing::Combine(::testing::Values(11, 1002),  // size
                                      ::testing::Values(1, 4),      // incX
                                      ::testing::Values(1, 3)       // incY
);
#endif

template <class T>
static std::string generate_name(
    const ::testing::TestParamInfo<combination_t<T>>& info) {
  int size, incX, incY;
  BLAS_GENERATE_NAME(info.param, size, incX, incY);
}

BLAS_REGISTER_TEST_ALL(Dot, combination_t, combi, generate_name);
