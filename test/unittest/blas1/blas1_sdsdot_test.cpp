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
 *  portBLAS: BLAS implementation using SYCL
 *
 *  @filename blas1_sdsdot_test.cpp
 *
 **************************************************************************/

#include "blas_test.hpp"
#include <iostream>
#include <type_traits>

template <typename scalar_t>
using combination_t =
    std::tuple<std::string, api_type, int, scalar_t, int, int>;

template <typename scalar_t, helper::AllocType mem_alloc>
void run_test(const combination_t<scalar_t> combi) {
  std::string alloc;
  index_t N;
  float sb;
  index_t incX;
  index_t incY;
  api_type api;
  std::tie(alloc, api, N, sb, incX, incY) = combi;

  /* Sycl Buffers do not work with size = 0. So setting input vectors to size
   * one to test the edge case where if size equals 0 then sb should be
   * returned. */
  index_t vectorSize = N > 0 ? N : 1;

  // Input vectors
  std::vector<scalar_t> x_v(vectorSize * incX);
  fill_random(x_v);
  std::vector<scalar_t> y_v(vectorSize * incY);
  fill_random(y_v);

  // Output scalar
  scalar_t out_s = 10.0;

  // Reference implementation
  auto out_cpu_s =
      reference_blas::sdsdot(N, sb, x_v.data(), incX, y_v.data(), incY);

  // SYCL implementation
  auto q = make_queue();
  blas::SB_Handle sb_handle(q);

  // Iterators
  auto gpu_x_v =
      helper::allocate<mem_alloc, scalar_t>(int(vectorSize * incX), q);
  auto gpu_y_v =
      helper::allocate<mem_alloc, scalar_t>(int(vectorSize * incY), q);

  auto copy_x =
      helper::copy_to_device(q, x_v.data(), gpu_x_v, vectorSize * incX);
  auto copy_y =
      helper::copy_to_device(q, y_v.data(), gpu_y_v, vectorSize * incY);

  if (api == api_type::async) {
    auto gpu_out_s = helper::allocate<mem_alloc, scalar_t>(1, q);
    auto copy_out = helper::copy_to_device(q, &out_s, gpu_out_s, 1);
    auto sdsdot_event = _sdsdot(sb_handle, N, sb, gpu_x_v, incX, gpu_y_v, incY,
                                gpu_out_s, {copy_x, copy_y, copy_out});
    sb_handle.wait(sdsdot_event);
    auto event = helper::copy_to_host<scalar_t>(sb_handle.get_queue(),
                                                gpu_out_s, &out_s, 1);
    sb_handle.wait(event);
    helper::deallocate<mem_alloc>(gpu_out_s, q);
  } else {
    out_s = _sdsdot(sb_handle, N, sb, gpu_x_v, incX, gpu_y_v, incY,
                    {copy_x, copy_y});
  }

  // Validate the result
  const bool isAlmostEqual = utils::almost_equal(out_s, out_cpu_s);
  ASSERT_TRUE(isAlmostEqual);

  helper::deallocate<mem_alloc>(gpu_x_v, q);
  helper::deallocate<mem_alloc>(gpu_y_v, q);
}

template <typename scalar_t>
void run_test(const combination_t<scalar_t> combi) {
  /* sdsdot is only valid when using floats */
  static_assert(std::is_same<scalar_t, float>::value);

  std::string alloc;
  index_t N;
  float sb;
  index_t incX;
  index_t incY;
  api_type api;
  std::tie(alloc, api, N, sb, incX, incY) = combi;

  if (alloc == "usm") {  // usm alloc
#ifdef SB_ENABLE_USM
    run_test<scalar_t, helper::AllocType::usm>(combi);
#else
    GTEST_SKIP();
#endif
  } else {  // buffer alloc
    run_test<scalar_t, helper::AllocType::buffer>(combi);
  }
}

#ifdef STRESS_TESTING
template <typename scalar_t>
const auto combi = ::testing::Combine(
    ::testing::Values("usm", "buf"),                     // allocation type
    ::testing::Values(api_type::async, api_type::sync),  // Api
    ::testing::Values(11, 65, 1002, 1002400),            // N
    ::testing::Values<scalar_t>(9.5f, 0.5f),             // sb
    ::testing::Values(1, 4),                             // incX
    ::testing::Values(1, 3)                              // incY
);
#else
template <typename scalar_t>
const auto combi = ::testing::Combine(
    ::testing::Values("usm", "buf"),                     // allocation type
    ::testing::Values(api_type::async, api_type::sync),  // Api
    ::testing::Values(11, 1002, 0),                      // N
    ::testing::Values<scalar_t>(9.5f, 0.5f, 0.0f),       // sb
    ::testing::Values(1, 4),                             // incX
    ::testing::Values(1, 3)                              // incY

);
#endif

template <class T>
static std::string generate_name(
    const ::testing::TestParamInfo<combination_t<T>>& info) {
  std::string alloc;
  int size, incX, incY;
  float sb;
  api_type api;
  BLAS_GENERATE_NAME(info.param, alloc, api, size, sb, incX, incY);
}

BLAS_REGISTER_TEST_FLOAT(Sdsdot, combination_t, combi, generate_name);
