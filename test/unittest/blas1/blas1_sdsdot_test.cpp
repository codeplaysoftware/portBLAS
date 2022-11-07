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
#include <iostream>
#include <type_traits>

enum class api_type : int { event = 0, result = 1 };

/**
 * Operator overload to print the test name properly
 */
std::ostream& operator<<(std::ostream& os, const api_type& type) {
  if (type == api_type::event) {
    return os << "event";
  } else {
    return os << "result";
  }
}

template <typename scalar_t>
using combination_t = std::tuple<api_type, int, float, int, int>;

template <typename scalar_t>
void run_test(const combination_t<scalar_t> combi) {
  /* sdsdot is only valid when using floats */
  static_assert(std::is_same<scalar_t, float>::value);

  index_t N;
  float sb;
  index_t incX;
  index_t incY;
  api_type api;
  std::tie(api, N, sb, incX, incY) = combi;

  using data_t = utils::data_storage_t<scalar_t>;

  /* Sycl Buffers do not work with size = 0. So setting input vectors to size
   * one to test the edge case where if size equals 0 then sb should be
   * returned. */
  index_t vectorSize = N > 0 ? N : 1;

  // Input vectors
  std::vector<data_t> x_v(vectorSize * incX);
  fill_random(x_v);
  std::vector<data_t> y_v(vectorSize * incY);
  fill_random(y_v);

  // Output vector
  std::vector<data_t> out_s(1, 10.0);

  // Reference implementation
  auto out_cpu_s =
      reference_blas::sdsdot(N, sb, x_v.data(), incX, y_v.data(), incY);

  // SYCL implementation
  auto q = make_queue();
  test_executor_t ex(q);

  // Iterators
  auto gpu_x_v = utils::make_quantized_buffer<scalar_t>(ex, x_v);
  auto gpu_y_v = utils::make_quantized_buffer<scalar_t>(ex, y_v);

  if (api == api_type::event) {
    auto gpu_out_s = utils::make_quantized_buffer<scalar_t>(ex, out_s);
    _sdsdot(ex, N, sb, gpu_x_v, incX, gpu_y_v, incY, gpu_out_s);
    auto event = utils::quantized_copy_to_host<scalar_t>(ex, gpu_out_s, out_s);
    ex.get_policy_handler().wait(event);
  } else {
    out_s[0] = _sdsdot(ex, N, sb, gpu_x_v, incX, gpu_y_v, incY);
  }

  // Validate the result
  const bool isAlmostEqual =
      utils::almost_equal<data_t, scalar_t>(out_s[0], out_cpu_s);
  ASSERT_TRUE(isAlmostEqual);

  ex.get_policy_handler().get_queue().wait();
}

#ifdef STRESS_TESTING
const auto combi = ::testing::Combine(
    ::testing::Values(api_type::event, api_type::result),  // Api
    ::testing::Values(11, 65, 1002, 1002400),              // N
    ::testing::Values(9.5f, 0.5f),                         // sb
    ::testing::Values(1, 4),                               // incX
    ::testing::Values(1, 3)                                // incY
);
#else
const auto combi = ::testing::Combine(
    ::testing::Values(api_type::event, api_type::result),  // Api
    ::testing::Values(11, 1002, 0),                        // N
    ::testing::Values(9.5f, 0.5f, 0.0f),                   // sb
    ::testing::Values(1, 4),                               // incX
    ::testing::Values(1, 3)                                // incY

);
#endif

template <class T>
static std::string generate_name(
    const ::testing::TestParamInfo<combination_t<T>>& info) {
  int size, incX, incY;
  float sb;
  api_type api;
  BLAS_GENERATE_NAME(info.param, api, size, sb, incX, incY);
}

BLAS_REGISTER_TEST_FLOAT(Sdsdot, combination_t, combi, generate_name);
