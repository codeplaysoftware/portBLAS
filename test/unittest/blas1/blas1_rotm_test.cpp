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
 *  @filename blas1_rotm_test.cpp
 *
 **************************************************************************/

#include "blas_test.hpp"

template <typename scalar_t>
using combination_t = std::tuple<int, int, int, scalar_t>;

template <typename scalar_t>
void run_test(const combination_t<scalar_t> combi) {
  index_t size;
  index_t incX;
  index_t incY;
  scalar_t flag;
  std::tie(size, incX, incY, flag) = combi;

  using data_t = utils::data_storage_t<scalar_t>;

  // Setup param
  constexpr size_t param_size = 5;
  std::vector<data_t> param(param_size);
  fill_random(param);
  param[0] = flag;

  // Input vectors
  std::vector<data_t> x_v(size * incX);
  fill_random(x_v);

  std::vector<data_t> y_v(size * incY);
  fill_random(y_v);

  // Output vectors
  std::vector<data_t> out_s(1, 10.0);
  std::vector<data_t> x_cpu_v(x_v);
  std::vector<data_t> y_cpu_v(y_v);

  // Reference implementation
  reference_blas::rotm(size, x_cpu_v.data(), incX, y_cpu_v.data(), incY,
                       param.data());

  // SYCL implementation
  auto q = make_queue();
  test_executor_t ex(q);

  // Iterators
  auto gpu_x_v = utils::make_quantized_buffer<scalar_t>(ex, x_v);
  auto gpu_y_v = utils::make_quantized_buffer<scalar_t>(ex, y_v);
  auto gpu_param = utils::make_quantized_buffer<scalar_t>(ex, param);

  _rotm(ex, size, gpu_x_v, incX, gpu_y_v, incY, gpu_param);

  auto event1 = utils::quantized_copy_to_host<scalar_t>(ex, gpu_x_v, x_v);
  auto event2 = utils::quantized_copy_to_host<scalar_t>(ex, gpu_y_v, y_v);
  ex.get_policy_handler().wait(event1);
  ex.get_policy_handler().wait(event2);

  // Validate the result
  const bool isAlmostEqual =
      std::equal(x_cpu_v.begin(), x_cpu_v.end(), x_v.begin()) &&
      std::equal(y_cpu_v.begin(), y_cpu_v.end(), y_v.begin());
  ASSERT_TRUE(isAlmostEqual);
}

#ifdef STRESS_TESTING
const auto combi =
    ::testing::Combine(::testing::Values(11, 65, 1002, 1002400),        // size
                       ::testing::Values(1, 4),                         // incX
                       ::testing::Values(1, 3),                         // incY
                       ::testing::Values(-2.0, -1.0, 0.0, 1.0, -4.0)    // flag
    );
#else
const auto combi =
    ::testing::Combine(::testing::Values(11, 1002),                      // size
                       ::testing::Values(4),                             // incX
                       ::testing::Values(3),                             // incY
                       ::testing::Values(-2.0, -1.0, 0.0, 1.0, -4.0)     // flag
    );
#endif

template <class T>
static std::string generate_name(
    const ::testing::TestParamInfo<combination_t<T>>& info) {
  int size, incX, incY;
  T flag;
  BLAS_GENERATE_NAME(info.param, size, incX, incY, flag);
}

BLAS_REGISTER_TEST_ALL(Rotm, combination_t, combi, generate_name);
