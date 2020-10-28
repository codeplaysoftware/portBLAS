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
 *  @filename blas1_swap_test.cpp
 *
 **************************************************************************/

#include "blas_test.hpp"

template <typename scalar_t>
using combination_t = std::tuple<int, int, int>;

template <typename scalar_t>
void run_test(const combination_t<scalar_t> combi) {
  int size;
  int incX;
  int incY;
  std::tie(size, incX, incY) = combi;

  using data_t = utils::data_storage_t<scalar_t>;

  // Input/Output vector
  std::vector<data_t> x_v(size * incX);
  fill_random(x_v);
  std::vector<data_t> x_cpu_v(x_v);

  std::vector<data_t> y_v(size * incY);
  fill_random(y_v);
  std::vector<data_t> y_cpu_v(y_v);

  // Reference implementation
  reference_blas::swap(size, x_cpu_v.data(), incX, y_cpu_v.data(), incY);

  // SYCL implementation
  auto q = make_queue();
  test_executor_t ex(q);

  // Iterators
  auto gpu_x_v = utils::make_quantized_buffer<scalar_t>(ex, x_v);
  auto gpu_y_v = utils::make_quantized_buffer<scalar_t>(ex, y_v);

  _swap(ex, size, gpu_x_v, incX, gpu_y_v, incY);
  auto event = utils::quantized_copy_to_host<scalar_t>(ex, gpu_x_v, x_v);
  ex.get_policy_handler().wait(event);
  event = utils::quantized_copy_to_host<scalar_t>(ex, gpu_y_v, y_v);
  ex.get_policy_handler().wait(event);

  // Validate the result
  // Since this is just a swap operation, float tolerances are fine
  ASSERT_TRUE(utils::compare_vectors(y_v, y_cpu_v));
  ASSERT_TRUE(utils::compare_vectors(x_v, x_cpu_v));
}

#ifdef STRESS_TESTING
const auto combi =
    ::testing::Combine(::testing::Values(11, 65, 1002, 1002400),  // size
                       ::testing::Values(1, 4),                   // incX
                       ::testing::Values(1, 3)                    // incY
    );
#else
const auto combi = ::testing::Combine(::testing::Values(11, 1002),  // size
                                      ::testing::Values(1, 4),      // incX
                                      ::testing::Values(1, 3)       // incY
);
#endif

BLAS_REGISTER_TEST(Swap, combination_t, combi);
