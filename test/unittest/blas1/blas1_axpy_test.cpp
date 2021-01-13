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
 *  @filename blas1_axpy_test.cpp
 *
 **************************************************************************/

#include "blas_test.hpp"

template <typename scalar_t>
using combination_t = std::tuple<int, scalar_t, int, int>;

template <typename scalar_t>
void run_test(const combination_t<scalar_t> combi) {
  int size;
  scalar_t alpha;
  int incX;
  int incY;
  std::tie(size, alpha, incX, incY) = combi;

  using data_t = utils::data_storage_t<scalar_t>;

  // Input vector
  std::vector<data_t> x_v(size * incX);
  fill_random(x_v);

  // Output vector
  std::vector<data_t> y_v(size * incY, 10.0);
  std::vector<data_t> y_cpu_v(size * incY, 10.0);

  // Reference implementation
  reference_blas::axpy(size, static_cast<data_t>(alpha), x_v.data(), incX,
                       y_cpu_v.data(), incY);

  // SYCL implementation
  auto q = make_queue();
  test_executor_t ex(q);

  // Iterators
  auto gpu_x_v = utils::make_quantized_buffer<scalar_t>(ex, x_v);
  auto gpu_y_v = utils::make_quantized_buffer<scalar_t>(ex, y_v);

  _axpy(ex, size, alpha, gpu_x_v, incX, gpu_y_v, incY);
  auto event = utils::quantized_copy_to_host<scalar_t>(ex, gpu_y_v, y_v);
  ex.get_policy_handler().wait(event);

  // Validate the result
  const bool isAlmostEqual =
      utils::compare_vectors<data_t, scalar_t>(y_v, y_cpu_v);
  ASSERT_TRUE(isAlmostEqual);
}

#ifdef STRESS_TESTING
const auto combi =
    ::testing::Combine(::testing::Values(11, 65, 1002, 1002400),  // size
                       ::testing::Values(0.0, 1.0, 1.5),          // alpha
                       ::testing::Values(1, 4),                   // incX
                       ::testing::Values(1, 3)                    // incY
    );
#else
const auto combi = ::testing::Combine(::testing::Values(11, 1002),  // size
                                      ::testing::Values(0.0, 1.5),  // alpha
                                      ::testing::Values(1, 4),      // incX
                                      ::testing::Values(1, 3)       // incY
);
#endif

BLAS_REGISTER_TEST(Axpy, combination_t, combi);
