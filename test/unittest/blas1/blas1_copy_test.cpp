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
 *  @filename blas1_copy_test.cpp
 *
 **************************************************************************/

#include "blas_test.hpp"

using combination_t = std::tuple<int, int, int>;

template <typename scalar_t>
void run_test(const combination_t combi) {
  using type_t = blas_test_args<scalar_t, void>;
  using blas_test_t = BLAS_Test<type_t>;
  using executor_t = typename type_t::executor_t;

  int size;
  int incX;
  int incY;
  std::tie(size, incX, incY) = combi;

  // Input vector
  std::vector<scalar_t> x_v(size * incX);
  fill_random(x_v);

  // Output vector
  std::vector<scalar_t> y_v(size * incY, 10.0);
  std::vector<scalar_t> y_cpu_v(size * incY, 10.0);

  // Reference implementation
  reference_blas::copy(size, x_v.data(), incX, y_cpu_v.data(), incY);

  // SYCL implementation
  SYCL_DEVICE_SELECTOR d;
  auto q = blas_test_t::make_queue(d);
  Executor<executor_t> ex(q);

  // Iterators
  auto gpu_x_v = blas::make_sycl_iterator_buffer<scalar_t>(int(size * incX));
  ex.get_policy_handler().copy_to_device(x_v.data(), gpu_x_v, size * incX);
  auto gpu_y_v = blas::make_sycl_iterator_buffer<scalar_t>(int(size * incY));
  ex.get_policy_handler().copy_to_device(y_v.data(), gpu_y_v, size * incY);

  _copy(ex, size, gpu_x_v, incX, gpu_y_v, incY);
  auto event =
      ex.get_policy_handler().copy_to_host(gpu_y_v, y_v.data(), size * incY);
  ex.get_policy_handler().wait(event);

  // Validate the result
  ASSERT_TRUE(utils::compare_vectors(y_v, y_cpu_v));
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

class CopyFloat : public ::testing::TestWithParam<combination_t> {};
TEST_P(CopyFloat, test) { run_test<float>(GetParam()); };
INSTANTIATE_TEST_SUITE_P(copy, CopyFloat, combi);

#if DOUBLE_SUPPORT
class CopyDouble : public ::testing::TestWithParam<combination_t> {};
TEST_P(CopyDouble, test) { run_test<double>(GetParam()); };
INSTANTIATE_TEST_SUITE_P(copy, CopyDouble, combi);
#endif
