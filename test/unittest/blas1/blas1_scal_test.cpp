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
 *  @filename blas1_scal_test.cpp
 *
 **************************************************************************/

#include "blas_test.hpp"

template <typename scalar_t>
using combination_t = std::tuple<int, scalar_t, int>;

template <typename scalar_t>
void run_test(const combination_t<scalar_t> combi) {
  using type_t = blas_test_args<scalar_t, void>;
  using blas_test_t = BLAS_Test<type_t>;
  using executor_t = typename type_t::executor_t;

  int size;
  scalar_t alpha;
  int incX;
  std::tie(size, alpha, incX) = combi;

  // Input/output vector
  std::vector<scalar_t> x_v(size * incX);
  std::vector<scalar_t> x_cpu_v(x_v);

  // Reference implementation
  reference_blas::scal(size, alpha, x_cpu_v.data(), incX);

  // SYCL implementation
  SYCL_DEVICE_SELECTOR d;
  auto q = blas_test_t::make_queue(d);
  Executor<executor_t> ex(q);

  // Iterators
  auto gpu_x_v = blas::make_sycl_iterator_buffer<scalar_t>(int(size * incX));
  ex.get_policy_handler().copy_to_device(x_v.data(), gpu_x_v, size * incX);

  _scal(ex, size, alpha, gpu_x_v, incX);
  auto event =
      ex.get_policy_handler().copy_to_host(gpu_x_v, x_v.data(), size * incX);
  ex.get_policy_handler().wait(event);

  // Validate the result
  ASSERT_TRUE(utils::compare_vectors(x_v, x_cpu_v));
}

#ifdef STRESS_TESTING
const auto combi =
    ::testing::Combine(::testing::Values(11, 65, 1002, 1002400),  // size
                       ::testing::Values(0.0, 1.0, 1.5),          // alpha
                       ::testing::Values(1, 4)                    // incX
    );
#else
const auto combi = ::testing::Combine(::testing::Values(11, 1002),  // size
                                      ::testing::Values(0.0, 1.5),  // alpha
                                      ::testing::Values(4)          // incX
);
#endif

class ScalFloat : public ::testing::TestWithParam<combination_t<float>> {};
TEST_P(ScalFloat, test) { run_test<float>(GetParam()); };
INSTANTIATE_TEST_SUITE_P(scal, ScalFloat, combi);

#if DOUBLE_SUPPORT
class ScalDouble : public ::testing::TestWithParam<combination_t<double>> {};
TEST_P(ScalDouble, test) { run_test<double>(GetParam()); };
INSTANTIATE_TEST_SUITE_P(scal, ScalDouble, combi);
#endif
