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
 *  @filename blas1_iamin_test.cpp
 *
 **************************************************************************/

#include "blas_test.hpp"
#include <limits>

using combination_t = std::tuple<int, int, bool>;

template <typename scalar_t>
void run_test(const combination_t combi) {
  using tuple_t = IndexValueTuple<scalar_t, int>;

  int size;
  int incX;
  bool all_max;
  std::tie(size, incX, all_max) = combi;

  const scalar_t max = std::numeric_limits<scalar_t>::max();

  // Input vector
  std::vector<scalar_t> x_v(size * incX, max);
  if (!all_max) {
    fill_random(x_v);
  }
  for (int i = 0; i < size * incX; i++) {
    // There is a bug in Openblas where 0s are not handled correctly
    if (x_v[i] == 0.0) {
      x_v[i] = 1.0;
    }
  }

  // Output scalar
  std::vector<tuple_t> out_s(1, tuple_t(0, max));

  // Reference implementation
  scalar_t out_cpu_s = reference_blas::iamin(size, x_v.data(), incX);

  // SYCL implementation
  auto q = make_queue();
  test_executor_t ex(q);

  // Iterators
  auto gpu_x_v = blas::make_sycl_iterator_buffer<scalar_t>(int(size * incX));
  ex.get_policy_handler().copy_to_device(x_v.data(), gpu_x_v, size * incX);
  auto gpu_out_s = blas::make_sycl_iterator_buffer<tuple_t>(int(1));
  ex.get_policy_handler().copy_to_device(out_s.data(), gpu_out_s, 1);

  _iamin(ex, size, gpu_x_v, incX, gpu_out_s);
  auto event = ex.get_policy_handler().copy_to_host(gpu_out_s, out_s.data(), 1);
  ex.get_policy_handler().wait(event);

  // Validate the result
  ASSERT_EQ(out_cpu_s, out_s[0].ind);
}

#ifdef STRESS_TESTING
const auto combi =
    ::testing::Combine(::testing::Values(11, 65, 10000, 1002400),  // size
                       ::testing::Values(1, 5),                    // incX
                       ::testing::Values(true, false)  // All zero input
    );
#else
const auto combi =
    ::testing::Combine(::testing::Values(11, 65, 10000),  // size
                       ::testing::Values(5),              // incX
                       ::testing::Values(true, false)     // All max input
    );
#endif

class IaminFloat : public ::testing::TestWithParam<combination_t> {};
TEST_P(IaminFloat, test) { run_test<float>(GetParam()); };
INSTANTIATE_TEST_SUITE_P(iamin, IaminFloat, combi);

#if DOUBLE_SUPPORT
class IaminDouble : public ::testing::TestWithParam<combination_t> {};
TEST_P(IaminDouble, test) { run_test<double>(GetParam()); };
INSTANTIATE_TEST_SUITE_P(iamin, IaminDouble, combi);
#endif
