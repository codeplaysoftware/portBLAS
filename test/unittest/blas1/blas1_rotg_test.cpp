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
 *  @filename blas1_rotg_test.cpp
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
  int incA;
  int incB;
  std::tie(size, incA, incB) = combi;

  // Input vectors
  std::vector<scalar_t> a_v(size * incA);
  fill_random(a_v);
  std::vector<scalar_t> b_v(size * incB);
  fill_random(b_v);

  // Output vectors
  std::vector<scalar_t> out_s(1, 10.0);
  std::vector<scalar_t> a_cpu_v(a_v);
  std::vector<scalar_t> b_cpu_v(b_v);

  // Looks like we don't have a SYCL rotg implementation
  scalar_t c;
  scalar_t s;
  scalar_t sa = a_v[0];
  scalar_t sb = a_v[1];
  reference_blas::rotg(&sa, &sb, &c, &s);

  // Reference implementation
  std::vector<scalar_t> c_cpu_v(size * incA);
  std::vector<scalar_t> s_cpu_v(size * incB);
  reference_blas::rot(size, a_cpu_v.data(), incA, b_cpu_v.data(), incB, c, s);
  auto out_cpu_s =
      reference_blas::dot(size, a_cpu_v.data(), incA, b_cpu_v.data(), incB);

  // SYCL implementation
  SYCL_DEVICE_SELECTOR d;
  auto q = blas_test_t::make_queue(d);
  Executor<executor_t> ex(q);

  // Iterators
  auto gpu_a_v = blas::make_sycl_iterator_buffer<scalar_t>(int(size * incA));
  ex.get_policy_handler().copy_to_device(a_v.data(), gpu_a_v, size * incA);
  auto gpu_b_v = blas::make_sycl_iterator_buffer<scalar_t>(int(size * incB));
  ex.get_policy_handler().copy_to_device(b_v.data(), gpu_b_v, size * incB);
  auto gpu_out_s = blas::make_sycl_iterator_buffer<scalar_t>(int(1));
  ex.get_policy_handler().copy_to_device(out_s.data(), gpu_out_s, 1);

  _rot(ex, size, gpu_a_v, incA, gpu_b_v, incB, c, s);
  _dot(ex, size, gpu_a_v, incA, gpu_b_v, incB, gpu_out_s);
  auto event = ex.get_policy_handler().copy_to_host(gpu_out_s, out_s.data(), 1);
  ex.get_policy_handler().wait(event);

  // Validate the result
  ASSERT_TRUE(utils::almost_equal(out_s[0], out_cpu_s));
}

#ifdef STRESS_TESTING
const auto combi =
    ::testing::Combine(::testing::Values(11, 65, 1002, 1002400),  // size
                       ::testing::Values(1, 4),                   // incX
                       ::testing::Values(1, 3)                    // incY
    );
#else
const auto combi = ::testing::Combine(::testing::Values(11, 1002),  // size
                                      ::testing::Values(4),         // incA
                                      ::testing::Values(3)          // incB
);
#endif

class RotgFloat : public ::testing::TestWithParam<combination_t> {};
TEST_P(RotgFloat, test) { run_test<float>(GetParam()); };
INSTANTIATE_TEST_SUITE_P(rotg, RotgFloat, combi);

#if DOUBLE_SUPPORT
class RotgDouble : public ::testing::TestWithParam<combination_t> {};
TEST_P(RotgDouble, test) { run_test<double>(GetParam()); };
INSTANTIATE_TEST_SUITE_P(rotg, RotgDouble, combi);
#endif
