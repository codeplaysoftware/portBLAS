/***************************************************************************
 *
 *  @license
 *  Nrm2right (C) Codeplay Software Limited
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a nrm2 of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  For your convenience, a nrm2 of the License has been included in this
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
 *  @filename blas1_nrm2_test.cpp
 *
 **************************************************************************/

#include "blas_test.hpp"

using combination_t = std::tuple<int, int>;

template <typename scalar_t>
void run_test(const combination_t combi) {
  int size;
  int incX;
  std::tie(size, incX) = combi;

  // Input vectors
  std::vector<scalar_t> x_v(size * incX);
  fill_random(x_v);

  // Output vector
  std::vector<scalar_t> out_s(1, 10.0);

  // Reference implementation
  auto out_cpu_s = reference_blas::nrm2(size, x_v.data(), incX);

  // SYCL implementation
  auto q = make_queue();
  test_executor_t ex(q);

  // Iterators
  auto gpu_x_v = blas::make_sycl_iterator_buffer<scalar_t>(int(size * incX));
  ex.get_policy_handler().copy_to_device(x_v.data(), gpu_x_v, size * incX);
  auto gpu_out_s = blas::make_sycl_iterator_buffer<scalar_t>(int(1));
  ex.get_policy_handler().copy_to_device(out_s.data(), gpu_out_s, 1);

  _nrm2(ex, size, gpu_x_v, incX, gpu_out_s);
  auto event = ex.get_policy_handler().copy_to_host(gpu_out_s, out_s.data(), 1);
  ex.get_policy_handler().wait(event);

  // Validate the result
  ASSERT_TRUE(utils::almost_equal(out_s[0], out_cpu_s));
}

const auto combi = ::testing::Combine(::testing::Values(11, 1002),  // size
                                      ::testing::Values(1, 4)       // incX
);

class Nrm2Float : public ::testing::TestWithParam<combination_t> {};
TEST_P(Nrm2Float, test) { run_test<float>(GetParam()); };
INSTANTIATE_TEST_SUITE_P(nrm2, Nrm2Float, combi);

#if DOUBLE_SUPPORT
class Nrm2Double : public ::testing::TestWithParam<combination_t> {};
TEST_P(Nrm2Double, test) { run_test<double>(GetParam()); };
INSTANTIATE_TEST_SUITE_P(nrm2, Nrm2Double, combi);
#endif
