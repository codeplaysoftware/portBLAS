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
 *  @filename blas1_asum_test.cpp
 *
 **************************************************************************/

#include "blas_test.hpp"

template <typename scalar_t>
using combination_t = std::tuple<int, int>;

template <typename scalar_t>
void run_test(const combination_t<scalar_t> combi) {
  int size;
  int incX;
  std::tie(size, incX) = combi;

  using data_t = utils::data_storage_t<scalar_t>;

  // Input vector
  std::vector<data_t> x_v(size * incX);
  fill_random<data_t>(x_v);

  // We need to guarantee that cl::sycl::half can hold the sum
  // of x_v without overflow by making sum(x_v) to be 1.0
  std::transform(std::begin(x_v), std::end(x_v), std::begin(x_v),
                 [=](data_t x) { return x / x_v.size(); });

  // Output scalar
  data_t out_s = 0;

  // Reference implementation
  data_t out_cpu_s = reference_blas::asum(size, x_v.data(), incX);

  // SYCL implementation
  auto q = make_queue();
  test_executor_t ex(q);

  // Iterators
  auto gpu_x_v = utils::make_quantized_buffer<scalar_t>(ex, x_v);
  auto gpu_out_s = utils::make_quantized_buffer<scalar_t>(ex, out_s);

  _asum(ex, size, gpu_x_v, incX, gpu_out_s);
  auto event = utils::quantized_copy_to_host<scalar_t>(ex, gpu_out_s, out_s);
  ex.get_policy_handler().wait(event);

  // Validate the result
  const bool is_almost_equal =
      utils::almost_equal<data_t, scalar_t>(out_s, out_cpu_s);
  ASSERT_TRUE(is_almost_equal);

  ex.get_policy_handler().get_queue().wait();
}

const auto combi =
    ::testing::Combine(::testing::Values(11, 65, 10000, 1002400),  // size
                       ::testing::Values(1, 4)                     // incX
    );

BLAS_REGISTER_TEST(Asum, combination_t, combi);
