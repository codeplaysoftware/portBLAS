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

template <typename scalar_t>
using combination_t = std::tuple<int, int, int>;

template <typename scalar_t>
void run_test(const combination_t<scalar_t> combi) {
  int size;
  int incX;
  int incY;
  std::tie(size, incX, incY) = combi;

#ifdef SYCL_BLAS_USE_USM
  using data_t = scalar_t;
#else
  using data_t = utils::data_storage_t<scalar_t>;
#endif

  // Input vector
  std::vector<data_t> x_v(size * incX);
  fill_random(x_v);

  // Output vector
  std::vector<data_t> y_v(size * incY, 10.0);
  std::vector<data_t> y_cpu_v(size * incY, 10.0);

  // Reference implementation
  reference_blas::copy(size, x_v.data(), incX, y_cpu_v.data(), incY);

  // SYCL implementation
  auto q = make_queue();
  test_executor_t ex(q);

  // Iterators
#ifdef SYCL_BLAS_USE_USM
  data_t* gpu_x_v = cl::sycl::malloc_device<data_t>(size * incX, q);
  data_t* gpu_y_v = cl::sycl::malloc_device<data_t>(size * incY, q);

  q.memcpy(gpu_x_v, x_v.data(), sizeof(data_t) * size * incX).wait();
  q.memcpy(gpu_y_v, y_v.data(), sizeof(data_t) * size * incY).wait();
#else
  auto gpu_x_v = utils::make_quantized_buffer<scalar_t>(ex, x_v);
  auto gpu_y_v = utils::make_quantized_buffer<scalar_t>(ex, y_v);
#endif

  auto ev = _copy(ex, size, gpu_x_v, incX, gpu_y_v, incY);
  ex.get_policy_handler().wait(ev);

  auto event =
#ifdef SYCL_BLAS_USE_USM
  q.memcpy(y_v.data(), gpu_y_v, sizeof(data_t) * size * incY);
#else
  utils::quantized_copy_to_host<scalar_t>(ex, gpu_y_v, y_v);
#endif
  ex.get_policy_handler().wait({event});

  // Validate the result
  // For copy, the float tolerances are ok
  ASSERT_TRUE(utils::compare_vectors(y_v, y_cpu_v));

#ifdef SYCL_BLAS_USE_USM
  cl::sycl::free(gpu_x_v, q);
  cl::sycl::free(gpu_y_v, q);
#endif
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

BLAS_REGISTER_TEST(Copy, combination_t, combi);
