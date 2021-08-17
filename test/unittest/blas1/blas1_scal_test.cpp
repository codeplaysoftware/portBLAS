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
  int size;
  scalar_t alpha;
  int incX;
  std::tie(size, alpha, incX) = combi;

#ifdef SYCL_BLAS_USE_USM
  using data_t = scalar_t;
#else
  using data_t = utils::data_storage_t<scalar_t>;
#endif

  // Input/output vector
  std::vector<data_t> x_v(size * incX);
  std::vector<data_t> x_cpu_v(x_v);

  // Reference implementation
  reference_blas::scal(size, static_cast<data_t>(alpha), x_cpu_v.data(), incX);

  // SYCL implementation
  auto q = make_queue();
  test_executor_t ex(q);

  // Iterators
#ifdef SYCL_BLAS_USE_USM
  data_t* gpu_x_v = cl::sycl::malloc_device<data_t>(size * incX, q);

  q.memcpy(gpu_x_v, x_v.data(), sizeof(data_t) * size * incX).wait();
#else
  auto gpu_x_v = utils::make_quantized_buffer<scalar_t>(ex, x_v);
#endif

  auto ev = _scal(ex, size, alpha, gpu_x_v, incX);
  ex.get_policy_handler().wait(ev);

  auto event =
#ifdef SYCL_BLAS_USE_USM
  q.memcpy(x_v.data(), gpu_x_v, sizeof(data_t) * size * incX);
#else
  utils::quantized_copy_to_host<scalar_t>(ex, gpu_x_v, x_v);
#endif
  ex.get_policy_handler().wait({event});

  // Validate the result
  const bool isAlmostEqual =
      utils::compare_vectors<data_t, scalar_t>(x_v, x_cpu_v);
  ASSERT_TRUE(isAlmostEqual);

  ex.get_policy_handler().get_queue().wait();

#ifdef SYCL_BLAS_USE_USM
  cl::sycl::free(gpu_x_v, q);
#endif
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

BLAS_REGISTER_TEST(Scal, combination_t, combi);
