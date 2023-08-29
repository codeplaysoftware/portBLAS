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
 *  portBLAS: BLAS implementation using SYCL
 *
 *  @filename blas1_axpy_test.cpp
 *
 **************************************************************************/

#include "blas_test.hpp"

template <typename scalar_t>
using combination_t = std::tuple<std::string, int, scalar_t, int, int>;

template <typename scalar_t, helper::AllocType mem_alloc>
void run_test(const combination_t<scalar_t> combi) {
  std::string alloc;
  index_t size;
  scalar_t alpha;
  index_t incX;
  index_t incY;
  std::tie(alloc, size, alpha, incX, incY) = combi;

  auto x_size = size * std::abs(incX);
  auto y_size = size * std::abs(incY);
  // Input vector
  std::vector<scalar_t> x_v(x_size);
  fill_random(x_v);

  // Output vector
  std::vector<scalar_t> y_v(y_size, 10.0);
  std::vector<scalar_t> y_cpu_v(y_size, 10.0);

  // Reference implementation
  reference_blas::axpy(size, alpha, x_v.data(), incX, y_cpu_v.data(), incY);

  // SYCL implementation
  auto q = make_queue();
  blas::SB_Handle sb_handle(q);

  // Iterators
  auto gpu_x_v = helper::allocate<mem_alloc, scalar_t>(x_size, q);
  auto gpu_y_v = helper::allocate<mem_alloc, scalar_t>(y_size, q);

  auto copy_x = helper::copy_to_device(q, x_v.data(), gpu_x_v, x_size);
  auto copy_y = helper::copy_to_device(q, y_v.data(), gpu_y_v, y_size);

  auto axpy_event = _axpy(sb_handle, size, alpha, gpu_x_v, incX, gpu_y_v, incY,
                          {copy_x, copy_y});
  sb_handle.wait(axpy_event);

  auto event = helper::copy_to_host(q, gpu_y_v, y_v.data(), y_size);
  sb_handle.wait(event);

  // Validate the result
  const bool isAlmostEqual = utils::compare_vectors(y_v, y_cpu_v);
  ASSERT_TRUE(isAlmostEqual);

  helper::deallocate<mem_alloc>(gpu_x_v, q);
  helper::deallocate<mem_alloc>(gpu_y_v, q);
}

template <typename scalar_t>
void run_test(const combination_t<scalar_t> combi) {
  std::string alloc;
  index_t size;
  scalar_t alpha;
  index_t incX;
  index_t incY;
  std::tie(alloc, size, alpha, incX, incY) = combi;

  if (alloc == "usm") {  // usm alloc
#ifdef SB_ENABLE_USM
    run_test<scalar_t, helper::AllocType::usm>(combi);
#else
    GTEST_SKIP();
#endif
  } else {  // buffer alloc
    run_test<scalar_t, helper::AllocType::buffer>(combi);
  }
}

#ifdef STRESS_TESTING
template <typename scalar_t>
const auto combi =
    ::testing::Combine(::testing::Values("usm", "buf"),  // allocation type
                       ::testing::Values(11, 65, 1002, 1002400),    // size
                       ::testing::Values<scalar_t>(0.0, 1.0, 1.5),  // alpha
                       ::testing::Values(1, 4),                     // incX
                       ::testing::Values(1, 3)                      // incY
    );
#else
template <typename scalar_t>
const auto combi =
    ::testing::Combine(::testing::Values("usm", "buf"),  // allocation type
                       ::testing::Values(11, 1002),      // size
                       ::testing::Values<scalar_t>(0.0, 1.0),  // alpha
                       ::testing::Values(1, 4, -1, -3),        // incX
                       ::testing::Values(1, 3, -2)             // incY
    );
#endif

template <class T>
static std::string generate_name(
    const ::testing::TestParamInfo<combination_t<T>>& info) {
  std::string alloc;
  int size, incX, incY;
  T alpha;
  BLAS_GENERATE_NAME(info.param, alloc, size, alpha, incX, incY);
}

BLAS_REGISTER_TEST_ALL(Axpy, combination_t, combi, generate_name);
