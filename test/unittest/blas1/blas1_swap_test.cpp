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
 *  @filename blas1_swap_test.cpp
 *
 **************************************************************************/

#include "blas_test.hpp"

template <typename scalar_t>
using combination_t = std::tuple<std::string, int, int, int>;

template <typename scalar_t, helper::AllocType mem_alloc>
void run_test(const combination_t<scalar_t> combi) {
  std::string alloc;
  index_t size;
  index_t incX;
  index_t incY;
  std::tie(alloc, size, incX, incY) = combi;

  // Input/Output vector
  std::vector<scalar_t> x_v(size * incX);
  fill_random(x_v);
  std::vector<scalar_t> x_cpu_v(x_v);

  std::vector<scalar_t> y_v(size * incY);
  fill_random(y_v);
  std::vector<scalar_t> y_cpu_v(y_v);

  // Reference implementation
  reference_blas::swap(size, x_cpu_v.data(), incX, y_cpu_v.data(), incY);

  // SYCL implementation
  auto q = make_queue();
  blas::SB_Handle sb_handle(q);

  // Iterators
  auto gpu_x_v = helper::allocate<mem_alloc, scalar_t>(size * incX, q);
  auto gpu_y_v = helper::allocate<mem_alloc, scalar_t>(size * incY, q);

  auto copy_x = helper::copy_to_device(q, x_v.data(), gpu_x_v, size * incX);
  auto copy_y = helper::copy_to_device(q, y_v.data(), gpu_y_v, size * incY);

  auto swap_event =
      _swap(sb_handle, size, gpu_x_v, incX, gpu_y_v, incY, {copy_x, copy_y});
  sb_handle.wait(swap_event);

  auto event0 = helper::copy_to_host(sb_handle.get_queue(), gpu_x_v, x_v.data(),
                                     size * incX);
  auto event1 = helper::copy_to_host(sb_handle.get_queue(), gpu_y_v, y_v.data(),
                                     size * incY);
  sb_handle.wait({event0, event1});

  // Validate the result
  // Since this is just a swap operation, float tolerances are fine
  ASSERT_TRUE(utils::compare_vectors(y_v, y_cpu_v));
  ASSERT_TRUE(utils::compare_vectors(x_v, x_cpu_v));

  helper::deallocate<mem_alloc>(gpu_x_v, q);
  helper::deallocate<mem_alloc>(gpu_y_v, q);
}

template <typename scalar_t>
void run_test(const combination_t<scalar_t> combi) {
  std::string alloc;
  index_t size;
  index_t incX;
  index_t incY;
  std::tie(alloc, size, incX, incY) = combi;

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
                       ::testing::Values(11, 65, 1002,
                                         1002400),  // size
                       ::testing::Values(1, 4),     // incX
                       ::testing::Values(1, 3)      // incY
    );
#else
template <typename scalar_t>
const auto combi =
    ::testing::Combine(::testing::Values("usm", "buf"),  // allocation type
                       ::testing::Values(11, 1002),      // size
                       ::testing::Values(1, 4),          // incX
                       ::testing::Values(1, 3)           // incY
    );
#endif

template <class T>
static std::string generate_name(
    const ::testing::TestParamInfo<combination_t<T>>& info) {
  std::string alloc;
  int size, incX, incY;
  BLAS_GENERATE_NAME(info.param, alloc, size, incX, incY);
}

BLAS_REGISTER_TEST_ALL(Swap, combination_t, combi, generate_name);
