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
 *  @filename blas1_rotm_test.cpp
 *
 **************************************************************************/

#include "blas_test.hpp"

template <typename scalar_t>
using combination_t = std::tuple<std::string, int, int, int, scalar_t>;

template <typename scalar_t, helper::AllocType mem_alloc>
void run_test(const combination_t<scalar_t> combi) {
  std::string alloc;
  index_t size;
  index_t incX;
  index_t incY;
  scalar_t flag;
  std::tie(alloc, size, incX, incY, flag) = combi;

  // Setup param
  constexpr size_t param_size = 5;
  std::vector<scalar_t> param(param_size);
  fill_random(param);
  param[0] = flag;

  // Input vectors
  std::vector<scalar_t> x_v(size * incX);
  fill_random(x_v);

  std::vector<scalar_t> y_v(size * incY);
  fill_random(y_v);

  // Output vectors
  std::vector<scalar_t> out_s(1, 10.0);
  std::vector<scalar_t> x_cpu_v(x_v);
  std::vector<scalar_t> y_cpu_v(y_v);

  // Reference implementation
  reference_blas::rotm(size, x_cpu_v.data(), incX, y_cpu_v.data(), incY,
                       param.data());

  // SYCL implementation
  auto q = make_queue();
  blas::SB_Handle sb_handle(q);

  // Iterators
  auto gpu_x_v = helper::allocate<mem_alloc, scalar_t>(size * incX, q);
  auto gpu_y_v = helper::allocate<mem_alloc, scalar_t>(size * incY, q);
  auto gpu_param = helper::allocate<mem_alloc, scalar_t>(param_size, q);

  auto copy_x = helper::copy_to_device(q, x_v.data(), gpu_x_v, size * incX);
  auto copy_y = helper::copy_to_device(q, y_v.data(), gpu_y_v, size * incY);
  auto copy_param =
      helper::copy_to_device(q, param.data(), gpu_param, param_size);

  sb_handle.wait(copy_param);

  auto rotm_event = _rotm(sb_handle, size, gpu_x_v, incX, gpu_y_v, incY,
                          gpu_param, {copy_x, copy_y});
  sb_handle.wait(rotm_event);

  auto event1 =
      helper::copy_to_host<scalar_t>(q, gpu_x_v, x_v.data(), size * incX);
  auto event2 =
      helper::copy_to_host<scalar_t>(q, gpu_y_v, y_v.data(), size * incY);
  sb_handle.wait({event1, event2});

  // Validate the result
  const bool isAlmostEqual = utils::compare_vectors(x_cpu_v, x_v) &&
                             utils::compare_vectors(y_cpu_v, y_v);
  ASSERT_TRUE(isAlmostEqual);

  helper::deallocate<mem_alloc>(gpu_x_v, q);
  helper::deallocate<mem_alloc>(gpu_y_v, q);
  helper::deallocate<mem_alloc>(gpu_param, q);
}

template <typename scalar_t>
void run_test(const combination_t<scalar_t> combi) {
  std::string alloc;
  index_t size;
  index_t incX;
  index_t incY;
  scalar_t flag;
  std::tie(alloc, size, incX, incY, flag) = combi;

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
const auto combi = ::testing::Combine(
    ::testing::Values("usm", "buf"),                         // allocation type
    ::testing::Values(11, 65, 1002, 1002400),                // size
    ::testing::Values(1, 4),                                 // incX
    ::testing::Values(1, 3),                                 // incY
    ::testing::Values<scalar_t>(-2.0, -1.0, 0.0, 1.0, -4.0)  // flag
);
#else
template <typename scalar_t>
const auto combi =
    ::testing::Combine(::testing::Values("usm", "buf"),  // allocation type
                       ::testing::Values(11, 1002),      // size
                       ::testing::Values(4),             // incX
                       ::testing::Values(3),             // incY
                       ::testing::Values<scalar_t>(-2.0, -1.0, 0.0, 1.0,
                                                   -4.0)  // flag
    );
#endif

template <class T>
static std::string generate_name(
    const ::testing::TestParamInfo<combination_t<T>>& info) {
  std::string alloc;
  int size, incX, incY;
  T flag;
  BLAS_GENERATE_NAME(info.param, alloc, size, incX, incY, flag);
}

BLAS_REGISTER_TEST_ALL(Rotm, combination_t, combi, generate_name);
