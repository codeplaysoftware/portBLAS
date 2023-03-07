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
using combination_t = std::tuple<char, int, int, int>;

template <typename scalar_t, helper::AllocType mem_alloc>
void run_test(const combination_t<scalar_t> combi) {
  char alloc;
  index_t size;
  index_t incX;
  index_t incY;
  std::tie(alloc, size, incX, incY) = combi;

  // Input vector
  std::vector<scalar_t> x_v(size * incX);
  fill_random(x_v);

  // Output vector
  std::vector<scalar_t> y_v(size * incY, 10.0);
  std::vector<scalar_t> y_cpu_v(size * incY, 10.0);

  // Reference implementation
  reference_blas::copy(size, x_v.data(), incX, y_cpu_v.data(), incY);

  // SYCL implementation
  auto q = make_queue();
  blas::SB_Handle sb_handle(q);

  // Iterators
  auto gpu_x_v = blas::helper::allocate<mem_alloc, scalar_t>(size * incX, q);
  auto gpu_y_v = blas::helper::allocate<mem_alloc, scalar_t>(size * incY, q);

  auto copy_event_x =
      blas::helper::copy_to_device(q, x_v.data(), gpu_x_v, size * incX);

  auto copy_event_y =
      blas::helper::copy_to_device(q, y_v.data(), gpu_y_v, size * incY);
  sb_handle.wait({copy_event_x, copy_event_y});

  _copy(sb_handle, size, gpu_x_v, incX, gpu_y_v, incY);
  auto event = blas::helper::copy_to_host(sb_handle.get_queue(), gpu_y_v,
                                          y_v.data(), size * incY);
  sb_handle.wait(event);

  // Validate the result
  // For copy, the float tolerances are ok
  ASSERT_TRUE(utils::compare_vectors(y_v, y_cpu_v));
}

template <typename scalar_t>
void run_test(const combination_t<scalar_t> combi) {
  char alloc;
  index_t size;
  index_t incX;
  index_t incY;
  std::tie(alloc, size, incX, incY) = combi;

  if (alloc == 'u') {  // usm alloc
    run_test<scalar_t, helper::AllocType::usm>(combi);
  } else {  // buffer alloc
    run_test<scalar_t, helper::AllocType::buffer>(combi);
  }
}

#ifdef STRESS_TESTING
template <typename scalar_t>
const auto combi =
    ::testing::Combine(::testing::Values('u', 'b'),  // allocation type
                       ::testing::Values(11, 65, 1002,
                                         1002400),  // size
                       ::testing::Values(1, 4),     // incX
                       ::testing::Values(1, 3)      // incY
    );
#else
template <typename scalar_t>
const auto combi =
    ::testing::Combine(::testing::Values('u', 'b'),  // allocation type
                       ::testing::Values(11, 1002),  // size
                       ::testing::Values(1, 4),      // incX
                       ::testing::Values(1, 3)       // incY
    );
#endif

template <class T>
static std::string generate_name(
    const ::testing::TestParamInfo<combination_t<T>>& info) {
  char alloc;
  int size, incX, incY;
  BLAS_GENERATE_NAME(info.param, alloc, size, incX, incY);
}

BLAS_REGISTER_TEST_ALL(Copy, combination_t, combi, generate_name);
