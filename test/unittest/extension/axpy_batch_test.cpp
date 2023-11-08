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
 *  @filename axpy_batch_test.cpp
 *
 **************************************************************************/

#include "blas_test.hpp"

template <typename scalar_t>
using combination_t = std::tuple<std::string, index_t, scalar_t, index_t,
                                 index_t, index_t, index_t, index_t>;

template <typename scalar_t, helper::AllocType mem_alloc>
void run_test(const combination_t<scalar_t> combi) {
  std::string alloc;
  index_t size;
  scalar_t alpha;
  index_t incX;
  index_t incY;
  index_t stride_mul_x;
  index_t stride_mul_y;
  index_t batch_size;
  std::tie(alloc, size, alpha, incX, incY, stride_mul_x, stride_mul_y,
           batch_size) = combi;

  const index_t stride_x{size * std::abs(incX) * stride_mul_x};
  const index_t stride_y{size * std::abs(incY) * stride_mul_y};

  auto x_size = stride_x * batch_size;
  auto y_size = stride_y * batch_size;
  // Input vector
  std::vector<scalar_t> x_v(x_size);
  fill_random(x_v);

  // Output vector
  std::vector<scalar_t> y_v(y_size, 10.0);
  std::vector<scalar_t> y_cpu_v(y_size, 10.0);

  // Reference implementation
  for (index_t i = 0; i < batch_size; ++i) {
    reference_blas::axpy(size, alpha, x_v.data() + i * stride_x, incX,
                         y_cpu_v.data() + i * stride_y, incY);
  }

  // SYCL implementation
  auto q = make_queue();
  blas::SB_Handle sb_handle(q);

  // Iterators
  auto gpu_x_v = helper::allocate<mem_alloc, scalar_t>(x_size, q);
  auto gpu_y_v = helper::allocate<mem_alloc, scalar_t>(y_size, q);

  auto copy_x = helper::copy_to_device(q, x_v.data(), gpu_x_v, x_size);
  auto copy_y = helper::copy_to_device(q, y_v.data(), gpu_y_v, y_size);

  auto axpy_batch_event =
      _axpy_batch(sb_handle, size, alpha, gpu_x_v, incX, stride_x, gpu_y_v,
                  incY, stride_y, batch_size, {copy_x, copy_y});
  sb_handle.wait(axpy_batch_event);

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
  index_t stride_mul_x;
  index_t stride_mul_y;
  index_t batch_size;
  std::tie(alloc, size, alpha, incX, incY, stride_mul_x, stride_mul_y,
           batch_size) = combi;

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
                       ::testing::Values(11, 65, 10020, 1000240),   // size
                       ::testing::Values<scalar_t>(0.0, 1.3, 2.5),  // alpha
                       ::testing::Values(1, -1, 2, -7),             // incX
                       ::testing::Values(1, -1, 3, -5),             // incY
                       ::testing::Values(1, 2, 3),  // stride_mul_x
                       ::testing::Values(1, 2, 3),  // stride_mul_y
                       ::testing::Values(5)         // batch_size
    );
#else
template <typename scalar_t>
const auto combi =
    ::testing::Combine(::testing::Values("usm", "buf"),  // allocation type
                       ::testing::Values(11, 65, 1002, 10240),  // size
                       ::testing::Values<scalar_t>(0.0, 1.3),   // alpha
                       ::testing::Values(1, -1, 2, -4),         // incX
                       ::testing::Values(1, -1, 3, -5),         // incY
                       ::testing::Values(1, 2, 3),              // stride_mul_x
                       ::testing::Values(1, 2, 3),              // stride_mul_y
                       ::testing::Values(5)                     // batch_size
    );
#endif

template <class T>
static std::string generate_name(
    const ::testing::TestParamInfo<combination_t<T>>& info) {
  std::string alloc;
  index_t size, incX, incY, stride_mul_x, stride_mul_y, batch_size;
  T alpha;
  BLAS_GENERATE_NAME(info.param, alloc, size, alpha, incX, incY, stride_mul_x,
                     stride_mul_y, batch_size);
}

BLAS_REGISTER_TEST_ALL(Axpy_batch, combination_t, combi, generate_name);
