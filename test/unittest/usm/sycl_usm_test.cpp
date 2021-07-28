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
 *  @filename sycl_buffer_test.cpp
 *
 **************************************************************************/

#include "blas_test.hpp"
#include <CL/sycl.hpp>

template <typename scalar_t>
using combination_t = std::tuple<int>;

template <typename scalar_t>
void run_test(const combination_t<scalar_t> combi) {
  int dataSize;
  std::tie(dataSize) = combi;

  float a[dataSize], b[dataSize], r[dataSize], z=0.0f;
  for (int i = 0; i < dataSize; ++i) {
    a[i] = static_cast<float>(i);
    r[i] = 0.0f;
  }

  auto queue = make_queue();

  auto devicePtr = blas::sycl_usm_malloc_device(sizeof(float) * dataSize, queue);

  queue.memcpy(devicePtr, a, sizeof(float) * dataSize).wait();
  queue.memcpy(r, devicePtr, sizeof(float) * dataSize).wait();

  blas::sycl_usm_free(devicePtr, queue);

  queue.throw_asynchronous();

  for (int i = 0; i < dataSize; ++i) {
    z += r[i] - a[i];
  }

  EXPECT_EQ(z, 0);
}

const auto combi = ::testing::Combine(::testing::Values(100, 1024000));

BLAS_REGISTER_TEST(Usm, combination_t, combi);
