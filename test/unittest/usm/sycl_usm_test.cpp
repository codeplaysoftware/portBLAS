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
 *  @filename sycl_usm_test.cpp
 *
 **************************************************************************/

#include "blas_test.hpp"
#include <iostream>

template <typename scalar_t>
using combination_t = std::tuple<int>;

template <typename scalar_t>
void run_test(const combination_t<scalar_t> combi) {
  int dataSize;
  std::tie(dataSize) = combi;

  float a[dataSize], b[dataSize];
  for (int i = 0; i < dataSize; ++i) {
    a[i] = static_cast<float>(i);
    b[i] = 0.0f;
  }

#ifdef SYCL_BLAS_USE_USM
  std::cout << "SYCL_BLAS_USE_USM Defined" << std::endl;
#else
  std::cout << "SYCL_BLAS_USE_USM Not Defined" << std::endl;
#endif
  auto queue = make_queue();

  //auto devicePtr = blas::sycl_usm_malloc_device(sizeof(float) * dataSize, queue);
  auto devicePtr = cl::sycl::malloc_device(sizeof(float) * dataSize, queue);

  queue.memcpy(devicePtr, a, sizeof(float) * dataSize).wait();
  queue.memcpy(b, devicePtr, sizeof(float) * dataSize).wait();

  //blas::sycl_usm_free(devicePtr, queue);
  cl::sycl::free(devicePtr, queue);

  queue.throw_asynchronous();

  for (int i = 0; i < dataSize; ++i) {
    EXPECT_EQ(b[i], a[i]);
  }
}

const auto combi = ::testing::Combine(::testing::Values(100, 1024000));

BLAS_REGISTER_TEST(Usm, combination_t, combi);
