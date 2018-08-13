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

typedef ::testing::Types<blas_test_args<float>, blas_test_args<double> >
    BlasTypes;

TYPED_TEST_CASE(BLAS_Test, BlasTypes);

REGISTER_SIZE(::RANDOM_SIZE, copy_test)
REGISTER_STRD(::RANDOM_STRD, copy_test)

TYPED_TEST(BLAS_Test, copy_test) {
  using ScalarT = typename TypeParam::scalar_t;
  using ExecutorType = typename TypeParam::executor_t;
  using TestClass = BLAS_Test<TypeParam>;
  using test = class copy_test;

  size_t size = TestClass::template test_size<test>();
  long strd = TestClass::template test_strd<test>();

  DEBUG_PRINT(std::cout << "size == " << size << std::endl);
  DEBUG_PRINT(std::cout << "strd == " << strd << std::endl);

  // create two vectors: vX and vY
  std::vector<ScalarT> vX(size);
  std::vector<ScalarT> vY(size, 0);
  TestClass::set_rand(vX, size);

  SYCL_DEVICE_SELECTOR d;
  auto q = TestClass::make_queue(d);
  Executor<ExecutorType> ex(q);
  auto gpu_vX = blas::helper::make_sycl_iteator_buffer<ScalarT>(vX, size);
  auto gpu_vY = blas::helper::make_sycl_iteator_buffer<ScalarT>(size);
  _copy(ex, (size + strd - 1) / strd, gpu_vX, strd, gpu_vY, strd);
  auto event = ex.copy_to_host(gpu_vY, vY.data(), size);
  ex.sync(event);
  // check that vX and vY are the same
  for (size_t i = 0; i < size; ++i) {
    if (i % strd == 0) {
      ASSERT_EQ(vX[i], vY[i]);
    } else {
      ASSERT_EQ(0, vY[i]);
    }
  }
}

REGISTER_SIZE(::RANDOM_SIZE, copy_test_vpr)
REGISTER_STRD(::RANDOM_STRD, copy_test_vpr)

TYPED_TEST(BLAS_Test, copy_test_vpr) {
  using ScalarT = typename TypeParam::scalar_t;
  using ExecutorType = typename TypeParam::executor_t;
  using TestClass = BLAS_Test<TypeParam>;
  using test = class copy_test_vpr;

  size_t size = TestClass::template test_size<test>();
  long strd = TestClass::template test_strd<test>();

  DEBUG_PRINT(std::cout << "size == " << size << std::endl);
  DEBUG_PRINT(std::cout << "strd == " << strd << std::endl);

  // create two vectors: vX and vY
  std::vector<ScalarT> vX(size);
  std::vector<ScalarT> vY(size, 0);
  TestClass::set_rand(vX, size);

  SYCL_DEVICE_SELECTOR d;
  auto q = TestClass::make_queue(d);
  Executor<ExecutorType> ex(q);
  auto gpu_vX = ex.template allocate<ScalarT>(size);
  auto gpu_vY = ex.template allocate<ScalarT>(size);
  ex.copy_to_device(vX.data(), gpu_vX, size);
  ex.copy_to_device(vY.data(), gpu_vY, size);
  _copy(ex, (size + strd - 1) / strd, gpu_vX, strd, gpu_vY, strd);
  auto event = ex.copy_to_host(gpu_vY, vY.data(), size);
  ex.sync(event);

  // check that vX and vY are the same
  for (size_t i = 0; i < size; ++i) {
    if (i % strd == 0) {
      ASSERT_EQ(vX[i], vY[i]);
    } else {
      ASSERT_EQ(0, vY[i]);
    }
  }

  ex.template deallocate<ScalarT>(gpu_vX);
  ex.template deallocate<ScalarT>(gpu_vY);
}
