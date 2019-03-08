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
 *  @filename blas1_swap_test.cpp
 *
 **************************************************************************/

#include "blas_test.hpp"

typedef ::testing::Types<blas_test_float<>, blas_test_double<> > BlasTypes;

TYPED_TEST_CASE(BLAS_Test, BlasTypes);

REGISTER_SIZE(::RANDOM_SIZE, swap_test)
REGISTER_STRD(::RANDOM_STRD, swap_test)

TYPED_TEST(BLAS_Test, swap_test) {
  using scalar_t = typename TypeParam::scalar_t;
  using ExecutorType = typename TypeParam::executor_t;
  using TestClass = BLAS_Test<TypeParam>;
  using test = class swap_test;

  int size = TestClass::template test_size<test>();
  int strd = TestClass::template test_strd<test>();

  DEBUG_PRINT(std::cout << "size == " << size << std::endl);
  DEBUG_PRINT(std::cout << "strd == " << strd << std::endl);

  // create two random vectors with the same size: vX and vY
  std::vector<scalar_t> vX(size);
  std::vector<scalar_t> vY(size);
  TestClass::set_rand(vX, size);
  TestClass::set_rand(vY, size);

  // create two more vectors equal to vX and vY
  std::vector<scalar_t> vZ = vX;
  std::vector<scalar_t> vT = vY;

  SYCL_DEVICE_SELECTOR d;
  auto q = TestClass::make_queue(d);
  Executor<ExecutorType> ex(q);
  auto gpu_vX = ex.get_policy_handler().template allocate<scalar_t>(size);
  auto gpu_vY = ex.get_policy_handler().template allocate<scalar_t>(size);
  ex.get_policy_handler().copy_to_device(vX.data(), gpu_vX, size);
  ex.get_policy_handler().copy_to_device(vY.data(), gpu_vY, size);
  _swap(ex, (size + strd - 1) / strd, gpu_vX, strd, gpu_vY, strd);
  auto event0 = ex.get_policy_handler().copy_to_host(gpu_vX, vX.data(), size);
  auto event1 = ex.get_policy_handler().copy_to_host(gpu_vY, vY.data(), size);
  ex.get_policy_handler().wait(event0, event1);
  // check that new vX is equal to the copy of the original vY and
  // that new vY is equal to the copy of the original vX
  for (int i = 0; i < size; ++i) {
    if (i % strd == 0) {
      ASSERT_EQ(vZ[i], vY[i]);
      ASSERT_EQ(vT[i], vX[i]);
    } else {
      ASSERT_EQ(vZ[i], vX[i]);
      ASSERT_EQ(vT[i], vY[i]);
    }
  }
  ex.get_policy_handler().template deallocate<scalar_t>(gpu_vX);
  ex.get_policy_handler().template deallocate<scalar_t>(gpu_vY);
}
