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

typedef ::testing::Types<blas_test_float<>, blas_test_double<> > BlasTypes;

TYPED_TEST_CASE(BLAS_Test, BlasTypes);

REGISTER_SIZE(::RANDOM_SIZE, scal_test)
REGISTER_STRD(::RANDOM_STRD, scal_test)
REGISTER_PREC(float, 1e-4, scal_test)
REGISTER_PREC(double, 1e-6, scal_test)

TYPED_TEST(BLAS_Test, scal_test) {
  using scalar_t = typename TypeParam::scalar_t;
  using ExecutorType = typename TypeParam::executor_t;
  using TestClass = BLAS_Test<TypeParam>;
  using test = class scal_test;

  int size = TestClass::template test_size<test>();
  int strd = TestClass::template test_strd<test>();

  DEBUG_PRINT(std::cout << "size == " << size << std::endl);
  DEBUG_PRINT(std::cout << "strd == " << strd << std::endl);

  scalar_t prec = TestClass::template test_prec<test>();

  scalar_t alpha(1.54);
  // create two vectors: vX and vY
  std::vector<scalar_t> vX(size);
  std::vector<scalar_t> vY(size, 0);
  TestClass::set_rand(vX, size);

  // compute vector scalar product vX * alpha in a for loop and put it into vY
  for (int i = 0; i < size; ++i) {
    if (i % strd == 0) {
      vY[i] = alpha * vX[i];
    } else {
      vY[i] = vX[i];
    }
  }

  SYCL_DEVICE_SELECTOR d;
  auto q = TestClass::make_queue(d);
  Executor<ExecutorType> ex(q);
  auto gpu_vX = ex.get_policy_handler().template allocate<scalar_t>(size);
  ex.get_policy_handler().copy_to_device(vX.data(), gpu_vX, size);
  _scal(ex, (size + strd - 1) / strd, alpha, gpu_vX, strd);
  auto event = ex.get_policy_handler().copy_to_host(gpu_vX, vX.data(), size);
  ex.get_policy_handler().wait(event);

  // check that the result is the same
  for (int i = 0; i < size; ++i) {
    ASSERT_NEAR(vY[i], vX[i], prec);
  }
  ex.get_policy_handler().template deallocate<scalar_t>(gpu_vX);
}
