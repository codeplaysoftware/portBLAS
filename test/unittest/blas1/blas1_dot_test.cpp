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
 *  @filename blas1_dot_test.cpp
 *
 **************************************************************************/

#include "blas_test.hpp"
typedef ::testing::Types<blas_test_args<float>, blas_test_args<double> >
    BlasTypes;

TYPED_TEST_CASE(BLAS_Test, BlasTypes);

REGISTER_SIZE(::RANDOM_SIZE, dot_test)
REGISTER_STRD(::RANDOM_STRD, dot_test)
REGISTER_PREC(float, 1e-4, dot_test)
REGISTER_PREC(double, 1e-6, dot_test)
REGISTER_PREC(long double, 1e-7, dot_test)

TYPED_TEST(BLAS_Test, dot_test) {
  using ScalarT = typename TypeParam::scalar_t;
  using ExecutorType = typename TypeParam::executor_t;
  using TestClass = BLAS_Test<TypeParam>;
  using test = class dot_test;

  size_t size = TestClass::template test_size<test>();
  long strd = TestClass::template test_strd<test>();
  ScalarT prec = TestClass::template test_prec<test>();

  DEBUG_PRINT(std::cout << "size == " << size << std::endl);
  DEBUG_PRINT(std::cout << "strd == " << strd << std::endl);

  // create two random vectors: vX and vY
  std::vector<ScalarT> vX(size);
  std::vector<ScalarT> vY(size);
  // create a vector of size 1 for the result
  std::vector<ScalarT> vR(1, ScalarT(0));
  TestClass::set_rand(vX, size);
  TestClass::set_rand(vY, size);

  ScalarT res(0);
  // compute dot(vX, vY) into res with a for loop
  for (size_t i = 0; i < size; i += strd) {
    res += vX[i] * vY[i];
  }

  SYCL_DEVICE_SELECTOR d;
  auto q = TestClass::make_queue(d);
  Executor<ExecutorType> ex(q);
  auto gpu_vX = blas::helper::make_sycl_iteator_buffer<ScalarT>(vX, size);
  auto gpu_vY = blas::helper::make_sycl_iteator_buffer<ScalarT>(vY, size);
  auto gpu_vR = blas::helper::make_sycl_iteator_buffer<ScalarT>(size_t(1));
  _dot(ex, (size + strd - 1) / strd, gpu_vX, strd, gpu_vY, strd, gpu_vR);
  ex.copy_to_host(gpu_vR, vR.data());
  ASSERT_NEAR(res, vR[0], prec);
}

REGISTER_SIZE(::RANDOM_SIZE, dot_test_vpr)
REGISTER_STRD(::RANDOM_STRD, dot_test_vpr)
REGISTER_PREC(float, 1e-4, dot_test_vpr)
REGISTER_PREC(double, 1e-6, dot_test_vpr)
REGISTER_PREC(long double, 1e-7, dot_test_vpr)

TYPED_TEST(BLAS_Test, dot_test_vpr) {
  using ScalarT = typename TypeParam::scalar_t;
  using ExecutorType = typename TypeParam::executor_t;
  using TestClass = BLAS_Test<TypeParam>;
  using test = class dot_test_vpr;

  size_t size = TestClass::template test_size<test>();
  long strd = TestClass::template test_strd<test>();
  ScalarT prec = TestClass::template test_prec<test>();

  DEBUG_PRINT(std::cout << "size == " << size << std::endl);
  DEBUG_PRINT(std::cout << "strd == " << strd << std::endl);

  // create two random vectors: vX and vY
  std::vector<ScalarT> vX(size);
  std::vector<ScalarT> vY(size);
  // create a vector of size 1 for the result
  std::vector<ScalarT> vR(1, ScalarT(0));
  TestClass::set_rand(vX, size);
  TestClass::set_rand(vY, size);

  ScalarT res(0);
  // compute dot(vX, vY) into res with a for loop
  for (size_t i = 0; i < size; i += strd) {
    res += vX[i] * vY[i];
  }

  SYCL_DEVICE_SELECTOR d;
  auto q = TestClass::make_queue(d);
  Executor<ExecutorType> ex(q);
  auto gpu_vX = ex.template allocate<ScalarT>(size);
  auto gpu_vY = ex.template allocate<ScalarT>(size);
  auto gpu_vR = ex.template allocate<ScalarT>(1);
  printf("inside the test: %p\n", gpu_vR);
  ex.copy_to_device(vX.data(), gpu_vX, size);
  ex.copy_to_device(vY.data(), gpu_vY, size);
  _dot(ex, (size + strd - 1) / strd, gpu_vX, strd, gpu_vY, strd, gpu_vR);
  ex.copy_to_host(gpu_vR, vR.data(), 1);
  ASSERT_NEAR(res, vR[0], prec);
  ex.template deallocate<ScalarT>(gpu_vX);
  ex.template deallocate<ScalarT>(gpu_vY);
  ex.template deallocate<ScalarT>(gpu_vR);
}
