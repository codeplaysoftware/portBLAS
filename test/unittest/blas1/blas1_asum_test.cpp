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
 *  @filename blas1_asum_test.cpp
 *
 **************************************************************************/

#include "blas_test.hpp"

typedef ::testing::Types<blas_test_float<>, blas_test_double<> > BlasTypes;

TYPED_TEST_CASE(BLAS_Test, BlasTypes);

REGISTER_SIZE(::RANDOM_SIZE, asum_test)
REGISTER_STRD(::RANDOM_STRD, asum_test)
REGISTER_PREC(float, 1e-4, asum_test)
REGISTER_PREC(double, 1e-6, asum_test)

TYPED_TEST(BLAS_Test, asum_test) {
  using ScalarT = typename TypeParam::scalar_t;
  using ExecutorType = typename TypeParam::executor_t;
  using TestClass = BLAS_Test<TypeParam>;
  using test = class asum_test;

  int size = TestClass::template test_size<test>();
  int strd = TestClass::template test_strd<test>();
  ScalarT prec = TestClass::template test_prec<test>();

  DEBUG_PRINT(std::cout << "size == " << size << std::endl);
  DEBUG_PRINT(std::cout << "strd == " << strd << std::endl);

  std::vector<ScalarT> vX(size);
  TestClass::set_rand(vX, size);

  std::vector<ScalarT> vR(1, ScalarT(0));
  ScalarT res(0);
  for (int i = 0; i < size; i += strd) {
    res += std::abs(vX[i]);
  }

  SYCL_DEVICE_SELECTOR d;
  auto q = TestClass::make_queue(d);
  Executor<ExecutorType> ex(q);
  auto gpu_vX = blas::make_sycl_iterator_buffer<ScalarT>(vX, size);
  auto gpu_vR = blas::make_sycl_iterator_buffer<ScalarT>(int(1));
  _asum(ex, (size + strd - 1) / strd, gpu_vX, strd, gpu_vR);
  auto event = ex.get_policy_handler().copy_to_host(gpu_vR, vR.data(), 1);
  ex.get_policy_handler().wait(event);
  ASSERT_NEAR(res, vR[0], prec);
}

REGISTER_SIZE(::RANDOM_SIZE, asum_test_auto_return)
REGISTER_STRD(::RANDOM_STRD, asum_test_auto_return)
REGISTER_PREC(float, 1e-4, asum_test_auto_return)
REGISTER_PREC(double, 1e-6, asum_test_auto_return)

TYPED_TEST(BLAS_Test, asum_test_auto_return) {
  using ScalarT = typename TypeParam::scalar_t;
  using ExecutorType = typename TypeParam::executor_t;
  using TestClass = BLAS_Test<TypeParam>;
  using test = class asum_test_auto_return;

  int size = TestClass::template test_size<test>();
  int strd = TestClass::template test_strd<test>();
  ScalarT prec = TestClass::template test_prec<test>();

  DEBUG_PRINT(std::cout << "size == " << size << std::endl);
  DEBUG_PRINT(std::cout << "strd == " << strd << std::endl);

  std::vector<ScalarT> vX(size);
  TestClass::set_rand(vX, size);

  std::vector<ScalarT> vR(1, ScalarT(0));
  ScalarT res(0);
  for (int i = 0; i < size; i += strd) {
    res += std::abs(vX[i]);
  }

  SYCL_DEVICE_SELECTOR d;
  auto q = TestClass::make_queue(d);
  Executor<ExecutorType> ex(q);
  {
    auto gpu_vX = blas::make_sycl_iterator_buffer<ScalarT>(vX, size);
    auto gpu_vR = blas::make_sycl_iterator_buffer<ScalarT>(vR, int(1));
    _asum(ex, (size + strd - 1) / strd, gpu_vX, strd, gpu_vR);
  }
  ASSERT_NEAR(res, vR[0], prec);
}

REGISTER_SIZE(::RANDOM_SIZE, asum_test_virtual_pointer)
REGISTER_STRD(::RANDOM_STRD, asum_test_virtual_pointer)
REGISTER_PREC(float, 1e-4, asum_test_virtual_pointer)
REGISTER_PREC(double, 1e-6, asum_test_virtual_pointer)

TYPED_TEST(BLAS_Test, asum_test_virtual_pointer) {
  using ScalarT = typename TypeParam::scalar_t;
  using ExecutorType = typename TypeParam::executor_t;
  using TestClass = BLAS_Test<TypeParam>;
  using test = class asum_test_virtual_pointer;

  int size = TestClass::template test_size<test>();
  int strd = TestClass::template test_strd<test>();
  ScalarT prec = TestClass::template test_prec<test>();

  DEBUG_PRINT(std::cout << "size == " << size << std::endl);
  DEBUG_PRINT(std::cout << "strd == " << strd << std::endl);

  std::vector<ScalarT> vX(size);
  TestClass::set_rand(vX, size);

  std::vector<ScalarT> vR(1, ScalarT(0));
  ScalarT res(0);
  for (int i = 0; i < size; i += strd) {
    res += std::abs(vX[i]);
  }

  SYCL_DEVICE_SELECTOR d;
  auto q = TestClass::make_queue(d);
  Executor<ExecutorType> ex(q);
  auto gpu_vX = ex.get_policy_handler().template allocate<ScalarT>(size);
  auto gpu_vR = ex.get_policy_handler().template allocate<ScalarT>(1);
  ex.get_policy_handler().copy_to_device(vX.data(), gpu_vX, size);
  ex.get_policy_handler().copy_to_device(vR.data(), gpu_vR, 1);
  _asum(ex, (size + strd - 1) / strd, gpu_vX, strd, gpu_vR);
  auto event = ex.get_policy_handler().copy_to_host(gpu_vR, vR.data(), 1);
  ex.get_policy_handler().wait(event);

  printf("vR[0] %f\n", vR[0]);
  ASSERT_NEAR(res, vR[0], prec);
  ex.get_policy_handler().template deallocate<ScalarT>(gpu_vX);
  ex.get_policy_handler().template deallocate<ScalarT>(gpu_vR);
}

REGISTER_SIZE(::RANDOM_SIZE, asum_test_combined_vp_buffer)
REGISTER_STRD(::RANDOM_STRD, asum_test_combined_vp_buffer)
REGISTER_PREC(float, 1e-4, asum_test_combined_vp_buffer)
REGISTER_PREC(double, 1e-6, asum_test_combined_vp_buffer)

TYPED_TEST(BLAS_Test, asum_test_combined_vp_buffer) {
  using ScalarT = typename TypeParam::scalar_t;
  using ExecutorType = typename TypeParam::executor_t;
  using TestClass = BLAS_Test<TypeParam>;
  using test = class asum_test_combined_vp_buffer;

  int size = TestClass::template test_size<test>();
  int strd = TestClass::template test_strd<test>();
  ScalarT prec = TestClass::template test_prec<test>();

  DEBUG_PRINT(std::cout << "size == " << size << std::endl);
  DEBUG_PRINT(std::cout << "strd == " << strd << std::endl);

  std::vector<ScalarT> vX(size);
  TestClass::set_rand(vX, size);

  std::vector<ScalarT> vR(1, ScalarT(0));
  ScalarT res(0);
  for (int i = 0; i < size; i += strd) {
    res += std::abs(vX[i]);
  }

  SYCL_DEVICE_SELECTOR d;
  auto q = TestClass::make_queue(d);
  Executor<ExecutorType> ex(q);
  auto gpu_vX = ex.get_policy_handler().template allocate<ScalarT>(size);
  auto gpu_vR = blas::make_sycl_iterator_buffer<ScalarT>(int(1));
  ex.get_policy_handler().copy_to_device(vX.data(), gpu_vX, size);
  ex.get_policy_handler().copy_to_device(vR.data(), gpu_vR, 1);
  _asum(ex, (size + strd - 1) / strd, gpu_vX, strd, gpu_vR);
  auto event = ex.get_policy_handler().copy_to_host(gpu_vR, vR.data(), 1);
  ex.get_policy_handler().wait(event);

  printf("vR[0] %f\n", vR[0]);
  ASSERT_NEAR(res, vR[0], prec);
  ex.get_policy_handler().template deallocate<ScalarT>(gpu_vX);
}

REGISTER_SIZE(::RANDOM_SIZE, asum_test_combined_vp_buffer_return_buff)
REGISTER_STRD(::RANDOM_STRD, asum_test_combined_vp_buffer_return_buff)
REGISTER_PREC(float, 1e-4, asum_test_combined_vp_buffer_return_buff)
REGISTER_PREC(double, 1e-6, asum_test_combined_vp_buffer_return_buff)

TYPED_TEST(BLAS_Test, asum_test_combined_vp_buffer_return_buff) {
  using ScalarT = typename TypeParam::scalar_t;
  using ExecutorType = typename TypeParam::executor_t;
  using TestClass = BLAS_Test<TypeParam>;
  using test = class asum_test_combined_vp_buffer_return_buff;

  int size = TestClass::template test_size<test>();
  int strd = TestClass::template test_strd<test>();
  ScalarT prec = TestClass::template test_prec<test>();

  DEBUG_PRINT(std::cout << "size == " << size << std::endl);
  DEBUG_PRINT(std::cout << "strd == " << strd << std::endl);

  std::vector<ScalarT> vX(size);
  TestClass::set_rand(vX, size);

  std::vector<ScalarT> vR(1, ScalarT(0));
  ScalarT res(0);
  for (int i = 0; i < size; i += strd) {
    res += std::abs(vX[i]);
  }

  SYCL_DEVICE_SELECTOR d;
  auto q = TestClass::make_queue(d);
  Executor<ExecutorType> ex(q);
  auto gpu_vX = ex.get_policy_handler().template allocate<ScalarT>(size);
  auto gpu_vR = blas::make_sycl_iterator_buffer<ScalarT>(int(1));
  ex.get_policy_handler().copy_to_device(vX.data(), gpu_vX, size);
  ex.get_policy_handler().copy_to_device(vR.data(), gpu_vR, 1);
  _asum(ex, (size + strd - 1) / strd, gpu_vX, strd, gpu_vR);
  auto event = ex.get_policy_handler().copy_to_host(gpu_vR, vR.data(), 1);
  ex.get_policy_handler().wait(event);

  printf("vR[0] %f\n", vR[0]);
  ASSERT_NEAR(res, vR[0], prec);
  ex.get_policy_handler().template deallocate<ScalarT>(gpu_vX);
}