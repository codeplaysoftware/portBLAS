/***************************************************************************
 *
 *  @license
 *  Copyright (C) 2016 Codeplay Software Limited
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
 *  @filename blas1_nrm2_test.cpp
 *
 **************************************************************************/

#include "blas1_test.hpp"

typedef ::testing::Types<blas1_test_args<float>, blas1_test_args<double> >
    BlasTypes;

TYPED_TEST_CASE(BLAS1_Test, BlasTypes);

REGISTER_SIZE(::RANDOM_SIZE, nrm2_test)
REGISTER_STRD(::RANDOM_STRD, nrm2_test)
REGISTER_PREC(float, 1e-4, nrm2_test)
REGISTER_PREC(double, 1e-6, nrm2_test)
REGISTER_PREC(long double, 1e-7, nrm2_test)

TYPED_TEST(BLAS1_Test, nrm2_test) {
  using ScalarT = typename TypeParam::scalar_t;
  using ExecutorType = typename TypeParam::executor_t;
  using TestClass = BLAS1_Test<TypeParam>;
  using test = class nrm2_test;

  size_t size = TestClass::template test_size<test>();
  size_t strd = TestClass::template test_strd<test>();
  ScalarT prec = TestClass::template test_prec<test>();

  DEBUG_PRINT(std::cout << "size == " << size << std::endl);
  DEBUG_PRINT(std::cout << "strd == " << strd << std::endl);

  // create a random vector
  std::vector<ScalarT> vX(size);
  // create a vector which will hold the result
  std::vector<ScalarT> vR(1, 0);
  TestClass::set_rand(vX, size);

  ScalarT res(0);
  // compute nrm2 (euclidean length) of vX into res in a for loop
  for (size_t i = 0; i < size; i += strd) {
    res += vX[i] * vX[i];
  }
  res = std::sqrt(res);

  SYCL_DEVICE_SELECTOR d;
  auto q = TestClass::make_queue(d);
  Executor<ExecutorType> ex(q);
  {
    // compute nrm2 of a vX into vR
    auto buf_vX = TestClass::make_buffer(vX);
    auto buf_vR = TestClass::make_buffer(vR);
    auto view_vX = TestClass::make_vview(buf_vX);
    auto view_vR = TestClass::make_vview(buf_vR);
    _nrm2(ex, (size+strd-1)/strd, view_vX, strd, view_vR);
  }
  // check that the result is the same
  ASSERT_NEAR(res, vR[0], prec);
}
