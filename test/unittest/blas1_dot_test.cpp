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
 *  @filename blas1_dot_test.cpp
 *
 **************************************************************************/

#include "blas1_test.hpp"

typedef ::testing::Types<blas1_test_args<float>, blas1_test_args<double> >
    BlasTypes;

TYPED_TEST_CASE(BLAS1_Test, BlasTypes);

REGISTER_SIZE(::RANDOM_SIZE, dot_test)
REGISTER_STRD(::RANDOM_STRD, dot_test)
REGISTER_PREC(float, 1e-4, dot_test)
REGISTER_PREC(double, 1e-6, dot_test)
REGISTER_PREC(long double, 1e-7, dot_test)

B1_TEST(dot_test) {
  UNPACK_PARAM(dot_test);
  size_t size = TEST_SIZE;
  size_t strd = TEST_STRD;
  ScalarT prec = TEST_PREC;

  std::vector<ScalarT> vX(size);
  std::vector<ScalarT> vY(size);
  std::vector<ScalarT> vR(1, 0);
  TestClass::set_rand(vX, size);
  TestClass::set_rand(vY, size);

  ScalarT res(0);
  for (size_t i = 0; i < size; i += strd) {
    res += vX[i] * vY[i];
  }

  Device dev;
  {
    auto buf_vX = TestClass::make_buffer(vX);
    auto buf_vY = TestClass::make_buffer(vY);
    auto buf_vR = TestClass::make_buffer(vR);
    blas::execute(dev, _dot((size+strd-1)/strd, buf_vX, 0, strd, buf_vY, 0, strd, buf_vR));
  }
  ASSERT_NEAR(res, vR[0], prec);
}
