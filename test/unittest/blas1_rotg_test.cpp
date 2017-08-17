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
 *  @filename blas1_rotg_test.cpp
 *
 **************************************************************************/

#include "blas1_test.hpp"

typedef ::testing::Types<blas1_test_args<float>, blas1_test_args<double> >
    BlasTypes;

TYPED_TEST_CASE(BLAS1_Test, BlasTypes);

REGISTER_SIZE(2, rotg_test)
REGISTER_STRD(1, rotg_test)
REGISTER_PREC(float, 1e-4, rotg_test)
REGISTER_PREC(double, 1e-7, rotg_test)

TYPED_TEST(BLAS1_Test, rotg_test) {
  using ScalarT = typename TypeParam::scalar_t;
  using ExecutorType = typename TypeParam::executor_t;
  using TestClass = BLAS1_Test<TypeParam>;
  using test = class rotg_test;

  DEBUG_PRINT(std::cout << "size == " << size << std::endl);
  DEBUG_PRINT(std::cout << "strd == " << strd << std::endl);
  size_t size = TestClass::template test_size<test>();
  size_t strd = TestClass::template test_strd<test>();
  ScalarT prec = TestClass::template test_prec<test>();

  std::vector<ScalarT> vX(size);
  std::vector<ScalarT> vY(size);
  std::vector<ScalarT> vR(1, 0);
  TestClass::set_rand(vX, size);
  TestClass::set_rand(vY, size);

  SYCL_DEVICE_SELECTOR d;
  ScalarT _cos, _sin;

  ScalarT giv = 0;
  ScalarT diff = 0;
  for(size_t i = 0; i < size; i += strd) {
    ScalarT x = vX[i], y = vY[i];
    if(i == 0) {
      diff = vY[i] - vX[i];
      _rotg(x, y, _cos, _sin);
    }
    x = vX[i], y = vY[i];
    giv += ((vX[i] * _cos + vY[i] * _sin) * (vY[i] * _cos - vX[i] * _sin));
    if(i == 0) {
      diff = (vY[i] * _cos - vX[i] * _sin) - ((vX[i] * _cos + vY[i] * _sin));
    } else {
      diff += (vY[i] * _cos - vX[i] * _sin) - ((vX[i] * _cos + vY[i] * _sin));
    }
  }

  auto q = TestClass::make_queue(d);
  Executor<ExecutorType> ex(q);
  {
    auto buf_vX = TestClass::make_buffer(vX);
    auto buf_vY = TestClass::make_buffer(vY);
    auto buf_res = TestClass::make_buffer(vR);
    auto view_vX = TestClass::make_vview(buf_vX);
    auto view_vY = TestClass::make_vview(buf_vY);
    auto view_res = TestClass::make_vview(buf_res);
    _dot(ex, size/strd, view_vX, strd, view_vY, strd, view_res);
  }
  ASSERT_NEAR(giv, vR[0], prec);
}
