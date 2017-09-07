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
 *  @filename blas1_fusion_test.cpp
 *
 **************************************************************************/

#include "blas1_test.hpp"

typedef ::testing::Types<blas1_test_args<float>, blas1_test_args<double> > BlasTypes;

TYPED_TEST_CASE(BLAS1_Test, BlasTypes);

REGISTER_SIZE(::RANDOM_SIZE, fusion_test)
REGISTER_STRD(::RANDOM_STRD, fusion_test)
REGISTER_PREC(float, 1e-4, fusion_test)
REGISTER_PREC(double, 1e-6, fusion_test)
REGISTER_PREC(std::complex<float>, 1e-4, fusion_test)
REGISTER_PREC(std::complex<double>, 1e-6, fusion_test)

TYPED_TEST(BLAS1_Test, fusion_test) {
  using ScalarT = typename TypeParam::scalar_t;
  using Device = typename TypeParam::device_t;
  using TestClass = BLAS1_Test<TypeParam>;
  using test = class fusion_test;

  size_t size = TestClass::template test_size<test>();
  size_t strd = TestClass::template test_strd<test>();
  ScalarT prec = TestClass::template test_prec<test>();

  DEBUG_PRINT(std::cout << "size == " << size << std::endl);
  DEBUG_PRINT(std::cout << "strd == " << strd << std::endl);

  // axpy(alpha, scal(beta, vX), vY) = (vY = alpha * vX + vY)
  // setting _scal and _axpy scalars to some values
  ScalarT alpha(1.54);
  ScalarT beta(.32);
  // creating four vectors: vX, vY, vZ and vT.
  // the for loop will compute axpy for vX, vY
  std::vector<ScalarT> vX(size);
  std::vector<ScalarT> vY(size);
  std::vector<ScalarT> vZ(size, 0);
  std::vector<ScalarT> vT(size, 0);
  TestClass::set_rand(vX, size);
  TestClass::set_rand(vY, size);

  SYCL_DEVICE_SELECTOR d;
  // compute axpy(scal) in a for loop and put the results into vZ and vT
  for (size_t i = 0; i < size; ++i) {
    auto &x = vX[i];
    auto &y = vY[i];
    auto &_x = vZ[i];
    auto &_y = vT[i];
    if (i % strd == 0) {
      _x = alpha * x;
      _y = beta * _x + y;
    } else {
      _x = x;
      _y = y;
    }
  }

  auto q = TestClass::make_queue(d);
  Device dev(q);
  {
    // compute scal, axpy(scal) with syclblas and put the result into vZ, vT
    auto buf_vX = TestClass::make_buffer(vX);
    auto buf_vY = TestClass::make_buffer(vY);

    auto scal = _scal((size+strd-1)/strd, alpha, buf_vX, 0, strd);
    auto scaxpy = _axpy((size+strd-1)/strd, beta, scal, 0, strd, buf_vY, 0, strd);

    blas::execute(dev, scaxpy);
  }
  // check that both results are the same
  for (size_t i = 0; i < size; ++i) {
    ASSERT_NEAR(vZ[i], vX[i], prec);
    ASSERT_NEAR(vT[i], vY[i], prec);
  }
}
