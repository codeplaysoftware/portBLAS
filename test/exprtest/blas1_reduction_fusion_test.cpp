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
 *  @filename blas1_reduction_reduction_fusion_test.cpp
 *
 **************************************************************************/

#include "blas1_test.hpp"

typedef ::testing::Types<blas1_test_args<float>, blas1_test_args<double> > BlasTypes;

TYPED_TEST_CASE(BLAS1_Test, BlasTypes);

REGISTER_SIZE(::RANDOM_SIZE, reduction_fusion_test)
REGISTER_STRD(::RANDOM_STRD, reduction_fusion_test)
REGISTER_PREC(float, 1e-4, reduction_fusion_test)
REGISTER_PREC(double, 1e-6, reduction_fusion_test)
REGISTER_PREC(std::complex<float>, 1e-4, reduction_fusion_test)
REGISTER_PREC(std::complex<double>, 1e-6, reduction_fusion_test)

/*
 * Checks that the fusion of scal(alpha, dot(nrm2(vX), asum(vX))) works correct
 */
TYPED_TEST(BLAS1_Test, reduction_fusion_test) {
  using ScalarT = typename TypeParam::scalar_t;
  using Device = typename TypeParam::device_t;
  using TestClass = BLAS1_Test<TypeParam>;
  using test = class reduction_fusion_test;

  size_t size = TestClass::template test_size<test>();
  size_t strd = TestClass::template test_strd<test>();
  ScalarT prec = TestClass::template test_prec<test>();

  DEBUG_PRINT(std::cout << "size == " << size << std::endl);
  DEBUG_PRINT(std::cout << "strd == " << strd << std::endl);

  // scalar for _scal operation
  ScalarT alpha(2);
  // creating two vectors: vX (for nrm2) and vY (for asum)
  std::vector<ScalarT> vX(size);
  std::vector<ScalarT> vY(size);
  TestClass::set_rand(vX, size);
  TestClass::set_rand(vY, size);

  // these scalars will carry the result of nrm2 and asum respectively
  ScalarT a(.0), b(.0), c(.0);
  // this vector will carry the results of the same operations computed with sycl-blas
  std::vector<ScalarT> vR1(1), vR2(1), vR3(1);

  SYCL_DEVICE_SELECTOR d;
  // compute axpy(scal) in a for loop and put the results into vZ and vT
  for (size_t i = 0; i < size; i+=strd) {
    auto x = vX[i];
    auto y = vY[i];
    a += x*x;
    b += std::fabs(y);
  }
  a = std::sqrt(a);
  c = alpha * a * b;

  auto q = TestClass::make_queue(d);
  Device dev(q);
  {
    // compute scal, axpy(scal) with syclblas and put the result into vZ, vT
    auto buf_vX = TestClass::make_buffer(vX);
    auto buf_vY = TestClass::make_buffer(vY);
    auto buf_vR1 = TestClass::make_buffer(vR1);
    auto buf_vR2 = TestClass::make_buffer(vR2);
    auto buf_vR3 = TestClass::make_buffer(vR3);

    size_t N = (size+strd-1)/strd;
    auto nrm2 = _nrm2(N, buf_vX, 0, strd, buf_vR1);
    auto asum = _asum(N, buf_vY, 0, strd, buf_vR2);
    auto dot = _dot(1, nrm2, 0, 1, asum, 0, 1, buf_vR3);
    auto scal = _scal(1, alpha, dot, 0, 1);
    blas::execute(dev, scal);
  }
  // check that both results are the same
  ASSERT_NEAR(a, vR1[0], prec);
  ASSERT_NEAR(b, vR2[0], prec);
  ASSERT_NEAR(c, vR3[0], prec);
}
