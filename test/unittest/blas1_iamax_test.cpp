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
 *  @filename blas1_iamax_test.cpp
 *
 **************************************************************************/

#include "blas1_test.hpp"

typedef ::testing::Types<blas1_test_args<double>> BlasTypes;

TYPED_TEST_CASE(BLAS1_Test, BlasTypes);

REGISTER_SIZE(::RANDOM_SIZE, iamax_test)
REGISTER_STRD(::RANDOM_STRD, iamax_test)

B1_TEST(iamax_test) {
  UNPACK_PARAM(iamax_test);
  size_t size = TEST_SIZE;
  size_t strd = TEST_STRD;

  std::vector<ScalarT> vX(size);
  TestClass::set_rand(vX, size);
  std::vector<IndVal<ScalarT>> vI(
      1, constant<IndVal<ScalarT>, const_val::imax>::value);

  std::cout << strd << std::endl;
  ScalarT max = 0.;
  size_t imax = std::numeric_limits<size_t>::max();
  for (size_t i = 0; i < size; i += strd) {
    if (std::abs(vX[i]) > std::abs(max)) {
      max = vX[i];
      imax = i;
    }
  }
  IndVal<ScalarT> res(imax, max);

  Device dev;
  {
    auto buf_vX = TestClass::make_buffer(vX);
    auto buf_vI = TestClass::make_buffer(vI);
    blas::execute(dev, _iamax((size+strd-1)/strd, buf_vX, 0, strd, buf_vI));
  }
  IndVal<ScalarT> res2(vI[0]);
  ASSERT_EQ(res.getVal(), res2.getVal());
  ASSERT_EQ(res.getInd(), res2.getInd());
}
