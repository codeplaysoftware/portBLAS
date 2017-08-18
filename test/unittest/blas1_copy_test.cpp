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
 *  @filename blas1_copy_test.cpp
 *
 **************************************************************************/

#include "blas1_test.hpp"

typedef ::testing::Types<blas1_test_args<float>, blas1_test_args<double> >
    BlasTypes;

TYPED_TEST_CASE(BLAS1_Test, BlasTypes);

REGISTER_SIZE(::RANDOM_SIZE, copy_test)
REGISTER_STRD(::RANDOM_STRD, copy_test)

B1_TEST(copy_test) {
  UNPACK_PARAM(copy_test);
  size_t size = TEST_SIZE;
  size_t strd = TEST_STRD;

  std::vector<ScalarT> vX(size);
  TestClass::set_rand(vX, size);

  std::vector<ScalarT> vY(size, 0);
  Device dev;
  {
    auto buf_vX = TestClass::make_buffer(vX);
    auto buf_vY = TestClass::make_buffer(vY);
    blas::execute(dev, _copy((size+strd-1)/strd, buf_vX, 0, strd, buf_vY, 0, strd));
  }
  for (size_t i = 0; i < size; ++i) {
    if (i % strd == 0) {
      ASSERT_EQ(vX[i], vY[i]);
    } else {
      ASSERT_EQ(0, vY[i]);
    }
  }
}
