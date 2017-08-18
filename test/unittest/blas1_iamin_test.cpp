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
 *  @filename blas1_iamin_test.cpp
 *
 **************************************************************************/

#include "blas1_test.hpp"

typedef ::testing::Types<blas1_test_args<double>> BlasTypes;

TYPED_TEST_CASE(BLAS1_Test, BlasTypes);

REGISTER_SIZE(::RANDOM_SIZE, iamin_test)
REGISTER_STRD(1, iamin_test)

TYPED_TEST(BLAS1_Test, iamin_test) {
  using ScalarT = typename TypeParam::scalar_t;
  using ExecutorType = typename TypeParam::executor_t;
  using TestClass = BLAS1_Test<TypeParam>;
  using test = class iamin_test;

  size_t size = TestClass::template test_size<test>();
  size_t strd = TestClass::template test_strd<test>();

  DEBUG_PRINT(std::cout << "size == " << size << std::endl);
  DEBUG_PRINT(std::cout << "strd == " << strd << std::endl);

  std::vector<ScalarT> vX(size);
  TestClass::set_rand(vX, size);
  std::vector<IndVal<ScalarT>> vI(
      1, constant<IndVal<ScalarT>, const_val::imin>::value);

  // compute iamin of vX into res with a for loop
  ScalarT min = std::numeric_limits<ScalarT>::max();
  size_t imin = std::numeric_limits<size_t>::max();
  for (size_t i = 0; i < size; i += strd) {
    if (std::abs(vX[i]) < std::abs(min)) {
      min = vX[i];
      imin = i;
    }
  }
  IndVal<ScalarT> res(imin, min);

  SYCL_DEVICE_SELECTOR d;
  auto q = TestClass::make_queue(d);
  Executor<ExecutorType> ex(q);
  {
    // compute iamax of vX into vI with sycl blas
    auto buf_vX = TestClass::make_buffer(vX);
    auto buf_vI = TestClass::make_buffer(vI);
    auto view_vX = TestClass::make_vview(buf_vX);
    auto view_vI = TestClass::make_vview(buf_vI);
    _iamin(ex, (size+strd-1)/strd, view_vX, strd, view_vI);
  }
  IndVal<ScalarT> res2(vI[0]);
  // check that the result value is the same
  ASSERT_EQ(res.getVal(), res2.getVal());
  // check that the result index is the same
  ASSERT_EQ(res.getInd(), res2.getInd());
}
