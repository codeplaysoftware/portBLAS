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
 *  @filename sycl_buffer_test.cpp
 *
 **************************************************************************/

#include "blas_test.hpp"
#include "queue/sycl_iterator.hpp"
typedef ::testing::Types<blas_test_args<float>, blas_test_args<double> >
    BlasTypes;

TYPED_TEST_CASE(BLAS_Test, BlasTypes);

REGISTER_SIZE(::RANDOM_SIZE, sycl_buffer_test)
REGISTER_STRD(::RANDOM_STRD, sycl_buffer_test)
REGISTER_PREC(float, 1e-4, sycl_buffer_test)
REGISTER_PREC(double, 1e-6, sycl_buffer_test)
REGISTER_PREC(long double, 1e-7, sycl_buffer_test)

TYPED_TEST(BLAS_Test, sycl_buffer_test) {
  using ScalarT = typename TypeParam::scalar_t;
  using ExecutorType = typename TypeParam::executor_t;
  using TestClass = BLAS_Test<TypeParam>;
  using test = class sycl_buffer_test;

  size_t size = TestClass::template test_size<test>();
  std::ptrdiff_t offset = TestClass::template test_strd<test>();
  ScalarT prec = TestClass::template test_prec<test>();
  size_t strd = TestClass::template test_strd<test>();

  DEBUG_PRINT(std::cout << "size == " << size << std::endl);
  DEBUG_PRINT(std::cout << "strd == " << strd << std::endl);

  std::vector<ScalarT> vX(size, ScalarT(1));
  TestClass::set_rand(vX, size);

  std::vector<ScalarT> vR(size - offset, ScalarT(0));

  SYCL_DEVICE_SELECTOR d;
  auto q = TestClass::make_queue(d);
  Executor<ExecutorType> ex(q);
  auto a = blas::helper::make_sycl_iteator_buffer<ScalarT>(vX.data(), size);
  auto event = ex.copy_to_host((a + offset), vR.data(), size - offset);
  ex.sync(event);

  for (auto i = 0; i < size; i++) {
    ASSERT_NEAR(vX[i + offset], vR[i], prec);
  }
}
