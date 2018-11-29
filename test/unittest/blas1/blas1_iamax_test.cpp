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
 *  @filename blas1_iamax_test.cpp
 *
 **************************************************************************/

#include "blas_test.hpp"
typedef ::testing::Types<blas_test_float<>, blas_test_double<>> BlasTypes;

TYPED_TEST_CASE(BLAS_Test, BlasTypes);
REGISTER_SIZE(::RANDOM_SIZE, iamax_test)
REGISTER_STRD(1, iamax_test)
REGISTER_PREC(float, 1e-4, iamax_test)
REGISTER_PREC(double, 1e-6, iamax_test)
REGISTER_PREC(long double, 1e-7, iamax_test)

TYPED_TEST(BLAS_Test, iamax_test) {
  using ScalarT = typename TypeParam::scalar_t;
  using ExecutorType = typename TypeParam::executor_t;
  using TestClass = BLAS_Test<TypeParam>;
  using test = class iamax_test;
  using IndexType = int;
  IndexType size = TestClass::template test_size<test>();
  IndexType strd = TestClass::template test_strd<test>();

  DEBUG_PRINT(std::cout << "size == " << size << std::endl);
  DEBUG_PRINT(std::cout << "strd == " << strd << std::endl);

  // create a random vector vX
  std::vector<ScalarT> vX(size);
  TestClass::set_rand(vX, size);
  constexpr auto val =
      constant<IndexValueTuple<ScalarT, IndexType>, const_val::imax>::value;
  // create a vector which will hold the result
  std::vector<IndexValueTuple<ScalarT, IndexType>> vI(1, val);

  ScalarT max = ScalarT(0);
  IndexType imax = std::numeric_limits<IndexType>::max();
  // compute index and value of the element with biggest absolute value
  for (IndexType i = 0; i < size; i += strd) {
    if (std::abs(vX[i]) > std::abs(max)) {
      max = vX[i];
      imax = i;
    }
  }
  IndexValueTuple<ScalarT, IndexType> res(imax, max);

  SYCL_DEVICE_SELECTOR d;
  auto q = TestClass::make_queue(d);
  Executor<ExecutorType> ex(q);
  auto gpu_vX = blas::helper::make_sycl_iterator_buffer<ScalarT>(vX, size);
  auto gpu_vI = blas::helper::make_sycl_iterator_buffer<
      IndexValueTuple<ScalarT, IndexType>>(IndexType(1));
  _iamax(ex, (size + strd - 1) / strd, gpu_vX, strd, gpu_vI);
  auto event = ex.copy_to_host(gpu_vI, vI.data(), 1);
  ex.wait(event);

  // check that the result value is the same
  ASSERT_EQ(res.get_value(), vI[0].get_value());
  // check that the result index is the same
  ASSERT_EQ(res.get_index(), vI[0].get_index());
}

REGISTER_SIZE(::RANDOM_SIZE, iamax_test_vpr)
REGISTER_STRD(1, iamax_test_vpr)

TYPED_TEST(BLAS_Test, iamax_test_vpr) {
  using ScalarT = typename TypeParam::scalar_t;
  using ExecutorType = typename TypeParam::executor_t;
  using TestClass = BLAS_Test<TypeParam>;
  using test = class iamax_test_vpr;
  using IndexType = int;

  IndexType size = TestClass::template test_size<test>();
  IndexType strd = TestClass::template test_strd<test>();

  DEBUG_PRINT(std::cout << "size == " << size << std::endl);
  DEBUG_PRINT(std::cout << "strd == " << strd << std::endl);

  // create a random vector vX
  std::vector<ScalarT> vX(size);
  TestClass::set_rand(vX, size);
  constexpr auto val =
      constant<IndexValueTuple<ScalarT, IndexType>, const_val::imax>::value;
  // create a vector which will hold the result
  std::vector<IndexValueTuple<ScalarT, IndexType>> vI(1, val);

  ScalarT max = ScalarT(0);
  IndexType imax = std::numeric_limits<IndexType>::max();
  // compute index and value of the element with biggest absolute value
  for (IndexType i = 0; i < size; i += strd) {
    if (std::abs(vX[i]) > std::abs(max)) {
      max = vX[i];
      imax = i;
    }
  }
  IndexValueTuple<ScalarT, IndexType> res(imax, max);

  SYCL_DEVICE_SELECTOR d;
  auto q = TestClass::make_queue(d);
  Executor<ExecutorType> ex(q);
  auto gpu_vX = ex.template allocate<ScalarT>(size);
  auto gpu_vI = ex.template allocate<IndexValueTuple<ScalarT, IndexType>>(1);
  ex.copy_to_device(vX.data(), gpu_vX, size);
  _iamax(ex, (size + strd - 1) / strd, gpu_vX, strd, gpu_vI);
  auto event = ex.copy_to_host(gpu_vI, vI.data(), 1);
  ex.wait(event);

  IndexValueTuple<ScalarT, IndexType> res2(vI[0]);
  // check that the result value is the same
  ASSERT_EQ(res.get_value(), res2.get_value());
  // check that the result index is the same
  ASSERT_EQ(res.get_index(), res2.get_index());
  ex.template deallocate<ScalarT>(gpu_vX);
  ex.template deallocate<IndexValueTuple<ScalarT, IndexType>>(gpu_vI);
}