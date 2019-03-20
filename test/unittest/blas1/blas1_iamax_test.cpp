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

TYPED_TEST(BLAS_Test, iamax_test) {
  using scalar_t = typename TypeParam::scalar_t;
  using ExecutorType = typename TypeParam::executor_t;
  using TestClass = BLAS_Test<TypeParam>;
  using test = class iamax_test;
  using index_t = int;
  index_t size = TestClass::template test_size<test>();
  index_t strd = TestClass::template test_strd<test>();

  DEBUG_PRINT(std::cout << "size == " << size << std::endl);
  DEBUG_PRINT(std::cout << "strd == " << strd << std::endl);

  // create a random vector vX
  std::vector<scalar_t> vX(size);
  TestClass::set_rand(vX, size);
  constexpr auto val =
      constant<IndexValueTuple<scalar_t, index_t>, const_val::imax>::value();
  // create a vector which will hold the result
  std::vector<IndexValueTuple<scalar_t, index_t>> vI(1, val);

  scalar_t max = scalar_t(0);
  index_t imax = std::numeric_limits<index_t>::max();
  // compute index and value of the element with biggest absolute value
  for (index_t i = 0; i < size; i += strd) {
    if (std::abs(vX[i]) > std::abs(max)) {
      max = vX[i];
      imax = i;
    }
  }
  IndexValueTuple<scalar_t, index_t> res(imax, max);

  SYCL_DEVICE_SELECTOR d;
  auto q = TestClass::make_queue(d);
  Executor<ExecutorType> ex(q);
  auto gpu_vX = blas::make_sycl_iterator_buffer<scalar_t>(vX, size);
  auto gpu_vI =
      blas::make_sycl_iterator_buffer<IndexValueTuple<scalar_t, index_t>>(
          index_t(1));
  _iamax(ex, (size + strd - 1) / strd, gpu_vX, strd, gpu_vI);
  auto event = ex.get_policy_handler().copy_to_host(gpu_vI, vI.data(), 1);
  ex.get_policy_handler().wait(event);

  // check that the result value is the same
  ASSERT_EQ(res.get_value(), vI[0].get_value());
  // check that the result index is the same
  ASSERT_EQ(res.get_index(), vI[0].get_index());
}

REGISTER_SIZE(::RANDOM_SIZE, iamax_test_vpr)
REGISTER_STRD(1, iamax_test_vpr)

TYPED_TEST(BLAS_Test, iamax_test_vpr) {
  using scalar_t = typename TypeParam::scalar_t;
  using ExecutorType = typename TypeParam::executor_t;
  using TestClass = BLAS_Test<TypeParam>;
  using test = class iamax_test_vpr;
  using index_t = int;

  index_t size = TestClass::template test_size<test>();
  index_t strd = TestClass::template test_strd<test>();

  DEBUG_PRINT(std::cout << "size == " << size << std::endl);
  DEBUG_PRINT(std::cout << "strd == " << strd << std::endl);

  // create a random vector vX
  std::vector<scalar_t> vX(size);
  TestClass::set_rand(vX, size);
  constexpr auto val =
      constant<IndexValueTuple<scalar_t, index_t>, const_val::imax>::value();
  // create a vector which will hold the result
  std::vector<IndexValueTuple<scalar_t, index_t>> vI(1, val);

  scalar_t max = scalar_t(0);
  index_t imax = std::numeric_limits<index_t>::max();
  // compute index and value of the element with biggest absolute value
  for (index_t i = 0; i < size; i += strd) {
    if (std::abs(vX[i]) > std::abs(max)) {
      max = vX[i];
      imax = i;
    }
  }
  IndexValueTuple<scalar_t, index_t> res(imax, max);

  SYCL_DEVICE_SELECTOR d;
  auto q = TestClass::make_queue(d);
  Executor<ExecutorType> ex(q);
  auto gpu_vX = ex.get_policy_handler().template allocate<scalar_t>(size);
  auto gpu_vI = ex.get_policy_handler()
                    .template allocate<IndexValueTuple<scalar_t, index_t>>(1);
  ex.get_policy_handler().copy_to_device(vX.data(), gpu_vX, size);
  _iamax(ex, (size + strd - 1) / strd, gpu_vX, strd, gpu_vI);
  auto event = ex.get_policy_handler().copy_to_host(gpu_vI, vI.data(), 1);
  ex.get_policy_handler().wait(event);

  IndexValueTuple<scalar_t, index_t> res2(vI[0]);
  // check that the result value is the same
  ASSERT_EQ(res.get_value(), res2.get_value());
  // check that the result index is the same
  ASSERT_EQ(res.get_index(), res2.get_index());
  ex.get_policy_handler().template deallocate<scalar_t>(gpu_vX);
  ex.get_policy_handler()
      .template deallocate<IndexValueTuple<scalar_t, index_t>>(gpu_vI);
}
