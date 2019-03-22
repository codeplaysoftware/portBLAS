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
 *  @filename blas1_iamin_test.cpp
 *
 **************************************************************************/

#include "blas_test.hpp"

typedef ::testing::Types<blas_test_float<>, blas_test_double<>> BlasTypes;

TYPED_TEST_CASE(BLAS_Test, BlasTypes);

REGISTER_SIZE(::RANDOM_SIZE, iamin_test)
REGISTER_STRD(1, iamin_test)

TYPED_TEST(BLAS_Test, iamin_test) {
  using scalar_t = typename TypeParam::scalar_t;
  using ExecutorType = typename TypeParam::executor_t;
  using TestClass = BLAS_Test<TypeParam>;
  using test = class iamin_test;
  using index_t = int;
  index_t size = TestClass::template test_size<test>();
  index_t strd = TestClass::template test_strd<test>();

  DEBUG_PRINT(std::cout << "size == " << size << std::endl);
  DEBUG_PRINT(std::cout << "strd == " << strd << std::endl);

  std::vector<scalar_t> vX(size);
  TestClass::set_rand(vX, size);
  constexpr auto val =
      constant<IndexValueTuple<scalar_t, index_t>, const_val::imin>::value();
  std::vector<IndexValueTuple<scalar_t, index_t>> vI(1, val);

  // compute iamin of vX into res with a for loop
  scalar_t min = std::numeric_limits<scalar_t>::max();
  index_t imin = std::numeric_limits<index_t>::max();
  for (index_t i = 0; i < size; i += strd) {
    if (std::abs(vX[i]) < std::abs(min)) {
      min = vX[i];
      imin = i;
    }
  }
  IndexValueTuple<scalar_t, index_t> res(imin, min);

  SYCL_DEVICE_SELECTOR d;
  auto q = TestClass::make_queue(d);
  Executor<ExecutorType> ex(q);
  auto gpu_vX = ex.get_policy_handler().template allocate<scalar_t>(size);
  auto gpu_vI =
      ex.get_policy_handler()
          .template allocate<IndexValueTuple<scalar_t, index_t>>(index_t(1));
  ex.get_policy_handler().copy_to_device(vX.data(), gpu_vX, size);
  _iamin(ex, (size + strd - 1) / strd, gpu_vX, strd, gpu_vI);
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
