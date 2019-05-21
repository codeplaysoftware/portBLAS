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
 *  @filename blas1_reduction_asum_test.cpp
 *
 **************************************************************************/
#include "blas_test.hpp"
#include "sycl_blas.hpp"
typedef ::testing::Types<blas_test_float<>, blas_test_double<> > BlasTypes;

TYPED_TEST_CASE(BLAS_Test, BlasTypes);

REGISTER_SIZE(::RANDOM_SIZE, reduction_asum_test)
REGISTER_STRD(::RANDOM_STRD, reduction_asum_test)
REGISTER_PREC(float, 1e-4, reduction_asum_test)
REGISTER_PREC(double, 1e-6, reduction_asum_test)
//REGISTER_PREC(std::complex<float>, 1e-4, reduction_asum_test)
//REGISTER_PREC(std::complex<double>, 1e-6, reduction_asum_test)

TYPED_TEST(BLAS_Test, reduction_asum_test) {
  using scalar_t = typename TypeParam::scalar_t;
  using ExecutorType = typename TypeParam::executor_t;
  using TestClass = BLAS_Test<TypeParam>;
  using test = class reduction_asum_test;

  int size = TestClass::template test_size<test>();
  int strd = TestClass::template test_strd<test>();
  scalar_t prec = TestClass::template test_prec<test>();

  DEBUG_PRINT(std::cout << "size == " << size << std::endl);
  DEBUG_PRINT(std::cout << "strd == " << strd << std::endl);

  std::vector<scalar_t> vX(size);
  TestClass::set_rand(vX, size);

  std::vector<scalar_t> vR(1, scalar_t(0));
  scalar_t result = 0;
  for (int i = 0; i < size; i += strd) {
    result += std::abs(vX[i]);
  }

  auto q = make_queue();
  Executor<ExecutorType> ex(q);

  auto gpu_vX = blas::make_sycl_iterator_buffer<scalar_t>(vX, size);
  auto gpu_vR = blas::make_sycl_iterator_buffer<scalar_t>(int(1));

  auto view_vX = make_vector_view(ex, gpu_vX, strd, (size + strd - 1) / strd);
  auto view_vR = make_vector_view(ex, gpu_vR, strd, (size + strd - 1) / strd);

  ex.get_policy_handler().copy_to_device(vX.data(), gpu_vX, size);

  const auto local_size = ex.get_policy_handler().get_work_group_size();
  const auto nWG = 2 * local_size;

  auto asum_x_op = make_AssignReduction<AbsoluteAddOperator>
                          (view_vX, view_vR, local_size, local_size * nWG);

  auto event = ex.get_policy_handler().copy_to_host(gpu_vR, vR.data(), 1);
  ex.get_policy_handler().wait(event);
  ASSERT_NEAR(result, vR[0], prec);

  
}