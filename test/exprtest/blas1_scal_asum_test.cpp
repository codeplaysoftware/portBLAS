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
REGISTER_PREC(float, 1e-3, reduction_asum_test) // Lowered precision for _scal
REGISTER_PREC(double, 1e-4, reduction_asum_test)
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

  // Select some alpha for _scal
  scalar_t alpha = 1.12;

  std::vector<scalar_t> vX(size);
  std::vector<scalar_t> vXResult(size);
  std::vector<scalar_t> vY(1);
  TestClass::set_rand(vX, size);

  // Compute the _scal
  for (int i = 0; i < size; i += strd) {
    vXResult[i] = vX[i] * alpha;
  }

  // Compute the _asum
  scalar_t scal_asum_result = 0;
  for (int i = 0; i < size; i += strd) {
    scal_asum_result += vXResult[i];
  }

  // Now repeat the same operations using SYCL BLAS
  auto q = make_queue();
  Executor<ExecutorType> ex(q);

  auto gpu_vX = blas::make_sycl_iterator_buffer<scalar_t>(vX, size);
  auto gpu_vY = blas::make_sycl_iterator_buffer<scalar_t>(vY, 1);
  auto view_vX = make_vector_view(ex, gpu_vX, strd, (size + strd - 1) / strd);
  auto view_vY = make_vector_view(ex, gpu_vY, 1, 1);

  ex.get_policy_handler().copy_to_device(vX.data(), gpu_vX, size);

  auto scal_op = make_op<ScalarOp, ProductOperator>(alpha, view_vX);
  auto assign_op = make_op<Assign>(view_vX, scal_op);

  const auto work_group_size = ex.get_policy_handler().get_work_group_size();
  const auto nWG = 2 * work_group_size;

  auto sacl_asum_op_tree = make_AssignReduction<AddOperator>
                    (view_vY, assign_op, work_group_size, work_group_size * nWG);

  auto ev = ex.execute(sacl_asum_op_tree);
  ex.get_policy_handler().wait(ev);

  auto event = ex.get_policy_handler().copy_to_host(gpu_vY, vY.data(), 1);
  ex.get_policy_handler().wait(event);
  ASSERT_NEAR(scal_asum_result, vY[0], prec);
}