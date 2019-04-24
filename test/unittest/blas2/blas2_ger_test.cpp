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
 *  @filename blas2_ger_test.cpp
 *
 **************************************************************************/

#include "blas_test.hpp"

typedef ::testing::Types<blas_test_float<>, blas_test_double<> > BlasTypes;

TYPED_TEST_CASE(BLAS_Test, BlasTypes);

REGISTER_PREC(float, 1e-4, ger_test)
REGISTER_PREC(double, 1e-8, ger_test)

TYPED_TEST(BLAS_Test, ger_test) {
  using scalar_t = typename TypeParam::scalar_t;
  using ExecutorType = typename TypeParam::executor_t;
  using TestClass = BLAS_Test<TypeParam>;
  using test = class ger_test;

  int m = 125;
  int n = 127;
  int lda = m;
  int incX = 1;
  int incY = 1;
  scalar_t prec = TestClass::template test_prec<test>();
  scalar_t alpha = scalar_t(3);

  // Input matrix
  std::vector<scalar_t> a_v(m);
  // Input Vector
  std::vector<scalar_t> b_v(n);
  // output Vector
  std::vector<scalar_t> c_m_gpu_result(m * n, scalar_t(0));
  // output system vector
  std::vector<scalar_t> c_m_cpu(m * n, scalar_t(0));
  TestClass::set_rand(a_v, m);
  TestClass::set_rand(b_v, n);

  // SYSTEM GEMMV
  reference_blas::ger(m, n, alpha, a_v.data(), incX, b_v.data(), incY,
                      c_m_cpu.data(), m);

  SYCL_DEVICE_SELECTOR d;
  auto q = TestClass::make_queue(d);
  Executor<ExecutorType> ex(q);
  auto v_a_gpu = ex.get_policy_handler().template allocate<scalar_t>(m);
  auto v_b_gpu = ex.get_policy_handler().template allocate<scalar_t>(n);
  auto m_c_gpu = ex.get_policy_handler().template allocate<scalar_t>(m * n);
  ex.get_policy_handler().copy_to_device(a_v.data(), v_a_gpu, m);
  ex.get_policy_handler().copy_to_device(b_v.data(), v_b_gpu, n);
  ex.get_policy_handler().copy_to_device(c_m_gpu_result.data(), m_c_gpu, m * n);
  // SYCLger
  _ger(ex, m, n, alpha, v_a_gpu, incX, v_b_gpu, incY, m_c_gpu, m);

  auto event = ex.get_policy_handler().copy_to_host(
      m_c_gpu, c_m_gpu_result.data(), m * n);
  ex.get_policy_handler().wait(event);

  for (int i = 0; i < m * n; ++i) {
    ASSERT_NEAR(c_m_gpu_result[i], c_m_cpu[i], prec);
  }
  ex.get_policy_handler().template deallocate<scalar_t>(v_a_gpu);
  ex.get_policy_handler().template deallocate<scalar_t>(v_b_gpu);
  ex.get_policy_handler().template deallocate<scalar_t>(m_c_gpu);
}
