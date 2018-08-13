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

typedef ::testing::Types<blas_test_args<float>, blas_test_args<double>>
    BlasTypes;

TYPED_TEST_CASE(BLAS_Test, BlasTypes);

REGISTER_PREC(float, 1e-4, ger_test)
REGISTER_PREC(double, 1e-8, ger_test)
REGISTER_PREC(long double, 1e-8, ger_test)

TYPED_TEST(BLAS_Test, ger_test) {
  using ScalarT = typename TypeParam::scalar_t;
  using ExecutorType = typename TypeParam::executor_t;
  using TestClass = BLAS_Test<TypeParam>;
  using test = class ger_test;

  size_t m = 125;
  size_t n = 127;
  size_t lda = m;
  long incX = 1;
  long incY = 1;
  ScalarT prec = TestClass::template test_prec<test>();
  ScalarT alpha = ScalarT(3);

  // Input matrix
  std::vector<ScalarT> a_v(m);
  // Input Vector
  std::vector<ScalarT> b_v(n);
  // output Vector
  std::vector<ScalarT> c_m_gpu_result(m * n, ScalarT(0));
  // output system vector
  std::vector<ScalarT> c_m_cpu(m * n, ScalarT(0));
  TestClass::set_rand(a_v, m);
  TestClass::set_rand(b_v, n);

  // SYSTEM GEMMV
  ger(m, n, alpha, a_v.data(), incX, b_v.data(), incY, c_m_cpu.data(), m);

  SYCL_DEVICE_SELECTOR d;
  auto q = TestClass::make_queue(d);
  Executor<ExecutorType> ex(q);
  auto v_a_gpu = ex.template allocate<ScalarT>(m);
  auto v_b_gpu = ex.template allocate<ScalarT>(n);
  auto m_c_gpu = ex.template allocate<ScalarT>(m * n);
  ex.copy_to_device(a_v.data(), v_a_gpu, m);
  ex.copy_to_device(b_v.data(), v_b_gpu, n);
  ex.copy_to_device(c_m_gpu_result.data(), m_c_gpu, m * n);
  // SYCLger
  _ger(ex, m, n, alpha, v_a_gpu, incX, v_b_gpu, incY, m_c_gpu, m);

  auto event = ex.copy_to_host(m_c_gpu, c_m_gpu_result.data(), m * n);
  ex.sync(event);

  for (size_t i = 0; i < m * n; ++i) {
    ASSERT_NEAR(c_m_gpu_result[i], c_m_cpu[i], prec);
  }
  ex.template deallocate<ScalarT>(v_a_gpu);
  ex.template deallocate<ScalarT>(v_b_gpu);
  ex.template deallocate<ScalarT>(m_c_gpu);
}
