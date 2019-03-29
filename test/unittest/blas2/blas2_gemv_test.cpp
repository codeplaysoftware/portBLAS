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
 *  @filename blas2_gemv_test.cpp
 *
 **************************************************************************/

#include "blas_test.hpp"

typedef ::testing::Types<blas_test_float<>, blas_test_double<> > BlasTypes;

TYPED_TEST_CASE(BLAS_Test, BlasTypes);

REGISTER_PREC(float, 1e-4, gemv_test)
REGISTER_PREC(double, 1e-8, gemv_test)

TYPED_TEST(BLAS_Test, gemv_test) {
  using scalar_t = typename TypeParam::scalar_t;
  using ExecutorType = typename TypeParam::executor_t;
  using TestClass = BLAS_Test<TypeParam>;
  using test = class gemv_test;

  int m = 125;
  int n = 127;
  int lda = m;
  int incX = 1;
  int incY = 1;
  const char* t_str = "n";  // Testing the no transpose matrix
  scalar_t prec = TestClass::template test_prec<test>();
  scalar_t alpha = scalar_t(1);
  scalar_t beta = scalar_t(1);

  // Input matrix
  std::vector<scalar_t> a_m(m * n);
  // Input Vector
  std::vector<scalar_t> b_v(n);
  // output Vector
  std::vector<scalar_t> c_v_gpu_result(m, scalar_t(0));
  // output system vector
  std::vector<scalar_t> c_v_cpu(m, scalar_t(0));
  TestClass::set_rand(a_m, m * n);
  TestClass::set_rand(b_v, n);

  // SYSTEM GEMMV
  reference_blas::gemv(t_str, m, n, alpha, a_m.data(), m, b_v.data(), incX,
                       beta, c_v_cpu.data(), incY);

  SYCL_DEVICE_SELECTOR d;
  auto q = TestClass::make_queue(d);
  Executor<ExecutorType> ex(q);
  auto m_a_gpu = ex.get_policy_handler().template allocate<scalar_t>(m * n);
  auto v_b_gpu = ex.get_policy_handler().template allocate<scalar_t>(n);
  auto v_c_gpu = ex.get_policy_handler().template allocate<scalar_t>(m);
  ex.get_policy_handler().copy_to_device(a_m.data(), m_a_gpu, m * n);
  ex.get_policy_handler().copy_to_device(b_v.data(), v_b_gpu, n);
  ex.get_policy_handler().copy_to_device(c_v_gpu_result.data(), v_c_gpu, m);
  // SYCLGEMV
  _gemv(ex, *t_str, m, n, alpha, m_a_gpu, m, v_b_gpu, incX, beta, v_c_gpu,
        incY);
  auto event =
      ex.get_policy_handler().copy_to_host(v_c_gpu, c_v_gpu_result.data(), m);
  ex.get_policy_handler().wait(event);

  for (int i = 0; i < m; ++i) {
    ASSERT_NEAR(c_v_gpu_result[i], c_v_cpu[i], prec);
  }
  ex.get_policy_handler().template deallocate<scalar_t>(m_a_gpu);
  ex.get_policy_handler().template deallocate<scalar_t>(v_b_gpu);
  ex.get_policy_handler().template deallocate<scalar_t>(v_c_gpu);
}

REGISTER_PREC(float, 1e-4, gemv_test_transposed)
REGISTER_PREC(double, 1e-8, gemv_test_transposed)

TYPED_TEST(BLAS_Test, gemv_test_transposed) {
  using scalar_t = typename TypeParam::scalar_t;
  using ExecutorType = typename TypeParam::executor_t;
  using TestClass = BLAS_Test<TypeParam>;
  using test = class gemv_test_transposed;

  int m = 125;
  int n = 127;
  int lda = m;
  int incX = 1;
  int incY = 1;
  const char* t_str = "t";  // Testing the no transpose matrix
  scalar_t prec = TestClass::template test_prec<test>();
  scalar_t alpha = scalar_t(1);
  scalar_t beta = scalar_t(1);

  // Input matrix
  std::vector<scalar_t> a_m(m * n);
  // Input Vector
  std::vector<scalar_t> b_v(m);
  // output Vector
  std::vector<scalar_t> c_v_gpu_result(n, scalar_t(0));
  // output system vector
  std::vector<scalar_t> c_v_cpu(n, scalar_t(0));
  TestClass::set_rand(a_m, m * n);
  TestClass::set_rand(b_v, m);

  // SYSTEM GEMMV
  reference_blas::gemv(t_str, m, n, alpha, a_m.data(), m, b_v.data(), incX,
                       beta, c_v_cpu.data(), incY);

  SYCL_DEVICE_SELECTOR d;
  auto q = TestClass::make_queue(d);
  Executor<ExecutorType> ex(q);
  auto m_a_gpu = ex.get_policy_handler().template allocate<scalar_t>(m * n);
  auto v_b_gpu = ex.get_policy_handler().template allocate<scalar_t>(m);
  auto v_c_gpu = ex.get_policy_handler().template allocate<scalar_t>(n);
  ex.get_policy_handler().copy_to_device(a_m.data(), m_a_gpu, m * n);
  ex.get_policy_handler().copy_to_device(b_v.data(), v_b_gpu, m);
  ex.get_policy_handler().copy_to_device(c_v_gpu_result.data(), v_c_gpu, n);
  // SYCLGEMV
  _gemv(ex, *t_str, m, n, alpha, m_a_gpu, m, v_b_gpu, incX, beta,
               v_c_gpu, incY);
  auto event =
      ex.get_policy_handler().copy_to_host(v_c_gpu, c_v_gpu_result.data(), n);
  ex.get_policy_handler().wait(event);

  ex.get_policy_handler().template deallocate<scalar_t>(m_a_gpu);
  ex.get_policy_handler().template deallocate<scalar_t>(v_b_gpu);
  ex.get_policy_handler().template deallocate<scalar_t>(v_c_gpu);
}
