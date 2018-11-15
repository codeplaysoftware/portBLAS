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
REGISTER_PREC(long double, 1e-8, gemv_test)

TYPED_TEST(BLAS_Test, gemv_test) {
  using ScalarT = typename TypeParam::scalar_t;
  using ExecutorType = typename TypeParam::executor_t;
  using TestClass = BLAS_Test<TypeParam>;
  using test = class gemv_test;

  size_t m = 125;
  size_t n = 127;
  size_t lda = m;
  long incX = 1;
  long incY = 1;
  const char* t_str = "n";  // Testing the no transpose matrix
  ScalarT prec = TestClass::template test_prec<test>();
  ScalarT alpha = ScalarT(1);
  ScalarT beta = ScalarT(1);

  // Input matrix
  std::vector<ScalarT> a_m(m * n);
  // Input Vector
  std::vector<ScalarT> b_v(n);
  // output Vector
  std::vector<ScalarT> c_v_gpu_result(m, ScalarT(0));
  // output system vector
  std::vector<ScalarT> c_v_cpu(m, ScalarT(0));
  TestClass::set_rand(a_m, m * n);
  TestClass::set_rand(b_v, n);

  // SYSTEM GEMMV
  gemv(t_str, m, n, alpha, a_m.data(), m, b_v.data(), incX, beta,
       c_v_cpu.data(), incY);

  SYCL_DEVICE_SELECTOR d;
  auto q = TestClass::make_queue(d);
  Executor<ExecutorType> ex(q);
  auto m_a_gpu = ex.template allocate<ScalarT>(m * n);
  auto v_b_gpu = ex.template allocate<ScalarT>(n);
  auto v_c_gpu = ex.template allocate<ScalarT>(m);
  ex.copy_to_device(a_m.data(), m_a_gpu, m * n);
  ex.copy_to_device(b_v.data(), v_b_gpu, n);
  ex.copy_to_device(c_v_gpu_result.data(), v_c_gpu, m);
  // SYCLGEMV
  _gemv(ex, *t_str, m, n, alpha, m_a_gpu, m, v_b_gpu, incX, beta, v_c_gpu,
        incY);
  auto event = ex.copy_to_host(v_c_gpu, c_v_gpu_result.data(), m);
  ex.wait(event);

  for (size_t i = 0; i < m; ++i) {
    ASSERT_NEAR(c_v_gpu_result[i], c_v_cpu[i], prec);
  }
  ex.template deallocate<ScalarT>(m_a_gpu);
  ex.template deallocate<ScalarT>(v_b_gpu);
  ex.template deallocate<ScalarT>(v_c_gpu);
}

REGISTER_PREC(float, 1e-4, gemv_test_transposed)
REGISTER_PREC(double, 1e-8, gemv_test_transposed)
REGISTER_PREC(long double, 1e-8, gemv_test_transposed)

TYPED_TEST(BLAS_Test, gemv_test_transposed) {
  using ScalarT = typename TypeParam::scalar_t;
  using ExecutorType = typename TypeParam::executor_t;
  using TestClass = BLAS_Test<TypeParam>;
  using test = class gemv_test_transposed;

  size_t m = 125;
  size_t n = 127;
  size_t lda = m;
  long incX = 1;
  long incY = 1;
  const char* t_str = "t";  // Testing the no transpose matrix
  ScalarT prec = TestClass::template test_prec<test>();
  ScalarT alpha = ScalarT(1);
  ScalarT beta = ScalarT(1);

  // Input matrix
  std::vector<ScalarT> a_m(m * n);
  // Input Vector
  std::vector<ScalarT> b_v(m);
  // output Vector
  std::vector<ScalarT> c_v_gpu_result(n, ScalarT(0));
  // output system vector
  std::vector<ScalarT> c_v_cpu(n, ScalarT(0));
  TestClass::set_rand(a_m, m * n);
  TestClass::set_rand(b_v, m);

  // SYSTEM GEMMV
  gemv(t_str, m, n, alpha, a_m.data(), m, b_v.data(), incX, beta,
       c_v_cpu.data(), incY);

  SYCL_DEVICE_SELECTOR d;
  auto q = TestClass::make_queue(d);
  Executor<ExecutorType> ex(q);
  auto m_a_gpu = ex.template allocate<ScalarT>(m * n);
  auto v_b_gpu = ex.template allocate<ScalarT>(m);
  auto v_c_gpu = ex.template allocate<ScalarT>(n);
  ex.copy_to_device(a_m.data(), m_a_gpu, m * n);
  ex.copy_to_device(b_v.data(), v_b_gpu, m);
  ex.copy_to_device(c_v_gpu_result.data(), v_c_gpu, n);
  // SYCLGEMV
  _gemv(ex, *t_str, m, n, alpha, m_a_gpu, m, v_b_gpu, incX, beta, v_c_gpu,
        incY);
  auto event = ex.copy_to_host(v_c_gpu, c_v_gpu_result.data(), n);
  ex.wait(event);

  for (size_t i = 0; i < n; ++i) {
    ASSERT_NEAR(c_v_gpu_result[i], c_v_cpu[i], prec);
  }
  ex.template deallocate<ScalarT>(m_a_gpu);
  ex.template deallocate<ScalarT>(v_b_gpu);
  ex.template deallocate<ScalarT>(v_c_gpu);
}

// ***********************
// Testing legacy versions:
// ***********************

REGISTER_PREC(float, 1e-4, gemv_test_legacy)
REGISTER_PREC(double, 1e-8, gemv_test_legacy)
REGISTER_PREC(long double, 1e-8, gemv_test_legacy)

TYPED_TEST(BLAS_Test, gemv_test_legacy) {
  using ScalarT = typename TypeParam::scalar_t;
  using ExecutorType = typename TypeParam::executor_t;
  using TestClass = BLAS_Test<TypeParam>;
  using test = class gemv_test_legacy;

  size_t m = 125;
  size_t n = 127;
  size_t lda = m;
  long incX = 1;
  long incY = 1;
  const char* t_str = "n";  // Testing the no transpose matrix
  ScalarT prec = TestClass::template test_prec<test>();
  ScalarT alpha = ScalarT(1);
  ScalarT beta = ScalarT(1);

  // Input matrix
  std::vector<ScalarT> a_m(m * n);
  // Input Vector
  std::vector<ScalarT> b_v(n);
  // output Vector
  std::vector<ScalarT> c_v_gpu_result(m, ScalarT(0));
  // output system vector
  std::vector<ScalarT> c_v_cpu(m, ScalarT(0));
  TestClass::set_rand(a_m, m * n);
  TestClass::set_rand(b_v, n);

  // SYSTEM GEMMV
  gemv(t_str, m, n, alpha, a_m.data(), m, b_v.data(), incX, beta,
       c_v_cpu.data(), incY);

  SYCL_DEVICE_SELECTOR d;
  auto q = TestClass::make_queue(d);
  Executor<ExecutorType> ex(q);
  auto m_a_gpu = ex.template allocate<ScalarT>(m * n);
  auto v_b_gpu = ex.template allocate<ScalarT>(n);
  auto v_c_gpu = ex.template allocate<ScalarT>(m);
  ex.copy_to_device(a_m.data(), m_a_gpu, m * n);
  ex.copy_to_device(b_v.data(), v_b_gpu, n);
  ex.copy_to_device(c_v_gpu_result.data(), v_c_gpu, m);
  // SYCLGEMV
  _gemv_legacy(ex, *t_str, m, n, alpha, m_a_gpu, m, v_b_gpu, incX, beta,
               v_c_gpu, incY);
  auto event = ex.copy_to_host(v_c_gpu, c_v_gpu_result.data(), m);
  ex.wait(event);

  for (size_t i = 0; i < m; ++i) {
    ASSERT_NEAR(c_v_gpu_result[i], c_v_cpu[i], prec);
  }

  ex.template deallocate<ScalarT>(m_a_gpu);
  ex.template deallocate<ScalarT>(v_b_gpu);
  ex.template deallocate<ScalarT>(v_c_gpu);
}

REGISTER_PREC(float, 1e-4, gemv_test_legacy_transposed)
REGISTER_PREC(double, 1e-8, gemv_test_legacy_transposed)
REGISTER_PREC(long double, 1e-8, gemv_test_legacy_transposed)

TYPED_TEST(BLAS_Test, gemv_test_legacy_transposed) {
  using ScalarT = typename TypeParam::scalar_t;
  using ExecutorType = typename TypeParam::executor_t;
  using TestClass = BLAS_Test<TypeParam>;
  using test = class gemv_test_legacy_transposed;

  size_t m = 125;
  size_t n = 127;
  size_t lda = m;
  long incX = 1;
  long incY = 1;
  const char* t_str = "t";  // Testing the no transpose matrix
  ScalarT prec = TestClass::template test_prec<test>();
  ScalarT alpha = ScalarT(1);
  ScalarT beta = ScalarT(1);

  // Input matrix
  std::vector<ScalarT> a_m(m * n);
  // Input Vector
  std::vector<ScalarT> b_v(m);
  // output Vector
  std::vector<ScalarT> c_v_gpu_result(n, ScalarT(0));
  // output system vector
  std::vector<ScalarT> c_v_cpu(n, ScalarT(0));
  TestClass::set_rand(a_m, m * n);
  TestClass::set_rand(b_v, m);

  // SYSTEM GEMMV
  gemv(t_str, m, n, alpha, a_m.data(), m, b_v.data(), incX, beta,
       c_v_cpu.data(), incY);

  SYCL_DEVICE_SELECTOR d;
  auto q = TestClass::make_queue(d);
  Executor<ExecutorType> ex(q);
  auto m_a_gpu = ex.template allocate<ScalarT>(m * n);
  auto v_b_gpu = ex.template allocate<ScalarT>(m);
  auto v_c_gpu = ex.template allocate<ScalarT>(n);
  ex.copy_to_device(a_m.data(), m_a_gpu, m * n);
  ex.copy_to_device(b_v.data(), v_b_gpu, m);
  ex.copy_to_device(c_v_gpu_result.data(), v_c_gpu, n);
  // SYCLGEMV
  _gemv_legacy(ex, *t_str, m, n, alpha, m_a_gpu, m, v_b_gpu, incX, beta,
               v_c_gpu, incY);
  auto event = ex.copy_to_host(v_c_gpu, c_v_gpu_result.data(), n);
  ex.wait(event);

  ex.template deallocate<ScalarT>(m_a_gpu);
  ex.template deallocate<ScalarT>(v_b_gpu);
  ex.template deallocate<ScalarT>(v_c_gpu);
}
