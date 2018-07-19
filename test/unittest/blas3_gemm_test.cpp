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
 *  @filename blas3_gemm_test.cpp
 *
 **************************************************************************/

#include "blas_test.hpp"

typedef ::testing::Types<blas_test_args<float>, blas_test_args<double>>
    BlasTypes;

TYPED_TEST_CASE(BLAS_Test, BlasTypes);

template <typename TypeParam> 
void _gemm_test_impl(typename TypeParam::scalar_t prec, const char* ta_str, const char* tb_str) { 
  using ScalarT = typename TypeParam::scalar_t;
  using ExecutorType = typename TypeParam::executor_t;
  using TestClass = BLAS_Test<TypeParam>;

  std::array<size_t, 2> dim_a = {127, 127};
  std::array<size_t, 2> dim_b = {127, 127};
  std::array<size_t, 2> dim_c = {127, 127};

  ScalarT alpha = ScalarT(1);
  ScalarT beta = ScalarT(1);

  std::vector<ScalarT> a_m(dim_a[0] * dim_a[1]);
  std::vector<ScalarT> b_m(dim_b[0] * dim_b[1]);
  std::vector<ScalarT> c_m_gpu_result(dim_c[0] * dim_c[1], ScalarT(0));
  std::vector<ScalarT> c_m_cpu(dim_c[0] * dim_c[1], ScalarT(0));
  TestClass::set_rand(a_m, dim_a[0] * dim_a[1]);
  TestClass::set_rand(b_m, dim_b[0] * dim_b[1]);
  auto lda = dim_a[0];
  auto ldb = dim_b[0];
  auto ldc = dim_c[0];
  auto m = dim_c[0];
  auto n = dim_c[1];
  auto k = dim_a[1];
  gemm(ta_str, tb_str, m, n, k, alpha, a_m.data(), lda, b_m.data(), ldb, beta,
       c_m_cpu.data(), m);
  SYCL_DEVICE_SELECTOR d;
  auto q = TestClass::make_queue(d);
  Executor<ExecutorType> ex(q);
  auto m_a_gpu = ex.template allocate<ScalarT>(dim_a[0] * dim_a[1]);
  auto m_b_gpu = ex.template allocate<ScalarT>(dim_b[0] * dim_b[1]);
  auto m_c_gpu = ex.template allocate<ScalarT>(dim_c[0] * dim_c[1]);
  ex.copy_to_device(a_m.data(), m_a_gpu, dim_a[0] * dim_a[1]);
  ex.copy_to_device(b_m.data(), m_b_gpu, dim_b[0] * dim_b[1]);
  ex.copy_to_device(c_m_gpu_result.data(), m_c_gpu, dim_c[0] * dim_c[1]);
  _gemm(ex, *ta_str, *tb_str, m, n, k, alpha, m_a_gpu, lda, m_b_gpu, ldb, beta,
        m_c_gpu, ldc);
  ex.copy_to_host(m_c_gpu, c_m_gpu_result.data(), dim_c[0] * dim_c[1]);
  for (size_t i = 0; i < dim_c[0] * dim_c[1]; ++i) {
    ASSERT_NEAR(c_m_gpu_result[i], c_m_cpu[i], prec);
  }
  ex.template deallocate<ScalarT>(m_a_gpu);
  ex.template deallocate<ScalarT>(m_b_gpu);
  ex.template deallocate<ScalarT>(m_c_gpu);
}

// Lightweight macro to make defining tests a little smoother.
// as test is an incomplete type, we must declare and immediately use it
// in this function as we cannot "pass" it down to _gemm_test_impl in any way
#define _GEMM_TEST(name, ta_str, tb_str) \
  REGISTER_PREC(float, 1e-4, name) \
  REGISTER_PREC(double, 1e-8, name) \
  REGISTER_PREC(long double, 1e-8, name) \
  TYPED_TEST(BLAS_Test, name) { \
    using test = class name; \
    typename TypeParam::scalar_t prec =  \
      BLAS_Test<TypeParam>::template test_prec<test>(); \
    _gemm_test_impl<TypeParam>(prec, ta_str, tb_str); \
  } \

_GEMM_TEST(gemm_test_na_nb, "n", "n")

_GEMM_TEST(gemm_test_ta_nb, "t", "n")
_GEMM_TEST(gemm_test_na_tb, "n", "t")
_GEMM_TEST(gemm_test_ta_tb, "t", "t")

_GEMM_TEST(gemm_test_ca_nb, "c", "n")
_GEMM_TEST(gemm_test_na_cb, "n", "c")
_GEMM_TEST(gemm_test_ca_cb, "c", "c")

_GEMM_TEST(gemm_test_ta_cb, "t", "c")
_GEMM_TEST(gemm_test_ca_tb, "c", "t")