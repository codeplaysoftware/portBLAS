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
 *  @filename blas3_gemm_test.hpp
 *
 **************************************************************************/

#include "blas_test.hpp"

#ifndef BlasTypes
#error "BlasTypes not defined before including blas3_gemm_def.hpp"
#endif

TYPED_TEST_CASE(BLAS_Test, BlasTypes);

REGISTER_PREC(float, 1e-4, gemm)
REGISTER_PREC(double, 1e-8, gemm)
REGISTER_PREC(long double, 1e-8, gemm)
TYPED_TEST(BLAS_Test, gemm) {
  using test = class gemm;

  using ScalarT = typename TypeParam::scalar_t;
  using ExecutorType = typename TypeParam::executor_t;
  using TestClass = BLAS_Test<TypeParam>;

  using MatAType = typename TypeParam::metadata_t::a_format;
  using MatBType = typename TypeParam::metadata_t::b_format;

  ScalarT prec = BLAS_Test<TypeParam>::template test_prec<test>();

  const char* ta_str = MatAType::str;
  const char* tb_str = MatBType::str;

  auto _TransA = tolower(*ta_str);
  auto _TransB = tolower(*tb_str);
  bool _TrA = _TransA != 'n';
  bool _TrB = _TransB != 'n';
  ScalarT alpha = ScalarT(1);
  ScalarT beta = ScalarT(1);
  SYCL_DEVICE_SELECTOR d;
  auto q = TestClass::make_queue(d);
  Executor<ExecutorType> ex(q);

  int m_sizes[] = {2,   11,  14,  31,  39,  63,  64,   65,   127,
                   129, 255, 257, 511, 512, 513, 1023, 1024, 1025};

  int n_sizes[] = {2,   11,  14,  31,  39,  63,  64,   65,   127,
                   129, 255, 257, 511, 512, 513, 1023, 1024, 1025};

  int k_sizes[] = {2,   11,  14,  31,  39,  63,  64,  65,   95,   96,
                   127, 129, 255, 257, 511, 512, 513, 1023, 1024, 1025};

  for (int i = 0; i < 18; i++) {
    for (int j = 0; j < 18; j++) {
      for (int l = 0; l < 20; l++) {
        std::array<int, 2> dim_a = {m_sizes[i], k_sizes[l]};
        std::array<int, 2> dim_b = {k_sizes[l], n_sizes[j]};
        std::array<int, 2> dim_c = {m_sizes[i], n_sizes[j]};

        std::vector<ScalarT> a_m(dim_a[0] * dim_a[1]);
        std::vector<ScalarT> b_m(dim_b[0] * dim_b[1]);
        std::vector<ScalarT> c_m_gpu_result(dim_c[0] * dim_c[1], ScalarT(0));
        std::vector<ScalarT> c_m_cpu(dim_c[0] * dim_c[1], ScalarT(0));
        TestClass::set_rand(a_m, dim_a[0] * dim_a[1]);
        TestClass::set_rand(b_m, dim_b[0] * dim_b[1]);
        int lda = (_TrA) ? dim_a[1] : dim_a[0];
        int ldb = (_TrB) ? dim_b[1] : dim_b[0];
        int ldc = dim_c[0];
        int m = dim_c[0];
        int n = dim_c[1];
        int k = dim_a[1];
        gemm(ta_str, tb_str, m, n, k, alpha, a_m.data(), lda, b_m.data(), ldb,
             beta, c_m_cpu.data(), m);

        auto m_a_gpu = ex.template allocate<ScalarT>(dim_a[0] * dim_a[1]);
        auto m_b_gpu = ex.template allocate<ScalarT>(dim_b[0] * dim_b[1]);
        auto m_c_gpu = ex.template allocate<ScalarT>(dim_c[0] * dim_c[1]);
        ex.copy_to_device(a_m.data(), m_a_gpu, dim_a[0] * dim_a[1]);
        ex.copy_to_device(b_m.data(), m_b_gpu, dim_b[0] * dim_b[1]);
        ex.copy_to_device(c_m_gpu_result.data(), m_c_gpu, dim_c[0] * dim_c[1]);
        _gemm(ex, *ta_str, *tb_str, m, n, k, alpha, m_a_gpu, lda, m_b_gpu, ldb,
              beta, m_c_gpu, ldc);
        auto event = ex.copy_to_host(m_c_gpu, c_m_gpu_result.data(),
                                     dim_c[0] * dim_c[1]);
        ex.wait(event);

        for (int i = 0; i < dim_c[0] * dim_c[1]; ++i) {
          ASSERT_NEAR(c_m_gpu_result[i], c_m_cpu[i], prec);
        }
        ex.template deallocate<ScalarT>(m_a_gpu);
        ex.template deallocate<ScalarT>(m_b_gpu);
        ex.template deallocate<ScalarT>(m_c_gpu);
      }
    }
  }
}