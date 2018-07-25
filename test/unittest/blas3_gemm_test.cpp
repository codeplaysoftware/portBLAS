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

#include "../blas_test.hpp"

class Normal {
  public:
  static constexpr char const* str = "n";
};
// const std::string Normal::str = "n";

class Transposed {
  public:
  static constexpr char const* str = "t";
};
// const std::string Transposed::str = "t";

class Conjugate {
  public:
   static constexpr char const* str = "c";
};
// const std::string Conjugate::str = "c";



template <class AT_ = Normal, class BT_ = Normal>
struct MatrixFormats
{
  using a_format = AT_;
  using b_format = BT_;
};

typedef ::testing::Types<
    blas_test_args<float, MatrixFormats<Normal, Normal>>,

    blas_test_args<float, MatrixFormats<Transposed, Normal>>,
    blas_test_args<float, MatrixFormats<Normal, Transposed>>,
    blas_test_args<float, MatrixFormats<Transposed, Transposed>>,

    blas_test_args<float, MatrixFormats<Conjugate, Normal>>,
    blas_test_args<float, MatrixFormats<Normal, Conjugate>>,
    blas_test_args<float, MatrixFormats<Conjugate, Conjugate>>,

    blas_test_args<float, MatrixFormats<Transposed, Conjugate>>,
    blas_test_args<float, MatrixFormats<Conjugate, Transposed>>,

    blas_test_args<double, MatrixFormats<Normal, Normal>>,

    blas_test_args<double, MatrixFormats<Transposed, Normal>>,
    blas_test_args<double, MatrixFormats<Normal, Transposed>>,
    blas_test_args<double, MatrixFormats<Transposed, Transposed>>,

    blas_test_args<double, MatrixFormats<Conjugate, Normal>>,
    blas_test_args<double, MatrixFormats<Normal, Conjugate>>,
    blas_test_args<double, MatrixFormats<Conjugate, Conjugate>>,

    blas_test_args<double, MatrixFormats<Transposed, Conjugate>>,
    blas_test_args<double, MatrixFormats<Conjugate, Transposed>>>
    BlasTypes;

TYPED_TEST_CASE(BLAS_Test, BlasTypes);

REGISTER_PREC(float, 1e-4, gemm_default)
REGISTER_PREC(double, 1e-8, gemm_default)
REGISTER_PREC(long double, 1e-8, gemm_default)
TYPED_TEST(BLAS_Test, gemm_default) {
  using test = class gemm_default;


  using ScalarT = typename TypeParam::scalar_t;
  using ExecutorType = typename TypeParam::executor_t;
  using TestClass = BLAS_Test<TypeParam>;

  using MatAType = typename TypeParam::metadata_t::a_format;
  using MatBType = typename TypeParam::metadata_t::b_format;

  ScalarT prec =
    BLAS_Test<TypeParam>::template test_prec<test>();

  const char* ta_str = MatAType::str;
  const char* tb_str = MatBType::str;


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