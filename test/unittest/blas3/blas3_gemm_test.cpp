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

#include "blas3_gemm_common.hpp"
#include "blas_test.hpp"

template <typename T>
using combination_t =
    std::tuple<int, int, int, int, char, char, T, T, int, int, int>;

template <typename scalar_t>
void run_test(const combination_t<scalar_t> combi) {
  using type_t = blas_test_args<scalar_t, void>;
  using blas_test_t = BLAS_Test<type_t>;
  using executor_t = typename type_t::executor_t;

  int batch_size;
  int m;
  int n;
  int k;
  char transa;
  char transb;
  scalar_t alpha;
  scalar_t beta;
  int lda_mul;
  int ldb_mul;
  int ldc_mul;
  std::tie(batch_size, m, n, k, transa, transb, alpha, beta, lda_mul, ldb_mul,
           ldc_mul) = combi;

  const char ta_str[2] = {transa, '\0'};
  const char tb_str[2] = {transb, '\0'};

  auto _size = [=](const std::array<int, 2> &dim, int ld_mul) {
    return dim[0] * dim[1] * batch_size * ld_mul;
  };
  auto _size_batch = [=](const std::array<int, 2> &dim, int ld_mul) {
    return dim[0] * dim[1] * ld_mul;
  };
  auto _base = [=](const std::array<int, 2> &dim, int ld_mul, int bs) {
    return dim[0] * dim[1] * ld_mul * bs;
  };

  auto q = make_queue();
  Executor<executor_t> ex(q);

  auto policy_handler = ex.get_policy_handler();

  std::array<int, 2> dim_a = {m, k};
  std::array<int, 2> dim_b = {k, n};
  std::array<int, 2> dim_c = {m, n};

  int lda = ((transa != 'n') ? dim_a[1] : dim_a[0]) * lda_mul;
  int ldb = ((transb != 'n') ? dim_b[1] : dim_b[0]) * ldb_mul;
  int ldc = dim_c[0] * ldc_mul;

  std::vector<scalar_t> a_m(_size(dim_a, lda_mul));
  std::vector<scalar_t> b_m(_size(dim_b, ldb_mul));
  std::vector<scalar_t> c_m_gpu(_size(dim_c, ldc_mul));
  std::vector<scalar_t> c_m_cpu(_size(dim_c, ldc_mul));

  fill_random(a_m);
  fill_random(b_m);
  fill_random(c_m_gpu);
  std::copy(c_m_gpu.begin(), c_m_gpu.end(), c_m_cpu.begin());

  auto m_a_gpu =
      policy_handler.template allocate<scalar_t>(_size(dim_a, lda_mul));
  auto m_b_gpu =
      policy_handler.template allocate<scalar_t>(_size(dim_b, ldb_mul));
  auto m_c_gpu =
      policy_handler.template allocate<scalar_t>(_size(dim_c, ldc_mul));

  for (int bs = 0; bs < batch_size; bs++) {
    // Use system blas to create a reference output
    reference_blas::gemm(ta_str, tb_str, m, n, k, alpha,
                         a_m.data() + _base(dim_a, lda_mul, bs), lda,
                         b_m.data() + _base(dim_b, ldb_mul, bs), ldb, beta,
                         c_m_cpu.data() + _base(dim_c, ldc_mul, bs), ldc);

    policy_handler.copy_to_device(a_m.data() + _base(dim_a, lda_mul, bs),
                                  m_a_gpu + _base(dim_a, lda_mul, bs),
                                  _size_batch(dim_a, lda_mul));
    policy_handler.copy_to_device(b_m.data() + _base(dim_b, ldb_mul, bs),
                                  m_b_gpu + _base(dim_b, ldb_mul, bs),
                                  _size_batch(dim_b, ldb_mul));
    policy_handler.copy_to_device(c_m_gpu.data() + _base(dim_c, ldc_mul, bs),
                                  m_c_gpu + _base(dim_c, ldc_mul, bs),
                                  _size_batch(dim_c, ldc_mul));

    // SYCL BLAS GEMM implementation
    _gemm(ex, transa, transb, m, n, k, alpha,
          m_a_gpu + _base(dim_a, lda_mul, bs), lda,
          m_b_gpu + _base(dim_b, ldb_mul, bs), ldb, beta,
          m_c_gpu + _base(dim_c, ldc_mul, bs), ldc);

    auto event =
        policy_handler.copy_to_host(m_c_gpu + _base(dim_c, ldc_mul, bs),
                                    c_m_gpu.data() + _base(dim_c, ldc_mul, bs),
                                    _size_batch(dim_c, ldc_mul));
    policy_handler.wait(event);
  }

  ASSERT_TRUE(utils::compare_vectors(c_m_gpu, c_m_cpu));

  policy_handler.template deallocate<scalar_t>(m_a_gpu);
  policy_handler.template deallocate<scalar_t>(m_b_gpu);
  policy_handler.template deallocate<scalar_t>(m_c_gpu);
}

class GemmFloat : public ::testing::TestWithParam<combination_t<float>> {};
TEST_P(GemmFloat, test) { run_test<float>(GetParam()); };
INSTANTIATE_TEST_SUITE_P(gemm, GemmFloat, combi);

#if DOUBLE_SUPPORT
class GemmDouble : public ::testing::TestWithParam<combination_t<double>> {};
TEST_P(GemmDouble, test) { run_test<double>(GetParam()); };
INSTANTIATE_TEST_SUITE_P(gemm, GemmDouble, combi);
#endif
