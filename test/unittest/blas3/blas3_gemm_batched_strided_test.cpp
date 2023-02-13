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
 *  @filename blas3_gemm_batched_strided_test.hpp
 *
 **************************************************************************/

#include <utility>

#include "blas_test.hpp"

template <typename T>
using gemm_combination_t = std::tuple<int, int, int, int, int, char, char, T, T,
                                      int, int, int, int, int, int>;

template <typename scalar_t>
inline void verify_gemm_strided(const gemm_combination_t<scalar_t> arguments) {
  index_t offset;
  index_t batch;
  index_t m;
  index_t n;
  index_t k;
  char transa;
  char transb;
  scalar_t alpha;
  scalar_t beta;
  index_t lda_mul;
  index_t ldb_mul;
  index_t ldc_mul;
  index_t stride_a_mul;
  index_t stride_b_mul;
  index_t stride_c_mul;
  std::tie(offset, batch, m, n, k, transa, transb, alpha, beta, lda_mul,
           ldb_mul, ldc_mul, stride_a_mul, stride_b_mul, stride_c_mul) =
      arguments;

  const char ta_str[2] = {transa, '\0'};
  const char tb_str[2] = {transb, '\0'};

  auto q = make_queue();
  blas::SB_Handle sb_handle(q);

  const index_t lda = ((transa != 'n') ? k : m) * lda_mul;
  const index_t ldb = ((transb != 'n') ? n : k) * ldb_mul;
  const index_t ldc = m * ldc_mul;

  const index_t size_a = m * k * lda_mul;
  const index_t size_b = k * n * ldb_mul;
  const index_t size_c = m * n * ldc_mul;

  const index_t stride_a = stride_a_mul * size_a;
  const index_t stride_b = stride_b_mul * size_b;
  const index_t stride_c = stride_c_mul * size_c;

  const index_t buffer_size_a = size_a + (batch - 1) * stride_a + offset;
  const index_t buffer_size_b = size_b + (batch - 1) * stride_b + offset;
  const index_t buffer_size_c = size_c + (batch - 1) * stride_c + offset;

  std::vector<scalar_t> a_m(buffer_size_a);
  std::vector<scalar_t> b_m(buffer_size_b);
  std::vector<scalar_t> c_m_gpu(buffer_size_c);

  fill_random(a_m);
  fill_random(b_m);
  fill_random(c_m_gpu);
  std::vector<scalar_t> c_m_cpu = c_m_gpu;

  // Use system blas to create a reference output
  for (int i = 0; i < batch; ++i) {
    reference_blas::gemm(ta_str, tb_str, m, n, k, alpha,
                         a_m.data() + i * stride_a + offset, lda,
                         b_m.data() + i * stride_b + offset, ldb, beta,
                         c_m_cpu.data() + i * stride_c + offset, ldc);
  }

  auto m_a_gpu = blas::make_sycl_iterator_buffer<scalar_t>(buffer_size_a);
  auto m_b_gpu = blas::make_sycl_iterator_buffer<scalar_t>(buffer_size_b);
  auto m_c_gpu = blas::make_sycl_iterator_buffer<scalar_t>(buffer_size_c);

  blas::helper::copy_to_device(sb_handle.get_queue(), a_m.data(), m_a_gpu,
                               buffer_size_a);
  blas::helper::copy_to_device(sb_handle.get_queue(), b_m.data(), m_b_gpu,
                               buffer_size_b);
  blas::helper::copy_to_device(sb_handle.get_queue(), c_m_gpu.data(), m_c_gpu,
                               buffer_size_c);

  // SYCL BLAS GEMM BATCHED STRIDED implementation
  _gemm_batched_strided(sb_handle, transa, transb, m, n, k, alpha,
                        m_a_gpu + offset, lda, stride_a, m_b_gpu + offset, ldb,
                        stride_b, beta, m_c_gpu + offset, ldc, stride_c, batch);

  auto event = blas::helper::copy_to_host(sb_handle.get_queue(), m_c_gpu,
                                          c_m_gpu.data(), buffer_size_c);
  sb_handle.wait(event);

  const bool isAlmostEqual =
      (batch == 1 || stride_c_mul == 1)
          ? utils::compare_vectors(c_m_gpu, c_m_cpu)
          : utils::compare_vectors_strided(c_m_gpu, c_m_cpu, stride_c, size_c);
  ASSERT_TRUE(isAlmostEqual);
}

template <typename scalar_t>
const auto WithoutStride =
    ::testing::Combine(::testing::Values(0),              // offset
                       ::testing::Values(5),              // batch
                       ::testing::Values(63),             // m
                       ::testing::Values(63),             // n
                       ::testing::Values(63),             // k
                       ::testing::Values('n', 't'),       // transa
                       ::testing::Values('n', 't'),       // transb
                       ::testing::Values<scalar_t>(3.0),  // alpha
                       ::testing::Values<scalar_t>(7.0),  // beta
                       ::testing::Values(1, 3),           // lda_mul
                       ::testing::Values(1, 2),           // ldb_mul
                       ::testing::Values(1, 4),           // ldc_mul
                       ::testing::Values(0),              // stride_a_mul
                       ::testing::Values(0),              // stride_b_mul
                       ::testing::Values(1)               // stride_c_mul
    );

template <typename scalar_t>
const auto WithStride =
    ::testing::Combine(::testing::Values(0),              // offset
                       ::testing::Values(3),              // batch
                       ::testing::Values(63, 128),        // m
                       ::testing::Values(63, 128),        // n
                       ::testing::Values(63, 129),        // k
                       ::testing::Values('n', 't'),       // transa
                       ::testing::Values('n', 't'),       // transb
                       ::testing::Values<scalar_t>(3.0),  // alpha
                       ::testing::Values<scalar_t>(7.0),  // beta
                       ::testing::Values(2),              // lda_mul
                       ::testing::Values(3),              // ldb_mul
                       ::testing::Values(4),              // ldc_mul
                       ::testing::Values(1, 3),           // stride_a_mul
                       ::testing::Values(1, 3),           // stride_b_mul
                       ::testing::Values(1, 2)            // stride_c_mul
    );

template <typename scalar_t>
const auto WithAndWithoutStride =
    ::testing::Combine(::testing::Values(0),              // offset
                       ::testing::Values(3),              // batch
                       ::testing::Values(63, 128),        // m
                       ::testing::Values(63, 128),        // n
                       ::testing::Values(63, 129),        // k
                       ::testing::Values('n', 't'),       // transa
                       ::testing::Values('n', 't'),       // transb
                       ::testing::Values<scalar_t>(3.0),  // alpha
                       ::testing::Values<scalar_t>(7.0),  // beta
                       ::testing::Values(2),              // lda_mul
                       ::testing::Values(3),              // ldb_mul
                       ::testing::Values(4),              // ldc_mul
                       ::testing::Values(0, 1),           // stride_a_mul
                       ::testing::Values(0, 2),           // stride_b_mul
                       ::testing::Values(1, 3)            // stride_c_mul
    );

template <class T>
static std::string generate_name_strided(
    const ::testing::TestParamInfo<gemm_combination_t<T>>& info) {
  int offset, batch, m, n, k, ldaMul, ldbMul, ldcMul, stride_a_mul,
      stride_b_mul, stride_c_mul;
  char transa, transb;
  T alpha, beta;
  BLAS_GENERATE_NAME(info.param, offset, batch, m, n, k, transa, transb, alpha,
                     beta, ldaMul, ldbMul, ldcMul, stride_a_mul, stride_b_mul,
                     stride_c_mul);
}

/** Registers GEMM test for all supported data types
 * @param test_suite Name of the test suite
 * @param combination Combinations object
 * @see BLAS_REGISTER_TEST_CUSTOM_NAME
 */
#define GENERATE_GEMM_TEST(test_suite, combination)                       \
  BLAS_REGISTER_TEST_CUSTOM_NAME(test_suite, test_suite##combination,     \
                                 verify_gemm_strided, gemm_combination_t, \
                                 combination, generate_name_strided);

GENERATE_GEMM_TEST(GemmBatchedStrided, WithoutStride);
GENERATE_GEMM_TEST(GemmBatchedStrided, WithStride);
GENERATE_GEMM_TEST(GemmBatchedStrided, WithAndWithoutStride);
