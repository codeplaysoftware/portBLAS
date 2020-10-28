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
 *  @filename blas3_gemm_values.hpp
 *
 **************************************************************************/

#include <utility>

#include "blas_test.hpp"

template <typename T>
using gemm_arguments_t = std::tuple<int, int, int, int, int, char, char, T, T,
                                    int, int, int, gemm_batch_type_t>;

// Convert batch_type=strided to interleaved on the host
template <typename scalar_t>
inline std::vector<scalar_t> strided_to_interleaved(
    const std::vector<scalar_t>& input, int offset, int ld_rows, int ld_cols,
    int batchs) {
  std::vector<scalar_t> output(input.size());
  for (int c = 0; c < ld_cols; ++c) {
    for (int r = 0; r < ld_rows; ++r) {
      for (int b = 0; b < batchs; ++b) {
        output[c * ld_rows * batchs + r * batchs + b + offset] =
            input[b * ld_cols * ld_rows + c * ld_rows + r + offset];
      }
    }
  }
  return output;
}

// Convert batch_type=interleaved to strided on the host
template <typename scalar_t>
inline std::vector<scalar_t> interleaved_to_strided(
    const std::vector<scalar_t>& input, int offset, int ld_rows, int ld_cols,
    int batchs) {
  std::vector<scalar_t> output(input.size());
  for (int b = 0; b < batchs; ++b) {
    for (int c = 0; c < ld_cols; ++c) {
      for (int r = 0; r < ld_rows; ++r) {
        output[b * ld_cols * ld_rows + c * ld_rows + r + offset] =
            input[c * ld_rows * batchs + r * batchs + b + offset];
      }
    }
  }
  return output;
}

template <typename scalar_t>
inline void verify_gemm(const gemm_arguments_t<scalar_t> arguments) {
  int offset;
  int batch;
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
  gemm_batch_type_t batch_type;
  std::tie(offset, batch, m, n, k, transa, transb, alpha, beta, lda_mul,
           ldb_mul, ldc_mul, batch_type) = arguments;

  using data_t = utils::data_storage_t<scalar_t>;

  const char ta_str[2] = {transa, '\0'};
  const char tb_str[2] = {transb, '\0'};

  auto q = make_queue();
  test_executor_t ex(q);

  auto policy_handler = ex.get_policy_handler();

  const int lda = ((transa != 'n') ? k : m) * lda_mul;
  const int ldb = ((transb != 'n') ? n : k) * ldb_mul;
  const int ldc = m * ldc_mul;

  const int size_a = m * k * lda_mul;
  const int size_b = k * n * ldb_mul;
  const int size_c = m * n * ldc_mul;

  const int buffer_size_a = batch * size_a + offset;
  const int buffer_size_b = batch * size_b + offset;
  const int buffer_size_c = batch * size_c + offset;

  std::vector<data_t> a_m(buffer_size_a);
  std::vector<data_t> b_m(buffer_size_b);
  std::vector<data_t> c_m_gpu(buffer_size_c);

  fill_random(a_m);
  fill_random(b_m);
  fill_random(c_m_gpu);
  std::vector<data_t> c_m_cpu = c_m_gpu;

  // Use system blas to create a reference output
  for (int i = 0; i < batch; ++i) {
    reference_blas::gemm(ta_str, tb_str, m, n, k, static_cast<data_t>(alpha),
                         a_m.data() + i * size_a + offset, lda,
                         b_m.data() + i * size_b + offset, ldb,
                         static_cast<data_t>(beta),
                         c_m_cpu.data() + i * size_c + offset, ldc);
  }

  if (batch > 1 && batch_type == gemm_batch_type_t::interleaved) {
    a_m =
        strided_to_interleaved(a_m, offset, lda, transa == 't' ? m : k, batch);
    b_m =
        strided_to_interleaved(b_m, offset, ldb, transb == 't' ? k : n, batch);
    c_m_gpu = strided_to_interleaved(c_m_gpu, offset, ldc, n, batch);
  }

  auto m_a_gpu = utils::make_quantized_buffer<scalar_t>(ex, a_m);
  auto m_b_gpu = utils::make_quantized_buffer<scalar_t>(ex, b_m);
  auto m_c_gpu = utils::make_quantized_buffer<scalar_t>(ex, c_m_gpu);

  // SYCL BLAS GEMM implementation
  if (batch == 1) {
    _gemm(ex, transa, transb, m, n, k, alpha, m_a_gpu + offset, lda,
          m_b_gpu + offset, ldb, beta, m_c_gpu + offset, ldc);
  } else {
    _gemm_batched(ex, transa, transb, m, n, k, alpha, m_a_gpu + offset, lda,
                  m_b_gpu + offset, ldb, beta, m_c_gpu + offset, ldc, batch,
                  batch_type);
  }

  auto event = utils::quantized_copy_to_host<scalar_t>(ex, m_c_gpu, c_m_gpu);
  policy_handler.wait(event);

  if (batch > 1 && batch_type == gemm_batch_type_t::interleaved) {
    c_m_gpu = interleaved_to_strided(c_m_gpu, offset, ldc, n, batch);
  }

  ex.get_policy_handler().wait();

  const bool isAlmostEqual =
      utils::compare_vectors<data_t, scalar_t>(c_m_gpu, c_m_cpu);
  ASSERT_TRUE(isAlmostEqual);

}

/** Registers GEMM test for all supported data types
 * @param test_suite Name of the test suite
 * @param combination Combinations object
 * @see BLAS_REGISTER_TEST_CUSTOM_NAME
 */
#define GENERATE_GEMM_TEST(test_suite, combination)                   \
  BLAS_REGISTER_TEST_CUSTOM_NAME(test_suite, test_suite##combination, \
                                 verify_gemm, gemm_arguments_t, combination);
