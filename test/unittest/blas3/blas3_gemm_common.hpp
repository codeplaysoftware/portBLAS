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
 *  portBLAS: BLAS implementation using SYCL
 *
 *  @filename blas3_gemm_common.hpp
 *
 **************************************************************************/

#include <utility>

#include "blas_test.hpp"

template <typename T>
using gemm_arguments_t =
    std::tuple<std::string, int, int, int, int, int, char, char, T, T, int, int,
               int, gemm_batch_type_t>;

template <typename T>
using gemm_batched_strided_arguments_t =
    std::tuple<std::string, int, int, int, int, int, char, char, T, T, int, int,
               int, int, int, int>;

#ifdef BLAS_ENABLE_COMPLEX
template <typename T>
using gemm_cplx_arguments_t =
    std::tuple<std::string, int, int, int, int, int, char, char,
               std::complex<T>, std::complex<T>, int, int, int,
               gemm_batch_type_t>;

template <typename T>
using gemm_cplx_batched_strided_arguments_t =
    std::tuple<std::string, int, int, int, int, int, char, char,
               std::complex<T>, std::complex<T>, int, int, int, int, int, int>;
#endif

// Convert batch_type=strided to interleaved on the host
template <typename scalar_t>
inline std::vector<scalar_t> strided_to_interleaved(
    const std::vector<scalar_t>& input, int offset, int ld_rows, int ld_cols,
    int batchs) {
  std::vector<scalar_t> output(input.size());
  for (int o = 0; o < offset; ++o) output[o] = input[o];
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
  for (int o = 0; o < offset; ++o) output[o] = input[o];
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

template <typename scalar_t, helper::AllocType mem_alloc>
inline void verify_gemm(const gemm_arguments_t<scalar_t> arguments) {
  std::string alloc;
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
  gemm_batch_type_t batch_type;
  std::tie(alloc, offset, batch, m, n, k, transa, transb, alpha, beta, lda_mul,
           ldb_mul, ldc_mul, batch_type) = arguments;

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

  const index_t buffer_size_a = batch * size_a + offset;
  const index_t buffer_size_b = batch * size_b + offset;
  const index_t buffer_size_c = batch * size_c + offset;

  std::vector<scalar_t> a_m(buffer_size_a);
  std::vector<scalar_t> b_m(buffer_size_b);
  std::vector<scalar_t> c_m_gpu(buffer_size_c);

  fill_random(a_m);
  fill_random(b_m);
  fill_random(c_m_gpu);

  const char* en_joint_matrix = std::getenv("SB_ENABLE_JOINT_MATRIX");
  if (en_joint_matrix != NULL && std::is_same<scalar_t, float>::value &&
      *en_joint_matrix == '1') {
    set_to_zero_last_nbits(a_m);
    set_to_zero_last_nbits(b_m);
    set_to_zero_last_nbits(c_m_gpu);
    set_to_zero_last_nbits(alpha);
    set_to_zero_last_nbits(beta);
  }

  std::vector<scalar_t> c_m_cpu = c_m_gpu;

  // Use system blas to create a reference output
  for (int i = 0; i < batch; ++i) {
    reference_blas::gemm(ta_str, tb_str, m, n, k, alpha,
                         a_m.data() + i * size_a + offset, lda,
                         b_m.data() + i * size_b + offset, ldb, beta,
                         c_m_cpu.data() + i * size_c + offset, ldc);
  }

  if (batch > 1 && batch_type == gemm_batch_type_t::interleaved) {
    a_m =
        strided_to_interleaved(a_m, offset, lda, transa == 't' ? m : k, batch);
    b_m =
        strided_to_interleaved(b_m, offset, ldb, transb == 't' ? k : n, batch);
    c_m_gpu = strided_to_interleaved(c_m_gpu, offset, ldc, n, batch);
  }

  auto m_a_gpu = blas::helper::allocate<mem_alloc, scalar_t>(buffer_size_a, q);
  auto m_b_gpu = blas::helper::allocate<mem_alloc, scalar_t>(buffer_size_b, q);
  auto m_c_gpu = blas::helper::allocate<mem_alloc, scalar_t>(buffer_size_c, q);

  auto copy_a =
      blas::helper::copy_to_device(q, a_m.data(), m_a_gpu, buffer_size_a);
  auto copy_b =
      blas::helper::copy_to_device(q, b_m.data(), m_b_gpu, buffer_size_b);
  auto copy_c =
      blas::helper::copy_to_device(q, c_m_gpu.data(), m_c_gpu, buffer_size_c);

  // portBLAS GEMM implementation
  typename blas::SB_Handle::event_t gemm_event;
  if (batch == 1) {
    gemm_event = _gemm(sb_handle, transa, transb, m, n, k, alpha,
                       m_a_gpu + offset, lda, m_b_gpu + offset, ldb, beta,
                       m_c_gpu + offset, ldc, {copy_a, copy_b, copy_c});
  } else {
    gemm_event = _gemm_batched(sb_handle, transa, transb, m, n, k, alpha,
                               m_a_gpu + offset, lda, m_b_gpu + offset, ldb,
                               beta, m_c_gpu + offset, ldc, batch, batch_type,
                               {copy_a, copy_b, copy_c});
  }
  sb_handle.wait(gemm_event);

  auto event =
      blas::helper::copy_to_host(q, m_c_gpu, c_m_gpu.data(), buffer_size_c);
  sb_handle.wait(event);

  if (batch > 1 && batch_type == gemm_batch_type_t::interleaved) {
    c_m_gpu = interleaved_to_strided(c_m_gpu, offset, ldc, n, batch);
  }

  sb_handle.wait();

  const bool isAlmostEqual = utils::compare_vectors(c_m_gpu, c_m_cpu);
  ASSERT_TRUE(isAlmostEqual);

  helper::deallocate<mem_alloc>(m_a_gpu, q);
  helper::deallocate<mem_alloc>(m_b_gpu, q);
  helper::deallocate<mem_alloc>(m_c_gpu, q);
}

template <typename scalar_t>
inline void verify_gemm(const gemm_arguments_t<scalar_t> arguments) {
  std::string alloc;
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
  gemm_batch_type_t batch_type;
  std::tie(alloc, offset, batch, m, n, k, transa, transb, alpha, beta, lda_mul,
           ldb_mul, ldc_mul, batch_type) = arguments;

  if (alloc == "usm") {
#ifdef SB_ENABLE_USM
    verify_gemm<scalar_t, helper::AllocType::usm>(arguments);
#else
    GTEST_SKIP();
#endif
  } else {
    verify_gemm<scalar_t, helper::AllocType::buffer>(arguments);
  }
}

template <>
inline void dump_arg<gemm_batch_type_t>(std::ostream& ss,
                                        gemm_batch_type_t batch_type) {
  ss << (int)batch_type;
}

template <class T>
static std::string generate_name(
    const ::testing::TestParamInfo<gemm_arguments_t<T>>& info) {
  std::string alloc;
  int offset, batch, m, n, k, ldaMul, ldbMul, ldcMul;
  char transa, transb;
  T alpha, beta;
  gemm_batch_type_t batchType;
  BLAS_GENERATE_NAME(info.param, alloc, offset, batch, m, n, k, transa, transb,
                     alpha, beta, ldaMul, ldbMul, ldcMul, batchType);
}

template <typename scalar_t, helper::AllocType mem_alloc>
inline void verify_gemm(
    const gemm_batched_strided_arguments_t<scalar_t> arguments) {
  std::string alloc;
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
  std::tie(alloc, offset, batch, m, n, k, transa, transb, alpha, beta, lda_mul,
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

  const char* en_joint_matrix = std::getenv("SB_ENABLE_JOINT_MATRIX");
  if (en_joint_matrix != NULL && std::is_same<scalar_t, float>::value &&
      *en_joint_matrix == '1') {
    set_to_zero_last_nbits(a_m);
    set_to_zero_last_nbits(b_m);
    set_to_zero_last_nbits(c_m_gpu);
    set_to_zero_last_nbits(alpha);
    set_to_zero_last_nbits(beta);
  }

  std::vector<scalar_t> c_m_cpu = c_m_gpu;

  // Use system blas to create a reference output
  for (int i = 0; i < batch; ++i) {
    reference_blas::gemm(ta_str, tb_str, m, n, k, alpha,
                         a_m.data() + i * stride_a + offset, lda,
                         b_m.data() + i * stride_b + offset, ldb, beta,
                         c_m_cpu.data() + i * stride_c + offset, ldc);
  }

  auto m_a_gpu = blas::helper::allocate<mem_alloc, scalar_t>(buffer_size_a, q);
  auto m_b_gpu = blas::helper::allocate<mem_alloc, scalar_t>(buffer_size_b, q);
  auto m_c_gpu = blas::helper::allocate<mem_alloc, scalar_t>(buffer_size_c, q);

  auto copy_a =
      blas::helper::copy_to_device(q, a_m.data(), m_a_gpu, buffer_size_a);
  auto copy_b =
      blas::helper::copy_to_device(q, b_m.data(), m_b_gpu, buffer_size_b);
  auto copy_c =
      blas::helper::copy_to_device(q, c_m_gpu.data(), m_c_gpu, buffer_size_c);

  // portBLAS GEMM STRIDED BATCHED implementation
  auto gemm_batched_event = _gemm_strided_batched(
      sb_handle, transa, transb, m, n, k, alpha, m_a_gpu + offset, lda,
      stride_a, m_b_gpu + offset, ldb, stride_b, beta, m_c_gpu + offset, ldc,
      stride_c, batch, {copy_a, copy_b, copy_c});

  sb_handle.wait({gemm_batched_event});
  auto event =
      blas::helper::copy_to_host(q, m_c_gpu, c_m_gpu.data(), buffer_size_c);
  sb_handle.wait(event);

  const bool isAlmostEqual =
      (stride_c_mul == 1)
          ? utils::compare_vectors(c_m_gpu, c_m_cpu)
          : utils::compare_vectors_strided(c_m_gpu, c_m_cpu, stride_c, size_c);
  ASSERT_TRUE(isAlmostEqual);

  helper::deallocate<mem_alloc>(m_a_gpu, q);
  helper::deallocate<mem_alloc>(m_b_gpu, q);
  helper::deallocate<mem_alloc>(m_c_gpu, q);
}

template <typename scalar_t>
inline void verify_gemm(
    const gemm_batched_strided_arguments_t<scalar_t> arguments) {
  std::string alloc;
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
  std::tie(alloc, offset, batch, m, n, k, transa, transb, alpha, beta, lda_mul,
           ldb_mul, ldc_mul, stride_a_mul, stride_b_mul, stride_c_mul) =
      arguments;

  if (alloc == "usm") {
#ifdef SB_ENABLE_USM
    verify_gemm<scalar_t, helper::AllocType::usm>(arguments);
#endif
  } else {
    verify_gemm<scalar_t, helper::AllocType::buffer>(arguments);
  }
}

template <class T>
static std::string generate_batched_strided_name(
    const ::testing::TestParamInfo<gemm_batched_strided_arguments_t<T>>& info) {
  std::string alloc;
  int offset, batch, m, n, k, ldaMul, ldbMul, ldcMul, stride_a_mul,
      stride_b_mul, stride_c_mul;
  char transa, transb;
  T alpha, beta;
  BLAS_GENERATE_NAME(info.param, alloc, offset, batch, m, n, k, transa, transb,
                     alpha, beta, ldaMul, ldbMul, ldcMul, stride_a_mul,
                     stride_b_mul, stride_c_mul);
}

/** Registers GEMM test for all supported data types
 * @param test_suite Name of the test suite
 * @param combination Combinations object
 * @see BLAS_REGISTER_TEST_CUSTOM_NAME
 */
#define GENERATE_GEMM_TEST(test_suite, combination)                          \
  BLAS_REGISTER_TEST_CUSTOM_NAME(test_suite, test_suite##combination,        \
                                 verify_gemm, gemm_arguments_t, combination, \
                                 generate_name);

/** Registers GEMM Strided Batched test for all supported data types
 * @param test_suite Name of the test suite
 * @param combination Combinations object
 * @see BLAS_REGISTER_TEST_CUSTOM_NAME
 */
#define GENERATE_GEMM_STRIDED_BATCHED_TEST(test_suite, combination)   \
  BLAS_REGISTER_TEST_CUSTOM_NAME(test_suite, test_suite##combination, \
                                 verify_gemm,                         \
                                 gemm_batched_strided_arguments_t,    \
                                 combination, generate_batched_strided_name);

#ifdef BLAS_ENABLE_COMPLEX

template <typename scalar_t, helper::AllocType mem_alloc>
inline void verify_gemm(const gemm_cplx_arguments_t<scalar_t> arguments) {
  std::string alloc;
  index_t offset;
  index_t batch;
  index_t m;
  index_t n;
  index_t k;
  char transa;
  char transb;
  std::complex<scalar_t> alpha;
  std::complex<scalar_t> beta;
  index_t lda_mul;
  index_t ldb_mul;
  index_t ldc_mul;
  gemm_batch_type_t batch_type;
  std::tie(alloc, offset, batch, m, n, k, transa, transb, alpha, beta, lda_mul,
           ldb_mul, ldc_mul, batch_type) = arguments;

  if (batch > 1 && batch_type == gemm_batch_type_t::interleaved) {
    // Interleaved batched gemm unsupported with complex data types
    GTEST_SKIP();
  }

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

  const index_t buffer_size_a = batch * size_a + offset;
  const index_t buffer_size_b = batch * size_b + offset;
  const index_t buffer_size_c = batch * size_c + offset;

  std::vector<std::complex<scalar_t>> a_m(buffer_size_a);
  std::vector<std::complex<scalar_t>> b_m(buffer_size_b);
  std::vector<std::complex<scalar_t>> c_m_gpu(buffer_size_c);

  fill_random(a_m);
  fill_random(b_m);
  fill_random(c_m_gpu);
  std::vector<std::complex<scalar_t>> c_m_cpu = c_m_gpu;

  // Use system blas to create a reference output
  for (int i = 0; i < batch; ++i) {
    reference_blas::cgemm<scalar_t>(
        ta_str, tb_str, m, n, k, reinterpret_cast<const void*>(&alpha),
        reinterpret_cast<const void*>(a_m.data() + i * size_a + offset), lda,
        reinterpret_cast<const void*>(b_m.data() + i * size_b + offset), ldb,
        reinterpret_cast<const void*>(&beta),
        reinterpret_cast<void*>(c_m_cpu.data() + i * size_c + offset), ldc);
  }

  auto m_a_gpu = blas::helper::allocate<mem_alloc, complex_sycl<scalar_t>>(
      buffer_size_a, q);
  auto m_b_gpu = blas::helper::allocate<mem_alloc, complex_sycl<scalar_t>>(
      buffer_size_b, q);
  auto m_c_gpu = blas::helper::allocate<mem_alloc, complex_sycl<scalar_t>>(
      buffer_size_c, q);

  auto copy_a = blas::helper::copy_to_device(
      q, reinterpret_cast<complex_sycl<scalar_t>*>(a_m.data()), m_a_gpu,
      buffer_size_a);
  auto copy_b = blas::helper::copy_to_device(
      q, reinterpret_cast<complex_sycl<scalar_t>*>(b_m.data()), m_b_gpu,
      buffer_size_b);
  auto copy_c = blas::helper::copy_to_device(
      q, reinterpret_cast<complex_sycl<scalar_t>*>(c_m_gpu.data()), m_c_gpu,
      buffer_size_c);

  complex_sycl<scalar_t> alpha_sycl(alpha);
  complex_sycl<scalar_t> beta_sycl(beta);

  // portBLAS GEMM implementation
  typename blas::SB_Handle::event_t gemm_event;
  if (batch == index_t(1)) {
    gemm_event = _gemm(sb_handle, transa, transb, m, n, k, alpha_sycl,
                       m_a_gpu + offset, lda, m_b_gpu + offset, ldb, beta_sycl,
                       m_c_gpu + offset, ldc, {copy_a, copy_b, copy_c});
  } else {
    return;
    _gemm_batched(sb_handle, transa, transb, m, n, k, alpha, m_a_gpu + offset,
                  lda, m_b_gpu + offset, ldb, beta, m_c_gpu + offset, ldc,
                  batch, batch_type, {copy_a, copy_b, copy_c});
  }
  sb_handle.wait(gemm_event);

  auto event = blas::helper::copy_to_host(
      q, m_c_gpu, reinterpret_cast<complex_sycl<scalar_t>*>(c_m_gpu.data()),
      buffer_size_c);
  sb_handle.wait(event);

  const bool isAlmostEqual = utils::compare_vectors<scalar_t>(c_m_gpu, c_m_cpu);
  ASSERT_TRUE(isAlmostEqual);

  helper::deallocate<mem_alloc>(m_a_gpu, q);
  helper::deallocate<mem_alloc>(m_b_gpu, q);
  helper::deallocate<mem_alloc>(m_c_gpu, q);
}

template <typename scalar_t>
inline void verify_gemm(const gemm_cplx_arguments_t<scalar_t> arguments) {
  std::string alloc;
  index_t offset;
  index_t batch;
  index_t m;
  index_t n;
  index_t k;
  char transa;
  char transb;
  std::complex<scalar_t> alpha;
  std::complex<scalar_t> beta;
  index_t lda_mul;
  index_t ldb_mul;
  index_t ldc_mul;
  gemm_batch_type_t batch_type;
  std::tie(alloc, offset, batch, m, n, k, transa, transb, alpha, beta, lda_mul,
           ldb_mul, ldc_mul, batch_type) = arguments;

  if (alloc == "usm") {
#ifdef SB_ENABLE_USM
    verify_gemm<scalar_t, helper::AllocType::usm>(arguments);
#else
    GTEST_SKIP();
#endif
  } else {
    verify_gemm<scalar_t, helper::AllocType::buffer>(arguments);
  }
}

template <class T>
static std::string generate_cplx_name(
    const ::testing::TestParamInfo<gemm_cplx_arguments_t<T>>& info) {
  std::string alloc;
  int offset, batch, m, n, k, ldaMul, ldbMul, ldcMul;
  char transa, transb;
  std::complex<T> alpha, beta;
  gemm_batch_type_t batchType;
  BLAS_GENERATE_NAME(info.param, alloc, offset, batch, m, n, k, transa, transb,
                     alpha, beta, ldaMul, ldbMul, ldcMul, batchType);
}

template <typename scalar_t, helper::AllocType mem_alloc>
inline void verify_gemm(
    const gemm_cplx_batched_strided_arguments_t<scalar_t> arguments) {
  std::string alloc;
  index_t offset;
  index_t batch;
  index_t m;
  index_t n;
  index_t k;
  char transa;
  char transb;
  std::complex<scalar_t> alpha;
  std::complex<scalar_t> beta;
  index_t lda_mul;
  index_t ldb_mul;
  index_t ldc_mul;
  index_t stride_a_mul;
  index_t stride_b_mul;
  index_t stride_c_mul;
  std::tie(alloc, offset, batch, m, n, k, transa, transb, alpha, beta, lda_mul,
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

  std::vector<std::complex<scalar_t>> a_m(buffer_size_a);
  std::vector<std::complex<scalar_t>> b_m(buffer_size_b);
  std::vector<std::complex<scalar_t>> c_m_gpu(buffer_size_c);

  fill_random(a_m);
  fill_random(b_m);
  fill_random(c_m_gpu);
  std::vector<std::complex<scalar_t>> c_m_cpu = c_m_gpu;

  // Use system blas to create a reference output
  for (int i = 0; i < batch; ++i) {
    reference_blas::cgemm<scalar_t>(
        ta_str, tb_str, m, n, k, reinterpret_cast<const void*>(&alpha),
        reinterpret_cast<const void*>(a_m.data() + i * stride_a + offset), lda,
        reinterpret_cast<const void*>(b_m.data() + i * stride_b + offset), ldb,
        reinterpret_cast<const void*>(&beta),
        reinterpret_cast<void*>(c_m_cpu.data() + i * stride_c + offset), ldc);
  }

  auto m_a_gpu = blas::helper::allocate<mem_alloc, complex_sycl<scalar_t>>(
      buffer_size_a, q);
  auto m_b_gpu = blas::helper::allocate<mem_alloc, complex_sycl<scalar_t>>(
      buffer_size_b, q);
  auto m_c_gpu = blas::helper::allocate<mem_alloc, complex_sycl<scalar_t>>(
      buffer_size_c, q);

  auto copy_a = blas::helper::copy_to_device(
      q, reinterpret_cast<complex_sycl<scalar_t>*>(a_m.data()), m_a_gpu,
      buffer_size_a);
  auto copy_b = blas::helper::copy_to_device(
      q, reinterpret_cast<complex_sycl<scalar_t>*>(b_m.data()), m_b_gpu,
      buffer_size_b);
  auto copy_c = blas::helper::copy_to_device(
      q, reinterpret_cast<complex_sycl<scalar_t>*>(c_m_gpu.data()), m_c_gpu,
      buffer_size_c);

  complex_sycl<scalar_t> alpha_sycl(alpha);
  complex_sycl<scalar_t> beta_sycl(beta);

  // portBLAS GEMM STRIDED BATCHED implementation
  auto gemm_batched_event = _gemm_strided_batched(
      sb_handle, transa, transb, m, n, k, alpha_sycl, m_a_gpu + offset, lda,
      stride_a, m_b_gpu + offset, ldb, stride_b, beta_sycl, m_c_gpu + offset,
      ldc, stride_c, batch, {copy_a, copy_b, copy_c});

  sb_handle.wait({gemm_batched_event});
  auto event = blas::helper::copy_to_host(
      q, m_c_gpu, reinterpret_cast<complex_sycl<scalar_t>*>(c_m_gpu.data()),
      buffer_size_c);
  sb_handle.wait(event);

  const bool isAlmostEqual =
      (stride_c_mul == 1)
          ? utils::compare_vectors(c_m_gpu, c_m_cpu)
          : utils::compare_vectors_strided(c_m_gpu, c_m_cpu, stride_c, size_c);
  ASSERT_TRUE(isAlmostEqual);

  helper::deallocate<mem_alloc>(m_a_gpu, q);
  helper::deallocate<mem_alloc>(m_b_gpu, q);
  helper::deallocate<mem_alloc>(m_c_gpu, q);
}

template <typename scalar_t>
inline void verify_gemm(
    const gemm_cplx_batched_strided_arguments_t<scalar_t> arguments) {
  std::string alloc;
  index_t offset;
  index_t batch;
  index_t m;
  index_t n;
  index_t k;
  char transa;
  char transb;
  std::complex<scalar_t> alpha;
  std::complex<scalar_t> beta;
  index_t lda_mul;
  index_t ldb_mul;
  index_t ldc_mul;
  index_t stride_a_mul;
  index_t stride_b_mul;
  index_t stride_c_mul;
  std::tie(alloc, offset, batch, m, n, k, transa, transb, alpha, beta, lda_mul,
           ldb_mul, ldc_mul, stride_a_mul, stride_b_mul, stride_c_mul) =
      arguments;

  if (alloc == "usm") {
#ifdef SB_ENABLE_USM
    verify_gemm<scalar_t, helper::AllocType::usm>(arguments);
#endif
  } else {
    verify_gemm<scalar_t, helper::AllocType::buffer>(arguments);
  }
}

template <class T>
static std::string generate_cplx_batched_strided_name(
    const ::testing::TestParamInfo<gemm_cplx_batched_strided_arguments_t<T>>&
        info) {
  std::string alloc;
  int offset, batch, m, n, k, ldaMul, ldbMul, ldcMul, stride_a_mul,
      stride_b_mul, stride_c_mul;
  char transa, transb;
  std::complex<T> alpha, beta;
  BLAS_GENERATE_NAME(info.param, alloc, offset, batch, m, n, k, transa, transb,
                     alpha, beta, ldaMul, ldbMul, ldcMul, stride_a_mul,
                     stride_b_mul, stride_c_mul);
}

/** Registers GEMM test for all supported complex data types
 * @param test_suite Name of the test suite
 * @param combination Combinations object
 * @see BLAS_REGISTER_TEST_CUSTOM_NAME
 */
#define GENERATE_CPLX_GEMM_TEST(test_suite, combination)                   \
  BLAS_REGISTER_CPLX_TEST_CUSTOM_NAME(test_suite, test_suite##combination, \
                                      verify_gemm, gemm_cplx_arguments_t,  \
                                      combination, generate_cplx_name);

#define GENERATE_CPLXGEMM_STRIDED_BATCHED_TEST(test_suite, combination) \
  BLAS_REGISTER_CPLX_TEST_CUSTOM_NAME(                                  \
      test_suite, test_suite##combination, verify_gemm,                 \
      gemm_cplx_batched_strided_arguments_t, combination,               \
      generate_cplx_batched_strided_name);

#endif
