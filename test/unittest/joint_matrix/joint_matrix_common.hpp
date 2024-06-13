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
 *  @filename joint_matrix_common.hpp
 *
 **************************************************************************/

#include "launch_gemm.hpp"

template <typename T>
using joint_matrix_arguments_t =
    std::tuple<std::string, std::string, index_t, index_t, index_t, std::string,
               index_t, index_t, index_t, index_t, index_t, char, char, T, T,
               index_t, index_t, index_t, gemm_batch_type_t>;

template <typename scalar_t, helper::AllocType mem_alloc>
inline void verify_gemm(const joint_matrix_arguments_t<scalar_t> arguments) {
  std::string jm_inType, jm_outType;
  index_t jm_m, jm_n, jm_k;
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
  std::tie(jm_inType, jm_outType, jm_m, jm_n, jm_k, alloc, offset, batch, m, n,
           k, transa, transb, alpha, beta, lda_mul, ldb_mul, ldc_mul,
           batch_type) = arguments;

  assert(batch_type == gemm_batch_type_t::strided);

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

  if (jm_outType == "half") {
    // initialize the vectors with positive values
    // to avoid test failures for half precision
    // accumulation
    fill_random_with_range(a_m, scalar_t{1}, scalar_t{2});
    fill_random_with_range(b_m, scalar_t{1}, scalar_t{2});
    fill_random_with_range(c_m_gpu, scalar_t{1}, scalar_t{2});
  } else {
    fill_random(a_m);
    fill_random(b_m);
    fill_random(c_m_gpu);
  }

  index_t nbits = 13;
  if (jm_inType == "bfloat16") {
    nbits = 16;
  }
  set_to_zero_last_nbits(a_m, nbits);
  set_to_zero_last_nbits(b_m, nbits);
  set_to_zero_last_nbits(c_m_gpu, nbits);
  set_to_zero_last_nbits(alpha, nbits);
  set_to_zero_last_nbits(beta, nbits);

  std::vector<scalar_t> c_m_cpu = c_m_gpu;

  // Use system blas to create a reference output
  for (int i = 0; i < batch; ++i) {
    reference_blas::gemm(ta_str, tb_str, m, n, k, alpha,
                         a_m.data() + i * size_a + offset, lda,
                         b_m.data() + i * size_b + offset, ldb, beta,
                         c_m_cpu.data() + i * size_c + offset, ldc);
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
  if (jm_inType == "half" && jm_outType == "float") {
    if (jm_m == 16 && jm_n == 16) {
      gemm_event = launch_gemm_with_beta<16, 16, 16, sycl::half, float>(
          sb_handle, transa, transb, m, n, k, alpha, m_a_gpu + offset, lda,
          size_a, m_b_gpu + offset, ldb, size_b, beta, m_c_gpu + offset, ldc,
          size_c, batch, batch_type, {copy_a, copy_b, copy_c});
    } else if (jm_m == 32 && jm_n == 8) {
      gemm_event = launch_gemm_with_beta<32, 8, 16, sycl::half, float>(
          sb_handle, transa, transb, m, n, k, alpha, m_a_gpu + offset, lda,
          size_a, m_b_gpu + offset, ldb, size_b, beta, m_c_gpu + offset, ldc,
          size_c, batch, batch_type, {copy_a, copy_b, copy_c});
    } else if (jm_n == 32 && jm_m == 8) {
      gemm_event = launch_gemm_with_beta<8, 32, 16, sycl::half, float>(
          sb_handle, transa, transb, m, n, k, alpha, m_a_gpu + offset, lda,
          size_a, m_b_gpu + offset, ldb, size_b, beta, m_c_gpu + offset, ldc,
          size_c, batch, batch_type, {copy_a, copy_b, copy_c});
    }
  } else if (jm_inType == "half" && jm_outType == "half") {
    if (jm_m == 16 && jm_n == 16) {
      gemm_event = launch_gemm_with_beta<16, 16, 16, sycl::half, sycl::half>(
          sb_handle, transa, transb, m, n, k, alpha, m_a_gpu + offset, lda,
          size_a, m_b_gpu + offset, ldb, size_b, beta, m_c_gpu + offset, ldc,
          size_c, batch, batch_type, {copy_a, copy_b, copy_c});
    } else if (jm_m == 32 && jm_n == 8) {
      gemm_event = launch_gemm_with_beta<32, 8, 16, sycl::half, sycl::half>(
          sb_handle, transa, transb, m, n, k, alpha, m_a_gpu + offset, lda,
          size_a, m_b_gpu + offset, ldb, size_b, beta, m_c_gpu + offset, ldc,
          size_c, batch, batch_type, {copy_a, copy_b, copy_c});
    } else if (jm_n == 32 && jm_m == 8) {
      gemm_event = launch_gemm_with_beta<8, 32, 16, sycl::half, sycl::half>(
          sb_handle, transa, transb, m, n, k, alpha, m_a_gpu + offset, lda,
          size_a, m_b_gpu + offset, ldb, size_b, beta, m_c_gpu + offset, ldc,
          size_c, batch, batch_type, {copy_a, copy_b, copy_c});
    }
  } else if (jm_inType == "bfloat16" && jm_outType == "float") {
    if (jm_m == 16 && jm_n == 16) {
      gemm_event =
          launch_gemm_with_beta<16, 16, 16, sycl::ext::oneapi::bfloat16, float>(
              sb_handle, transa, transb, m, n, k, alpha, m_a_gpu + offset, lda,
              size_a, m_b_gpu + offset, ldb, size_b, beta, m_c_gpu + offset,
              ldc, size_c, batch, batch_type, {copy_a, copy_b, copy_c});
    } else if (jm_m == 32 && jm_n == 8) {
      gemm_event =
          launch_gemm_with_beta<32, 8, 16, sycl::ext::oneapi::bfloat16, float>(
              sb_handle, transa, transb, m, n, k, alpha, m_a_gpu + offset, lda,
              size_a, m_b_gpu + offset, ldb, size_b, beta, m_c_gpu + offset,
              ldc, size_c, batch, batch_type, {copy_a, copy_b, copy_c});
    } else if (jm_n == 32 && jm_m == 8) {
      gemm_event =
          launch_gemm_with_beta<8, 32, 16, sycl::ext::oneapi::bfloat16, float>(
              sb_handle, transa, transb, m, n, k, alpha, m_a_gpu + offset, lda,
              size_a, m_b_gpu + offset, ldb, size_b, beta, m_c_gpu + offset,
              ldc, size_c, batch, batch_type, {copy_a, copy_b, copy_c});
    }
  } else if (jm_inType == "tf32" && jm_outType == "float") {
    using namespace sycl::ext::oneapi::experimental::matrix;
    gemm_event = launch_gemm_with_beta<16, 16, 8, precision::tf32, float>(
        sb_handle, transa, transb, m, n, k, alpha, m_a_gpu + offset, lda,
        size_a, m_b_gpu + offset, ldb, size_b, beta, m_c_gpu + offset, ldc,
        size_c, batch, batch_type, {copy_a, copy_b, copy_c});
  }
  sb_handle.wait(gemm_event);

  auto event =
      blas::helper::copy_to_host(q, m_c_gpu, c_m_gpu.data(), buffer_size_c);
  sb_handle.wait(event);

  const bool isAlmostEqual = utils::compare_vectors(
      c_m_gpu, c_m_cpu, std::cerr, "\n", jm_outType == "half" ? 3 : 1);
  ASSERT_TRUE(isAlmostEqual);

  helper::deallocate<mem_alloc>(m_a_gpu, q);
  helper::deallocate<mem_alloc>(m_b_gpu, q);
  helper::deallocate<mem_alloc>(m_c_gpu, q);
}

template <typename scalar_t>
inline void verify_gemm(const joint_matrix_arguments_t<scalar_t> arguments) {
  std::string jm_inType, jm_OutType;
  index_t jm_m, jm_n, jm_k;
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
  std::tie(jm_inType, jm_OutType, jm_m, jm_n, jm_k, alloc, offset, batch, m, n,
           k, transa, transb, alpha, beta, lda_mul, ldb_mul, ldc_mul,
           batch_type) = arguments;

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
    const ::testing::TestParamInfo<joint_matrix_arguments_t<T>>& info) {
  std::string jm_inType, jm_OutType;
  int jm_m, jm_n, jm_k;
  std::string alloc;
  int offset, batch, m, n, k, ldaMul, ldbMul, ldcMul;
  char transa, transb;
  T alpha, beta;
  gemm_batch_type_t batchType;
  BLAS_GENERATE_NAME(info.param, jm_inType, jm_OutType, jm_m, jm_n, jm_k, alloc,
                     offset, batch, m, n, k, transa, transb, alpha, beta,
                     ldaMul, ldbMul, ldcMul, batchType);
}

/** Registers Joint Matrix test for all supported data types (only float for
 * now)
 * @param test_suite Name of the test suite
 * @param combination Combinations object
 * @see BLAS_REGISTER_TEST_CUSTOM_NAME
 */
#define GENERATE_JOINTMATRIX_TEST(test_suite, combination)                    \
  BLAS_REGISTER_TEST_FLOAT_CUSTOM_NAME(test_suite, test_suite##combination,   \
                                       verify_gemm, joint_matrix_arguments_t, \
                                       combination, generate_name);
