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
 *  @filename omatadd_batched_test.cpp
 *
 **************************************************************************/

#include "blas_test.hpp"
#include "extension_reference.hpp"

template <typename scalar_t>
using combination_t =
    std::tuple<std::string, char, char, index_t, index_t, scalar_t, scalar_t,
               index_t, index_t, index_t, index_t, index_t, index_t, index_t>;

template <typename scalar_t, helper::AllocType mem_alloc>
void run_test(const combination_t<scalar_t> combi) {
  std::string alloc;
  char trans_a, trans_b;
  index_t m, n, ld_a_mul, ld_b_mul, ld_c_mul, stride_a_m, stride_b_m,
      stride_c_m, batch_size;
  scalar_t alpha, beta;

  std::tie(alloc, trans_a, trans_b, m, n, alpha, beta, ld_a_mul, ld_b_mul,
           ld_c_mul, stride_a_m, stride_b_m, stride_c_m, batch_size) = combi;

  // Rows & Cols per matrix
  const index_t a_rows = (trans_a == 't') ? n : m;
  const index_t a_cols = (trans_a == 't') ? m : n;
  const index_t b_rows = (trans_b == 't') ? n : m;
  const index_t b_cols = (trans_b == 't') ? m : n;

  index_t lda = ld_a_mul * a_rows;
  index_t ldb = ld_b_mul * b_rows;
  index_t ldc = ld_c_mul * m;

  // Base sizes of matrices
  index_t base_size = m * n;

  // Compute Strides using size-stride multipliers
  index_t stride_a = stride_a_m * base_size * ld_a_mul;
  index_t stride_b = stride_b_m * base_size * ld_b_mul;
  index_t stride_c = stride_c_m * base_size * ld_c_mul;

  auto q = make_queue();
  blas::SB_Handle sb_handle(q);

  std::vector<scalar_t> A(stride_a * batch_size);
  std::vector<scalar_t> B(stride_b * batch_size);
  std::vector<scalar_t> C(stride_c * batch_size, (scalar_t)0);

  fill_random(A);
  fill_random(B);

  std::vector<scalar_t> C_ref = C;

  // Reference implementation
  for (int i = 0; i < batch_size; ++i) {
    reference_blas::ext_omatadd(
        trans_a, trans_b, m, n, alpha, A.data() + i * stride_a, lda, beta,
        B.data() + i * stride_b, ldb, C_ref.data() + i * stride_c, ldc);
  }

  auto m_a_gpu =
      blas::helper::allocate<mem_alloc, scalar_t>(stride_a * batch_size, q);
  auto m_b_gpu =
      blas::helper::allocate<mem_alloc, scalar_t>(stride_b * batch_size, q);
  auto m_c_gpu =
      blas::helper::allocate<mem_alloc, scalar_t>(stride_c * batch_size, q);

  auto copy_m_a =
      blas::helper::copy_to_device(q, A.data(), m_a_gpu, stride_a * batch_size);
  auto copy_m_b =
      blas::helper::copy_to_device(q, B.data(), m_b_gpu, stride_b * batch_size);
  auto copy_m_c =
      blas::helper::copy_to_device(q, C.data(), m_c_gpu, stride_c * batch_size);

  auto op_event = blas::_omatadd_batch(
      sb_handle, trans_a, trans_b, m, n, alpha, m_a_gpu, lda, stride_a, beta,
      m_b_gpu, ldb, stride_b, m_c_gpu, ldc, stride_c, batch_size,
      {copy_m_a, copy_m_b, copy_m_c});

  sb_handle.wait(op_event);

  auto event = blas::helper::copy_to_host<scalar_t>(
      sb_handle.get_queue(), m_c_gpu, C.data(), stride_c * batch_size);
  sb_handle.wait(event);

  // Validate the result
  const bool isAlmostEqual = utils::compare_vectors(C, C_ref);
  ASSERT_TRUE(isAlmostEqual);

  helper::deallocate<mem_alloc>(m_a_gpu, q);
  helper::deallocate<mem_alloc>(m_b_gpu, q);
  helper::deallocate<mem_alloc>(m_c_gpu, q);
}

template <typename scalar_t>
void run_test(const combination_t<scalar_t> combi) {
  std::string alloc;
  char trans_a, trans_b;
  index_t m, n, lda_mul, ldb_mul, ldc_mul, stride_a_m, stride_b_m, stride_c_m,
      batch_size;
  scalar_t alpha, beta;

  std::tie(alloc, trans_a, trans_b, m, n, alpha, beta, lda_mul, ldb_mul,
           ldc_mul, stride_a_m, stride_b_m, stride_c_m, batch_size) = combi;

  if (alloc == "usm") {
#ifdef SB_ENABLE_USM
    run_test<scalar_t, helper::AllocType::usm>(combi);
#else
    GTEST_SKIP();
#endif
  } else {
    run_test<scalar_t, helper::AllocType::buffer>(combi);
  }
}

#ifdef STRESS_TESTING
template <typename scalar_t>
const auto combi =
    ::testing::Combine(::testing::Values("usm", "buf"),    // allocation_type
                       ::testing::Values<char>('n', 't'),  // trans_a
                       ::testing::Values<char>('n', 't'),  // trans_b
                       ::testing::Values<index_t>(1024, 4050, 16380),  // m
                       ::testing::Values<index_t>(1024, 4050, 16380),  // n
                       ::testing::Values<scalar_t>(0, 1.05, 2.01),     // alpha
                       ::testing::Values<scalar_t>(0, 1.05, 2.01),     // beta
                       ::testing::Values<index_t>(1, 2),      // lda_mul
                       ::testing::Values<index_t>(1, 2),      // ldb_mul
                       ::testing::Values<index_t>(1, 2, 3),   // ldc_mul
                       ::testing::Values<index_t>(1, 3),      // stride_a_m
                       ::testing::Values<index_t>(1, 3),      // stride_b_m
                       ::testing::Values<index_t>(1, 3),      // stride_c_m
                       ::testing::Values<index_t>(1, 2, 3));  // batch_size
#else
template <typename scalar_t>
const auto combi =
    ::testing::Combine(::testing::Values("usm", "buf"),    // allocation_type
                       ::testing::Values<char>('n', 't'),  // trans_a
                       ::testing::Values<char>('n', 't'),  // trans_b
                       ::testing::Values<index_t>(64, 129, 255),  // m
                       ::testing::Values<index_t>(64, 129, 255),  // n
                       ::testing::Values<scalar_t>(2.5),            // alpha
                       ::testing::Values<scalar_t>(3.5),            // beta
                       ::testing::Values<index_t>(1, 2),          // lda_mul
                       ::testing::Values<index_t>(1, 2),          // ldb_mul
                       ::testing::Values<index_t>(1, 2, 3),       // ldc_mul
                       ::testing::Values<index_t>(1, 3),          // stride_a_m
                       ::testing::Values<index_t>(1, 3),          // stride_b_m
                       ::testing::Values<index_t>(1, 3),          // stride_c_m
                       ::testing::Values<index_t>(2, 3));         // batch_size
#endif

template <class T>
static std::string generate_name(
    const ::testing::TestParamInfo<combination_t<T>> &info) {
  std::string alloc;
  char trans_a, trans_b;
  index_t m, n, lda_mul, ldb_mul, ldc_mul, stride_a_m, stride_b_m, stride_c_m,
      batch_size;
  T alpha, beta;
  BLAS_GENERATE_NAME(info.param, alloc, trans_a, trans_b, m, n, alpha, beta,
                     lda_mul, ldb_mul, ldc_mul, stride_a_m, stride_b_m,
                     stride_c_m, batch_size);
}

BLAS_REGISTER_TEST_ALL(OmatAddBatched, combination_t, combi, generate_name);
