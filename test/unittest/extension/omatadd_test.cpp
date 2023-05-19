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
 *  @filename omatadd_test.cpp
 *
 **************************************************************************/

#include "blas_test.hpp"

using index_t = int;

template <typename scalar_t>
using combination_t = std::tuple<char, char, index_t, index_t, scalar_t,
                                 scalar_t, index_t, index_t>;

template <typename T>
[[clang::always_inline]] inline void otp_transpose_matrix(
    std::vector<T> &m, const index_t rows, const index_t cols,
    const index_t ldm, const index_t ldm_out) {
  std::vector<T> b(m.size());
  for (int i = 0; i < cols; ++i) {
    for (int j = 0; j < rows; ++j) {
      b[i + j * ldm_out] = m[j + i * ldm];
    }
  }
  std::swap(m, b);
}

template <typename T>
void omatadd(const char trans_a, const char trans_b, const index_t m,
             const index_t n, const T alpha, std::vector<T> &a,
             const index_t lda, const T beta, std::vector<T> &b,
             const index_t ldb, std::vector<T> &c, const index_t ldc) {
  if (trans_a) {
    otp_transpose_matrix(a, n, m, lda, ldc);
  }
  if (trans_b) {
    otp_transpose_matrix(b, n, m, ldb, ldc);
  }

  for (int i = 0; i < a.size(); ++i) {
    c[i] = alpha * a[i] + beta * b[i];
  }
  return;
}

template <typename scalar_t>
void run_test(const combination_t<scalar_t> combi) {
  char trans_a, trans_b;
  index_t m, n, ld_in, ld_out;
  scalar_t alpha, beta;

  std::tie(trans_a, trans_b, m, n, alpha, beta, ld_in, ld_out) = combi;

  // bail out early if the leading dimensions are not correct
  if (ld_in < m || ld_out < (trans_a == 't' ? n : m)) return;

  auto q = make_queue();
  blas::SB_Handle sb_handle(q);

  index_t size = m * n;
  // std::max(ld_in, ld_out) * (trans_a == 't' ? std::max(m, n) : n);
  std::vector<scalar_t> A(size);
  std::vector<scalar_t> B(size);
  std::vector<scalar_t> C(size);

  fill_random(A);
  fill_random(B);

  std::vector<scalar_t> A_ref = A;
  std::vector<scalar_t> B_ref = B;
  std::vector<scalar_t> C_ref = C;

  const index_t lda = (trans_a == 'n') ? m : n;
  const index_t ldb = (trans_b == 'n') ? m : n;

  // Reference implementation
  omatadd((trans_a == 't'), (trans_b == 't'), m, n, alpha, A_ref, lda, beta,
          B_ref, ldb, C_ref, m);

  auto m_a_gpu = blas::make_sycl_iterator_buffer<scalar_t>(A, size);
  auto m_b_gpu = blas::make_sycl_iterator_buffer<scalar_t>(B, size);
  auto m_c_gpu = blas::make_sycl_iterator_buffer<scalar_t>(C, size);

  blas::extension::_omatadd(sb_handle, trans_a, trans_b, m, n, alpha, m_a_gpu,
                            lda, beta, m_b_gpu, ldb, m_c_gpu, m);

  auto event = blas::helper::copy_to_host<scalar_t>(sb_handle.get_queue(),
                                                    m_c_gpu, C.data(), size);
  sb_handle.wait(event);

  // Validate the result
  const bool isAlmostEqual = utils::compare_vectors(C, C_ref);
  ASSERT_TRUE(isAlmostEqual);
}

template <typename scalar_t>
const auto combi = ::testing::Combine(
    ::testing::Values<char>('n'), ::testing::Values<char>('n'),
    ::testing::Values<index_t>(64, 128), ::testing::Values<index_t>(64, 128),
    ::testing::Values<scalar_t>(0, 1, 2), ::testing::Values<scalar_t>(0, 1, 2),
    ::testing::Values<index_t>(64, 128), ::testing::Values<index_t>(64, 128));

template <class T>
static std::string generate_name(
    const ::testing::TestParamInfo<combination_t<T>> &info) {
  char trans_a, trans_b;
  index_t m, n, ld_in, ld_out;
  T alpha, beta;
  BLAS_GENERATE_NAME(info.param, trans_a, trans_b, m, n, alpha, beta, ld_in,
                     ld_out);
}

BLAS_REGISTER_TEST_ALL(OmatAdd, combination_t, combi, generate_name);
