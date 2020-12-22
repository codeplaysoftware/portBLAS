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
 **************************************************************************/

#include "blas_test.hpp"

template <typename scalar_t>
using combination_t =
    std::tuple<int, int, char, char, char, char, scalar_t, scalar_t>;

template <typename scalar_t>
void run_test(const combination_t<scalar_t> combi) {
  int m;
  int n;
  char transA;
  char side;
  char diag;
  char triangle;
  scalar_t alpha;
  scalar_t ldaMul;
  std::tie(m, n, transA, side, diag, triangle, alpha, ldaMul) = combi;

  using data_t = utils::data_storage_t<scalar_t>;

  int lda = side == 'l' ? m : n;
  int ldb = m;
  const int k = side == 'l' ? m : n;

  // Scale LDA with the value from the test suite
  lda *= ldaMul;

  const int sizeA = k * lda;
  const int sizeB = n * ldb;

  std::vector<scalar_t> A(sizeA);
  std::vector<scalar_t> B(sizeB);

  const scalar_t diagValue =
      diag == 'u' ? scalar_t{1} : random_scalar(scalar_t{1}, scalar_t{10});

  fill_trsm_matrix(A, k, lda, triangle, diagValue);
  fill_random(B);

  auto q = make_queue();
  test_executor_t ex(q);
  auto a_gpu = blas::make_sycl_iterator_buffer<scalar_t>(A, A.size());
  auto b_gpu = blas::make_sycl_iterator_buffer<scalar_t>(B, B.size());

  _trsm(ex, side, triangle, transA, diag, m, n, alpha, a_gpu, lda, b_gpu, ldb);

  std::vector<data_t> X(B.size(), data_t{0});

  auto x_gpu = blas::make_sycl_iterator_buffer<scalar_t>(X.size());
  if (side == 'l') {
    // Verification of the result. AX = alpha*B
    _gemm(ex, transA, 'n', m, n, m, data_t{1}, a_gpu, lda, b_gpu, ldb,
          data_t{0}, x_gpu, ldb);
  } else {
    // Verification of the result. XA = alpha*B
    _gemm(ex, 'n', transA, m, n, n, data_t{1}, b_gpu, ldb, a_gpu, lda,
          data_t{0}, x_gpu, ldb);
  }

  // Scale X by 1/alpha
  _scal(ex, sizeB, data_t{1} / alpha, x_gpu, 1);

  // Copy the verification result to X
  ex.get_policy_handler().wait(
      ex.get_policy_handler().copy_to_host(x_gpu, X.data(), X.size()));

  ASSERT_TRUE(utils::compare_vectors(X, B));
  ex.get_policy_handler().wait();
}

const auto combi = ::testing::Combine(::testing::Values(7, 16, 70, 300),  // m
                                      ::testing::Values(7, 16, 70, 300),  // n
                                      ::testing::Values('n', 't'),  // transA
                                      ::testing::Values('l', 'r'),  // side
                                      ::testing::Values('u', 'n'),  // diag
                                      ::testing::Values('l', 'u'),  // triangle
                                      ::testing::Values(1.0, 2.0),  // alpha
                                      ::testing::Values(1.0, 2.0)   // lda_mul
);

BLAS_REGISTER_TEST(Trsm, combination_t, combi);
