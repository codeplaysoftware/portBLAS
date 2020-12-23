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
using combination_t = std::tuple<int, int, char, char, char, char, scalar_t,
                                 scalar_t, scalar_t, scalar_t>;

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
  scalar_t ldbMul;
  scalar_t unusedValue;
  std::tie(m, n, transA, side, diag, triangle, alpha, ldaMul, ldbMul,
           unusedValue) = combi;

  using data_t = utils::data_storage_t<scalar_t>;

  const int lda = (side == 'l' ? m : n) * ldaMul;
  const int ldb = m * ldbMul;
  const int k = side == 'l' ? m : n;

  const int sizeA = k * lda;
  const int sizeB = n * ldb;

  std::vector<scalar_t> A(sizeA);
  std::vector<scalar_t> B(sizeB);
  std::vector<scalar_t> cpu_B(sizeB);

  const scalar_t diagValue =
      diag == 'u' ? scalar_t{1} : random_scalar(scalar_t{1}, scalar_t{10});

  fill_trsm_matrix(A, k, lda, triangle, diagValue, unusedValue);
  fill_random(B);

  // Create a copy of B to calculate the reference outputs
  cpu_B = B;
  reference_blas::trsm(&side, &triangle, &transA, &diag, m, n, alpha, A.data(),
                       lda, cpu_B.data(), ldb);

  auto q = make_queue();
  test_executor_t ex(q);
  auto a_gpu = blas::make_sycl_iterator_buffer<scalar_t>(A, A.size());
  auto b_gpu = blas::make_sycl_iterator_buffer<scalar_t>(B, B.size());

  _trsm(ex, side, triangle, transA, diag, m, n, alpha, a_gpu, lda, b_gpu, ldb);

  ex.get_policy_handler().wait(
      ex.get_policy_handler().copy_to_host(b_gpu, B.data(), B.size()));

  ASSERT_TRUE(utils::compare_vectors(cpu_B, B));
  ex.get_policy_handler().wait();
}

static constexpr double NaN = std::numeric_limits<double>::quiet_NaN();

const auto combi = ::testing::Combine(::testing::Values(7, 513, 1027),  // m
                                      ::testing::Values(7, 513, 1027),  // n
                                      ::testing::Values('n', 't'),  // transA
                                      ::testing::Values('l', 'r'),  // side
                                      ::testing::Values('u', 'n'),  // diag
                                      ::testing::Values('l', 'u'),  // triangle
                                      ::testing::Values(1.0, 2.0),  // alpha
                                      ::testing::Values(1.0, 2.0),  // lda_mul
                                      ::testing::Values(1.0, 2.0),  // ldb_mul
                                      ::testing::Values(0.0, NaN)   // unused
);

BLAS_REGISTER_TEST(Trsm, combination_t, combi);
