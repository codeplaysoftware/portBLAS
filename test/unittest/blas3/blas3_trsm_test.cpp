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
  index_t m;
  index_t n;
  char trans;
  char side;
  char diag;
  char uplo;
  scalar_t alpha;
  scalar_t ldaMul;
  scalar_t ldbMul;
  scalar_t unusedValue;
  std::tie(m, n, trans, side, diag, uplo, alpha, ldaMul, ldbMul, unusedValue) =
      combi;

  const index_t lda = (side == 'l' ? m : n) * ldaMul;
  const index_t ldb = m * ldbMul;
  const int k = side == 'l' ? m : n;

  const int sizeA = k * lda;
  const int sizeB = n * ldb;

  std::vector<scalar_t> A(sizeA);
  std::vector<scalar_t> B(sizeB);
  std::vector<scalar_t> cpu_B(sizeB);

  const scalar_t diagValue =
      diag == 'u' ? scalar_t{1} : random_scalar(scalar_t{1}, scalar_t{10});

  fill_trsm_matrix(A, k, lda, uplo, diagValue,
                   static_cast<scalar_t>(unusedValue));
  fill_random(B);

  // Create a copy of B to calculate the reference outputs
  cpu_B = B;
  reference_blas::trsm(&side, &uplo, &trans, &diag, m, n,
                       static_cast<scalar_t>(alpha), A.data(), lda,
                       cpu_B.data(), ldb);

  auto q = make_queue();
  blas::SB_Handle sb_handle(q);
  auto a_gpu = blas::make_sycl_iterator_buffer<scalar_t>(A, A.size());
  auto b_gpu = blas::make_sycl_iterator_buffer<scalar_t>(B, B.size());

  _trsm(sb_handle, side, uplo, trans, diag, m, n, alpha, a_gpu, lda, b_gpu,
        ldb);

  auto event = blas::helper::copy_to_host<scalar_t>(sb_handle.get_queue(),
                                                    b_gpu, B.data(), B.size());
  sb_handle.wait(event);

  bool isAlmostEqual = utils::compare_vectors(cpu_B, B);

  ASSERT_TRUE(isAlmostEqual);
}

static constexpr double NaN = std::numeric_limits<double>::quiet_NaN();

template <typename scalar_t>
const auto combi =
    ::testing::Combine(::testing::Values(7, 513, 1027),        // m
                       ::testing::Values(7, 513, 1027),        // n
                       ::testing::Values('n', 't'),            // trans
                       ::testing::Values('l', 'r'),            // side
                       ::testing::Values('u', 'n'),            // diag
                       ::testing::Values('l', 'u'),            // uplo
                       ::testing::Values<scalar_t>(1.0, 2.0),  // alpha
                       ::testing::Values<scalar_t>(1.0, 2.0),  // lda_mul
                       ::testing::Values<scalar_t>(1.0, 2.0),  // ldb_mul
                       ::testing::Values<scalar_t>(0.0, NaN)   // unused
    );

// unused is a value that will be placed in the input matrix and is not meant to
// be accessed by the trsm implementation

template <class T>
static std::string generate_name(
    const ::testing::TestParamInfo<combination_t<T>>& info) {
  int m, n;
  char trans, side, diag, uplo;
  T alpha, ldaMul, ldbMul, unusedValue;
  BLAS_GENERATE_NAME(info.param, m, n, trans, side, diag, uplo, alpha, ldaMul,
                     ldbMul, unusedValue);
}

BLAS_REGISTER_TEST_ALL(Trsm, combination_t, combi, generate_name);
