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
  char trans;
  char side;
  char diag;
  char uplo;
  scalar_t alpha;
  scalar_t ldaMul;
  scalar_t ldbMul;
  scalar_t unusedValue;
  std::tie(m, n, trans, side, diag, uplo, alpha, ldaMul, ldbMul,
           unusedValue) = combi;

  using data_t = utils::data_storage_t<scalar_t>;

  const int lda = (side == 'l' ? m : n) * ldaMul;
  const int ldb = m * ldbMul;
  const int k = side == 'l' ? m : n;

  const int sizeA = k * lda;
  const int sizeB = n * ldb;

  std::vector<data_t> A(sizeA);
  std::vector<data_t> B(sizeB);
  std::vector<data_t> cpu_B(sizeB);

  const data_t diagValue =
      diag == 'u' ? data_t{1} : random_scalar(data_t{1}, data_t{10});

  fill_trsm_matrix(A, k, lda, uplo, diagValue,
                   static_cast<data_t>(unusedValue));
  fill_random(B);

  // Create a copy of B to calculate the reference outputs
  cpu_B = B;
  reference_blas::trsm(&side, &uplo, &trans, &diag, m, n,
                       static_cast<data_t>(alpha), A.data(), lda, cpu_B.data(),
                       ldb);

  auto q = make_queue();
  test_executor_t ex(q);
  auto a_gpu = utils::make_quantized_buffer<scalar_t>(
      ex, A);  //::make_sycl_iterator_buffer<scalar_t>(A, A.size());
  auto b_gpu = utils::make_quantized_buffer<scalar_t>(
      ex, B);  // blas::make_sycl_iterator_buffer<scalar_t>(B, B.size());

  _trsm(ex, side, uplo, trans, diag, m, n, alpha, a_gpu, lda, b_gpu, ldb);

  auto event = utils::quantized_copy_to_host<scalar_t>(ex, b_gpu, B);

  ex.get_policy_handler().wait(event);

  bool isAlmostEqual = utils::compare_vectors<data_t, scalar_t>(cpu_B, B);

  ASSERT_TRUE(isAlmostEqual);
  ex.get_policy_handler().wait();
}

static constexpr double NaN = std::numeric_limits<double>::quiet_NaN();

const auto combi = ::testing::Combine(::testing::Values(7, 513, 1027),  // m
                                      ::testing::Values(7, 513, 1027),  // n
                                      ::testing::Values('n', 't'),  // trans
                                      ::testing::Values('l', 'r'),  // side
                                      ::testing::Values('u', 'n'),  // diag
                                      ::testing::Values('l', 'u'),  // uplo
                                      ::testing::Values(1.0, 2.0),  // alpha
                                      ::testing::Values(1.0, 2.0),  // lda_mul
                                      ::testing::Values(1.0, 2.0),  // ldb_mul
                                      ::testing::Values(0.0, NaN)   // unused
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

BLAS_REGISTER_TEST(Trsm, combination_t, combi, generate_name);
