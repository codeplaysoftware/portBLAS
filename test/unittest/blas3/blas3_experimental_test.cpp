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
 *  @filename blas3_experimental_test.cpp
 *
 **************************************************************************/

#include "blas_test.hpp"
// #define MULTI_TEST
#ifdef MULTI_TEST
const auto combi = ::testing::Combine(::testing::Values(5),  // batch_size
                                      ::testing::Values(1, 11, 39, 64),   // m
                                      ::testing::Values(63, 257, 2, 31),  // n
                                      ::testing::Values(2, 67, 253, 65),  // k
                                      ::testing::Values('n', 't'),  // transa
                                      ::testing::Values('n', 't'),  // transb
                                      ::testing::Values(1.5),       // alpha
                                      ::testing::Values(1.0),       // beta
                                      ::testing::Values(2),         // lda_mul
                                      ::testing::Values(3),         // ldb_mul
                                      ::testing::Values(2)          // ldc_mul
);
bool print_matrices = false;
#else
const auto combi = ::testing::Combine(::testing::Values(1),       // batch_size
                                      ::testing::Values(128),     // m
                                      ::testing::Values(576),     // n
                                      ::testing::Values(401408),  // k
                                      ::testing::Values('n'),     // transa
                                      ::testing::Values('n'),     // transb
                                      ::testing::Values(1.0),     // alpha
                                      ::testing::Values(1.0),     // beta
                                      ::testing::Values(1),       // lda_mul
                                      ::testing::Values(1),       // ldb_mul
                                      ::testing::Values(1)        // ldc_mul
);
bool print_matrices = false;
#endif
template <typename T>
using combination_t =
    std::tuple<int, int, int, int, char, char, T, T, int, int, int>;

template <typename scalar_t>
void run_test(const combination_t<scalar_t> combi) {
  int batch_size;
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
  std::tie(batch_size, m, n, k, transa, transb, alpha, beta, lda_mul, ldb_mul,
           ldc_mul) = combi;

  const char ta_str[2] = {transa, '\0'};
  const char tb_str[2] = {transb, '\0'};

  auto _size = [=](const std::array<int, 2> &dim, int ld_mul) {
    return dim[0] * dim[1] * batch_size * ld_mul;
  };
  auto _size_batch = [=](const std::array<int, 2> &dim, int ld_mul) {
    return dim[0] * dim[1] * ld_mul;
  };
  auto _base = [=](const std::array<int, 2> &dim, int ld_mul, int bs) {
    return dim[0] * dim[1] * ld_mul * bs;
  };

  auto q = make_queue();
  test_executor_t ex(q);

  auto policy_handler = ex.get_policy_handler();

  std::array<int, 2> dim_a = {m, k};
  std::array<int, 2> dim_b = {k, n};
  std::array<int, 2> dim_c = {m, n};

  int lda = ((transa != 'n') ? dim_a[1] : dim_a[0]) * lda_mul;
  int ldb = ((transb != 'n') ? dim_b[1] : dim_b[0]) * ldb_mul;
  int ldc = dim_c[0] * ldc_mul;
  bool trans_a = transa != 'n';
  bool trans_b = transb != 'n';
  std::vector<scalar_t> a_m(_size(dim_a, lda_mul));
  std::vector<scalar_t> b_m(_size(dim_b, ldb_mul));
  std::vector<scalar_t> c_m_gpu(_size(dim_c, ldc_mul));
  std::vector<scalar_t> c_m_cpu(_size(dim_c, ldc_mul));
  // printf("%d, %d, %d, %d, %c, %c, %f, %f, %d, %d, %d\n", batch_size, m, n, k,
  //        transa, transb, alpha, beta, lda, ldb, ldc);
  // std::fill(a_m.begin(), a_m.end(), 0.0f);
  // std::fill(b_m.begin(), b_m.end(), 0.0f);
  // std::fill(c_m_gpu.begin(), c_m_gpu.end(), 0.0f);
  const auto increment = [](bool trans, std::vector<scalar_t> &vec,
                            std::array<int, 2> &dim, int ld_mul,
                            scalar_t starting_value = 0,
                            scalar_t increment_amount = 1) {
    scalar_t new_value = starting_value;
    for (int i = 0; i < dim[1]; ++i) {
      for (int j = 0; j < dim[0]; ++j) {
        size_t index = 0;
        if (!trans) {
          index = i * dim[0] * ld_mul + j;
        } else {
          index = j * dim[1] * ld_mul + i;
        }
        new_value += increment_amount;
        vec[index] = new_value;
      }
    }
  };
  const auto print_matrix = [&print_matrices](
                                bool trans, std::vector<scalar_t> &vec,
                                std::array<int, 2> &dim, int ld_mul,
                                int width = 8, const std::string &title = "") {
    if (!print_matrices) return;
    if (!title.empty()) {
      std::cout << title << ": \n---------------------------------\n";
    }
    for (int i = 0; i < dim[0] * ld_mul; ++i) {
      for (int j = 0; j < dim[1] * ld_mul; ++j) {
        if (!trans) {
          std::cout << std::setw(width) << vec[j * dim[0] * ld_mul + i] << " ";
        } else {
          std::cout << std::setw(width) << vec[i * dim[1] * ld_mul + j] << " ";
        }
      }
      std::cout << "\n";
    }
    std::cout << "\n";
  };
  increment(trans_a, a_m, dim_a, lda_mul);
  increment(trans_b, b_m, dim_b, ldb_mul, 0.5f);
  // increment(false,c_m_gpu, dim_c[0], dim_c[1], ldc_mul);
  fill_random(a_m);
  fill_random(b_m);
  // fill_random(c_m_gpu);
  std::copy(c_m_gpu.begin(), c_m_gpu.end(), c_m_cpu.begin());
  print_matrix(trans_a, a_m, dim_a, lda_mul, 3, "Matrix A");
  print_matrix(trans_b, b_m, dim_b, ldb_mul, 3, "Matrix B");

  auto m_a_gpu =
      policy_handler.template allocate<scalar_t>(_size(dim_a, lda_mul));
  auto m_b_gpu =
      policy_handler.template allocate<scalar_t>(_size(dim_b, ldb_mul));
  auto m_c_gpu =
      policy_handler.template allocate<scalar_t>(_size(dim_c, ldc_mul));

  for (int bs = 0; bs < batch_size; bs++) {
    // Use system blas to create a reference output
    reference_blas::gemm(ta_str, tb_str, m, n, k, alpha,
                         a_m.data() + _base(dim_a, lda_mul, bs), lda,
                         b_m.data() + _base(dim_b, ldb_mul, bs), ldb, beta,
                         c_m_cpu.data() + _base(dim_c, ldc_mul, bs), ldc);

    policy_handler.copy_to_device(a_m.data() + _base(dim_a, lda_mul, bs),
                                  m_a_gpu + _base(dim_a, lda_mul, bs),
                                  _size_batch(dim_a, lda_mul));
    policy_handler.copy_to_device(b_m.data() + _base(dim_b, ldb_mul, bs),
                                  m_b_gpu + _base(dim_b, ldb_mul, bs),
                                  _size_batch(dim_b, ldb_mul));
    policy_handler.copy_to_device(c_m_gpu.data() + _base(dim_c, ldc_mul, bs),
                                  m_c_gpu + _base(dim_c, ldc_mul, bs),
                                  _size_batch(dim_c, ldc_mul));

    // SYCL BLAS GEMM implementation
    _gemm(ex, transa, transb, m, n, k, alpha,
          m_a_gpu + _base(dim_a, lda_mul, bs), lda,
          m_b_gpu + _base(dim_b, ldb_mul, bs), ldb, beta,
          m_c_gpu + _base(dim_c, ldc_mul, bs), ldc);

    auto event =
        policy_handler.copy_to_host(m_c_gpu + _base(dim_c, ldc_mul, bs),
                                    c_m_gpu.data() + _base(dim_c, ldc_mul, bs),
                                    _size_batch(dim_c, ldc_mul));
    policy_handler.wait(event);
  }

  print_matrix(false, c_m_gpu, dim_c, ldc_mul, 3, "Matrix C");
  print_matrix(false, c_m_cpu, dim_c, ldc_mul, 3, "Matrix C(CPU)");

  ASSERT_TRUE(utils::compare_vectors(c_m_gpu, c_m_cpu));

  policy_handler.template deallocate<scalar_t>(m_a_gpu);
  policy_handler.template deallocate<scalar_t>(m_b_gpu);
  policy_handler.template deallocate<scalar_t>(m_c_gpu);
}

class GemmExperiment : public ::testing::TestWithParam<combination_t<float>> {};
TEST_P(GemmExperiment, test) { run_test<float>(GetParam()); };
INSTANTIATE_TEST_SUITE_P(gemm, GemmExperiment, combi);

#if DOUBLE_SUPPORT
class GemmDouble : public ::testing::TestWithParam<combination_t<double>> {};
TEST_P(GemmDouble, test) { run_test<double>(GetParam()); };
INSTANTIATE_TEST_SUITE_P(gemm, GemmDouble, combi);
#endif
