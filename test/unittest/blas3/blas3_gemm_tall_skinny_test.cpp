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
 *  @filename blas3_gemm_tall_skinny_test.cpp
 *
 **************************************************************************/

// TODO: cleanup

#include "blas_test.hpp"

template <typename scalar_t>
using combination_t =
    std::tuple<int, int, int, char, char, scalar_t, scalar_t, int, int, int>;

const auto combi = ::testing::Combine(::testing::Values(7),    // m
                                      ::testing::Values(13),    // n
                                      ::testing::Values(120),    // k
                                      ::testing::Values('n'),  // transa
                                      ::testing::Values('t'),  // transb
                                      ::testing::Values(1.0),  // alpha
                                      ::testing::Values(0.0),  // beta
                                      ::testing::Values(2),    // lda_mul
                                      ::testing::Values(3),    // ldb_mul
                                      ::testing::Values(1)     // ldc_mul
);

// ---------------------------
// Utilities to print matrices
// ---------------------------
template <bool ColMajor>
struct MatrixPrinter {
  // SFINAE
};

template <>
struct MatrixPrinter<true> {
  template <typename index_t, typename VectorT>
  static inline void eval(index_t w, index_t h, VectorT v, index_t ld) {
    for (index_t i = 0; i < h; i++) {
      std::cerr << "[";
      for (index_t j = 0; j < w; j++) {
        if (j != 0) {
          std::cerr << ", ";
        }
        std::cerr << v[i + (j * ld)];
      }
      std::cerr << "]\n";
    }
  }
};

template <>
struct MatrixPrinter<false> {
  template <typename index_t, typename VectorT>
  static inline void eval(index_t w, index_t h, VectorT v, index_t ld) {
    for (index_t i = 0; i < h; i++) {
      std::cerr << "[";
      for (index_t j = 0; j < w; j++) {
        if (j != 0) {
          std::cerr << ", ";
        }
        std::cerr << v[(i * ld) + j];
      }
      std::cerr << "]\n";
    }
  }
};

/// TESTING IMPLEMENTATION

// #define TESTING_GUARD
// #include <operations/blas3_trees.hpp>

using index_t = int;

class TSGEMMKernel;

template <typename scalar_t>
void run_test(const combination_t<scalar_t> combi) {
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
  std::tie(m, n, k, transa, transb, alpha, beta, lda_mul, ldb_mul, ldc_mul) =
      combi;

  const char ta_str[2] = {transa, '\0'};
  const char tb_str[2] = {transb, '\0'};

  auto q = make_queue();
  test_executor_t ex(q);

  auto policy_handler = ex.get_policy_handler();

  int lda = ((transa != 'n') ? k : m) * lda_mul;
  int ldb = ((transb != 'n') ? n : k) * ldb_mul;
  int ldc = m * ldc_mul;

  std::vector<scalar_t> a_m(m * k * lda_mul);
  std::vector<scalar_t> b_m(k * n * ldb_mul);
  std::vector<scalar_t> c_m_gpu(m * n * ldc_mul);
  std::vector<scalar_t> c_m_cpu(m * n * ldc_mul);

  fill_random(a_m);
  fill_random(b_m);
  fill_random(c_m_gpu);
  std::copy(c_m_gpu.begin(), c_m_gpu.end(), c_m_cpu.begin());

  // system gemm implementation

  reference_blas::gemm(ta_str, tb_str, m, n, k, (scalar_t)alpha, a_m.data(), lda,
                       b_m.data(), ldb, (scalar_t)beta, c_m_cpu.data(), ldc);

  {
    auto m_a_gpu =
        blas::make_sycl_iterator_buffer<scalar_t>(a_m, m * k * lda_mul);
    auto m_b_gpu =
        blas::make_sycl_iterator_buffer<scalar_t>(b_m, k * n * ldb_mul);
    auto m_c_gpu =
        blas::make_sycl_iterator_buffer<scalar_t>(c_m_gpu, m * n * ldc_mul);
    try {
      _gemm(ex, transa, transb, m, n, k, alpha, m_a_gpu, lda, m_b_gpu, ldb, beta,
            m_c_gpu, ldc);
    } catch (cl::sycl::exception &e) {
      std::cerr << "Exception occured:" << std::endl;
      std::cerr << e.what() << std::endl;
    }
  }

  std::cerr << "A before: " << std::endl;
  if(transa == 'n') MatrixPrinter<true>::eval(k, m, a_m, lda);
  else MatrixPrinter<false>::eval(k, m, a_m, lda);

  std::cerr << "B before: " << std::endl;
  if(transb == 'n') MatrixPrinter<true>::eval(n, k, b_m, ldb);
  else MatrixPrinter<false>::eval(n, k, b_m, ldb);

  // the matrix is now in tsgf._C
  std::cerr << "C expected: " << std::endl;
  MatrixPrinter<true>::eval(n, m, c_m_cpu, ldc);

  std::cerr << "C afterwards: " << std::endl;
  MatrixPrinter<true>::eval(n, m, c_m_gpu, ldc);

  ASSERT_TRUE(utils::compare_vectors(c_m_gpu, c_m_cpu));
}

class GemmFloat : public ::testing::TestWithParam<combination_t<float>> {};
TEST_P(GemmFloat, test) { run_test<float>(GetParam()); };
INSTANTIATE_TEST_SUITE_P(gemm, GemmFloat, combi);

#if DOUBLE_SUPPORT
class GemmDouble : public ::testing::TestWithParam<combination_t<double>> {};
TEST_P(GemmDouble, test) { run_test<double>(GetParam()); };
INSTANTIATE_TEST_SUITE_P(gemm, GemmDouble, combi);
#endif
