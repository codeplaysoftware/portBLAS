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
 *  @filename blas2_tsgemm.cpp
 *
 **************************************************************************/

// TODO: cleanup

#include "blas_test.hpp"

template <typename scalar_t>
using combination_t =
    std::tuple<int, int, int, char, char, scalar_t, scalar_t, int, int, int>;

const auto combi = ::testing::Combine(::testing::Values(6),    // m
                                      ::testing::Values(9),    // n
                                      ::testing::Values(14),    // k
                                      ::testing::Values('n'),  // transa
                                      ::testing::Values('n'),  // transb
                                      ::testing::Values(1.0),  // alpha
                                      ::testing::Values(0.0),  // beta
                                      ::testing::Values(1),    // lda_mul
                                      ::testing::Values(1),    // ldb_mul
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
  template <typename IxType, typename VectorT>
  static inline void eval(IxType w, IxType h, VectorT v) {
    for (IxType i = 0; i < h; i++) {
      std::cerr << "[";
      for (IxType j = 0; j < w; j++) {
        if (j != 0) {
          std::cerr << ", ";
        }
        std::cerr << v[i + (j * h)];
      }
      std::cerr << "]\n";
    }
  }
};

template <>
struct MatrixPrinter<false> {
  template <typename IxType, typename VectorT>
  static inline void eval(IxType w, IxType h, VectorT v) {
    for (IxType i = 0; i < h; i++) {
      std::cerr << "[";
      for (IxType j = 0; j < w; j++) {
        if (j != 0) {
          std::cerr << ", ";
        }
        std::cerr << v[(i * w) + j];
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

  std::array<int, 2> dim_a = {m, k};
  std::array<int, 2> dim_b = {k, n};
  std::array<int, 2> dim_c = {m, n};

  int lda = ((transa != 'n') ? dim_a[1] : dim_a[0]) * lda_mul;
  int ldb = ((transb != 'n') ? dim_b[1] : dim_b[0]) * ldb_mul;
  int ldc = dim_c[0] * ldc_mul;

  std::vector<scalar_t> a_m(m * k * lda_mul);
  std::vector<scalar_t> b_m(k * n * ldb_mul);
  std::vector<scalar_t> c_m_gpu(m * n * ldc_mul);
  std::vector<scalar_t> c_m_cpu(m * n * ldc_mul);

  fill_random(a_m);
  fill_random(b_m);
  fill_random(c_m_gpu);
  std::copy(c_m_gpu.begin(), c_m_gpu.end(), c_m_cpu.begin());

  // system gemm implementation

  reference_blas::gemm(ta_str, tb_str, m, n, k, (scalar_t)alpha, a_m.data(), m,
                       b_m.data(), k, (scalar_t)beta, c_m_cpu.data(), m);

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
      std::cerr << " I have caught an exception! " << std::endl;
      std::cerr << e.what() << std::endl;
    }
  }

  std::cerr << "A before: " << std::endl;
  MatrixPrinter<true>::eval(k, m, a_m);

  std::cerr << "B before: " << std::endl;
  MatrixPrinter<true>::eval(n, k, b_m);

  // the matrix is now in tsgf._C
  std::cerr << "C expected: " << std::endl;
  MatrixPrinter<true>::eval(n, m, c_m_cpu);

  std::cerr << "C afterwards: " << std::endl;
  MatrixPrinter<true>::eval(n, m, c_m_gpu);

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
