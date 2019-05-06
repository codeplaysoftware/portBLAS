/***************************************************************************
 *
 *  @license
 *  Copyright (C) 2018 Codeplay Software Limited
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
 *  @filename gemm_tuner.hpp
 *
 **************************************************************************/

#include <iostream>
#include <random>
#include <vector>

#include <chrono>
#include <cmath>
#include <functional>
#include <numeric>

#include "reference_gemm.hpp"
#include "sycl_blas.h"

using namespace cl::sycl;
using namespace blas;

struct TestResultEntry {
  std::string name;
  double sec;
  double gflops;
  double error;

  TestResultEntry(std::string name) : name(name) {}

  void print() const {
    std::cout << gflops << " gflops: " << name << " - Time: " << sec
              << " ms, Error: " << error << "\n";
  }

  bool operator<(const TestResultEntry &other) const {
    return gflops < other.gflops;
  }
  bool operator>(const TestResultEntry &other) const {
    return gflops > other.gflops;
  }
};

class TestResult : public std::vector<TestResultEntry> {
 public:
  void print_all() const {
    std::cout << "== Performance Results ==\n";
    for (auto &r : *this) {
      r.print();
    }
  }
};

template <bool _TransA, bool _TransB, typename _data_t, Gemm_t _Mode>
struct GemmConfig {
  static const bool TransA = _TransA;
  static const bool TransB = _TransB;
  static const Gemm_t Mode = _Mode;
  using data_t = _data_t;
};

template <typename element_t, typename container_t, typename executor_t>
struct GemmArgs {
  int m;
  int n;
  int k;
  element_t alpha;
  container_t &a;
  int lda;
  container_t &b;
  int ldb;
  element_t beta;
  container_t c;  // Not a reference - need new copy every time
  int ldc;
  int batch_size;
  container_t &refC;
  executor_t &ex;
  TestResult &results;
};

template <typename T, typename RndEngine>
std::vector<T> gen_matrix(int size, T lo, T hi, RndEngine rnd) {
  std::uniform_real_distribution<T> dst(lo, hi);
  std::vector<T> v(size);
  for (auto &e : v) {
    e = dst(rnd);
  }
  return v;
}

template <typename T>
T relative_diff(const std::vector<T> &ref, const std::vector<T> &obt) {
  T mag(0);
  for (auto x : ref) {
    mag += x * x;
  }
  T diff = std::inner_product(std::begin(ref), std::end(ref), std::begin(obt),
                              T(0), std::plus<T>(),
                              [](T x, T y) { return (x - y) * (x - y); });
  return std::sqrt(diff / mag);
}

template <typename TestOperator>
void run_tune(int rep, double flop_cnt, TestResultEntry &result,
              TestOperator op = TestOperator()) {
  // warmup
  op();
  auto start = std::chrono::steady_clock::now();
  for (int i = 0; i < rep; ++i) {
    op();
  }
  auto end = std::chrono::steady_clock::now();
  std::chrono::duration<double> sec_d = end - start;
  double sec = sec_d.count() / rep;
  result.sec = sec * 1e3;
  result.gflops = flop_cnt / sec / 1e9;
}

template <int Cls, typename Tile, bool DoubleBuffer, bool Nbca, bool Nbcb,
          typename Config, typename T, typename Container, typename Executor>
// a should not be a reference, the C buffer needs copied
void tune(int r, GemmArgs<T, Container, Executor> a) {
  using Gemm = Gemm<typename Config::data_t, typename Config::data_t,
                    DoubleBuffer, Nbca, Nbcb, Cls, Tile, Config::TransA,
                    Config::TransB, T, false, static_cast<int>(Config::Mode)>;

  using etype = typename Gemm::value_t;
  a.results.emplace_back(Gemm::get_type_string());
  TestResultEntry &result = a.results.back();
  {
    blas::BufferIterator<etype, codeplay_policy> m_a_gpu =
        blas::make_sycl_iterator_buffer<etype>(const_cast<etype *>(a.a.data()),
                                               a.a.size());

    blas::BufferIterator<etype, codeplay_policy> m_b_gpu =
        blas::make_sycl_iterator_buffer<etype>(const_cast<etype *>(a.b.data()),
                                               a.b.size());

    blas::BufferIterator<etype, codeplay_policy> m_c_gpu =
        blas::make_sycl_iterator_buffer<etype>(const_cast<etype *>(a.c.data()),
                                               a.c.size());

    auto accA =
        make_matrix_view(a.ex, m_a_gpu, a.m, a.k, a.lda, Access::col_major());
    auto accB =
        make_matrix_view(a.ex, m_b_gpu, a.k, a.n, a.ldb, Access::col_major());
    auto accC =
        make_matrix_view(a.ex, m_c_gpu, a.m, a.n, a.ldc, Access::col_major());
    auto gemm = Gemm(accA, accB, accC, a.alpha, a.beta, a.batch_size);
    run_tune(r, 2.0 * a.m * a.n * a.k * a.batch_size, result, [&] {
      auto event = a.ex.execute(gemm);
      a.ex.get_policy_handler().wait(event);
    });
  }
  result.error = relative_diff(a.refC, a.c);
}

template <typename T, typename Container, typename Executor>
void tune_syclblas(int r, char transA, char transB,
                   GemmArgs<T, Container, Executor> a) {
  using etype = typename Container::value_type;
  a.results.emplace_back("SYCL-BLAS gemm");
  TestResultEntry &result = a.results.back();
  {
    auto m_a_gpu = blas::make_sycl_iterator_buffer<etype>(
        const_cast<etype *>(a.a.data()), a.a.size());
    auto m_b_gpu = blas::make_sycl_iterator_buffer<etype>(
        const_cast<etype *>(a.b.data()), a.b.size());
    auto m_c_gpu = blas::make_sycl_iterator_buffer<etype>(
        const_cast<etype *>(a.c.data()), a.c.size());
    run_tune(r, 2.0 * a.m * a.n * a.k * a.batch_size, result, [&] {
      auto event = _gemm_batched(a.ex, transA, transB, a.m, a.n, a.k, a.alpha,
                                 m_a_gpu, a.lda, m_b_gpu, a.ldb, a.beta,
                                 m_c_gpu, a.ldc, a.batch_size);
      a.ex.get_policy_handler().wait(event);
    });
  }
  result.error = relative_diff(a.refC, a.c);
}

template <bool TransA, bool TransB, typename E>
void run_tune_gemm(int seed, int m, int k, int n, int batch_size, int rep) {
  std::cout << std::scientific;

  std::mt19937 rnd(seed);

  auto dataA = gen_matrix<E>(k * m * batch_size, -1, 1, rnd);
  auto dataB = gen_matrix<E>(n * k * batch_size, -1, 1, rnd);
  auto origC = gen_matrix<E>(m * n * batch_size, -1, 1, rnd);
  auto refC = origC;

  const char *ta_str = TransA ? "T" : "N";
  const char *tb_str = TransB ? "T" : "N";

  const int lda = TransA ? k : m;
  const int ldb = TransB ? n : k;
  const int ldc = m;

  TestResult results{};

  results.emplace_back("System GEMM implementation");
  TestResultEntry &ref_result = results.back();
  run_tune(rep, 2.0 * m * n * k * batch_size, ref_result, [&] {
    for (int bs = 0; bs < batch_size; bs++) {
      // system gemm implementation
      reference_gemm::gemm(ta_str, tb_str, m, n, k, E(1),
                           dataA.data() + (bs * m * k), lda,
                           dataB.data() + (bs * n * k), ldb, E(1),
                           refC.data() + (bs * m * n), m);
    }
  });
  ref_result.error = 0.0;

  cl::sycl::queue q([=](cl::sycl::exception_list eL) {
    try {
      for (auto &e : eL) {
        std::rethrow_exception(e);
      }
    } catch (cl::sycl::exception &e) {
      std::cout << " E " << e.what() << std::endl;
    } catch (...) {
      std::cout << " An exception " << std::endl;
    }
  });
  std::cout << "\nDevice: "
            << q.get_device().get_info<cl::sycl::info::device::name>()
            << std::endl;

  Executor<PolicyHandler<codeplay_policy>> ex(q);

  GemmArgs<E, decltype(dataA), decltype(ex)> args{
      m,    n,     k,   E(1),       dataA, lda, dataB,  ldb,
      E(1), origC, ldc, batch_size, refC,  ex,  results};

  using data_t =
      typename MatrixViewTypeFactory<codeplay_policy, E, int>::output_t;

  using Local = GemmConfig<TransA, TransB, data_t, Gemm_t::local_memory>;
  using NonLocal = GemmConfig<TransA, TransB, data_t, Gemm_t::no_local_memory>;
  using Naive = GemmConfig<TransA, TransB, data_t, Gemm_t::naive>;

  tune_syclblas(rep, *ta_str, *tb_str, args);

#include "generate_combinations.inc.hpp"

  std::sort(results.begin(), results.end());
  results.print_all();
}
