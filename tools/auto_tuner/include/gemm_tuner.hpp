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

#include "blas_meta.h"
#include "reference_gemm.hpp"
#include "sycl_blas.hpp"

using namespace cl::sycl;
using namespace blas;

using SYCLExecutor =
    ::blas::Executor<::blas::PolicyHandler<::blas::codeplay_policy>>;

template <typename DataType>
using HostContainer = std::vector<DataType>;

template <typename DataType>
using DeviceContainer =
    typename ::blas::MatrixViewTypeFactory<::blas::codeplay_policy, DataType,
                                           int, ::blas::col_major>::output_t;
SYCLExecutor make_sycl_executor() {
  cl::sycl::queue q([=](cl::sycl::exception_list ex_list) {
    try {
      for (auto &e_ptr : ex_list) {
        std::rethrow_exception(e_ptr);
      }
    } catch (cl::sycl::exception &e) {
      throw std::runtime_error(e.what());
    }
  });
  std::cout << "\nDevice: "
            << q.get_device().get_info<cl::sycl::info::device::name>()
            << std::endl;

  SYCLExecutor ex(q);
  return ex;
}

SYCLExecutor &get_sycl_executor() {
  static SYCLExecutor executor = make_sycl_executor();
  return executor;
}

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
      if (r.error < 0.1) {
        r.print();
      }
    }
  }
};

template <bool _TransA, bool _TransB, gemm_memory_t _MemoryMode,
          gemm_algorithm_t _ShapeMode>
struct GemmConfig {
  static const bool TransA = _TransA;
  static const bool TransB = _TransB;
  static const gemm_memory_t MemoryMode = _MemoryMode;
  static const gemm_algorithm_t ShapeMode = _ShapeMode;
};

template <typename element_t>
struct GemmArgs {
  int m;
  int n;
  int k;
  element_t alpha;
  HostContainer<element_t> &a;
  int lda;
  HostContainer<element_t> &b;
  int ldb;
  element_t beta;
  HostContainer<element_t> c;  // Not a reference - need new copy every time
  int ldc;
  int batch_size;
  HostContainer<element_t> &refC;
  SYCLExecutor &ex;
  TestResult &results;
};

template <typename T, typename RndEngine>
HostContainer<T> gen_matrix(int size, T lo, T hi, RndEngine rnd) {
  std::uniform_real_distribution<T> dst(lo, hi);
  HostContainer<T> v(size);
  for (auto &e : v) {
    e = dst(rnd);
  }
  return v;
}

template <typename T>
T relative_diff(const HostContainer<T> &ref, const HostContainer<T> &obt) {
  using std::begin;
  using std::end;
  T mag(0);
  for (auto x : ref) {
    mag += x * x;
  }
  T diff =
      std::inner_product(begin(ref), end(ref), begin(obt), T(0), std::plus<T>(),
                         [](T x, T y) { return (x - y) * (x - y); });
  return std::sqrt(diff / mag);
}

template <typename TestOperator>
void run_tune(int rep, double flop_cnt, TestResultEntry &result,
              TestOperator op = TestOperator()) {
  using Seconds = std::chrono::duration<double>;
  using MilliSeconds = std::chrono::duration<double, std::milli>;
  Seconds runtime_secs;
  // warmup
  try {
    op();
    auto start = std::chrono::steady_clock::now();
    for (int i = 0; i < rep; ++i) {
      op();
    }
    auto end = std::chrono::steady_clock::now();
    runtime_secs = end - start;
  } catch (std::exception const &e) {
    // If an error is detected when running a kernel, return without setting the
    // time in the result.
    std::cerr << "Error detected: " << e.what() << "\n";
    return;
  }
  auto seconds_per_iter = runtime_secs / rep;
  auto milliseconds =
      std::chrono::duration_cast<MilliSeconds>(seconds_per_iter);
  result.sec = milliseconds.count();
  auto gigaflop_count = flop_cnt / 1e9;
  result.gflops = gigaflop_cnt / seconds_per_iter.count();
}

template <int Cls, typename Tile, bool DoubleBuffer, bool Nbca, bool Nbcb,
          typename Config, typename T>
// a should not be a reference, the C buffer needs copied
void tune(int r, GemmArgs<T> a) {
  using Gemm = Gemm<DeviceContainer<T>, DeviceContainer<T>, DoubleBuffer, Nbca,
                    Nbcb, Cls, Tile, Config::TransA, Config::TransB, T, false,
                    static_cast<int>(Config::MemoryMode),
                    static_cast<int>(Config::ShapeMode), 1, false>;

  a.results.emplace_back(Gemm::get_type_string());
  TestResultEntry &result = a.results.back();
  {
    auto m_a_gpu = blas::make_sycl_iterator_buffer(a.a, a.a.size());
    auto m_b_gpu = blas::make_sycl_iterator_buffer(a.b, a.b.size());
    auto m_c_gpu = blas::make_sycl_iterator_buffer(a.c, a.c.size());

    auto accA = make_matrix_view<col_major>(a.ex, m_a_gpu, a.m, a.k, a.lda);
    auto accB = make_matrix_view<col_major>(a.ex, m_b_gpu, a.k, a.n, a.ldb);
    auto accC = make_matrix_view<col_major>(a.ex, m_c_gpu, a.m, a.n, a.ldc);
    auto gemm = Gemm(accA, accB, accC, a.alpha, a.beta, a.batch_size);
    run_tune(r, 2.0 * a.m * a.n * a.k * a.batch_size, result, [&] {
      auto event_list = a.ex.execute(gemm);
      for (auto &event : event_list) {
        event.wait_and_throw();
      }
    });
  }
  result.error = relative_diff(a.refC, a.c);
}

template <typename T>
void tune_syclblas(int r, char transA, char transB, GemmArgs<T> a) {
  a.results.emplace_back("SYCL-BLAS gemm");
  TestResultEntry &result = a.results.back();
  {
    auto m_a_gpu = blas::make_sycl_iterator_buffer(a.a, a.a.size());
    auto m_b_gpu = blas::make_sycl_iterator_buffer(a.b, a.b.size());
    auto m_c_gpu = blas::make_sycl_iterator_buffer(a.c, a.c.size());
    run_tune(r, 2.0 * a.m * a.n * a.k * a.batch_size, result, [&] {
      auto event_list = _gemm_batched(a.ex, transA, transB, a.m, a.n, a.k,
                                      a.alpha, m_a_gpu, a.lda, m_b_gpu, a.ldb,
                                      a.beta, m_c_gpu, a.ldc, a.batch_size);
      for (auto &event : event_list) {
        event.wait_and_throw();
      }
    });
  }
  result.error = relative_diff(a.refC, a.c);
}

template <bool TransA, bool TransB, typename DataType>
void run_tune_gemm(int seed, int m, int k, int n, int batch_size, int rep) {
  std::cout << std::scientific;

  std::mt19937 rnd(seed);

  auto dataA = gen_matrix<DataType>(k * m * batch_size, -1, 1, rnd);
  auto dataB = gen_matrix<DataType>(n * k * batch_size, -1, 1, rnd);
  auto origC = gen_matrix<DataType>(m * n * batch_size, -1, 1, rnd);
  auto refC = origC;

  const char *ta_str = TransA ? "T" : "N";
  const char *tb_str = TransB ? "T" : "N";

  const int lda = TransA ? k : m;
  const int ldb = TransB ? n : k;
  const int ldc = m;

  const DataType alpha = 1;
  const DataType beta = 1;

  TestResult results{};

  results.emplace_back("System GEMM implementation");
  TestResultEntry &ref_result = results.back();
  run_tune(rep, 2.0 * m * n * k * batch_size, ref_result, [&] {
    for (int bs = 0; bs < batch_size; bs++) {
      // system gemm implementation
      reference_gemm::gemm(ta_str, tb_str, m, n, k, alpha,
                           dataA.data() + (bs * m * k), lda,
                           dataB.data() + (bs * n * k), ldb, beta,
                           refC.data() + (bs * m * n), m);
    }
  });
  ref_result.error = 0.0;

  auto ex = get_sycl_executor();

  GemmArgs<DataType> args{m,    n,     k,   alpha,      dataA, lda, dataB,  ldb,
                          beta, origC, ldc, batch_size, refC,  ex,  results};

  using Local = GemmConfig<TransA, TransB, gemm_memory_t::local,
                           gemm_algorithm_t::standard>;
  using NonLocal = GemmConfig<TransA, TransB, gemm_memory_t::no_local,
                              gemm_algorithm_t::standard>;
  using Naive = GemmConfig<TransA, TransB, gemm_memory_t::no_local,
                           gemm_algorithm_t::naive>;

  tune_syclblas(rep, *ta_str, *tb_str, args);

#include "generate_combinations.inc.hpp"

  std::sort(results.begin(), results.end());
  results.print_all();
}
