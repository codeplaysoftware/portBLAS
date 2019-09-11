/**************************************************************************
 *
 *  @license
 *  Copyright (C) 2016 Codeplay Software Limited
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
 *  @filename gemm.cpp
 *
 **************************************************************************/

#include "utils.hpp"

template <typename scalar_t>
std::string get_name(std::string t1, std::string t2, int m, int k, int n) {
  std::ostringstream str{};
  str << "BM_Gemm<" << blas_benchmark::utils::get_type_name<scalar_t>() << ">/"
      << t1 << "/" << t2 << "/" << m << "/" << k << "/" << n;
  return str.str();
}

template <typename scalar_t>
void run(benchmark::State& state, ExecutorType* executorPtr, int t1, int t2,
         index_t m, index_t k, index_t n, scalar_t alpha, scalar_t beta,
         bool* success) {
  // Standard test setup.
  std::string t1s = blas_benchmark::utils::from_transpose_enum(
      static_cast<blas_benchmark::utils::Transposition>(t1));
  std::string t2s = blas_benchmark::utils::from_transpose_enum(
      static_cast<blas_benchmark::utils::Transposition>(t2));
  const char* t_a = t1s.c_str();
  const char* t_b = t2s.c_str();

  index_t lda = t_a[0] == 'n' ? m : k;
  index_t ldb = t_b[0] == 'n' ? k : n;
  index_t ldc = m;

  // The counters are double. We convert m, n and k to double to avoid
  // integer overflows for n_fl_ops and bytes_processed
  double m_d = static_cast<double>(m);
  double n_d = static_cast<double>(n);
  double k_d = static_cast<double>(k);

  state.counters["m"] = m_d;
  state.counters["k"] = k_d;
  state.counters["n"] = n_d;

  {
    double nflops_AtimesB = (2 * k_d - 1) * m_d * n_d;
    double nflops_timesAlpha = m_d * n_d;
    double nflops_addBetaC = (beta != 0) ? 2 * m_d * n_d : 0;
    state.counters["n_fl_ops"] =
        nflops_AtimesB + nflops_timesAlpha + nflops_addBetaC;
  }
  {
    double mem_readA = m_d * k_d;
    double mem_readB = k_d * n_d;
    double mem_writeC = m_d * n_d;
    double mem_readC = (beta != 0) ? m_d * n_d : 0;
    state.counters["bytes_processed"] =
        (mem_readA + mem_readB + mem_readC + mem_writeC) * sizeof(scalar_t);
  }

  ExecutorType& ex = *executorPtr;

  // Matrices
  std::vector<scalar_t> a = blas_benchmark::utils::random_data<scalar_t>(m * k);
  std::vector<scalar_t> b = blas_benchmark::utils::random_data<scalar_t>(k * n);
  std::vector<scalar_t> c =
      blas_benchmark::utils::const_data<scalar_t>(m * n, 0);

  auto a_gpu = blas::make_sycl_iterator_buffer<scalar_t>(a, m * k);
  auto b_gpu = blas::make_sycl_iterator_buffer<scalar_t>(b, k * n);
  auto c_gpu = blas::make_sycl_iterator_buffer<scalar_t>(c, m * n);

#ifdef BLAS_VERIFY_BENCHMARK
  // Run a first time with a verification of the results
  std::vector<scalar_t> c_ref = c;
  reference_blas::gemm(t_a, t_b, m, n, k, alpha, a.data(), lda, b.data(), ldb,
                       beta, c_ref.data(), ldc);
  std::vector<scalar_t> c_temp = c;
  {
    auto c_temp_gpu = blas::make_sycl_iterator_buffer<scalar_t>(c_temp, m * n);
    auto event = _gemm(ex, *t_a, *t_b, m, n, k, alpha, a_gpu, lda, b_gpu, ldb,
                       beta, c_temp_gpu, ldc);
    ex.get_policy_handler().wait(event);
  }

  std::ostringstream err_stream;
  if (!utils::compare_vectors<scalar_t>(c_temp, c_ref, err_stream, "")) {
    const std::string& err_str = err_stream.str();
    state.SkipWithError(err_str.c_str());
    *success = false;
  };
#endif
  printf("m: %d n: %d k: %d alpha: %f beta: %f lda: %d ldb: %d ldc: %d\n", m, n,
         k, alpha, beta, lda, ldb, ldc);
  auto blas_method_def = [&]() -> std::vector<cl::sycl::event> {
    auto event = _gemm(ex, *t_a, *t_b, m, n, k, alpha, a_gpu, lda, b_gpu, ldb,
                       beta, c_gpu, ldc);
    ex.get_policy_handler().wait(event);
    return event;
  };

  // Warmup
  blas_benchmark::utils::warmup(blas_method_def);
  ex.get_policy_handler().wait();

  blas_benchmark::utils::init_counters(state);

  // Measure
  for (auto _ : state) {
    // Run
    std::tuple<double, double> times =
        blas_benchmark::utils::timef(blas_method_def);

    // Report
    blas_benchmark::utils::update_counters(state, times);
  }

  blas_benchmark::utils::calc_avg_counters(state);
};

template <typename scalar_t>
void register_benchmark(blas_benchmark::Args& args, ExecutorType* exPtr,
                        bool* success) {
  auto gemm_params = blas_benchmark::utils::get_blas3_params<scalar_t>(args);

  for (auto p : gemm_params) {
    std::string t1s, t2s;
    index_t m, n, k;
    scalar_t alpha, beta;
    std::tie(t1s, t2s, m, k, n, alpha, beta) = p;
    int t1 = static_cast<int>(blas_benchmark::utils::to_transpose_enum(t1s));
    int t2 = static_cast<int>(blas_benchmark::utils::to_transpose_enum(t2s));

    auto BM_lambda = [&](benchmark::State& st, ExecutorType* exPtr, int t1,
                         int t2, index_t m, index_t k, index_t n,
                         scalar_t alpha, scalar_t beta, bool* success) {
      run<scalar_t>(st, exPtr, t1, t2, m, k, n, alpha, beta, success);
    };
    benchmark::RegisterBenchmark(get_name<scalar_t>(t1s, t2s, m, k, n).c_str(),
                                 BM_lambda, exPtr, t1, t2, m, k, n, alpha, beta,
                                 success);
  }
}

namespace blas_benchmark {
void create_benchmark(blas_benchmark::Args& args, ExecutorType* exPtr,
                      bool* success) {
  register_benchmark<float>(args, exPtr, success);
#ifdef DOUBLE_SUPPORT
  register_benchmark<double>(args, exPtr, success);
#endif
}
}  // namespace blas_benchmark
