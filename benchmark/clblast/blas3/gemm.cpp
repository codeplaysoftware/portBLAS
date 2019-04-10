/***************************************************************************
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
  return "BM_Gemm<" + blas_benchmark::utils::get_type_name<scalar_t>() + ">/" +
         t1 + "/" + t2 + "/" + std::to_string(m) + "/" + std::to_string(k) +
         "/" + std::to_string(n);
}

template <typename scalar_t>
void run(benchmark::State& state, ExecutorType* executorPtr, int t1,
         int t2, int mi, int ki, int ni) {
  // Standard test setup.
  std::string t1s = blas_benchmark::utils::from_transpose_enum(
      static_cast<blas_benchmark::utils::Transposition>(t1));
  std::string t2s = blas_benchmark::utils::from_transpose_enum(
      static_cast<blas_benchmark::utils::Transposition>(t2));
  const char* t_a = t1s.c_str();
  const char* t_b = t2s.c_str();
  const index_t m = static_cast<index_t>(mi);
  const index_t k = static_cast<index_t>(ki);
  const index_t n = static_cast<index_t>(ni);

  index_t lda = t_a[0] == 'n' ? m : k;
  index_t ldb = t_b[0] == 'n' ? k : n;
  index_t ldc = m;

  state.counters["m"] = m;
  state.counters["k"] = k;
  state.counters["n"] = n;

  // The counters are double. We convert m, n and k to double to avoid
  // integer overflows and write them in the counters
  double m_d = static_cast<double>(m);
  double n_d = static_cast<double>(n);
  double k_d = static_cast<double>(k);

  state.counters["n_fl_ops"] = 3 * (m_d * n_d * k_d) + 2 * (m_d * n_d);
  state.counters["bytes_processed"] =
      (m_d * k_d + k_d * n_d + 2 * m_d * n_d) * sizeof(scalar_t);

  // Create data
  // Scalars
  scalar_t alpha = blas_benchmark::utils::random_scalar<scalar_t>();
  scalar_t beta = blas_benchmark::utils::random_scalar<scalar_t>();

  // Matrices
  std::vector<scalar_t> a = blas_benchmark::utils::random_data<scalar_t>(m * k);
  std::vector<scalar_t> b = blas_benchmark::utils::random_data<scalar_t>(k * n);
  std::vector<scalar_t> c =
      blas_benchmark::utils::const_data<scalar_t>(m * n, 0);

  // Specify the transpositions
  clblast::Transpose a_tr = blas_benchmark::utils::translate_transposition(t_a);
  clblast::Transpose b_tr = blas_benchmark::utils::translate_transposition(t_b);

  // Specify the layout. As with GEMV, this needs to be kColMajor, and results
  // in errors otherwise. It may be that this is incorrect (especially for
  // performance reasons), so may need to be revisited.
  auto layout = clblast::Layout::kColMajor;

  // Device matrices
  MemBuffer<scalar_t> a_gpu(executorPtr, a.data(), static_cast<size_t>(m * k));
  MemBuffer<scalar_t> b_gpu(executorPtr, b.data(), static_cast<size_t>(k * n));
  MemBuffer<scalar_t> c_gpu(executorPtr, c.data(), static_cast<size_t>(m * n));

  // Create a utility lambda describing the blas method that we want to run.
  auto blas_method_def = [&]() -> std::vector<cl_event> {
    cl_event event;
    clblast::Gemm<scalar_t>(layout, a_tr, b_tr, m, n, k, alpha, a_gpu.dev(), 0,
                            lda, b_gpu.dev(), 0, ldb, beta, c_gpu.dev(), 0, ldc,
                            executorPtr->_queue(), &event);
    CLEventHandler::wait(event);
    return {event};
  };

  // Warm up to avoid benchmarking data transfer
  blas_benchmark::utils::warmup(blas_method_def);

  blas_benchmark::utils::init_counters(state);

  // Measure
  for (auto _ : state) {
    std::tuple<double, double> times =
        blas_benchmark::utils::timef(blas_method_def);

    // Report
    blas_benchmark::utils::update_counters(state, times);
  }

  blas_benchmark::utils::calc_avg_counters(state);
};

template <typename scalar_t>
void register_benchmark(blas_benchmark::Args& args, ExecutorType* exPtr) {
  auto gemm_params = blas_benchmark::utils::get_params<blas3_param_t>(args);

  for (auto p : gemm_params) {
    std::string t1s, t2s;
    int m, n, k;
    std::tie(t1s, t2s, m, k, n) = p;
    int t1 = (int)blas_benchmark::utils::to_transpose_enum(t1s);
    int t2 = (int)blas_benchmark::utils::to_transpose_enum(t2s);

    auto BM_lambda = [&](benchmark::State& st, ExecutorType* exPtr, int t1,
                         int t2, int m, int k,
                         int n) { run<scalar_t>(st, exPtr, t1, t2, m, k, n); };
    benchmark::RegisterBenchmark(get_name<scalar_t>(t1s, t2s, m, k, n).c_str(),
                                 BM_lambda, exPtr, t1, t2, m, k, n);
  }
}

namespace blas_benchmark {
void create_benchmark(blas_benchmark::Args& args, ExecutorType* exPtr) {
  register_benchmark<float>(args, exPtr);
#ifdef DOUBLE_SUPPORT
  register_benchmark<double>(args, exPtr);
#endif
}
}  // namespace blas_benchmark
