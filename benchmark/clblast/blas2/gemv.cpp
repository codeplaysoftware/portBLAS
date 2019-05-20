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
 *  @filename gemv.cpp
 *
 **************************************************************************/

#include "utils.hpp"

template <typename scalar_t>
std::string get_name(std::string t, int m, int n) {
  std::ostringstream str{};
  str << "BM_Gemv<" << blas_benchmark::utils::get_type_name<scalar_t>() << ">/"
      << t << "/" << m << "/" << n;
  return str.str();
}

template <typename scalar_t>
void run(benchmark::State& state, ExecutorType* executorPtr, int ti, index_t m,
         index_t n, scalar_t alpha, scalar_t beta) {
  // Standard test setup.
  std::string ts = blas_benchmark::utils::from_transpose_enum(
      static_cast<blas_benchmark::utils::Transposition>(ti));
  const char* t_str = ts.c_str();

  index_t vlen = t_str[0] == 'n' ? n : m;
  index_t rlen = t_str[0] == 'n' ? m : n;

  index_t lda = m;
  index_t incX = 1;
  index_t incY = 1;

  // The counters are double. We convert m and n to double to avoid
  // integer overflows for n_fl_ops and bytes_processed
  double m_d = static_cast<double>(m);
  double n_d = static_cast<double>(n);

  state.counters["m"] = m_d;
  state.counters["n"] = n_d;

  {
    double nflops_AtimesX = 2.0 * m_d * n_d;
    double nflops_timesAlpha = m_d;
    double nflops_addBetaY = (beta != 0) ? 2 * m_d : 0;
    state.counters["n_fl_ops"] =
        nflops_AtimesX + nflops_timesAlpha + nflops_addBetaY;
  }
  {
    double mem_readA = m_d * n_d;
    double mem_readX = n_d;
    double mem_writeY = m_d;
    double mem_readY = (beta != 0) ? m_d : 0;
    state.counters["bytes_processed"] =
        (mem_readA + mem_readX + mem_writeY + mem_readY) * sizeof(scalar_t);
  }

  // Input matrix/vector, output vector.
  std::vector<scalar_t> m_a =
      blas_benchmark::utils::random_data<scalar_t>(m * n);
  std::vector<scalar_t> v_b =
      blas_benchmark::utils::random_data<scalar_t>(vlen);
  std::vector<scalar_t> v_c =
      blas_benchmark::utils::const_data<scalar_t>(rlen, 0);

  // Specify the transposition
  clblast::Transpose a_tr =
      blas_benchmark::utils::translate_transposition(t_str);

  // Specify the layout. This needs to be kColMajor, and results in errors
  // otherwise. It may be that this is incorrect (especially for performance
  // reasons), so may need to be revisited.
  auto layout = clblast::Layout::kColMajor;

  // Device matrices
  MemBuffer<scalar_t> m_a_gpu(executorPtr, m_a.data(),
                              static_cast<size_t>(m * n));
  MemBuffer<scalar_t> v_b_gpu(executorPtr, v_b.data(),
                              static_cast<size_t>(vlen));
  MemBuffer<scalar_t> v_c_gpu(executorPtr, v_c.data(),
                              static_cast<size_t>(rlen));

  // Create a utility lambda describing the blas method that we want to run.
  auto blas_method_def = [&]() -> std::vector<cl_event> {
    cl_event event;
    clblast::Gemv<scalar_t>(layout, a_tr, m, n, alpha, m_a_gpu.dev(), 0, lda,
                            v_b_gpu.dev(), 0, incX, beta, v_c_gpu.dev(), 0,
                            incY, executorPtr->_queue(), &event);
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
  auto gemm_params = blas_benchmark::utils::get_blas2_params<scalar_t>(args);

  for (auto p : gemm_params) {
    std::string ts;
    index_t m, n;
    scalar_t alpha, beta;
    std::tie(ts, m, n, alpha, beta) = p;
    int t = static_cast<int>(blas_benchmark::utils::to_transpose_enum(ts));

    auto BM_lambda = [&](benchmark::State& st, ExecutorType* exPtr, int t,
                         index_t m, index_t n, scalar_t alpha, scalar_t beta) {
      run<scalar_t>(st, exPtr, t, m, n, alpha, beta);
    };
    benchmark::RegisterBenchmark(get_name<scalar_t>(ts, m, n).c_str(),
                                 BM_lambda, exPtr, t, m, n, alpha, beta);
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
