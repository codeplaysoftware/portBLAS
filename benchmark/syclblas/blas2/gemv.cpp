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
 *  @filename gemv.cpp
 *
 **************************************************************************/

#include "../utils.hpp"

template <typename scalar_t>
std::string get_name(std::string t, int m, int n) {
  std::ostringstream str{};
  str << "BM_Gemv<" << blas_benchmark::utils::get_type_name<scalar_t>() << ">/"
      << t << "/" << m << "/" << n;
  return str.str();
}

template <typename scalar_t>
void run(benchmark::State& state, ExecutorType* executorPtr, int ti, index_t m,
         index_t n, scalar_t alpha, scalar_t beta, bool* success) {
  // Standard test setup.
  std::string ts = blas_benchmark::utils::from_transpose_enum(
      static_cast<blas_benchmark::utils::Transposition>(ti));
  const char* t_str = ts.c_str();

  index_t xlen = t_str[0] == 'n' ? n : m;
  index_t ylen = t_str[0] == 'n' ? m : n;

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
    double nflops_timesAlpha = ylen;
    double nflops_addBetaY = (beta != scalar_t{0}) ? 2 * ylen : 0;
    state.counters["n_fl_ops"] =
        nflops_AtimesX + nflops_timesAlpha + nflops_addBetaY;
  }
  {
    double mem_readA = m_d * n_d;
    double mem_readX = xlen;
    double mem_writeY = ylen;
    double mem_readY = (beta != scalar_t{0}) ? ylen : 0;
    state.counters["bytes_processed"] =
        (mem_readA + mem_readX + mem_writeY + mem_readY) * sizeof(scalar_t);
  }

  ExecutorType& ex = *executorPtr;

  using data_t = utils::data_storage_t<scalar_t>;

  // Input matrix/vector, output vector.
  std::vector<data_t> m_a = blas_benchmark::utils::random_data<data_t>(m * n);
  std::vector<data_t> v_x = blas_benchmark::utils::random_data<data_t>(xlen);
  std::vector<data_t> v_y = blas_benchmark::utils::random_data<data_t>(ylen);

  auto m_a_gpu = utils::make_quantized_buffer<scalar_t>(ex, m_a);
  auto v_x_gpu = utils::make_quantized_buffer<scalar_t>(ex, v_x);
  auto v_y_gpu = utils::make_quantized_buffer<scalar_t>(ex, v_y);

#ifdef BLAS_VERIFY_BENCHMARK
  // Run a first time with a verification of the results
  std::vector<data_t> v_y_ref = v_y;
  reference_blas::gemv(t_str, m, n, static_cast<data_t>(alpha), m_a.data(), m,
                       v_x.data(), incX, static_cast<data_t>(beta),
                       v_y_ref.data(), incY);
  std::vector<data_t> v_y_temp = v_y;
  {
    auto v_y_temp_gpu = utils::make_quantized_buffer<scalar_t>(ex, v_y_temp);
    _gemv(ex, *t_str, m, n, alpha, m_a_gpu, m, v_x_gpu, incX, beta,
          v_y_temp_gpu, incY);
    auto event =
        utils::quantized_copy_to_host<scalar_t>(ex, v_y_temp_gpu, v_y_temp);
    ex.get_policy_handler().wait();
  }

  std::ostringstream err_stream;
  if (!utils::compare_vectors<data_t, scalar_t>(v_y_temp, v_y_ref, err_stream,
                                                "")) {
    const std::string& err_str = err_stream.str();
    state.SkipWithError(err_str.c_str());
    *success = false;
  };
#endif

  auto blas_method_def = [&]() -> std::vector<cl::sycl::event> {
    auto event = _gemv(ex, *t_str, m, n, alpha, m_a_gpu, m, v_x_gpu, incX, beta,
                       v_y_gpu, incY);
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
}

template <typename scalar_t>
void register_benchmark(blas_benchmark::Args& args, ExecutorType* exPtr,
                        bool* success) {
  auto gemm_params = blas_benchmark::utils::get_blas2_params<scalar_t>(args);

  for (auto p : gemm_params) {
    std::string ts;
    index_t m, n;
    scalar_t alpha, beta;
    std::tie(ts, m, n, alpha, beta) = p;
    int t = static_cast<int>(blas_benchmark::utils::to_transpose_enum(ts));

    auto BM_lambda = [&](benchmark::State& st, ExecutorType* exPtr, int t,
                         index_t m, index_t n, scalar_t alpha, scalar_t beta,
                         bool* success) {
      run<scalar_t>(st, exPtr, t, m, n, alpha, beta, success);
    };
    benchmark::RegisterBenchmark(get_name<scalar_t>(ts, m, n).c_str(),
                                 BM_lambda, exPtr, t, m, n, alpha, beta,
                                 success);
  }
}

namespace blas_benchmark {
void create_benchmark(blas_benchmark::Args& args, ExecutorType* exPtr,
                      bool* success) {
  BLAS_REGISTER_BENCHMARK(args, exPtr, success);
}
}  // namespace blas_benchmark
