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

  state.counters["n_fl_ops"] = 2.0 * m_d * n_d + 3 * m_d;
  state.counters["bytes_processed"] =
      (m_d * n_d + n_d + 2 * m_d) * sizeof(scalar_t);

  if (beta == 0.0) {
    // not adding beta * Y
    state.counters["n_fl_ops"] -= 2 * m_d;
    // not reading Y
    state.counters["bytes_processed"] -= m_d * sizeof(scalar_t);
  }

  ExecutorType& ex = *executorPtr;

  // Input matrix/vector, output vector.
  std::vector<scalar_t> m_a =
      blas_benchmark::utils::random_data<scalar_t>(m * n);
  std::vector<scalar_t> v_b =
      blas_benchmark::utils::random_data<scalar_t>(vlen);
  std::vector<scalar_t> v_c =
      blas_benchmark::utils::const_data<scalar_t>(rlen, 0);

  auto m_a_gpu = blas::make_sycl_iterator_buffer<scalar_t>(m_a, m * n);
  auto v_b_gpu = blas::make_sycl_iterator_buffer<scalar_t>(v_b, vlen);
  auto v_c_gpu = blas::make_sycl_iterator_buffer<scalar_t>(v_c, rlen);

#ifdef BLAS_VERIFY_BENCHMARK
  // Run a first time with a verification of the results
  std::vector<scalar_t> v_c_ref = v_c;
  reference_blas::gemv(t_str, m, n, alpha, m_a.data(), m, v_b.data(), incX,
                       beta, v_c_ref.data(), incY);
  std::vector<scalar_t> v_c_temp = v_c;
  {
    auto v_c_temp_gpu = blas::make_sycl_iterator_buffer<scalar_t>(v_c_temp, m);
    auto event = _gemv(ex, *t_str, m, n, alpha, m_a_gpu, m, v_b_gpu, incX, beta,
                       v_c_temp_gpu, incY);
    ex.get_policy_handler().wait(event);
  }

  if (!utils::compare_vectors<scalar_t>(v_c_temp, v_c_ref)) {
    exit(1);
  };
#endif

  auto blas_method_def = [&]() -> std::vector<cl::sycl::event> {
    auto event = _gemv(ex, *t_str, m, n, alpha, m_a_gpu, m, v_b_gpu, incX, beta,
                       v_c_gpu, incY);
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
