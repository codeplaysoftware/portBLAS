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
  return "BM_Gemv<" + blas_benchmark::utils::get_type_name<scalar_t>() + ">/" +
         t + "/" + std::to_string(m) + "/" + std::to_string(n);
}

template <typename scalar_t>
void run(benchmark::State& state, ExecutorType* executorPtr, int ti, int mi,
         int ni) {
  // Standard test setup.
  std::string ts = blas_benchmark::utils::from_transpose_enum(
      static_cast<blas_benchmark::utils::Transposition>(ti));
  const char* t_str = ts.c_str();
  const index_t m = static_cast<index_t>(mi);
  const index_t n = static_cast<index_t>(ni);

  index_t vlen = t_str[0] == 'n' ? n : m;
  index_t rlen = t_str[0] == 'n' ? m : n;

  index_t lda = m;
  index_t incX = 1;
  index_t incY = 1;

  state.counters["m"] = m;
  state.counters["n"] = n;

  // The counters are double. We convert m and n to double to avoid
  // integer overflows for n_fl_ops and bytes_processed
  double m_d = static_cast<double>(m);
  double n_d = static_cast<double>(n);
  state.counters["n_fl_ops"] = 2.0 * m_d * n_d;
  state.counters["bytes_processed"] =
      (m_d * n_d + m_d + n_d) * sizeof(scalar_t);

  ExecutorType& ex = *executorPtr;

  // Create data
  // Scalars
  scalar_t alpha = blas_benchmark::utils::random_scalar<scalar_t>();
  scalar_t beta = blas_benchmark::utils::random_scalar<scalar_t>();

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

  // Warmup
  for (int i = 0; i < 10; i++) {
    _gemv(ex, *t_str, m, n, alpha, m_a_gpu, m, v_b_gpu, incX, beta, v_c_gpu,
          incY);
  }
  ex.get_policy_handler().wait();

  blas_benchmark::utils::init_counters(state);

  // Measure
  for (auto _ : state) {
    // Run
    std::tuple<double, double> times =
        blas_benchmark::utils::timef([&]() -> std::vector<cl::sycl::event> {
          auto event = _gemv(ex, *t_str, m, n, alpha, m_a_gpu, m, v_b_gpu, incX,
                             beta, v_c_gpu, incY);
          ex.get_policy_handler().wait(event);
          return event;
        });

    // Report
    blas_benchmark::utils::update_counters(state, times);
  }

  blas_benchmark::utils::calc_avg_counters(state);
}

template <typename scalar_t>
void register_benchmark(blas_benchmark::Args& args, ExecutorType* exPtr) {
  auto gemm_params = blas_benchmark::utils::get_params<blas2_param_t>(args);

  for (auto p : gemm_params) {
    std::string ts;
    int m, n;
    std::tie(ts, m, n) = p;
    int t = (int)blas_benchmark::utils::to_transpose_enum(ts);

    auto BM_lambda = [&](benchmark::State& st, ExecutorType* exPtr, int t,
                         int m, int n) { run<scalar_t>(st, exPtr, t, m, n); };
    benchmark::RegisterBenchmark(get_name<scalar_t>(ts, m, n).c_str(),
                                 BM_lambda, exPtr, t, m, n);
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
