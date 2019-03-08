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
void BM_Gemv(benchmark::State& state) {
  // Standard test setup.
  const char* t_str = benchmark::utils::from_transpose_enum(
      static_cast<benchmark::utils::Transposition>(state.range(0)));
  const index_t m = static_cast<index_t>(state.range(1));
  const index_t n = static_cast<index_t>(state.range(2));

  index_t vlen = t_str[0] == 'n' ? n : m;
  index_t rlen = t_str[0] == 'n' ? m : n;

  index_t lda = m;
  index_t incX = 1;
  index_t incY = 1;

  state.counters["m"] = m;
  state.counters["n"] = n;

  SyclExecutorType ex = *getExecutor();

  // Create data
  // Scalars
  scalar_t alpha = benchmark::utils::random_scalar<scalar_t>();
  scalar_t beta = benchmark::utils::random_scalar<scalar_t>();

  // Input matrix/vector, output vector.
  std::vector<scalar_t> a_m = benchmark::utils::random_data<scalar_t>(m * n);
  std::vector<scalar_t> b_v = benchmark::utils::random_data<scalar_t>(vlen);
  std::vector<scalar_t> c_v_gpu_result =
      benchmark::utils::const_data<scalar_t>(rlen, 0);

  auto m_a_gpu = blas::make_sycl_iterator_buffer<scalar_t>(a_m, m * n);
  auto v_b_gpu = blas::make_sycl_iterator_buffer<scalar_t>(b_v, vlen);
  auto v_c_gpu = blas::make_sycl_iterator_buffer<scalar_t>(c_v_gpu_result, rlen);

  // Warmup
  for (int i = 0; i < 10; i++) {
    _gemv(ex, *t_str, m, n, alpha, m_a_gpu, m, v_b_gpu, incX, beta, v_c_gpu,
          incY);
  }

  //   std::cout << "Ran successfully! " << std::endl;

  // Measure
  for (auto _ : state) {
    // Run
    auto event = _gemv(ex, *t_str, m, n, alpha, m_a_gpu, m, v_b_gpu, incX, beta,
                       v_c_gpu, incY);
    ex.get_policy_handler().wait(event);

    // Report
    state.PauseTiming();
    state.counters["event_time"] = benchmark::utils::time_events(event);
    state.ResumeTiming();
  }
}

static void gemv_args(benchmark::internal::Benchmark* b) {
  for (auto t : benchmark::utils::possible_transpositions)
    for (int i = 2 << 5; i <= 2 << 18; i *= 2)
      for (int j = 2 << 5; j <= 2 << 18; j *= 2) {
        b->Args({(int)(t), i, j});
      }
}

BENCHMARK_TEMPLATE(BM_Gemv, float)->Apply(gemv_args);
#ifdef DOUBLE_SUPPORT
BENCHMARK_TEMPLATE(BM_Gemv, double)->Apply(gemv_args);
#endif
