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

template <typename ScalarT>
void BM_Gemm(benchmark::State& state) {
  // Standard test setup.
  char const* t_a = benchmark::utils::from_transpose_enum(
      static_cast<benchmark::utils::Transposition>(state.range(0)));
  char const* t_b = benchmark::utils::from_transpose_enum(
      static_cast<benchmark::utils::Transposition>(state.range(1)));
  const IndexType m = static_cast<IndexType>(state.range(2));
  const IndexType k = static_cast<IndexType>(state.range(3));
  const IndexType n = static_cast<IndexType>(state.range(4));

  IndexType lda = t_a[0] == 'n' ? m : k;
  IndexType ldb = t_b[0] == 'n' ? k : n;
  IndexType ldc = m;

  state.counters["m"] = m;
  state.counters["k"] = k;
  state.counters["n"] = n;

  SYCL_EXECUTOR_TYPE ex = *getExecutor();

  // Create data
  // Scalars
  ScalarT alpha = benchmark::utils::random_scalar<ScalarT>();
  ScalarT beta = benchmark::utils::random_scalar<ScalarT>();

  // Matrices
  std::vector<ScalarT> a = benchmark::utils::random_data<ScalarT>(m * k);
  std::vector<ScalarT> b = benchmark::utils::random_data<ScalarT>(k * n);
  std::vector<ScalarT> c = benchmark::utils::const_data<ScalarT>(m * n, 0);

  auto a_gpu = blas::make_sycl_iterator_buffer<ScalarT>(a, m * k);
  auto b_gpu = blas::make_sycl_iterator_buffer<ScalarT>(b, k * n);
  auto c_gpu = blas::make_sycl_iterator_buffer<ScalarT>(c, m * n);

  // Warmup
  for (int i = 0; i < 10; i++) {
    _gemm(ex, *t_a, *t_b, m, n, k, alpha, a_gpu, lda, b_gpu, ldb, beta, c_gpu,
          ldc);
  }

  // Measure
  for (auto _ : state) {
    // Run
    auto event = _gemm(ex, *t_a, *t_b, m, n, k, alpha, a_gpu, lda, b_gpu, ldb,
                       beta, c_gpu, ldc);
    ex.get_policy_handler().wait(event);

    // Report
    state.PauseTiming();
    state.counters["event_time"] = benchmark::utils::time_events(event);
    state.ResumeTiming();
  }
};

static void gemm_args(benchmark::internal::Benchmark* b) {
  for (auto t1 : benchmark::utils::possible_transpositions)
    for (auto t2 : benchmark::utils::possible_transpositions)
      for (int m = 2 << 5; m <= 2 << 12; m *= 2)
        for (int k = 2 << 5; k <= 2 << 12; k *= 2)
          for (int n = 2 << 5; n <= 2 << 12; n *= 2) {
            b->Args({(int)(t1), (int)(t2), m, k, n});
          }
}

BENCHMARK_TEMPLATE(BM_Gemm, float)->Apply(gemm_args);
#ifdef DOUBLE_SUPPORT
BENCHMARK_TEMPLATE(BM_Gemm, double)->Apply(gemm_args);
#endif
