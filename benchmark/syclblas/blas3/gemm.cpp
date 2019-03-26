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

#include "range.hpp"
#include "utils.hpp"

template <typename scalar_t>
void BM_Gemm(benchmark::State& state) {
  // Standard test setup.
  char const* t_a = benchmark::utils::from_transpose_enum(
      static_cast<benchmark::utils::Transposition>(state.range(0)));
  char const* t_b = benchmark::utils::from_transpose_enum(
      static_cast<benchmark::utils::Transposition>(state.range(1)));
  const index_t m = static_cast<index_t>(state.range(2));
  const index_t k = static_cast<index_t>(state.range(3));
  const index_t n = static_cast<index_t>(state.range(4));

  index_t lda = t_a[0] == 'n' ? m : k;
  index_t ldb = t_b[0] == 'n' ? k : n;
  index_t ldc = m;

  state.counters["m"] = m;
  state.counters["k"] = k;
  state.counters["n"] = n;

  // The counters are double. We convert m, n and k to double to avoid
  // integer overflows for n_fl_ops and bytes_processed
  double m_d = static_cast<double>(m);
  double n_d = static_cast<double>(n);
  double k_d = static_cast<double>(k);
  state.counters["n_fl_ops"] = 3 * (m_d * n_d * k_d) + 2 * (m_d * n_d);
  state.counters["bytes_processed"] =
      (m_d * k_d + k_d * n_d + 2 * m_d * n_d) * sizeof(scalar_t);

  SyclExecutorType ex = *Global::executorInstancePtr;

  // Create data
  // Scalars
  scalar_t alpha = benchmark::utils::random_scalar<scalar_t>();
  scalar_t beta = benchmark::utils::random_scalar<scalar_t>();

  // Matrices
  std::vector<scalar_t> a = benchmark::utils::random_data<scalar_t>(m * k);
  std::vector<scalar_t> b = benchmark::utils::random_data<scalar_t>(k * n);
  std::vector<scalar_t> c = benchmark::utils::const_data<scalar_t>(m * n, 0);

  auto a_gpu = blas::make_sycl_iterator_buffer<scalar_t>(a, m * k);
  auto b_gpu = blas::make_sycl_iterator_buffer<scalar_t>(b, k * n);
  auto c_gpu = blas::make_sycl_iterator_buffer<scalar_t>(c, m * n);

  // Warmup
  for (int i = 0; i < 10; i++) {
    _gemm(ex, *t_a, *t_b, m, n, k, alpha, a_gpu, lda, b_gpu, ldb, beta, c_gpu,
          ldc);
  }
  ex.get_policy_handler().wait();

  state.counters["best_event_time"] = ULONG_MAX;
  state.counters["best_overall_time"] = ULONG_MAX;

  // Measure
  for (auto _ : state) {
    // Run
    std::tuple<double, double> times =
        benchmark::utils::timef([&]() -> std::vector<cl::sycl::event> {
          auto event = _gemm(ex, *t_a, *t_b, m, n, k, alpha, a_gpu, lda, b_gpu,
                             ldb, beta, c_gpu, ldc);
          ex.get_policy_handler().wait(event);
          return event;
        });

    // Report
    state.PauseTiming();

    double overall_time, event_time;
    std::tie(overall_time, event_time) = times;

    state.counters["total_event_time"] += event_time;
    state.counters["best_event_time"] =
        std::min<double>(state.counters["best_event_time"], event_time);

    state.counters["total_overall_time"] += overall_time;
    state.counters["best_overall_time"] =
        std::min<double>(state.counters["best_overall_time"], overall_time);

    state.ResumeTiming();
  }

  state.counters["avg_event_time"] =
      state.counters["total_event_time"] / state.iterations();
  state.counters["avg_overall_time"] =
      state.counters["total_overall_time"] / state.iterations();
};

static void gemm_args(benchmark::internal::Benchmark* b) {
  // Matrix dimensions bounds
  constexpr const int dim_min = 2 << 5;
  constexpr const int dim_max = 2 << 10;
  // Matrix dimensions multiplier
  constexpr const int dim_mult = 2;

  auto gemm_range = nd_range(value_range({"n", "t"}), value_range({"n", "t"}),
                             size_range(dim_min, dim_max, dim_mult),
                             size_range(dim_min, dim_max, dim_mult),
                             size_range(dim_min, dim_max, dim_mult));

  do {
    auto p = gemm_range.yield();
    int t1 = (int)benchmark::utils::to_transpose_enum(std::get<0>(p));
    int t2 = (int)benchmark::utils::to_transpose_enum(std::get<1>(p));
    int m = std::get<2>(p);
    int k = std::get<3>(p);
    int n = std::get<4>(p);
    b->Args({t1, t2, m, k, n});

  } while (!gemm_range.finished());
}

BENCHMARK_TEMPLATE(BM_Gemm, float)
    ->Apply(gemm_args)
    ->Unit(benchmark::kNanosecond);
#ifdef DOUBLE_SUPPORT
BENCHMARK_TEMPLATE(BM_Gemm, double)
    ->Apply(gemm_args)
    ->Unit(benchmark::kNanosecond);
#endif
