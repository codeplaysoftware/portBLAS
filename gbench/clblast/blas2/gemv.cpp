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

#include "range.hpp"
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

  // The counters are double. We convert m and n to double to avoid
  // integer overflows for n_fl_ops and bytes_processed
  double m_d = static_cast<double>(m);
  double n_d = static_cast<double>(n);
  state.counters["n_fl_ops"] = 2.0 * m_d * n_d;
  state.counters["bytes_processed"] =
      (m_d * n_d + m_d + n_d) * sizeof(scalar_t);

  // Create data
  // Scalars
  scalar_t alpha = benchmark::utils::random_scalar<scalar_t>();
  scalar_t beta = benchmark::utils::random_scalar<scalar_t>();

  // Input matrix/vector, output vector.
  std::vector<scalar_t> m_a = benchmark::utils::random_data<scalar_t>(m * n);
  std::vector<scalar_t> v_b = benchmark::utils::random_data<scalar_t>(vlen);
  std::vector<scalar_t> v_c = benchmark::utils::const_data<scalar_t>(rlen, 0);

  // Specify the transposition
  clblast::Transpose a_tr = benchmark::utils::translate_transposition(t_str);

  // Specify the layout. This needs to be kColMajor, and results in errors
  // otherwise. It may be that this is incorrect (especially for performance
  // reasons), so may need to be revisited.
  auto layout = clblast::Layout::kColMajor;

  ExecutorType* ex = Global::executorInstancePtr.get();

  // Device matrices
  MemBuffer<scalar_t> m_a_gpu(ex, m_a.data(), static_cast<size_t>(m * n));
  MemBuffer<scalar_t> v_b_gpu(ex, v_b.data(), static_cast<size_t>(vlen));
  MemBuffer<scalar_t> v_c_gpu(ex, v_c.data(), static_cast<size_t>(rlen));

  // Create a utility lambda describing the blas method that we want to run.
  auto blas_method_def = [&]() -> std::vector<cl_event> {
    cl_event event;
    clblast::Gemv<scalar_t>(layout, a_tr, m, n, alpha, m_a_gpu.dev(), 0, lda,
                            v_b_gpu.dev(), 0, incX, beta, v_c_gpu.dev(), 0,
                            incY, ex->_queue(), &event);
    CLEventHandler::wait(event);
    return {event};
  };

  // Warm up to avoid benchmarking data transfer
  benchmark::utils::warmup(blas_method_def);

  state.counters["best_event_time"] = ULONG_MAX;
  state.counters["best_overall_time"] = ULONG_MAX;

  // Measure
  for (auto _ : state) {
    std::tuple<double, double> times = benchmark::utils::timef(blas_method_def);

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

static void gemv_args(benchmark::internal::Benchmark* b) {
  // Matrix dimensions bounds
  constexpr const int dim_min = 2 << 5;
  constexpr const int dim_max = 2 << 14;
  // Matrix dimensions multiplier
  constexpr const int dim_mult = 2;

  auto gemv_range =
      nd_range(value_range({"n", "t"}), size_range(dim_min, dim_max, dim_mult),
               size_range(dim_min, dim_max, dim_mult));

  do {
    auto p = gemv_range.yield();
    int t = (int)benchmark::utils::to_transpose_enum(std::get<0>(p));
    int m = std::get<1>(p);
    int n = std::get<2>(p);
    b->Args({t, m, n});

  } while (!gemv_range.finished());
}

BENCHMARK_TEMPLATE(BM_Gemv, float)
    ->Apply(gemv_args)
    ->Unit(benchmark::kNanosecond);
#ifdef DOUBLE_SUPPORT
BENCHMARK_TEMPLATE(BM_Gemv, double)
    ->Apply(gemv_args)
    ->Unit(benchmark::kNanosecond);
#endif
