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
 *  @filename axpy3op.cpp
 *
 **************************************************************************/

#include "utils.hpp"

template <typename scalar_t>
void BM_Axpy3op(benchmark::State& state) {
  // Standard test setup.
  const index_t size = static_cast<index_t>(state.range(0));
  double size_d = static_cast<double>(size);
  state.counters["size"] = size_d;
  state.counters["n_fl_ops"] = 6 * size_d;
  state.counters["bytes_processed"] = 9 * size_d * sizeof(scalar_t);

  SyclExecutorType ex = *Global::executorInstancePtr;

  // Create data
  std::array<scalar_t, 3> alphas = {1.78426458744, 2.187346575843,
                                    3.78164387328};
  std::vector<scalar_t> vsrc1 = benchmark::utils::random_data<scalar_t>(size);
  std::vector<scalar_t> vsrc2 = benchmark::utils::random_data<scalar_t>(size);
  std::vector<scalar_t> vsrc3 = benchmark::utils::random_data<scalar_t>(size);
  std::vector<scalar_t> vdst1 = benchmark::utils::random_data<scalar_t>(size);
  std::vector<scalar_t> vdst2 = benchmark::utils::random_data<scalar_t>(size);
  std::vector<scalar_t> vdst3 = benchmark::utils::random_data<scalar_t>(size);
  auto insrc1 = blas::make_sycl_iterator_buffer<scalar_t>(vsrc1, size);
  auto insrc2 = blas::make_sycl_iterator_buffer<scalar_t>(vsrc2, size);
  auto insrc3 = blas::make_sycl_iterator_buffer<scalar_t>(vsrc3, size);
  auto indst1 = blas::make_sycl_iterator_buffer<scalar_t>(vdst1, size);
  auto indst2 = blas::make_sycl_iterator_buffer<scalar_t>(vdst2, size);
  auto indst3 = blas::make_sycl_iterator_buffer<scalar_t>(vdst3, size);

  // Warmup
  for (int i = 0; i < 10; i++) {
    auto event0 = _axpy(ex, size, alphas[0], insrc1, 1, indst1, 1);
    auto event1 = _axpy(ex, size, alphas[1], insrc2, 1, indst2, 1);
    auto event2 = _axpy(ex, size, alphas[2], insrc3, 1, indst3, 1);
  }
  ex.get_policy_handler().wait();

  state.counters["best_event_time"] = ULONG_MAX;
  state.counters["best_overall_time"] = ULONG_MAX;

  // Measure
  for (auto _ : state) {
    // Run
    std::tuple<double, double> times = benchmark::utils::timef(
      [&]() -> std::vector<cl::sycl::event> {
        auto event0 = _axpy(ex, size, alphas[0], insrc1, 1, indst1, 1);
        auto event1 = _axpy(ex, size, alphas[1], insrc2, 1, indst2, 1);
        auto event2 = _axpy(ex, size, alphas[2], insrc3, 1, indst3, 1);
        ex.get_policy_handler().wait(event0, event1, event2);
        return blas::concatenate_vectors(event0, event1, event2);
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

  state.counters["avg_event_time"] = state.counters["total_event_time"]
                                   / state.iterations();
  state.counters["avg_overall_time"] = state.counters["total_overall_time"]
                                       / state.iterations();
}

BENCHMARK_TEMPLATE(BM_Axpy3op, float)
    ->RangeMultiplier(2)
    ->Range(2 << 5, 2 << 18);
#ifdef DOUBLE_SUPPORT
BENCHMARK_TEMPLATE(BM_Axpy3op, double)
    ->RangeMultiplier(2)
    ->Range(2 << 5, 2 << 18);
#endif
