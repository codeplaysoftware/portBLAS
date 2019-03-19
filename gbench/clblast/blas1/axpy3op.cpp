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
 *  @filename asum.cpp
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

  // Create data
  scalar_t alphas[] = {1.78426458744, 2.187346575843, 3.78164387328};
  size_t offsets[] = {0, size, size * 2};
  std::vector<scalar_t> src = benchmark::utils::random_data<scalar_t>(size * 3);
  std::vector<scalar_t> dst = benchmark::utils::random_data<scalar_t>(size * 3);
  scalar_t alpha = benchmark::utils::random_scalar<scalar_t>();

  ExecutorType* ex = getExecutor().get();

  // Device vectors
  MemBuffer<scalar_t, CL_MEM_READ_ONLY> bufsrc(ex, src.data(),
                                               size * 3);
  MemBuffer<scalar_t> bufdst(ex, dst.data(), size * 3);

  // Create a utility lambda describing the blas method that we want to run.
  auto blas_method_def = [&]() -> std::vector<Event> {
    Event event;
    clblast::AxpyBatched<scalar_t>(size, alphas, bufsrc.dev(), offsets, 1,
                                   bufdst.dev(), offsets, 1, 3,
                                   ex->_queue(), &event._cl());
    event.wait();
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
  state.counters["avg_overall_time"] = state.counters["total_overall_time"]
       / state.iterations();
};

BENCHMARK_TEMPLATE(BM_Axpy3op, float)->RangeMultiplier(2)->Range(2 << 5, 2 << 18);
#ifdef DOUBLE_SUPPORT
BENCHMARK_TEMPLATE(BM_Axpy3op, double)->RangeMultiplier(2)->Range(2 << 5, 2 << 18);
#endif
