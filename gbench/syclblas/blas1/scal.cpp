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
 *  @filename scal.cpp
 *
 **************************************************************************/

#include "utils.hpp"

template <typename ScalarT>
void BM_Scal(benchmark::State& state) {
  // Standard test setup.
  const IndexType size = static_cast<IndexType>(state.range(0));
  state.counters["size"] = size;

  SyclExecutorType ex = *getExecutor();

  // Create data
  std::vector<ScalarT> v1 = benchmark::utils::random_data<ScalarT>(size);
  ScalarT alpha = benchmark::utils::random_scalar<ScalarT>();

  auto in = blas::make_sycl_iterator_buffer<ScalarT>(v1, size);

  // Warmup
  for (int i = 0; i < 10; i++) {
    _scal(ex, size, alpha, in, 1);
  }

  // Measure
  for (auto _ : state) {
    // Run
    auto event = _scal(ex, size, alpha, in, 1);
    ex.get_policy_handler().wait(event);

    // Report
    state.PauseTiming();
    state.counters["event_time"] = benchmark::utils::time_events(event);
    state.ResumeTiming();
  };
}

BENCHMARK_TEMPLATE(BM_Scal, float)->RangeMultiplier(2)->Range(2 << 5, 2 << 18);
#ifdef DOUBLE_SUPPORT
BENCHMARK_TEMPLATE(BM_Scal, double)->RangeMultiplier(2)->Range(2 << 5, 2 << 18);
#endif
