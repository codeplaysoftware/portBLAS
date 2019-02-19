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
 *  @filename scal2op.cpp
 *
 **************************************************************************/

#include "utils.hpp"

template <typename ScalarT>
void BM_Scal2op(benchmark::State& state) {
  // Standard test setup.
  const IndexType size = static_cast<IndexType>(state.range(0));
  state.counters["size"] = size;

  SYCL_EXECUTOR_TYPE ex = *getExecutor();

  // Create data
  std::vector<ScalarT> v1 = benchmark::utils::random_data<ScalarT>(size);
  std::vector<ScalarT> v2 = benchmark::utils::random_data<ScalarT>(size);
  ScalarT alpha = benchmark::utils::random_scalar<ScalarT>();

  auto inx = blas::make_sycl_iterator_buffer<ScalarT>(v1, size);
  auto iny = blas::make_sycl_iterator_buffer<ScalarT>(v2, size);

  // Warmup
  for (int i = 0; i < 10; i++) {
    _scal(ex, size, alpha, inx, 1);
  }

  // Measure
  for (auto _ : state) {
    // Run
    auto event0 = _scal(ex, size, alpha, inx, 1);
    auto event1 = _scal(ex, size, alpha, iny, 1);
    ex.get_policy_handler().wait(event0, event1);

    // Report
    state.PauseTiming();
    state.counters["event_time"] =
        benchmark::utils::time_events(event0, event1);
    state.ResumeTiming();
  }
}

BENCHMARK_TEMPLATE(BM_Scal2op, float)
    ->RangeMultiplier(2)
    ->Range(2 << 5, 2 << 18);
#ifdef DOUBLE_SUPPORT
BENCHMARK_TEMPLATE(BM_Scal2op, double)
    ->RangeMultiplier(2)
    ->Range(2 << 5, 2 << 18);
#endif
