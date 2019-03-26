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
 *  @filename iamax.cpp
 *
 **************************************************************************/

#include "utils.hpp"

template <typename scalar_t>
void BM_Iamax(benchmark::State& state) {
  // Standard test setup.
  const index_t size = static_cast<index_t>(state.range(0));
  state.counters["size"] = size;

  SyclExecutorType ex = *Global::executorInstancePtr;

  // Create data
  std::vector<scalar_t> v1 = benchmark::utils::random_data<scalar_t>(size);
  blas::IndexValueTuple<scalar_t, index_t> out(-1, -1);

  auto inx = blas::make_sycl_iterator_buffer<scalar_t>(v1, size);
  auto outI =
      blas::make_sycl_iterator_buffer<blas::IndexValueTuple<scalar_t, index_t>>(
          &out, 1);

  // Warmup
  for (int i = 0; i < 10; i++) {
    _iamax(ex, size, inx, 1, outI);
  }

  // Measure
  for (auto _ : state) {
    // Run
    auto event = _iamax(ex, size, inx, 1, outI);
    ex.get_policy_handler().wait(event);

    // Report
    state.PauseTiming();
    state.counters["event_time"] = benchmark::utils::time_events(event);
    state.ResumeTiming();
  }
}

BENCHMARK_TEMPLATE(BM_Iamax, float)->RangeMultiplier(2)->Range(2 << 5, 2 << 18);
#ifdef DOUBLE_SUPPORT
BENCHMARK_TEMPLATE(BM_Iamax, double)
    ->RangeMultiplier(2)
    ->Range(2 << 5, 2 << 24);
#endif
