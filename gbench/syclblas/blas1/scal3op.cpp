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
 *  @filename scal3op.cpp
 *
 **************************************************************************/

#include "utils.hpp"

template <typename scalar_t>
void BM_Scal3op(benchmark::State& state) {
  // Standard test setup.
  const index_t size = static_cast<index_t>(state.range(0));
  state.counters["size"] = size;

  SyclExecutorType ex = *getExecutor();

  // Create data
  std::vector<scalar_t> v1 = benchmark::utils::random_data<scalar_t>(size);
  std::vector<scalar_t> v2 = benchmark::utils::random_data<scalar_t>(size);
  std::vector<scalar_t> v3 = benchmark::utils::random_data<scalar_t>(size);
  scalar_t alpha = benchmark::utils::random_scalar<scalar_t>();

  auto inx = blas::make_sycl_iterator_buffer<scalar_t>(v1, size);
  auto iny = blas::make_sycl_iterator_buffer<scalar_t>(v2, size);
  auto inz = blas::make_sycl_iterator_buffer<scalar_t>(v3, size);

  // Warmup
  for (int i = 0; i < 10; i++) {
    _scal(ex, size, alpha, inx, 1);
  }

  // Measure
  for (auto _ : state) {
    // Run
    auto event0 = _scal(ex, size, alpha, inx, 1);
    auto event1 = _scal(ex, size, alpha, iny, 1);
    auto event2 = _scal(ex, size, alpha, inz, 1);
    ex.get_policy_handler().wait(event0, event1, event2);

    // Report
    state.PauseTiming();
    state.counters["event_time"] =
        benchmark::utils::time_events(event0, event1, event2);
    state.ResumeTiming();
  }
}

BENCHMARK_TEMPLATE(BM_Scal3op, float)
    ->RangeMultiplier(2)
    ->Range(2 << 5, 2 << 18);
#ifdef DOUBLE_SUPPORT
BENCHMARK_TEMPLATE(BM_Scal3op, double)
    ->RangeMultiplier(2)
    ->Range(2 << 5, 2 << 18);
#endif
