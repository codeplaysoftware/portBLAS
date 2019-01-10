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
 *  @filename dot.cpp
 *
 **************************************************************************/

#include "utils.hpp"

#include <interface/blas1_interface.hpp>

template <typename ScalarT>
void BM_Dot(benchmark::State& state) {
  // Standard test setup.
  using IndexType = unsigned int;

  const IndexType size = static_cast<IndexType>(state.range(0));
  state.counters["size"] = size;

  blas::Executor<SYCL> ex = *getExecutor();

  // Create data
  std::vector<ScalarT> v1 = benchmark::utils::random_data<ScalarT>(size);
  std::vector<ScalarT> v2 = benchmark::utils::random_data<ScalarT>(size);
  ScalarT res;

  auto inx = blas::helper::make_sycl_iterator_buffer<ScalarT>(v1, size);
  auto iny = blas::helper::make_sycl_iterator_buffer<ScalarT>(v2, size);
  auto inr = blas::helper::make_sycl_iterator_buffer<ScalarT>(&res, 1);

  // Warmup
  for (int i = 0; i < 10; i++) {
    _dot(ex, size, inx, 1, iny, 1, inr);
  }

  // Measure
  for (auto _ : state) {
    // Run
    auto event = _dot(ex, size, inx, 1, iny, 1, inr);
    ex.wait(event);

    // Report
    state.PauseTiming();
    state.counters["event_time"] = benchmark::utils::time_event(event);
    state.ResumeTiming();
  }
}

BENCHMARK_TEMPLATE(BM_Dot, float)->RangeMultiplier(2)->Range(2 << 5, 2 << 18);
BENCHMARK_TEMPLATE(BM_Dot, double)->RangeMultiplier(2)->Range(2 << 5, 2 << 18);