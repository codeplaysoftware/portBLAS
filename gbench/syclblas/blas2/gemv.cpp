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
 *  @filename syclblas_benchmark.cpp
 *
 **************************************************************************/

#include "utils.hpp"

#include <interface/blas2_interface.hpp>

template <typename ScalarT>
void BM_Gemv(benchmark::State& state) {
  // Standard test setup.
  using IndexType = unsigned int;

  const char* t_str = benchmark::utils::from_transpose_enum(
      static_cast<benchmark::utils::Transposition>(state.range(0)));
  const IndexType m = static_cast<IndexType>(state.range(1));
  const IndexType n = static_cast<IndexType>(state.range(2));

  IndexType vlen = t_str[0] == 'n' ? n : m;
  IndexType rlen = t_str[0] == 'n' ? m : n;

  IndexType lda = m;
  long incX = 1;
  long incY = 1;

  state.counters["m"] = m;
  state.counters["n"] = n;

  blas::Executor<SYCL> ex = *getExecutor();

  // Create data
  // Scalars
  ScalarT alpha = benchmark::utils::random_scalar<ScalarT>();
  ScalarT beta = benchmark::utils::random_scalar<ScalarT>();

  // Input matrix/vector, output vector.
  std::vector<ScalarT> a_m = benchmark::utils::random_data<ScalarT>(m * n);
  std::vector<ScalarT> b_v = benchmark::utils::random_data<ScalarT>(vlen);
  std::vector<ScalarT> c_v_gpu_result =
      benchmark::utils::const_data<ScalarT>(rlen, 0);

  auto m_a_gpu = blas::helper::make_sycl_iterator_buffer<ScalarT>(a_m, m * n);
  auto v_b_gpu = blas::helper::make_sycl_iterator_buffer<ScalarT>(b_v, vlen);
  auto v_c_gpu =
      blas::helper::make_sycl_iterator_buffer<ScalarT>(c_v_gpu_result, rlen);

  // Warmup
  for (int i = 0; i < 10; i++) {
    _gemv(ex, *t_str, m, n, alpha, m_a_gpu, m, v_b_gpu, incX, beta, v_c_gpu,
          incY);
  }

  //   std::cout << "Ran successfully! " << std::endl;

  // Measure
  for (auto _ : state) {
    // Run
    auto event = _gemv(ex, *t_str, m, n, alpha, m_a_gpu, m, v_b_gpu, incX, beta,
                       v_c_gpu, incY);
    ex.wait(event);

    // Report
    state.PauseTiming();
    state.counters["event_time"] = benchmark::utils::time_event(event);
    state.ResumeTiming();
  }
}

static void gemv_args(benchmark::internal::Benchmark* b) {
  for (int i = 2 << 5; i <= 2 << 18; i *= 2)
    for (int j = 2 << 5; j <= 2 << 18; j *= 2) {
      b->Args({(int)benchmark::utils::to_transpose_enum("n"), i, j});
      b->Args({(int)benchmark::utils::to_transpose_enum("t"), i, j});
      b->Args({(int)benchmark::utils::to_transpose_enum("c"), i, j});
    }
}

BENCHMARK_TEMPLATE(BM_Gemv, float)->Apply(gemv_args);
BENCHMARK_TEMPLATE(BM_Gemv, double)->Apply(gemv_args);