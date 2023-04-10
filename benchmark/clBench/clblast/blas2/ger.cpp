/**************************************************************************
 *
 *  @license
 *  Copyright (C) Codeplay Software Limited
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
 *  @filename ger.cpp
 *
 **************************************************************************/

#include "../utils.hpp"

template <typename scalar_t>
std::string get_name(int m, int n) {
  std::ostringstream str{};
  str << "BM_Ger<" << blas_benchmark::utils::get_type_name<scalar_t>() << ">/"
      << m << "/" << n;
  return str.str();
}

template <typename scalar_t>
void run(benchmark::State& state, ExecutorType* executorPtr, index_t m,
         index_t n, scalar_t alpha, bool* success) {
  // Standard test setup.

  index_t xlen = m;
  index_t ylen = n;

  index_t lda = m;
  index_t incX = 1;
  index_t incY = 1;

  blas_benchmark::utils::init_level_2_counters<
      blas_benchmark::utils::Level2Op::ger, scalar_t>(state, m, n);

  ExecutorType& sb_handle = *executorPtr;

  // Input matrix/vector, output vector.
  std::vector<scalar_t> m_a =
      blas_benchmark::utils::random_data<scalar_t>(m * n);
  std::vector<scalar_t> v_x =
      blas_benchmark::utils::random_data<scalar_t>(xlen);
  std::vector<scalar_t> v_y =
      blas_benchmark::utils::random_data<scalar_t>(ylen);

  auto layout = clblast::Layout::kColMajor;

  MemBuffer<scalar_t> m_a_gpu(executorPtr, m_a.data(),
                              static_cast<size_t>(m * n));
  MemBuffer<scalar_t> v_x_gpu(executorPtr, v_x.data(),
                              static_cast<size_t>(xlen));
  MemBuffer<scalar_t> v_y_gpu(executorPtr, v_y.data(),
                              static_cast<size_t>(ylen));

#ifdef BLAS_VERIFY_BENCHMARK
  // Run a first time with a verification of the results
  std::vector<scalar_t> m_a_ref = m_a;
  reference_blas::ger(m, n, alpha, v_x.data(), incX, v_y.data(), incY,
                      m_a_ref.data(), lda);
  std::vector<scalar_t> m_a_temp(m_a);

  {
    MemBuffer<scalar_t> m_a_temp_gpu(executorPtr, m_a_temp.data(),
                                     static_cast<size_t>(m * n));
    cl_event event;
    clblast::Ger(layout, m, n, alpha, v_x_gpu.dev(), 0, incX, v_y_gpu.dev(), 0,
                 incY, m_a_temp_gpu.dev(), 0, lda, executorPtr->_queue(),
                 &event);
    CLEventHandler::wait(event);
  }

  std::ostringstream err_stream;
  if (!utils::compare_vectors(m_a_temp, m_a_ref, err_stream, "")) {
    const std::string& err_str = err_stream.str();
    state.SkipWithError(err_str.c_str());
    *success = false;
  };

#endif

  auto blas_method_def = [&]() -> std::vector<cl_event> {
    cl_event event;
    clblast::Ger(layout, m, n, alpha, v_x_gpu.dev(), 0, incX, v_y_gpu.dev(), 0,
                 incY, m_a_gpu.dev(), 0, lda, executorPtr->_queue(), &event);
    CLEventHandler::wait(event);
    return {event};
  };

  // Warmup
  blas_benchmark::utils::warmup(blas_method_def);

  blas_benchmark::utils::init_counters(state);

  // Measure
  for (auto _ : state) {
    // Run
    std::tuple<double, double> times =
        blas_benchmark::utils::timef(blas_method_def);

    // Report
    blas_benchmark::utils::update_counters(state, times);
  }

  state.SetItemsProcessed(state.iterations() * state.counters["n_fl_ops"]);
  state.SetBytesProcessed(state.iterations() *
                          state.counters["bytes_processed"]);

  blas_benchmark::utils::calc_avg_counters(state);
}

template <typename scalar_t>
void register_benchmark(blas_benchmark::Args& args, ExecutorType* executorPtr,
                        bool* success) {
  auto ger_params = blas_benchmark::utils::get_ger_params<scalar_t>(args);

  for (auto p : ger_params) {
    index_t m, n;
    scalar_t alpha;
    std::tie(m, n, alpha) = p;

    auto BM_lambda = [&](benchmark::State& st, ExecutorType* executorPtr,
                         index_t m, index_t n, scalar_t alpha, bool* success) {
      run<scalar_t>(st, executorPtr, m, n, alpha, success);
    };
    benchmark::RegisterBenchmark(get_name<scalar_t>(m, n).c_str(), BM_lambda,
                                 executorPtr, m, n, alpha, success)
        ->UseRealTime();
  }
}

namespace blas_benchmark {
void create_benchmark(blas_benchmark::Args& args, ExecutorType* executorPtr,
                      bool* success) {
  BLAS_REGISTER_BENCHMARK(args, executorPtr, success);
}
}  // namespace blas_benchmark
