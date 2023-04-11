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
 *  @filename syr.cpp
 *
 **************************************************************************/

#include "../utils.hpp"

template <typename scalar_t>
std::string get_name(std::string uplo, int n, scalar_t alpha) {
  std::ostringstream str{};
  str << "BM_Syr<" << blas_benchmark::utils::get_type_name<scalar_t>() << ">/"
      << uplo << "/" << n << "/" << alpha;
  return str.str();
}

template <typename scalar_t>
void run(benchmark::State& state, ExecutorType* executorPtr, std::string uplo,
         index_t n, scalar_t alpha, bool* success) {
  // Standard test setup.
  const char* uplo_str = uplo.c_str();

  index_t lda = n;
  index_t incX = 1;

  // The counters are double. We convert m and n to double to avoid
  // integer overflows for n_fl_ops and bytes_processed
  scalar_t size_d = static_cast<scalar_t>(n * (n + 1) / 2);

  state.counters["n"] = static_cast<double>(n);

  const double nflops_timesAlpha = static_cast<double>(n);
  const double nflops_AplusXtimeAlphaX = 2. * size_d;
  const double nflops_tot = nflops_AplusXtimeAlphaX + nflops_timesAlpha;
  state.counters["n_fl_ops"] = nflops_tot;

  const double mem_readWriteA = 2 * size_d;
  const double mem_readX = static_cast<double>(n);
  const double tot_mem_processed =
      (mem_readWriteA + mem_readX) * sizeof(scalar_t);
  state.counters["bytes_processed"] = tot_mem_processed;

  ExecutorType& ex = *executorPtr;

  // Input matrix/vector, output vector.
  std::vector<scalar_t> m_a =
      blas_benchmark::utils::random_data<scalar_t>(n * n);
  std::vector<scalar_t> v_x = blas_benchmark::utils::random_data<scalar_t>(n);

  // Specify the triangle.
  clblast::Triangle a_tr = blas_benchmark::utils::translate_triangle(uplo_str);

  // Specify the layout.
  auto layout = clblast::Layout::kColMajor;

  MemBuffer<scalar_t> m_a_gpu(executorPtr, m_a.data(),
                              static_cast<size_t>(n * n));
  MemBuffer<scalar_t> v_x_gpu(executorPtr, v_x.data(), static_cast<size_t>(n));

#ifdef BLAS_VERIFY_BENCHMARK
  // Run a first time with a verification of the results
  std::vector<scalar_t> m_a_ref = m_a;
  reference_blas::syr(uplo_str, n, alpha, v_x.data(), incX, m_a_ref.data(),
                      lda);
  std::vector<scalar_t> m_a_temp(m_a);

  {
    MemBuffer<scalar_t> m_a_temp_gpu(executorPtr, m_a_temp.data(),
                                     static_cast<size_t>(n * n));
    cl_event event;
    clblast::Syr(layout, a_tr, n, alpha, v_x_gpu.dev(), 0, incX,
                 m_a_temp_gpu.dev(), 0, lda, executorPtr->_queue(), &event);
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
    clblast::Syr(layout, a_tr, n, alpha, v_x_gpu.dev(), 0, incX, m_a_gpu.dev(),
                 0, lda, executorPtr->_queue(), &event);
    CLEventHandler::wait(event);
    return std::vector{event};
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

  state.SetBytesProcessed(state.iterations() * tot_mem_processed);
  state.SetItemsProcessed(state.iterations() * nflops_tot);

  blas_benchmark::utils::calc_avg_counters(state);
}

template <typename scalar_t>
void register_benchmark(blas_benchmark::Args& args, ExecutorType* executorPtr,
                        bool* success) {
  auto syr_params = blas_benchmark::utils::get_syr_params<scalar_t>(args);

  for (auto p : syr_params) {
    std::string uplo;
    index_t n;
    scalar_t alpha;
    std::tie(uplo, n, alpha) = p;

    auto BM_lambda = [&](benchmark::State& st, ExecutorType* executorPtr,
                         std::string uplo, index_t n, scalar_t alpha,
                         bool* success) {
      run<scalar_t>(st, executorPtr, uplo, n, alpha, success);
    };
    benchmark::RegisterBenchmark(get_name<scalar_t>(uplo, n, alpha).c_str(),
                                 BM_lambda, executorPtr, uplo, n, alpha,
                                 success);
  }
}

namespace blas_benchmark {
void create_benchmark(blas_benchmark::Args& args, ExecutorType* executorPtr,
                      bool* success) {
  BLAS_REGISTER_BENCHMARK(args, executorPtr, success);
}
}  // namespace blas_benchmark
