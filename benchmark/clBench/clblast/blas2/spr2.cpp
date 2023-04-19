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
 *  @filename spr2.cpp
 *
 **************************************************************************/

#include "../utils.hpp"

template <typename scalar_t>
std::string get_name(char uplo, int size, scalar_t alpha, int incX) {
  std::ostringstream str{};
  str << "BM_Spr2<" << blas_benchmark::utils::get_type_name<scalar_t>() << ">/"
      << uplo << "/" << size << "/" << alpha << "/" << incX;
  return str.str();
}

template <typename scalar_t>
void run(benchmark::State& state, ExecutorType* executorPtr, char uplo,
         int size, scalar_t alpha, int incX, bool* success) {
  // The counters are double. We convert size to double to avoid
  // integer overflows for n_fl_ops and bytes_processed
  const double size_d = static_cast<double>(size * (size + 1) / 2);
  const double n_d = static_cast<double>(size);

  // Temporarely set incY to 1 for all benchmarks
  index_t incY = 1;

  state.counters["n"] = n_d;
  state.counters["size_d"] = size_d;
  state.counters["alpha"] = static_cast<double>(alpha);
  state.counters["incX"] = incX;

  const double nflops_XtimesX = 4.0 * size_d;
  const double tot_nflops = size + nflops_XtimesX;
  state.counters["n_fl_ops"] = tot_nflops;

  const double mem_readWriteA = 2 * size_d;
  const double mem_readXreadY = 2 * static_cast<double>(n_d * std::abs(incX));
  const double tot_mem_processed =
      (mem_readWriteA + mem_readXreadY) * sizeof(scalar_t);
  state.counters["bytes_processed"] = tot_mem_processed;

  ExecutorType& ex = *executorPtr;

  const clblast::Triangle triangle =
      uplo == 'u' ? clblast::Triangle::kUpper : clblast::Triangle::kLower;

  const int m_size = size * size;
  const int v_x_size = 1 + (size - 1) * std::abs(incX);
  const int v_y_size = 1 + (size - 1) * std::abs(incY);

  // Input matrix/vector, output vector.
  std::vector<scalar_t> m_a =
      blas_benchmark::utils::random_data<scalar_t>(m_size);
  std::vector<scalar_t> v_x =
      blas_benchmark::utils::random_data<scalar_t>(v_x_size);
  std::vector<scalar_t> v_y =
      blas_benchmark::utils::random_data<scalar_t>(v_y_size);

  MemBuffer<scalar_t> m_a_gpu(executorPtr, m_a.data(),
                              static_cast<size_t>(m_size));
  MemBuffer<scalar_t> v_x_gpu(executorPtr, v_x.data(),
                              static_cast<size_t>(v_x_size));
  MemBuffer<scalar_t> v_y_gpu(executorPtr, v_y.data(),
                              static_cast<size_t>(v_y_size));

  clblast::Layout layout = clblast::Layout::kColMajor;
#ifdef BLAS_VERIFY_BENCHMARK
  // Run a first time with a verification of the results
  std::vector<scalar_t> x_ref = v_x;
  std::vector<scalar_t> y_ref = v_y;
  std::vector<scalar_t> m_a_ref = m_a;
  reference_blas::spr2<scalar_t>(&uplo, size, alpha, x_ref.data(), incX,
                                 y_ref.data(), incY, m_a_ref.data());
  std::vector<scalar_t> m_a_temp = m_a;
  {
    MemBuffer<scalar_t> m_a_temp_gpu(executorPtr, m_a_temp.data(),
                                     static_cast<size_t>(m_size));

    cl_event event;
    clblast::Spr2<scalar_t>(layout, triangle, size, alpha, v_x_gpu.dev(), 0,
                            incX, v_y_gpu.dev(), 0, incY, m_a_temp_gpu.dev(), 0,
                            executorPtr->_queue(), &event);
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
    clblast::Spr2<scalar_t>(layout, triangle, size, alpha, v_x_gpu.dev(), 0,
                           incX, v_y_gpu.dev(), 0, incY, m_a_gpu.dev(), 0,
                           executorPtr->_queue(), &event);
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

  state.SetBytesProcessed(state.iterations() * tot_mem_processed);
  state.SetItemsProcessed(state.iterations() * tot_nflops);

  blas_benchmark::utils::calc_avg_counters(state);
}

template <typename scalar_t>
void register_benchmark(blas_benchmark::Args& args, ExecutorType* exPtr,
                        bool* success) {
  auto spr2_params = blas_benchmark::utils::get_spr_params<scalar_t>(args);

  for (auto p : spr2_params) {
    int n, incX;
    std::string uplo;
    scalar_t alpha;
    std::tie(uplo, n, alpha, incX) = p;

    char uplo_c = uplo[0];
    auto BM_lambda_col = [&](benchmark::State& st, ExecutorType* exPtr,
                             char uplo, int size, scalar_t alpha, int incX,
                             bool* success) {
      run<scalar_t>(st, exPtr, uplo, size, alpha, incX, success);
    };
    benchmark::RegisterBenchmark(
        get_name<scalar_t>(uplo_c, n, alpha, incX).c_str(), BM_lambda_col,
        exPtr, uplo_c, n, alpha, incX, success);
  }
}

namespace blas_benchmark {
void create_benchmark(blas_benchmark::Args& args, ExecutorType* exPtr,
                      bool* success) {
  BLAS_REGISTER_BENCHMARK(args, exPtr, success);
}
}  // namespace blas_benchmark
