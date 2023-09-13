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
 *  portBLAS: BLAS implementation using SYCL
 *
 *  @filename syr2.cpp
 *
 **************************************************************************/

#include "../utils.hpp"

template <typename scalar_t>
std::string get_name(std::string uplo, int n, scalar_t alpha) {
  std::ostringstream str{};
  str << "BM_Syr2<" << blas_benchmark::utils::get_type_name<scalar_t>() << ">/"
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
  index_t incY = 1;

  blas_benchmark::utils::init_level_2_counters<
      blas_benchmark::utils::Level2Op::syr2, scalar_t>(state, "n", 0, 0, n);

  ExecutorType& sb_handle = *executorPtr;

  const int v_x_size = 1 + (n - 1) * std::abs(incX);
  const int v_y_size = 1 + (n - 1) * std::abs(incY);

  // Input matrix/vector, output vector.
  std::vector<scalar_t> m_a =
      blas_benchmark::utils::random_data<scalar_t>(n * n);
  std::vector<scalar_t> v_x =
      blas_benchmark::utils::random_data<scalar_t>(v_x_size);
  std::vector<scalar_t> v_y =
      blas_benchmark::utils::random_data<scalar_t>(v_y_size);

  // Specify the triangle.
  clblast::Triangle a_tr = blas_benchmark::utils::translate_triangle(uplo_str);

  // Specify the layout.
  auto layout = clblast::Layout::kColMajor;

  MemBuffer<scalar_t> m_a_gpu(executorPtr, m_a.data(),
                              static_cast<size_t>(n * n));
  MemBuffer<scalar_t> v_x_gpu(executorPtr, v_x.data(),
                              static_cast<size_t>(v_x_size));
  MemBuffer<scalar_t> v_y_gpu(executorPtr, v_y.data(),
                              static_cast<size_t>(v_y_size));

#ifdef BLAS_VERIFY_BENCHMARK
  // Run a first time with a verification of the results
  std::vector<scalar_t> m_a_ref = m_a;
  reference_blas::syr2(uplo_str, n, alpha, v_x.data(), incX, v_y.data(), incY,
                       m_a_ref.data(), lda);
  std::vector<scalar_t> m_a_temp(m_a);

  {
    MemBuffer<scalar_t> m_a_temp_gpu(executorPtr, m_a_temp.data(),
                                     static_cast<size_t>(n * n));
    cl_event event;
    clblast::Syr2(layout, a_tr, n, alpha, v_x_gpu.dev(), 0, incX, v_y_gpu.dev(),
                  0, incY, m_a_temp_gpu.dev(), 0, lda, executorPtr->_queue(),
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
    clblast::Syr2(layout, a_tr, n, alpha, v_x_gpu.dev(), 0, incX, v_y_gpu.dev(),
                  0, incY, m_a_gpu.dev(), 0, lda, executorPtr->_queue(),
                  &event);
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

  state.SetItemsProcessed(state.iterations() * state.counters["n_fl_ops"]);
  state.SetBytesProcessed(state.iterations() *
                          state.counters["bytes_processed"]);

  blas_benchmark::utils::calc_avg_counters(state);
}

template <typename scalar_t>
void register_benchmark(blas_benchmark::Args& args, ExecutorType* executorPtr,
                        bool* success) {
  // syr2 use the same parameters so reuse syr function
  auto syr2_params = blas_benchmark::utils::get_syr_params<scalar_t>(args);

  for (auto p : syr2_params) {
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
                                 success)
        ->UseRealTime();
  }
}

namespace blas_benchmark {
void create_benchmark(blas_benchmark::Args& args, ExecutorType* executorPtr,
                      bool* success) {
  BLAS_REGISTER_BENCHMARK(args, executorPtr, success);
}
}  // namespace blas_benchmark
