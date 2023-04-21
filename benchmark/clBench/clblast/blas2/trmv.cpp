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
 *  @filename trmv.cpp
 *
 **************************************************************************/

#include "../utils.hpp"

template <typename scalar_t>
std::string get_name(std::string uplo, std::string t, std::string diag, int n) {
  std::ostringstream str{};
  str << "BM_Trmv<" << blas_benchmark::utils::get_type_name<scalar_t>() << ">/"
      << uplo << "/" << t << "/" << diag << "/" << n;
  return str.str();
}

template <typename scalar_t>
void run(benchmark::State& state, ExecutorType* executorPtr, std::string uplo,
         std::string t, std::string diag, index_t n, bool* success) {
  // Standard test setup.
  const char* uplo_str = uplo.c_str();
  const char* t_str = t.c_str();
  const char* diag_str = diag.c_str();

  index_t xlen = n;
  index_t lda = n;
  index_t incX = 1;

  blas_benchmark::utils::init_level_2_counters<
      blas_benchmark::utils::Level2Op::trmv, scalar_t>(state, "n", 0, 0, n);

  ExecutorType& sb_handle = *executorPtr;

  // Input matrix/vector, output vector.
  std::vector<scalar_t> m_a =
      blas_benchmark::utils::random_data<scalar_t>(lda * n);
  std::vector<scalar_t> v_x =
      blas_benchmark::utils::random_data<scalar_t>(xlen);

  // Specify the triangle.
  clblast::Triangle a_uplo =
      blas_benchmark::utils::translate_triangle(uplo_str);

  // Specify the transposition.
  clblast::Transpose a_tr =
      blas_benchmark::utils::translate_transposition(t_str);

  // Specify the unit-diagonal.
  clblast::Diagonal a_diag =
      blas_benchmark::utils::translate_diagonal(diag_str);

  // Specify the layout.
  auto layout = clblast::Layout::kColMajor;

  MemBuffer<scalar_t> m_a_gpu(executorPtr, m_a.data(),
                              static_cast<size_t>(lda * n));
  MemBuffer<scalar_t> v_x_gpu(executorPtr, v_x.data(),
                              static_cast<size_t>(xlen));

#ifdef BLAS_VERIFY_BENCHMARK
  // Run a first time with a verification of the results
  std::vector<scalar_t> v_x_ref = v_x;
  reference_blas::trmv(uplo_str, t_str, diag_str, n, m_a.data(), lda,
                       v_x_ref.data(), incX);
  std::vector<scalar_t> v_x_temp = v_x;
  {
    MemBuffer<scalar_t> v_x_temp_gpu(executorPtr, v_x_temp.data(),
                                     static_cast<size_t>(xlen));
    cl_event event;
    clblast::Trmv<scalar_t>(layout, a_uplo, a_tr, a_diag, n, m_a_gpu.dev(), 0,
                            lda, v_x_temp_gpu.dev(), 0, incX,
                            executorPtr->_queue(), &event);
    CLEventHandler::wait(event);
  }

  std::ostringstream err_stream;
  if (!utils::compare_vectors(v_x_temp, v_x_ref, err_stream, "")) {
    const std::string& err_str = err_stream.str();
    state.SkipWithError(err_str.c_str());
    *success = false;
  };
#endif

  auto blas_method_def = [&]() -> std::vector<cl_event> {
    cl_event event;
    clblast::Trmv<scalar_t>(layout, a_uplo, a_tr, a_diag, n, m_a_gpu.dev(), 0,
                            lda, v_x_gpu.dev(), 0, incX, executorPtr->_queue(),
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
  // trmv uses the same parameters as trsv
  auto trmv_params = blas_benchmark::utils::get_trsv_params(args);

  for (auto p : trmv_params) {
    std::string uplos;
    std::string ts;
    std::string diags;
    index_t n;
    std::tie(uplos, ts, diags, n) = p;

    auto BM_lambda = [&](benchmark::State& st, ExecutorType* executorPtr,
                         std::string uplos, std::string ts, std::string diags,
                         index_t n, bool* success) {
      run<scalar_t>(st, executorPtr, uplos, ts, diags, n, success);
    };
    benchmark::RegisterBenchmark(
        get_name<scalar_t>(uplos, ts, diags, n).c_str(), BM_lambda, executorPtr,
        uplos, ts, diags, n, success)
        ->UseRealTime();
  }
}

namespace blas_benchmark {
void create_benchmark(blas_benchmark::Args& args, ExecutorType* executorPtr,
                      bool* success) {
  BLAS_REGISTER_BENCHMARK(args, executorPtr, success);
}
}  // namespace blas_benchmark
