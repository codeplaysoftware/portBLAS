/**************************************************************************
 *
 *  @license
 *  Copyright (C) 2023 Codeplay Software Limited
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
 *  @filename tbmv.cpp
 *
 **************************************************************************/

#include "../utils.hpp"

template <typename scalar_t>
std::string get_name(std::string uplo, std::string t, std::string diag, int n,
                     int k) {
  std::ostringstream str{};
  str << "BM_Tbmv<" << blas_benchmark::utils::get_type_name<scalar_t>() << ">/"
      << uplo << "/" << t << "/" << diag << "/" << n << "/" << k;
  return str.str();
}

template <typename scalar_t>
void run(benchmark::State& state, ExecutorType* executorPtr, std::string uplo,
         std::string t, std::string diag, index_t n, index_t k, bool* success) {
  // Standard test setup.
  const char* uplo_str = uplo.c_str();
  const char* t_str = t.c_str();
  const char* diag_str = diag.c_str();

  index_t xlen = n;
  index_t lda = (k + 1);
  index_t incX = 1;

  // The counters are double. We convert n to double to avoid
  // integer overflows for n_fl_ops and bytes_processed
  double n_d = static_cast<double>(n);
  double k_d = static_cast<double>(k);

  state.counters["n"] = n_d;
  state.counters["k"] = k_d;

  // Compute the number of A non-zero elements.
  const double A_validVal = (n_d * (k_d + 1.0)) - (0.5 * (k_d * (k_d + 1.0)));

  {
    double nflops_AtimesX = 2.0 * A_validVal;
    state.counters["n_fl_ops"] = nflops_AtimesX;
  }

  {
    double mem_readA = A_validVal;
    double mem_readX = xlen;
    double mem_writeX = xlen;
    state.counters["bytes_processed"] =
        (mem_readA + mem_readX + mem_writeX) * sizeof(scalar_t);
  }

  ExecutorType& ex = *executorPtr;

  // Input matrix/vector, output vector.
  std::vector<scalar_t> m_a =
      blas_benchmark::utils::random_data<scalar_t>(lda * n);
  std::vector<scalar_t> v_x =
      blas_benchmark::utils::random_data<scalar_t>(xlen);

  // Specify the layout.
  auto layout = clblast::Layout::kColMajor;

  MemBuffer<scalar_t> m_a_gpu(executorPtr, m_a.data(),
                              static_cast<size_t>(lda * n));
  MemBuffer<scalar_t> v_x_gpu(executorPtr, v_x.data(),
                              static_cast<size_t>(xlen));

  // Specify the triangle.
  clblast::Triangle a_uplo =
      blas_benchmark::utils::translate_triangle(uplo_str);

  // Specify the transposition.
  clblast::Transpose a_tr =
      blas_benchmark::utils::translate_transposition(t_str);

  // Specify the unit-diagonal.
  clblast::Diagonal a_diag =
      blas_benchmark::utils::translate_diagonal(diag_str);

#ifdef BLAS_VERIFY_BENCHMARK
  // Run a first time with a verification of the results
  std::vector<scalar_t> v_x_ref = v_x;
  reference_blas::tbmv(uplo_str, t_str, diag_str, n, k, m_a.data(), lda,
                       v_x_ref.data(), incX);
  std::vector<scalar_t> v_x_temp = v_x;
  {
    MemBuffer<scalar_t> v_x_temp_gpu(executorPtr, v_x_temp.data(),
                                     static_cast<size_t>(xlen));
    cl_event event;
    clblast::Tbmv<scalar_t>(layout, a_uplo, a_tr, a_diag, n, k, m_a_gpu.dev(),
                            0, lda, v_x_temp_gpu.dev(), 0, incX,
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
    clblast::Tbmv<scalar_t>(layout, a_uplo, a_tr, a_diag, n, k, m_a_gpu.dev(),
                            0, lda, v_x_gpu.dev(), 0, incX,
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

  blas_benchmark::utils::calc_avg_counters(state);
}

template <typename scalar_t>
void register_benchmark(blas_benchmark::Args& args, ExecutorType* exPtr,
                        bool* success) {
  auto tbmv_params = blas_benchmark::utils::get_tbmv_params(args);

  for (auto p : tbmv_params) {
    std::string uplos;
    std::string ts;
    std::string diags;
    index_t n;
    index_t k;
    std::tie(uplos, ts, diags, n, k) = p;

    auto BM_lambda = [&](benchmark::State& st, ExecutorType* exPtr,
                         std::string uplos, std::string ts, std::string diags,
                         index_t n, index_t k, bool* success) {
      run<scalar_t>(st, exPtr, uplos, ts, diags, n, k, success);
    };
    benchmark::RegisterBenchmark(
        get_name<scalar_t>(uplos, ts, diags, n, k).c_str(), BM_lambda, exPtr,
        uplos, ts, diags, n, k, success);
  }
}

namespace blas_benchmark {
void create_benchmark(blas_benchmark::Args& args, ExecutorType* exPtr,
                      bool* success) {
  BLAS_REGISTER_BENCHMARK(args, exPtr, success);
}
}  // namespace blas_benchmark
