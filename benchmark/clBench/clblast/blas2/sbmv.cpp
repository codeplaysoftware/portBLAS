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
 *  @filename sbmv.cpp
 *
 **************************************************************************/

#include "../utils.hpp"

template <typename scalar_t>
std::string get_name(std::string uplo, int n, int k) {
  std::ostringstream str{};
  str << "BM_Sbmv<" << blas_benchmark::utils::get_type_name<scalar_t>() << ">/"
      << uplo << "/" << n << "/" << k;
  return str.str();
}

template <typename scalar_t>
void run(benchmark::State& state, ExecutorType* executorPtr, std::string uplo,
         index_t n, index_t k, scalar_t alpha, scalar_t beta, bool* success) {
  // Standard test setup.
  const char* uplo_str = uplo.c_str();

  index_t xlen = n;
  index_t ylen = n;

  index_t lda = (k + 1);
  index_t incX = 1;
  index_t incY = 1;

  // The counters are double. We convert n to double to avoid
  // integer overflows for n_fl_ops and bytes_processed
  double n_d = static_cast<double>(n);
  double k_d = static_cast<double>(k);

  state.counters["n"] = n_d;
  state.counters["k"] = k_d;

  // Compute the number of A non-zero elements.
  const double A_validVal = (n_d * (2.0 * k_d + 1.0)) - (k_d * (k_d + 1.0));

  {
    double nflops_AtimesX = 2.0 * A_validVal;
    double nflops_timesAlpha = ylen;
    double nflops_addBetaY = (beta != scalar_t{0}) ? 2 * ylen : 0;
    state.counters["n_fl_ops"] =
        nflops_AtimesX + nflops_timesAlpha + nflops_addBetaY;
  }
  {
    double mem_readA = A_validVal;
    double mem_readX = xlen;
    double mem_writeY = ylen;
    double mem_readY = (beta != scalar_t{0}) ? ylen : 0;
    state.counters["bytes_processed"] =
        (mem_readA + mem_readX + mem_writeY + mem_readY) * sizeof(scalar_t);
  }

  ExecutorType& ex = *executorPtr;

  // Input matrix/vector, output vector.
  std::vector<scalar_t> m_a =
      blas_benchmark::utils::random_data<scalar_t>(lda * n);
  std::vector<scalar_t> v_x =
      blas_benchmark::utils::random_data<scalar_t>(xlen);
  std::vector<scalar_t> v_y =
      blas_benchmark::utils::random_data<scalar_t>(ylen);

  // Specify the triangle.
  clblast::Triangle a_tr = blas_benchmark::utils::translate_triangle(uplo_str);

  // Specify the layout.
  auto layout = clblast::Layout::kColMajor;

  MemBuffer<scalar_t> m_a_gpu(executorPtr, m_a.data(),
                              static_cast<size_t>(lda * n));
  MemBuffer<scalar_t> v_x_gpu(executorPtr, v_x.data(),
                              static_cast<size_t>(xlen));
  MemBuffer<scalar_t> v_y_gpu(executorPtr, v_y.data(),
                              static_cast<size_t>(ylen));

#ifdef BLAS_VERIFY_BENCHMARK
  // Run a first time with a verification of the results
  std::vector<scalar_t> v_y_ref = v_y;
  reference_blas::sbmv(uplo_str, n, k, alpha, m_a.data(), lda, v_x.data(), incX,
                       beta, v_y_ref.data(), incY);
  std::vector<scalar_t> v_y_temp = v_y;
  {
    MemBuffer<scalar_t> v_y_temp_gpu(executorPtr, v_y_temp.data(),
                                     static_cast<size_t>(ylen));
    cl_event event;
    clblast::Sbmv<scalar_t>(layout, a_tr, n, k, alpha, m_a_gpu.dev(), 0, lda,
                            v_x_gpu.dev(), 0, incX, beta, v_y_temp_gpu.dev(), 0,
                            incY, executorPtr->_queue(), &event);
    CLEventHandler::wait(event);
  }

  std::ostringstream err_stream;
  if (!utils::compare_vectors(v_y_temp, v_y_ref, err_stream, "")) {
    const std::string& err_str = err_stream.str();
    state.SkipWithError(err_str.c_str());
    *success = false;
  };
#endif

  auto blas_method_def = [&]() -> std::vector<cl_event> {
    cl_event event;
    clblast::Sbmv<scalar_t>(layout, a_tr, n, k, alpha, m_a_gpu.dev(), 0, lda,
                            v_x_gpu.dev(), 0, incX, beta, v_y_gpu.dev(), 0,
                            incY, executorPtr->_queue(), &event);
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
  auto sbmv_params = blas_benchmark::utils::get_sbmv_params<scalar_t>(args);

  for (auto p : sbmv_params) {
    std::string uplos;
    index_t n, k;
    scalar_t alpha, beta;
    std::tie(uplos, n, k, alpha, beta) = p;

    auto BM_lambda = [&](benchmark::State& st, ExecutorType* exPtr,
                         std::string uplos, index_t n, index_t k,
                         scalar_t alpha, scalar_t beta, bool* success) {
      run<scalar_t>(st, exPtr, uplos, n, k, alpha, beta, success);
    };
    benchmark::RegisterBenchmark(get_name<scalar_t>(uplos, n, k).c_str(),
                                 BM_lambda, exPtr, uplos, n, k, alpha, beta,
                                 success);
  }
}

namespace blas_benchmark {
void create_benchmark(blas_benchmark::Args& args, ExecutorType* exPtr,
                      bool* success) {
  BLAS_REGISTER_BENCHMARK(args, exPtr, success);
}
}  // namespace blas_benchmark
