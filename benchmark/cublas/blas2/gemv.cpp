/**************************************************************************
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
 *  @filename gemv.cpp
 *
 **************************************************************************/

#include "../utils.hpp"

template <typename scalar_t>
std::string get_name(std::string t, int m, int n) {
  std::ostringstream str{};
  str << "BM_Gemv<" << blas_benchmark::utils::get_type_name<scalar_t>() << ">/"
      << t << "/" << m << "/" << n;
  return str.str();
}

template <typename scalar_t, typename... args_t>
static inline void cublas_routine(args_t&&... args) {
  if constexpr (std::is_same_v<scalar_t, float>)
    cublasSgemv(std::forward<args_t>(args)...);
  else if constexpr (std::is_same_v<scalar_t, double>)
    cublasDgemv(std::forward(args)...);
  return;
}

template <typename scalar_t>
void run(benchmark::State& state, cublasHandle_t* cuda_handle_ptr, int ti,
         index_t m, index_t n, scalar_t alpha, scalar_t beta, bool* success) {
  // Standard test setup.
  std::string ts = blas_benchmark::utils::from_transpose_enum(
      static_cast<blas_benchmark::utils::Transposition>(ti));
  const char* t_str = ts.c_str();

  index_t xlen = t_str[0] == 'n' ? n : m;
  index_t ylen = t_str[0] == 'n' ? m : n;

  index_t lda = m;
  index_t incX = 1;
  index_t incY = 1;

  // The counters are double. We convert m and n to double to avoid
  // integer overflows for n_fl_ops and bytes_processed
  double m_d = static_cast<double>(m);
  double n_d = static_cast<double>(n);

  state.counters["m"] = m_d;
  state.counters["n"] = n_d;

  {
    double nflops_AtimesX = 2.0 * m_d * n_d;
    double nflops_timesAlpha = ylen;
    double nflops_addBetaY = (beta != scalar_t{0}) ? 2 * ylen : 0;
    state.counters["n_fl_ops"] =
        nflops_AtimesX + nflops_timesAlpha + nflops_addBetaY;
  }
  {
    double mem_readA = m_d * n_d;
    double mem_readX = xlen;
    double mem_writeY = ylen;
    double mem_readY = (beta != scalar_t{0}) ? ylen : 0;
    state.counters["bytes_processed"] =
        (mem_readA + mem_readX + mem_writeY + mem_readY) * sizeof(scalar_t);
  }

  cublasHandle_t& cuda_handle = *cuda_handle_ptr;

  // Input matrix/vector, output vector.
  std::vector<scalar_t> m_a =
      blas_benchmark::utils::random_data<scalar_t>(m * n);
  std::vector<scalar_t> v_x =
      blas_benchmark::utils::random_data<scalar_t>(xlen);
  std::vector<scalar_t> v_y =
      blas_benchmark::utils::random_data<scalar_t>(ylen);

  cublasSetPointerMode(cuda_handle, CUBLAS_POINTER_MODE_HOST);

  scalar_t* m_a_gpu = nullptr;
  scalar_t* v_x_gpu = nullptr;
  scalar_t* v_y_gpu = nullptr;

  cudaMalloc(&m_a_gpu, m * n * sizeof(scalar_t));
  cudaMalloc(&v_x_gpu, xlen * sizeof(scalar_t));
  cudaMalloc(&v_y_gpu, ylen * sizeof(scalar_t));

  cudaMemcpyAsync(m_a_gpu, m_a.data(), m * n * sizeof(scalar_t), cudaMemcpyHostToDevice);
  cudaMemcpyAsync(v_x_gpu, v_x.data(), xlen * sizeof(scalar_t), cudaMemcpyHostToDevice);
  cudaMemcpyAsync(v_y_gpu, v_y.data(), ylen * sizeof(scalar_t), cudaMemcpyHostToDevice);

  cublasOperation_t cuda_trans = (*t_str == 'n') ? CUBLAS_OP_N : CUBLAS_OP_T;

#ifdef BLAS_VERIFY_BENCHMARK
  // Run a first time with a verification of the results
  std::vector<scalar_t> v_y_ref = v_y;
  reference_blas::gemv(t_str, m, n, alpha, m_a.data(), m, v_x.data(), incX,
                       beta, v_y_ref.data(), incY);
  std::vector<scalar_t> v_y_temp = v_y;
  {
    scalar_t* v_y_temp_gpu = nullptr;
    cudaMalloc(&v_y_temp_gpu, ylen * sizeof(scalar_t));
    cudaMemcpy(v_y_temp_gpu, v_y_temp.data(), ylen * sizeof(scalar_t), cudaMemcpyHostToDevice);
    cublas_routine<scalar_t>(cuda_handle, cuda_trans, m, n, &alpha, m_a_gpu, m, v_x_gpu, incX, &beta, v_y_temp_gpu, incY);
    cudaMemcpy(v_y_temp.data(), v_y_temp_gpu, ylen * sizeof(scalar_t), cudaMemcpyDeviceToHost);
    cudaFree(v_y_temp_gpu);
  }

  std::ostringstream err_stream;
  if (!utils::compare_vectors(v_y_temp, v_y_ref, err_stream, "")) {
    const std::string& err_str = err_stream.str();
    state.SkipWithError(err_str.c_str());
    *success = false;
  };
#endif

  auto blas_method_def = [&]() -> std::vector<cudaEvent_t> {
    cudaEvent_t start;
    cudaEvent_t stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    cublas_routine<scalar_t>(cuda_handle, cuda_trans, m, n, &alpha, m_a_gpu, m, v_x_gpu, incX, &beta, v_y_gpu, incY);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    return std::vector{start, stop};
  };

  // Warmup
  blas_benchmark::utils::warmup(blas_method_def);

  cudaDeviceSynchronize();

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
void register_benchmark(blas_benchmark::Args& args, cublasHandle_t* cuda_handle_ptr,
                        bool* success) {
  auto gemm_params = blas_benchmark::utils::get_blas2_params<scalar_t>(args);

  for (auto p : gemm_params) {
    std::string ts;
    index_t m, n;
    scalar_t alpha, beta;
    std::tie(ts, m, n, alpha, beta) = p;
    int t = static_cast<int>(blas_benchmark::utils::to_transpose_enum(ts));

    auto BM_lambda = [&](benchmark::State& st, cublasHandle_t* cuda_handle_ptr, int t,
                         index_t m, index_t n, scalar_t alpha, scalar_t beta,
                         bool* success) {
      run<scalar_t>(st, cuda_handle_ptr, t, m, n, alpha, beta, success);
    };
    benchmark::RegisterBenchmark(get_name<scalar_t>(ts, m, n).c_str(),
                                 BM_lambda, cuda_handle_ptr, t, m, n, alpha, beta,
                                 success);
  }
}

namespace blas_benchmark {
void create_benchmark(blas_benchmark::Args& args, cublasHandle_t* cuda_handle_ptr,
                      bool* success) {
  BLAS_REGISTER_BENCHMARK(args, cuda_handle_ptr, success);
}
}  // namespace blas_benchmark
