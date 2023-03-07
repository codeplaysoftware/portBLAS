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

template <typename scalar_t, typename... args_t>
static inline void cublas_routine(args_t&&... args) {
  if constexpr (std::is_same_v<scalar_t, float>){
    CUBLAS_CHECK(cublasStbmv(std::forward<args_t>(args)...));
  }
  else if constexpr (std::is_same_v<scalar_t, double>){
    CUBLAS_CHECK(cublasDtbmv(std::forward<args_t>(args)...));
  }
  return;
}

template <typename scalar_t>
void run(benchmark::State& state, cublasHandle_t* cuda_handle_ptr,
         std::string uplo, std::string t, std::string diag, index_t n,
         index_t k, bool* success) {
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

  cublasHandle_t& cuda_handle = *cuda_handle_ptr;

  cublasFillMode_t cuda_uplo = (*uplo_str == 'u') ?
    CUBLAS_FILL_MODE_UPPER : CUBLAS_FILL_MODE_LOWER;

  cublasOperation_t cuda_t = (*t_str == 'n') ? CUBLAS_OP_N : CUBLAS_OP_T;
  cublasDiagType_t cuda_diag = (*diag_str == 'u') ? CUBLAS_DIAG_UNIT : CUBLAS_DIAG_NON_UNIT;

  // Input matrix/vector, output vector.
  std::vector<scalar_t> m_a =
      blas_benchmark::utils::random_data<scalar_t>(lda * n);
  std::vector<scalar_t> v_x =
      blas_benchmark::utils::random_data<scalar_t>(xlen);

  scalar_t* m_a_gpu = nullptr;
  scalar_t* v_x_gpu = nullptr;
  CUDA_CHECK(cudaMalloc(&m_a_gpu, lda * n * sizeof(scalar_t)));
  CUDA_CHECK(cudaMalloc(&v_x_gpu, xlen * sizeof(scalar_t)));
  CUDA_CHECK(cudaMemcpy(m_a_gpu, m_a.data(), lda * n * sizeof(scalar_t), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(v_x_gpu, v_x.data(), xlen * sizeof(scalar_t), cudaMemcpyHostToDevice));


#ifdef BLAS_VERIFY_BENCHMARK
  // Run a first time with a verification of the results
  std::vector<scalar_t> v_x_ref = v_x;
  reference_blas::tbmv(uplo_str, t_str, diag_str, n, k, m_a.data(), lda,
                       v_x_ref.data(), incX);
  std::vector<scalar_t> v_x_temp = v_x;
  {
    scalar_t* v_x_temp_gpu = nullptr;
    CUDA_CHECK(cudaMalloc(&v_x_temp_gpu, xlen * sizeof(scalar_t)));
    CUDA_CHECK(cudaMemcpy(v_x_temp_gpu, v_x_temp.data(), xlen * sizeof(scalar_t), cudaMemcpyHostToDevice));
    cublas_routine<scalar_t>(cuda_handle, cuda_uplo, cuda_t, cuda_diag, n, k, m_a_gpu, lda, v_x_temp_gpu, incX);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(v_x_temp.data(), v_x_temp_gpu, xlen * sizeof(scalar_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(v_x_temp_gpu));
  }

  std::ostringstream err_stream;
  if (!utils::compare_vectors(v_x_temp, v_x_ref, err_stream, "")) {
    const std::string& err_str = err_stream.str();
    state.SkipWithError(err_str.c_str());
    *success = false;
  };
#endif

  auto blas_warmup = [&]() -> void {
    cublas_routine<scalar_t>(cuda_handle, cuda_uplo, cuda_t, cuda_diag, n, k, m_a_gpu, lda, v_x_gpu, incX);
    CUDA_CHECK(cudaDeviceSynchronize());
    return;
  };
  cudaEvent_t start;
  cudaEvent_t stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));

  auto blas_method_def = [&]() -> std::vector<cudaEvent_t> {
    CUDA_CHECK(cudaEventRecord(start));
    cublas_routine<scalar_t>(cuda_handle, cuda_uplo, cuda_t, cuda_diag, n, k, m_a_gpu, lda, v_x_gpu, incX);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    return std::vector{start, stop};
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

  CUDA_CHECK(cudaFree(m_a_gpu));
  CUDA_CHECK(cudaFree(v_x_gpu));
  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
}

template <typename scalar_t>
void register_benchmark(blas_benchmark::Args& args,
                        cublasHandle_t* cuda_handle_ptr, bool* success) {
  auto tbmv_params = blas_benchmark::utils::get_tbmv_params(args);

  for (auto p : tbmv_params) {
    std::string uplos;
    std::string ts;
    std::string diags;
    index_t n;
    index_t k;
    std::tie(uplos, ts, diags, n, k) = p;

    auto BM_lambda = [&](benchmark::State& st, cublasHandle_t* cuda_handle_ptr,
                         std::string uplos, std::string ts, std::string diags,
                         index_t n, index_t k, bool* success) {
      run<scalar_t>(st, cuda_handle_ptr, uplos, ts, diags, n, k, success);
    };
    benchmark::RegisterBenchmark(
        get_name<scalar_t>(uplos, ts, diags, n, k).c_str(), BM_lambda,
        cuda_handle_ptr, uplos, ts, diags, n, k, success);
  }
}

namespace blas_benchmark {
void create_benchmark(blas_benchmark::Args& args,
                      cublasHandle_t* cuda_handle_ptr, bool* success) {
  BLAS_REGISTER_BENCHMARK(args, cuda_handle_ptr, success);
}
}  // namespace blas_benchmark
