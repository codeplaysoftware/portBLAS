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
 *  @filename tpmv.cpp
 *
 **************************************************************************/

#include "../utils.hpp"

constexpr blas_benchmark::utils::Level2Op benchmark_op =
    blas_benchmark::utils::Level2Op::tpmv;

template <typename scalar_t, typename... args_t>
static inline void cublas_routine(args_t&&... args) {
  if constexpr (std::is_same_v<scalar_t, float>) {
    CUBLAS_CHECK(cublasStpmv(std::forward<args_t>(args)...));
  } else if constexpr (std::is_same_v<scalar_t, double>) {
    CUBLAS_CHECK(cublasDtpmv(std::forward<args_t>(args)...));
  }
  return;
}

template <typename scalar_t>
void run(benchmark::State& state, cublasHandle_t* cuda_handle_ptr,
         std::string uplo, std::string t, std::string diag, index_t n,
         bool* success) {
  // initialize the state label
  blas_benchmark::utils::set_benchmark_label<scalar_t>(state);

  // Standard test setup.
  const char* uplo_str = uplo.c_str();
  const char* t_str = t.c_str();
  const char* diag_str = diag.c_str();

  index_t xlen = n;
  index_t incX = 1;

  blas_benchmark::utils::init_level_2_counters<
      blas_benchmark::utils::Level2Op::tpmv, scalar_t>(state, "n", 0, 0, n);

  cublasHandle_t& cuda_handle = *cuda_handle_ptr;

  // Input matrix/vector, output vector.
  std::vector<scalar_t> m_a =
      blas_benchmark::utils::random_data<scalar_t>(((n + 1) * n) / 2);
  std::vector<scalar_t> v_x =
      blas_benchmark::utils::random_data<scalar_t>(xlen);

  blas_benchmark::utils::CUDAVector<scalar_t> m_a_gpu(m_a.size(), m_a.data());
  blas_benchmark::utils::CUDAVector<scalar_t> v_x_gpu(xlen, v_x.data());

  cublasFillMode_t cuda_uplo =
      (*uplo_str == 'u') ? CUBLAS_FILL_MODE_UPPER : CUBLAS_FILL_MODE_LOWER;
  cublasOperation_t cuda_trans = (*t_str == 'n') ? CUBLAS_OP_N : CUBLAS_OP_T;
  cublasDiagType_t cuda_diag =
      (*diag_str == 'u') ? CUBLAS_DIAG_UNIT : CUBLAS_DIAG_NON_UNIT;

#ifdef BLAS_VERIFY_BENCHMARK
  // Run a first time with a verification of the results
  std::vector<scalar_t> v_x_ref = v_x;
  reference_blas::tpmv(uplo_str, t_str, diag_str, n, m_a.data(), v_x_ref.data(),
                       incX);
  std::vector<scalar_t> v_x_temp = v_x;
  {
    blas_benchmark::utils::CUDAVector<scalar_t, true> v_x_temp_gpu(
        xlen, v_x_temp.data());
    cublas_routine<scalar_t>(cuda_handle, cuda_uplo, cuda_trans, cuda_diag, n,
                             m_a_gpu, v_x_temp_gpu, incX);
  }

  std::ostringstream err_stream;
  if (!utils::compare_vectors(v_x_temp, v_x_ref, err_stream, "")) {
    const std::string& err_str = err_stream.str();
    state.SkipWithError(err_str.c_str());
    *success = false;
  };
#endif

  auto blas_warmup = [&]() -> void {
    cublas_routine<scalar_t>(cuda_handle, cuda_uplo, cuda_trans, cuda_diag, n,
                             m_a_gpu, v_x_gpu, incX);
    return;
  };
  cudaEvent_t start;
  cudaEvent_t stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));

  auto blas_method_def = [&]() -> std::vector<cudaEvent_t> {
    CUDA_CHECK(cudaEventRecord(start));
    cublas_routine<scalar_t>(cuda_handle, cuda_uplo, cuda_trans, cuda_diag, n,
                             m_a_gpu, v_x_gpu, incX);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    return std::vector{start, stop};
  };

  // Warmup
  blas_benchmark::utils::warmup(blas_warmup);
  CUDA_CHECK(cudaStreamSynchronize(NULL));

  blas_benchmark::utils::init_counters(state);

  // Measure
  for (auto _ : state) {
    // Run
    std::tuple<double, double> times =
        blas_benchmark::utils::timef_cuda(blas_method_def);

    // Report
    blas_benchmark::utils::update_counters(state, times);
  }

  state.SetItemsProcessed(state.iterations() * state.counters["n_fl_ops"]);
  state.SetBytesProcessed(state.iterations() *
                          state.counters["bytes_processed"]);

  blas_benchmark::utils::calc_avg_counters(state);

  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
}

template <typename scalar_t>
void register_benchmark(blas_benchmark::Args& args,
                        cublasHandle_t* cuda_handle_ptr, bool* success) {
  // tpmv uses the same parameters as trsv
  auto tpmv_params = blas_benchmark::utils::get_trsv_params(args);

  for (auto p : tpmv_params) {
    std::string uplos;
    std::string ts;
    std::string diags;
    index_t n;
    index_t k;
    std::tie(uplos, ts, diags, n) = p;

    auto BM_lambda = [&](benchmark::State& st, cublasHandle_t* cuda_handle_ptr,
                         std::string uplos, std::string ts, std::string diags,
                         index_t n, bool* success) {
      run<scalar_t>(st, cuda_handle_ptr, uplos, ts, diags, n, success);
    };
    benchmark::RegisterBenchmark(
        blas_benchmark::utils::get_name<benchmark_op, scalar_t>(
            uplos, ts, diags, n, blas_benchmark::utils::MEM_TYPE_USM)
            .c_str(),
        BM_lambda, cuda_handle_ptr, uplos, ts, diags, n, success)
        ->UseRealTime();
  }
}

namespace blas_benchmark {
void create_benchmark(blas_benchmark::Args& args,
                      cublasHandle_t* cuda_handle_ptr, bool* success) {
  BLAS_REGISTER_BENCHMARK(args, cuda_handle_ptr, success);
}
}  // namespace blas_benchmark
