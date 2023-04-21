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

template <typename scalar_t, typename... args_t>
static inline void cublas_routine(args_t&&... args) {
  if constexpr (std::is_same_v<scalar_t, float>) {
    CUBLAS_CHECK(cublasSsyr2(std::forward<args_t>(args)...));
  } else if constexpr (std::is_same_v<scalar_t, double>) {
    CUBLAS_CHECK(cublasDsyr2(std::forward<args_t>(args)...));
  }
  return;
}

template <typename scalar_t>
void run(benchmark::State& state, cublasHandle_t* cuda_handle_ptr,
         std::string uplo, index_t n, scalar_t alpha, bool* success) {
  // Standard test setup.
  const char* uplo_str = uplo.c_str();

  index_t lda = n;
  index_t incX = 1;
  index_t incY = 1;

  blas_benchmark::utils::init_level_2_counters<
      blas_benchmark::utils::Level2Op::syr2, scalar_t>(state, "n", 0, 0, n);

  cublasHandle_t& cuda_handle = *cuda_handle_ptr;

  const int v_x_size = 1 + (n - 1) * std::abs(incX);
  const int v_y_size = 1 + (n - 1) * std::abs(incY);

  // Input matrix/vector, output vector.
  std::vector<scalar_t> m_a =
      blas_benchmark::utils::random_data<scalar_t>(n * n);
  std::vector<scalar_t> v_x =
      blas_benchmark::utils::random_data<scalar_t>(v_x_size);
  std::vector<scalar_t> v_y =
      blas_benchmark::utils::random_data<scalar_t>(v_y_size);

  blas_benchmark::utils::CUDAVector<scalar_t> m_a_gpu(n * n, m_a.data());
  blas_benchmark::utils::CUDAVector<scalar_t> v_x_gpu(v_x_size, v_x.data());
  blas_benchmark::utils::CUDAVector<scalar_t> v_y_gpu(v_y_size, v_y.data());

  cublasFillMode_t c_uplo =
      (*uplo_str == 'u') ? CUBLAS_FILL_MODE_UPPER : CUBLAS_FILL_MODE_LOWER;

#ifdef BLAS_VERIFY_BENCHMARK
  // Run a first time with a verification of the results
  std::vector<scalar_t> m_a_ref = m_a;
  reference_blas::syr2(uplo_str, n, alpha, v_x.data(), incX, v_y.data(), incY,
                       m_a_ref.data(), lda);
  std::vector<scalar_t> m_a_temp(m_a);

  {
    blas_benchmark::utils::CUDAVector<scalar_t, true> m_a_temp_gpu(
        n * n, m_a_temp.data());
    cublas_routine<scalar_t>(cuda_handle, c_uplo, n, &alpha, v_x_gpu, incX,
                             v_y_gpu, incY, m_a_temp_gpu, lda);
  }

  std::ostringstream err_stream;
  if (!utils::compare_vectors(m_a_temp, m_a_ref, err_stream, "")) {
    const std::string& err_str = err_stream.str();
    state.SkipWithError(err_str.c_str());
    *success = false;
  };

#endif

  auto blas_warmup = [&]() -> void {
    cublas_routine<scalar_t>(cuda_handle, c_uplo, n, &alpha, v_x_gpu, incX,
                             v_y_gpu, incY, m_a_gpu, lda);
    return;
  };

  cudaEvent_t start;
  cudaEvent_t stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  auto blas_method_def = [&]() -> std::vector<cudaEvent_t> {
    cudaEventRecord(start);
    cublas_routine<scalar_t>(cuda_handle, c_uplo, n, &alpha, v_x_gpu, incX,
                             v_y_gpu, incY, m_a_gpu, lda);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
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
  // syr2 use the same parameters so reuse syr function
  auto syr2_params = blas_benchmark::utils::get_syr_params<scalar_t>(args);

  for (auto p : syr2_params) {
    std::string uplo;
    index_t n;
    scalar_t alpha;
    std::tie(uplo, n, alpha) = p;

    auto BM_lambda = [&](benchmark::State& st, cublasHandle_t* cuda_handle_ptr,
                         std::string uplo, index_t n, scalar_t alpha,
                         bool* success) {
      run<scalar_t>(st, cuda_handle_ptr, uplo, n, alpha, success);
    };
    benchmark::RegisterBenchmark(get_name<scalar_t>(uplo, n, alpha).c_str(),
                                 BM_lambda, cuda_handle_ptr, uplo, n, alpha,
                                 success)
        ->UseRealTime();
  }
}

namespace blas_benchmark {
void create_benchmark(blas_benchmark::Args& args,
                      cublasHandle_t* cuda_handle_ptr, bool* success) {
  BLAS_REGISTER_BENCHMARK(args, cuda_handle_ptr, success);
}
}  // namespace blas_benchmark
