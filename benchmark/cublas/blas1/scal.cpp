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
 *  @filename scal.cpp
 *
 **************************************************************************/

#include "../utils.hpp"

template <typename scalar_t>
std::string get_name(int size) {
  std::ostringstream str{};
  str << "BM_Scal<" << blas_benchmark::utils::get_type_name<scalar_t>() << ">/";
  str << size;
  return str.str();
}

template <typename scalar_t, typename... args_t>
static inline void cublas_routine(args_t&&... args) {
  if constexpr (std::is_same_v<scalar_t, float>)
    cublasSscal(std::forward<args_t>(args)...);
  else if constexpr (std::is_same_v<scalar_t, double>)
    cublasDscal(std::forward(args)...);
  return;
}

template <typename scalar_t>
void run(benchmark::State& state, cublasHandle_t* cuda_handle_ptr, index_t size,
         bool* success) {
  // Google-benchmark counters are double.
  double size_d = static_cast<double>(size);
  state.counters["size"] = size_d;
  state.counters["n_fl_ops"] = size_d;
  state.counters["bytes_processed"] = 2 * size_d * sizeof(scalar_t);

  cublasHandle_t& cuda_handle = *cuda_handle_ptr;

  // Create data
  std::vector<scalar_t> v1 = blas_benchmark::utils::random_data<scalar_t>(size);
  auto alpha = blas_benchmark::utils::random_scalar<scalar_t>();

  scalar_t* d_inx = nullptr;
  cudaMalloc(&d_inx, size * sizeof(scalar_t));
  cudaMemcpy(d_inx, v1.data(), size * sizeof(scalar_t), cudaMemcpyHostToDevice);

  scalar_t* d_alpha = nullptr;
  cudaMalloc(&d_alpha, sizeof(scalar_t));
  cudaMemcpy(d_alpha, &alpha, sizeof(scalar_t), cudaMemcpyHostToDevice);
  cudaDeviceSynchronize();

#ifdef BLAS_VERIFY_BENCHMARK
  // Run a first time with a verification of the results
  std::vector<scalar_t> v1_ref = v1;
  reference_blas::scal(size, alpha, v1_ref.data(), 1);
  std::vector<scalar_t> vr_temp(size);
  {
    scalar_t* vr_temp_gpu = nullptr;
    cudaMalloc(&vr_temp_gpu, size * sizeof(scalar_t));
    cudaMemcpy(vr_temp_gpu, v1.data(), size * sizeof(scalar_t),
               cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    cublasSetPointerMode(cuda_handle, CUBLAS_POINTER_MODE_DEVICE);
    cublas_routine<scalar_t>(cuda_handle, size, d_alpha, vr_temp_gpu, 1);
    cudaDeviceSynchronize();
    cudaMemcpy(vr_temp.data(), vr_temp_gpu, size * sizeof(scalar_t),
               cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    cudaFree(vr_temp_gpu);
  }

  std::ostringstream err_stream;
  if (!utils::compare_vectors(vr_temp, v1_ref, err_stream, "")) {
    const std::string& err_str = err_stream.str();
    state.SkipWithError(err_str.c_str());
    *success = false;
  };
#endif

  auto blas_warmup = [&]() -> void {
    cublas_routine<scalar_t>(cuda_handle, size, d_alpha, d_inx, 1);
    cudaDeviceSynchronize();
    return;
  };

  auto blas_method_def = [&]() -> std::vector<cudaEvent_t> {
    cudaEvent_t start;
    cudaEvent_t stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    cublas_routine<scalar_t>(cuda_handle, size, d_alpha, d_inx, 1);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
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
  };

  blas_benchmark::utils::calc_avg_counters(state);

  cudaFree(d_inx);
  cudaFree(d_alpha);
}

template <typename scalar_t>
void register_benchmark(blas_benchmark::Args& args,
                        cublasHandle_t* cuda_handle_ptr, bool* success) {
  auto scal_params = blas_benchmark::utils::get_blas1_params(args);

  for (auto size : scal_params) {
    auto BM_lambda = [&](benchmark::State& st, cublasHandle_t* cuda_handle_ptr,
                         index_t size, bool* success) {
      run<scalar_t>(st, cuda_handle_ptr, size, success);
    };
    benchmark::RegisterBenchmark(get_name<scalar_t>(size).c_str(), BM_lambda,
                                 cuda_handle_ptr, size, success);
  }
}

namespace blas_benchmark {
void create_benchmark(blas_benchmark::Args& args,
                      cublasHandle_t* cuda_handle_ptr, bool* success) {
  BLAS_REGISTER_BENCHMARK(args, cuda_handle_ptr, success);
}
}  // namespace blas_benchmark
