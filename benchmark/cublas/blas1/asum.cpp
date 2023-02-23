/***************************************************************************
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
 *  @filename asum.cpp
 *
 **************************************************************************/

#include "../utils.hpp"

template <typename scalar_t>
std::string get_name(int size) {
  std::ostringstream str{};
  str << "BM_Asum<" << blas_benchmark::utils::get_type_name<scalar_t>() << ">/";
  str << size;
  return str.str();
}

template <typename scalar_t, typename... args_t>
static inline void cublas_routine(args_t&&... args) {
  if constexpr (std::is_same_v<scalar_t, float>)
    cublasSasum(std::forward<args_t>(args)...);
  else if constexpr (std::is_same_v<scalar_t, double>)
    cublasDasum(std::forward(args)...);
  return;
}

template <typename scalar_t>
void run(benchmark::State& state, cublasHandle_t* cuda_handle_ptr, index_t size,
         bool* success) {
  // Google-benchmark counters are double.
  double size_d = static_cast<double>(size);
  state.counters["size"] = size_d;
  state.counters["n_fl_ops"] = 2.0 * size_d;
  state.counters["bytes_processed"] = size_d * sizeof(scalar_t);

  cublasHandle_t& cuda_handle = *cuda_handle_ptr;
  // Create data
  std::vector<scalar_t> v1 = blas_benchmark::utils::random_data<scalar_t>(size);
  // We need to guarantee that cl::sycl::half can hold the sum
  // of x_v without overflow by making sum(x_v) to be 1.0
  std::transform(std::begin(v1), std::end(v1), std::begin(v1),
                 [=](scalar_t x) { return x / v1.size(); });

  scalar_t vr;

  scalar_t* d_x = nullptr;
  cudaMalloc(&d_x, size * sizeof(scalar_t));
  scalar_t* d_r = nullptr;
  cudaMalloc(&d_r, sizeof(scalar_t));

  cudaMemcpyAsync(d_x, v1.data(), size * sizeof(scalar_t),
                  cudaMemcpyHostToDevice);

#ifdef BLAS_VERIFY_BENCHMARK
  // Run a first time with a verification of the results
  scalar_t vr_ref = reference_blas::asum(size, v1.data(), 1);
  scalar_t vr_temp = 0;
  {
    scalar_t* vr_temp_gpu = nullptr;
    cudaMalloc(&vr_temp_gpu, sizeof(scalar_t));
    cudaDeviceSynchronize();
    cublasSetPointerMode(cuda_handle, CUBLAS_POINTER_MODE_DEVICE);
    cublas_routine<scalar_t>(cuda_handle, size, d_x, 1, vr_temp_gpu);
    cudaDeviceSynchronize();
    cudaMemcpy(&vr_temp, vr_temp_gpu, sizeof(scalar_t), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    cudaFree(vr_temp_gpu);
  }

  if (!utils::almost_equal<scalar_t>(vr_temp, vr_ref)) {
    std::ostringstream err_stream;
    err_stream << "Value mismatch: " << vr_temp << "; expected " << vr_ref;
    const std::string& err_str = err_stream.str();
    state.SkipWithError(err_str.c_str());
    *success = false;
  };
#endif

  auto blas_warmup = [&]() -> void {
    cublas_routine<scalar_t>(cuda_handle, size, d_x, 1, d_r);
    cudaDeviceSynchronize();
    return;
  };

  auto blas_method_def = [&]() -> std::vector<cudaEvent_t> {
    cudaEvent_t start;
    cudaEvent_t stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    cublas_routine<scalar_t>(cuda_handle, size, d_x, 1, d_r);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    return std::vector{start, stop};
  };

  // Warmup
  blas_benchmark::utils::warmup(blas_warmup);

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

  cudaFree(d_x);
  cudaFree(d_r);
}

template <typename scalar_t>
void register_benchmark(blas_benchmark::Args& args,
                        cublasHandle_t* cuda_handle_ptr, bool* success) {
  auto asum_params = blas_benchmark::utils::get_blas1_params(args);

  for (auto size : asum_params) {
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
