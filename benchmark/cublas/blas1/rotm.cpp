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
 *  @filename rotm.cpp
 *
 **************************************************************************/

#include "../utils.hpp"

template <typename scalar_t>
std::string get_name(int size) {
  std::ostringstream str{};
  str << "BM_Rotm<" << blas_benchmark::utils::get_type_name<scalar_t>() << ">/";
  str << size;
  return str.str();
}

template <typename scalar_t, typename... args_t>
static inline void cublas_routine(args_t&&... args) {
  if constexpr (std::is_same_v<scalar_t, float>)
    cublasSrotm(std::forward<args_t>(args)...);
  else if constexpr (std::is_same_v<scalar_t, double>)
    cublasDrotm(std::forward(args)...);
  return;
}

template <typename scalar_t>
void run(benchmark::State& state, cublasHandle_t* cuda_handle_ptr, index_t size,
         bool* success) {
  // Google-benchmark counters are double.
  double size_d = static_cast<double>(size);

  state.counters["size"] = size_d;
  state.counters["n_fl_ops"] = 2 * size_d;
  state.counters["bytes_processed"] = 2 * size_d * sizeof(scalar_t);

  cublasHandle_t& cuda_handle = *cuda_handle_ptr;

  // Create data
  constexpr size_t param_size = 5;
  std::vector<scalar_t> param =
      blas_benchmark::utils::random_data<scalar_t>(param_size);
  param[0] =
      static_cast<scalar_t>(-1.0);  // Use -1.0 flag to use the whole matrix

  std::vector<scalar_t> x_v =
      blas_benchmark::utils::random_data<scalar_t>(size);
  std::vector<scalar_t> y_v =
      blas_benchmark::utils::random_data<scalar_t>(size);

  scalar_t* d_v_x = nullptr;
  scalar_t* d_v_y = nullptr;
  scalar_t* d_param = nullptr;
  cudaMalloc(&d_v_x, size * sizeof(scalar_t));
  cudaMalloc(&d_v_y, size * sizeof(scalar_t));
  cudaMalloc(&d_param, param_size * sizeof(scalar_t));
  cudaMemcpy(d_v_x, x_v.data(), size * sizeof(scalar_t),
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_v_y, y_v.data(), size * sizeof(scalar_t),
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_param, param.data(), param_size * sizeof(scalar_t),
             cudaMemcpyHostToDevice);
  cudaDeviceSynchronize();

#ifdef BLAS_VERIFY_BENCHMARK
  // Run a first time with a verification of the results
  std::vector<scalar_t> x_v_ref = x_v;
  std::vector<scalar_t> y_v_ref = y_v;

  std::vector<scalar_t> x_v_verify(size);
  std::vector<scalar_t> y_v_verify(size);

  reference_blas::rotm(size, x_v_ref.data(), 1, y_v_ref.data(), 1,
                       param.data());
  {
    scalar_t* d_v_x_verify = nullptr;
    scalar_t* d_v_y_verify = nullptr;
    cudaMalloc(&d_v_x_verify, size * sizeof(scalar_t));
    cudaMalloc(&d_v_y_verify, size * sizeof(scalar_t));
    cudaMemcpy(d_v_x_verify, x_v.data(), size * sizeof(scalar_t),
               cudaMemcpyHostToDevice);
    cudaMemcpy(d_v_y_verify, y_v.data(), size * sizeof(scalar_t),
               cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    cublasSetPointerMode_v2(cuda_handle, CUBLAS_POINTER_MODE_DEVICE);
    cublas_routine<scalar_t>(cuda_handle, size, d_v_x_verify, 1, d_v_y_verify,
                             1, d_param);
    cudaDeviceSynchronize();
    cudaMemcpy(x_v_verify.data(), d_v_x_verify, size * sizeof(scalar_t),
               cudaMemcpyDeviceToHost);
    cudaMemcpy(y_v_verify.data(), d_v_y_verify, size * sizeof(scalar_t),
               cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    cudaFree(d_v_x_verify);
    cudaFree(d_v_y_verify);
  }
  // Verify results
  std::ostringstream err_stream;
  const bool isAlmostEqual = utils::compare_vectors<scalar_t, scalar_t>(
                                 x_v_ref, x_v_verify, err_stream, "") &&
                             utils::compare_vectors<scalar_t, scalar_t>(
                                 y_v_ref, y_v_verify, err_stream, "");

  if (!isAlmostEqual) {
    const std::string& err_str = err_stream.str();
    state.SkipWithError(err_str.c_str());
    *success = false;
  };
#endif

  auto blas_warmup = [&]() -> void {
    cublas_routine<scalar_t>(cuda_handle, size, d_v_x, 1, d_v_y, 1, d_param);
    cudaDeviceSynchronize();
    return;
  };

  auto blas_method_def = [&]() -> std::vector<cudaEvent_t> {
    cudaEvent_t start;
    cudaEvent_t stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    cublas_routine<scalar_t>(cuda_handle, size, d_v_x, 1, d_v_y, 1, d_param);
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
  }

  blas_benchmark::utils::calc_avg_counters(state);

  cudaFree(d_v_x);
  cudaFree(d_v_y);
  cudaFree(d_param);
}

template <typename scalar_t>
void register_benchmark(blas_benchmark::Args& args,
                        cublasHandle_t* cuda_handle_ptr, bool* success) {
  auto rotm_params = blas_benchmark::utils::get_blas1_params(args);

  for (auto size : rotm_params) {
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
  BLAS_REGISTER_BENCHMARK_FLOAT(args, cuda_handle_ptr, success);
}
}  // namespace blas_benchmark
