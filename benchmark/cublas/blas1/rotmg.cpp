/***************************************************************************
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
 *  @filename rotmg.cpp
 *
 **************************************************************************/

#include "../utils.hpp"
#include "common/float_comparison.hpp"

template <typename scalar_t>
std::string get_name() {
  std::ostringstream str{};
  str << "BM_Rotmg<" << blas_benchmark::utils::get_type_name<scalar_t>()
      << ">/";
  return str.str();
}

template <typename scalar_t, typename... args_t>
static inline void cublas_routine(args_t&&... args) {
  if constexpr (std::is_same_v<scalar_t, float>)
    CUBLAS_CHECK(cublasSrotmg(std::forward<args_t>(args)...));
  else if constexpr (std::is_same_v<scalar_t, double>)
    CUBLAS_CHECK(cublasDrotmg(std::forward(args)...));
  return;
}

template <typename scalar_t>
void run(benchmark::State& state, cublasHandle_t* cuda_handle_ptr,
         bool* success) {
  // Create data
  constexpr size_t param_size = 5;
  std::vector<scalar_t> param = std::vector<scalar_t>(param_size);
  scalar_t d1 = blas_benchmark::utils::random_data<scalar_t>(1)[0];
  scalar_t d2 = blas_benchmark::utils::random_data<scalar_t>(1)[0];
  scalar_t x1 = blas_benchmark::utils::random_data<scalar_t>(1)[0];
  scalar_t y1 = blas_benchmark::utils::random_data<scalar_t>(1)[0];

  cublasHandle_t& cuda_handle = *cuda_handle_ptr;

  scalar_t* d_d1 = nullptr;
  scalar_t* d_d2 = nullptr;
  scalar_t* d_x1 = nullptr;
  scalar_t* d_y1 = nullptr;
  scalar_t* d_param = nullptr;
  CUDA_CHECK(cudaMalloc(&d_d1, sizeof(scalar_t)));
  CUDA_CHECK(cudaMalloc(&d_d2, sizeof(scalar_t)));
  CUDA_CHECK(cudaMalloc(&d_x1, sizeof(scalar_t)));
  CUDA_CHECK(cudaMalloc(&d_y1, sizeof(scalar_t)));
  CUDA_CHECK(cudaMalloc(&d_param, param_size * sizeof(scalar_t)));

  CUDA_CHECK(cudaMemcpy(d_d1, &d1, sizeof(scalar_t), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_d2, &d2, sizeof(scalar_t), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_x1, &x1, sizeof(scalar_t), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(d_y1, &y1, sizeof(scalar_t), cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaDeviceSynchronize());

  CUBLAS_CHECK(
      cublasSetPointerMode_v2(cuda_handle, CUBLAS_POINTER_MODE_DEVICE));

#ifdef BLAS_VERIFY_BENCHMARK
  // Run a first time with a verification of the results
  scalar_t d1_ref = d1;
  scalar_t d2_ref = d2;
  scalar_t x1_ref = x1;
  scalar_t y1_ref = y1;
  std::vector<scalar_t> param_ref = param;

  scalar_t d1_verify;
  scalar_t d2_verify;
  scalar_t x1_verify;
  scalar_t y1_verify;
  std::vector<scalar_t> param_verify(param_size);
  reference_blas::rotmg(&d1_ref, &d2_ref, &x1_ref, &y1_ref, param_ref.data());

  {
    scalar_t* d_d1_verify;
    scalar_t* d_d2_verify;
    scalar_t* d_x1_verify;
    scalar_t* d_y1_verify;
    scalar_t* d_param_verify;
    CUDA_CHECK(cudaMalloc(&d_d1_verify, sizeof(scalar_t)));
    CUDA_CHECK(cudaMalloc(&d_d2_verify, sizeof(scalar_t)));
    CUDA_CHECK(cudaMalloc(&d_x1_verify, sizeof(scalar_t)));
    CUDA_CHECK(cudaMalloc(&d_y1_verify, sizeof(scalar_t)));
    CUDA_CHECK(cudaMalloc(&d_param_verify, param_size * sizeof(scalar_t)));

    CUDA_CHECK(
        cudaMemcpy(d_d1_verify, &d1, sizeof(scalar_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(
        cudaMemcpy(d_d2_verify, &d2, sizeof(scalar_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(
        cudaMemcpy(d_x1_verify, &x1, sizeof(scalar_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(
        cudaMemcpy(d_y1_verify, &y1, sizeof(scalar_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_param_verify, param.data(),
                          param_size * sizeof(scalar_t),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaDeviceSynchronize());
    cublas_routine<scalar_t>(cuda_handle, d_d1_verify, d_d2_verify, d_x1_verify,
                             d_y1_verify, d_param_verify);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(&d1_verify, d_d1_verify, sizeof(scalar_t),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&d2_verify, d_d2_verify, sizeof(scalar_t),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&x1_verify, d_x1_verify, sizeof(scalar_t),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&y1_verify, d_y1_verify, sizeof(scalar_t),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(param_verify.data(), d_param_verify,
                          param_size * sizeof(scalar_t),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaFree(d_d1_verify));
    CUDA_CHECK(cudaFree(d_d2_verify));
    CUDA_CHECK(cudaFree(d_x1_verify));
    CUDA_CHECK(cudaFree(d_y1_verify));
    CUDA_CHECK(cudaFree(d_param_verify));
  }

  std::ostringstream err_stream;
  const bool isAlmostEqual =
      utils::almost_equal(d1_verify, d1_ref) &&
      utils::almost_equal(d2_verify, d2_ref) &&
      utils::almost_equal(x1_verify, x1_ref) &&
      utils::almost_equal(y1_verify, y1_ref) &&
      utils::compare_vectors(param_verify, param_ref, err_stream, "");

  if (!isAlmostEqual) {
    err_stream << "Value mismatch." << std::endl;
    const std::string& err_str = err_stream.str();
    state.SkipWithError(err_str.c_str());
    *success = false;
  };
#endif

  auto blas_warmup = [&]() -> void {
    cublas_routine<scalar_t>(cuda_handle, d_d1, d_d2, d_x1, d_y1, d_param);
    CUDA_CHECK(cudaDeviceSynchronize());
    return;
  };

  // Create a utility lambda describing the blas method that we want to run.
  auto blas_method_def = [&]() -> std::vector<cudaEvent_t> {
    cudaEvent_t start;
    cudaEvent_t stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));
    cublas_routine<scalar_t>(cuda_handle, d_d1, d_d2, d_x1, d_y1, d_param);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    return std::vector{start, stop};
  };

  // Warm up to avoid benchmarking data transfer
  blas_benchmark::utils::warmup(blas_warmup);

  blas_benchmark::utils::init_counters(state);

  // Measure
  for (auto _ : state) {
    std::tuple<double, double> times =
        blas_benchmark::utils::timef(blas_method_def);

    // Report
    blas_benchmark::utils::update_counters(state, times);
  }

  blas_benchmark::utils::calc_avg_counters(state);
  CUDA_CHECK(cudaFree(d_d1));
  CUDA_CHECK(cudaFree(d_d2));
  CUDA_CHECK(cudaFree(d_x1));
  CUDA_CHECK(cudaFree(d_y1));
  CUDA_CHECK(cudaFree(d_param));
};

template <typename scalar_t>
void register_benchmark(blas_benchmark::Args& args,
                        cublasHandle_t* cuda_handle_ptr, bool* success) {
  auto BM_lambda = [&](benchmark::State& st, cublasHandle_t* cuda_handle_ptr,
                       bool* success) {
    run<scalar_t>(st, cuda_handle_ptr, success);
  };
  benchmark::RegisterBenchmark(get_name<scalar_t>().c_str(), BM_lambda,
                               cuda_handle_ptr, success);
}

namespace blas_benchmark {
void create_benchmark(blas_benchmark::Args& args,
                      cublasHandle_t* cuda_handle_ptr, bool* success) {
  BLAS_REGISTER_BENCHMARK(args, cuda_handle_ptr, success);
}
}  // namespace blas_benchmark
