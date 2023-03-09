/***************************************************************************
 *
 *  @license
 *  Copyright (C) 2022 Codeplay Software Limited
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
 *  @filename rotg.cpp
 *
 **************************************************************************/

#include "../utils.hpp"

template <typename scalar_t>
std::string get_name() {
  std::ostringstream str{};
  str << "BM_Rotg<" << blas_benchmark::utils::get_type_name<scalar_t>() << ">/";
  return str.str();
}

template <typename scalar_t, typename... args_t>
static inline void cublas_routine(args_t&&... args) {
  if constexpr (std::is_same_v<scalar_t, float>)
    cublasSrotg(std::forward<args_t>(args)...);
  else if constexpr (std::is_same_v<scalar_t, double>)
    cublasDrotg(std::forward(args)...);
  return;
}

template <typename scalar_t>
void run(benchmark::State& state, cublasHandle_t* cuda_handle_ptr,
         bool* success) {
  // Create data
  scalar_t a = blas_benchmark::utils::random_data<scalar_t>(1)[0];
  scalar_t b = blas_benchmark::utils::random_data<scalar_t>(1)[0];
  scalar_t c = blas_benchmark::utils::random_data<scalar_t>(1)[0];
  scalar_t s = blas_benchmark::utils::random_data<scalar_t>(1)[0];

  cublasHandle_t& cuda_handle = *cuda_handle_ptr;

  scalar_t* d_a = nullptr;
  scalar_t* d_b = nullptr;
  scalar_t* d_c = nullptr;
  scalar_t* d_s = nullptr;
  cudaMalloc(&d_a, sizeof(scalar_t));
  cudaMemcpyAsync(d_a, &a, sizeof(scalar_t), cudaMemcpyHostToDevice);
  cudaMalloc(&d_b, sizeof(scalar_t));
  cudaMemcpyAsync(d_b, &b, sizeof(scalar_t), cudaMemcpyHostToDevice);
  cudaMalloc(&d_c, sizeof(scalar_t));
  cudaMemcpyAsync(d_c, &c, sizeof(scalar_t), cudaMemcpyHostToDevice);
  cudaMalloc(&d_s, sizeof(scalar_t));
  cudaMemcpyAsync(d_s, &s, sizeof(scalar_t), cudaMemcpyHostToDevice);
  cudaDeviceSynchronize();

#ifdef BLAS_VERIFY_BENCHMARK
  // Run a first time with a verification of the results
  scalar_t a_ref = a;
  scalar_t b_ref = b;
  scalar_t c_ref = c;
  scalar_t s_ref = s;

  scalar_t a_verify = a;
  scalar_t b_verify = b;
  scalar_t c_verify = c;
  scalar_t s_verify = s;

  reference_blas::rotg(&a_ref, &b_ref, &c_ref, &s_ref);
  {
    scalar_t* d_a_verify = nullptr;
    scalar_t* d_b_verify = nullptr;
    scalar_t* d_c_verify = nullptr;
    scalar_t* d_s_verify = nullptr;
    cudaMalloc(&d_a_verify, sizeof(scalar_t));
    cudaMemcpyAsync(d_a_verify, &a, sizeof(scalar_t), cudaMemcpyHostToDevice);
    cudaMalloc(&d_b_verify, sizeof(scalar_t));
    cudaMemcpyAsync(d_b_verify, &b, sizeof(scalar_t), cudaMemcpyHostToDevice);
    cudaMalloc(&d_c_verify, sizeof(scalar_t));
    cudaMemcpyAsync(d_c_verify, &c, sizeof(scalar_t), cudaMemcpyHostToDevice);
    cudaMalloc(&d_s_verify, sizeof(scalar_t));
    cudaMemcpyAsync(d_s_verify, &s, sizeof(scalar_t), cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    cublasSetPointerMode_v2(cuda_handle, CUBLAS_POINTER_MODE_DEVICE);
    cublas_routine<scalar_t>(cuda_handle, d_a_verify, d_b_verify, d_c_verify,
                             d_s_verify);
    cudaDeviceSynchronize();
    cudaMemcpy(&a_verify, d_a_verify, sizeof(scalar_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(&b_verify, d_b_verify, sizeof(scalar_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(&c_verify, d_c_verify, sizeof(scalar_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(&s_verify, d_s_verify, sizeof(scalar_t), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    cudaFree(d_a_verify);
    cudaFree(d_b_verify);
    cudaFree(d_c_verify);
    cudaFree(d_s_verify);
  }
  const bool isAlmostEqual =
      utils::almost_equal<scalar_t, scalar_t>(a_verify, a_ref) &&
      utils::almost_equal<scalar_t, scalar_t>(b_verify, b_ref) &&
      utils::almost_equal<scalar_t, scalar_t>(c_verify, c_ref) &&
      utils::almost_equal<scalar_t, scalar_t>(s_verify, s_ref);

  if (!isAlmostEqual) {
    std::ostringstream err_stream;
    err_stream << "Value mismatch: " << std::endl;
    err_stream << "Got: "
               << "A: " << a_verify << " B: " << b_verify << " C: " << c_verify
               << " S: " << s_verify << std::endl;
    err_stream << "Expected: "
               << "A: " << a_ref << " B: " << b_ref << " C: " << c_ref
               << " S: " << s_ref << std::endl;
    const std::string& err_str = err_stream.str();
    state.SkipWithError(err_str.c_str());
    *success = false;
  };
#endif

  auto blas_warmup = [&]() -> void {
    cublas_routine<scalar_t>(cuda_handle, d_a, d_b, d_c, d_s);
    cudaDeviceSynchronize();
    return;
  };

  // Create a utility lambda describing the blas method that we want to run.
  auto blas_method_def = [&]() -> std::vector<cudaEvent_t> {
    cudaEvent_t start;
    cudaEvent_t stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    cublas_routine<scalar_t>(cuda_handle, d_a, d_b, d_c, d_s);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
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

  cudaFree(d_a);
  cudaFree(d_b);
  cudaFree(d_c);
  cudaFree(d_s);
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
