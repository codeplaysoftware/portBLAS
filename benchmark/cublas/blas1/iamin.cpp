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
 *  @filename iamin.cpp
 *
 **************************************************************************/

#include "../utils.hpp"
#include "common/common_utils.hpp"

template <typename scalar_t>
std::string get_name(int size) {
  std::ostringstream str{};
  str << "BM_Iamin<" << blas_benchmark::utils::get_type_name<scalar_t>();
  str << ">/" << size;
  return str.str();
}

template <typename scalar_t, typename... args_t>
static inline void cublas_routine(args_t&&... args) {
  if constexpr (std::is_same_v<scalar_t, float>)
    CUBLAS_CHECK(cublasIsamin(std::forward<args_t>(args)...));
  else if constexpr (std::is_same_v<scalar_t, double>)
    CUBLAS_CHECK(cublasIdamin(std::forward(args)...));
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

  std::transform(std::begin(v1), std::end(v1), std::begin(v1), [](scalar_t v) {
    return utils::clamp_to_limits<scalar_t>(v);
  });

  scalar_t* d_inx = nullptr;
  CUDA_CHECK(cudaMalloc(&d_inx, size * sizeof(scalar_t)));
  CUDA_CHECK(cudaMemcpy(d_inx, v1.data(), size * sizeof(scalar_t),
                        cudaMemcpyHostToDevice));

  index_t* d_out = nullptr;
  CUDA_CHECK(cudaMalloc(&d_out, sizeof(index_t)));
  CUDA_CHECK(cudaDeviceSynchronize());

  cublasSetPointerMode_v2(cuda_handle, CUBLAS_POINTER_MODE_DEVICE);

#ifdef BLAS_VERIFY_BENCHMARK
  // Run a first time with a verification of the results
  index_t idx_ref =
      static_cast<index_t>(reference_blas::iamin(size, v1.data(), 1));
  index_t idx_temp;
  {
    index_t* idx_temp_gpu = nullptr;
    CUDA_CHECK(cudaMalloc(&idx_temp_gpu, sizeof(index_t)));
    CUDA_CHECK(cudaDeviceSynchronize());
    cublas_routine<scalar_t>(cuda_handle, size, d_inx, 1, idx_temp_gpu);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(&idx_temp, idx_temp_gpu, sizeof(index_t),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaFree(idx_temp_gpu));
  }

  // due to cuBLAS following FORTRAN indexes convention. I need to subtract
  // one from the given result.
  idx_temp -= 1;

  if (idx_temp != idx_ref) {
    std::ostringstream err_stream;
    err_stream << "Index mismatch: " << idx_temp << "; expected " << idx_ref;
    const std::string& err_str = err_stream.str();
    state.SkipWithError(err_str.c_str());
    *success = false;
  };
#endif

  auto blas_warmup = [&]() -> void {
    cublas_routine<scalar_t>(cuda_handle, size, d_inx, 1, d_out);
    CUDA_CHECK(cudaDeviceSynchronize());
    return;
  };

  auto blas_method_def = [&]() -> std::vector<cudaEvent_t> {
    cudaEvent_t start;
    cudaEvent_t stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    CUDA_CHECK(cudaEventRecord(start));
    cublas_routine<scalar_t>(cuda_handle, size, d_inx, 1, d_out);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
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
  CUDA_CHECK(cudaFree(d_inx));
  CUDA_CHECK(cudaFree(d_out));
}

template <typename scalar_t>
void register_benchmark(blas_benchmark::Args& args,
                        cublasHandle_t* cuda_handle_ptr, bool* success) {
  auto iamin_params = blas_benchmark::utils::get_blas1_params(args);

  for (auto size : iamin_params) {
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
