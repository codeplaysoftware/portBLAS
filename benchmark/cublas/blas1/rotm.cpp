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
 *  @filename rotm.cpp
 *
 **************************************************************************/

#include "../utils.hpp"

constexpr blas_benchmark::utils::Level1Op benchmark_op =
    blas_benchmark::utils::Level1Op::rotm;

template <typename scalar_t, typename... args_t>
static inline void cublas_routine(args_t&&... args) {
  if constexpr (std::is_same_v<scalar_t, float>) {
    CUBLAS_CHECK(cublasSrotm(std::forward<args_t>(args)...));
  } else if constexpr (std::is_same_v<scalar_t, double>) {
    CUBLAS_CHECK(cublasDrotm(std::forward<args_t>(args)...));
  }
  return;
}

template <typename scalar_t>
void run(benchmark::State& state, cublasHandle_t* cuda_handle_ptr, index_t size,
         bool* success) {
  // initialize the state label
  blas_benchmark::utils::set_benchmark_label<scalar_t>(state);

  // init Google-benchmark counters.
  blas_benchmark::utils::init_level_1_counters<
      blas_benchmark::utils::Level1Op::rotm, scalar_t>(state, size);

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

  blas_benchmark::utils::CUDAVector<scalar_t> d_v_x(size, x_v.data());
  blas_benchmark::utils::CUDAVector<scalar_t> d_v_y(size, y_v.data());
  blas_benchmark::utils::CUDAVector<scalar_t> d_param(param_size, param.data());

  CUBLAS_CHECK(cublasSetPointerMode(cuda_handle, CUBLAS_POINTER_MODE_DEVICE));

#ifdef BLAS_VERIFY_BENCHMARK
  // Run a first time with a verification of the results
  std::vector<scalar_t> x_v_ref = x_v;
  std::vector<scalar_t> y_v_ref = y_v;

  std::vector<scalar_t> x_v_verify(x_v);
  std::vector<scalar_t> y_v_verify(y_v);

  reference_blas::rotm(size, x_v_ref.data(), 1, y_v_ref.data(), 1,
                       param.data());
  {
    // Temp result on device copied back upon destruction
    blas_benchmark::utils::CUDAVector<scalar_t, true> d_v_x_verify(
        size, x_v_verify.data());
    blas_benchmark::utils::CUDAVector<scalar_t, true> d_v_y_verify(
        size, y_v_verify.data());
    cublas_routine<scalar_t>(cuda_handle, size, d_v_x_verify, 1, d_v_y_verify,
                             1, d_param);
  }
  // Verify results
  std::ostringstream err_stream;
  const bool isAlmostEqual = utils::compare_vectors<scalar_t, scalar_t>(
                                 x_v_verify, x_v_ref, err_stream, "") &&
                             utils::compare_vectors<scalar_t, scalar_t>(
                                 y_v_verify, y_v_ref, err_stream, "");

  if (!isAlmostEqual) {
    const std::string& err_str = err_stream.str();
    state.SkipWithError(err_str.c_str());
    *success = false;
  };
#endif

  auto blas_warmup = [&]() -> void {
    cublas_routine<scalar_t>(cuda_handle, size, d_v_x, 1, d_v_y, 1, d_param);
    return;
  };

  cudaEvent_t start;
  cudaEvent_t stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));

  auto blas_method_def = [&]() -> std::vector<cudaEvent_t> {
    CUDA_CHECK(cudaEventRecord(start));
    cublas_routine<scalar_t>(cuda_handle, size, d_v_x, 1, d_v_y, 1, d_param);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    return std::vector{start, stop};
  };

  // Warmup
  blas_benchmark::utils::warmup(blas_method_def);
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
  auto rotm_params = blas_benchmark::utils::get_blas1_params(args);

  for (auto size : rotm_params) {
    auto BM_lambda = [&](benchmark::State& st, cublasHandle_t* cuda_handle_ptr,
                         index_t size, bool* success) {
      run<scalar_t>(st, cuda_handle_ptr, size, success);
    };
    benchmark::RegisterBenchmark(
        blas_benchmark::utils::get_name<benchmark_op, scalar_t>(
            size, blas_benchmark::utils::MEM_TYPE_USM)
            .c_str(),
        BM_lambda, cuda_handle_ptr, size, success)
        ->UseRealTime();
  }
}

namespace blas_benchmark {
void create_benchmark(blas_benchmark::Args& args,
                      cublasHandle_t* cuda_handle_ptr, bool* success) {
  BLAS_REGISTER_BENCHMARK_FLOAT(args, cuda_handle_ptr, success);
}
}  // namespace blas_benchmark
