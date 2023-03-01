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
 *  @filename axpy.cpp
 *
 **************************************************************************/

#include "../utils.hpp"

template <typename scalar_t>
std::string get_name(int size) {
  std::ostringstream str{};
  str << "BM_Axpy" << blas_benchmark::utils::get_type_name<scalar_t>() << ">/";
  str << size;
  return str.str();
}

template <typename scalar_t, typename... args_t>
static inline void rocblas_saxpy_f(args_t&&... args) {
  if constexpr (std::is_same_v<scalar_t, float>) {
    CHECK_ROCBLAS_STATUS(rocblas_saxpy(std::forward<args_t>(args)...));
  } else if constexpr (std::is_same_v<scalar_t, double>) {
    CHECK_ROCBLAS_STATUS(rocblas_daxpy(std::forward<args_t>(args)...));
  }
  return;
}

template <typename scalar_t>
void run(benchmark::State& state, rocblas_handle& rb_handle_ptr, index_t size,
         bool* success) {
  // Google-benchmark counters are double.
  double size_d = static_cast<double>(size);
  state.counters["size"] = size_d;
  state.counters["n_fl_ops"] = 2 * size_d;
  state.counters["bytes_processed"] = 3 * size_d * sizeof(scalar_t);

  // Create data
  std::vector<scalar_t> v1 = blas_benchmark::utils::random_data<scalar_t>(size);
  std::vector<scalar_t> v2 = blas_benchmark::utils::random_data<scalar_t>(size);
  auto alpha = blas_benchmark::utils::random_scalar<scalar_t>();

  // Initialize HIP & rocBLAS errors for checking the HIP/Rocblas APIs status
  hipError_t herror = hipSuccess;
  rocblas_status rstatus = rocblas_status_success;

  {
    // Device memory allocation
    blas_benchmark::utils::DeviceVector<scalar_t> d_v1(size);
    blas_benchmark::utils::DeviceVector<scalar_t> d_v2(size);

    // Copy data (D2H)
    herror = hipMemcpy(d_v1, v1.data(), sizeof(scalar_t) * size,
                       hipMemcpyHostToDevice);
    CHECK_HIP_ERROR(herror);
    herror = hipMemcpy(d_v1, v1.data(), sizeof(scalar_t) * size,
                       hipMemcpyHostToDevice);
    CHECK_HIP_ERROR(herror);

#ifdef BLAS_VERIFY_BENCHMARK
    // Run a first time with a verification of the results
    std::vector<scalar_t> y_ref = v2;
    reference_blas::axpy(size, alpha, v1.data(), 1, y_ref.data(), 1);
    std::vector<scalar_t> y_temp = v2;

    {
      blas_benchmark::utils::DeviceVector<scalar_t> y_temp_gpu(size);
      herror = hipMemcpy(y_temp_gpu, y_temp.data(), sizeof(scalar_t) * size,
                         hipMemcpyHostToDevice);
      CHECK_HIP_ERROR(herror);

      // Enable passing alpha parameter from pointer to host memory
      rstatus =
          rocblas_set_pointer_mode(rb_handle_ptr, rocblas_pointer_mode_host);
      CHECK_ROCBLAS_STATUS(rstatus);

      rocblas_saxpy_f<scalar_t>(rb_handle_ptr, size, &alpha, d_v1, 1,
                                y_temp_gpu, 1);

      herror = hipMemcpy(y_temp.data(), y_temp_gpu, sizeof(scalar_t) * size,
                         hipMemcpyDeviceToHost);

      CHECK_HIP_ERROR(herror);
    }

    std::ostringstream err_stream;
    if (!utils::compare_vectors(y_temp, y_ref, err_stream, "")) {
      const std::string& err_str = err_stream.str();
      state.SkipWithError(err_str.c_str());
      *success = false;
    };
#endif

    auto blas_warmup = [&]() -> void {
      rocblas_saxpy_f<scalar_t>(rb_handle_ptr, size, &alpha, d_v1, 1, d_v2, 1);
      // hipDeviceSynchronize();
      CHECK_HIP_ERROR(hipStreamSynchronize(NULL));
      return;
    };

    auto blas_method_def = [&]() -> std::vector<hipEvent_t> {
      hipEvent_t start;
      hipEvent_t stop;
      CHECK_HIP_ERROR(hipEventCreate(&start));
      CHECK_HIP_ERROR(hipEventCreate(&stop));

      // Assuming the NULL (default) stream is the only one in use
      CHECK_HIP_ERROR(hipEventRecord(start, NULL));

      rocblas_saxpy_f<scalar_t>(rb_handle_ptr, size, &alpha, d_v1, 1, d_v2, 1);

      CHECK_HIP_ERROR(hipEventRecord(stop, NULL));

      CHECK_HIP_ERROR(hipEventSynchronize(stop));

      return std::vector{start, stop};
    };

    // Warmup
    blas_benchmark::utils::warmup(blas_warmup);

    blas_benchmark::utils::init_counters(state);

    // Measure
    for (auto _ : state) {
      // Run
      std::tuple<double, double> times =
          blas_benchmark::utils::timef_hip(blas_method_def);

      // Report
      blas_benchmark::utils::update_counters(state, times);
    }

    blas_benchmark::utils::calc_avg_counters(state);
  }  // release device memory via utils::DeviceVector destructors
};

template <typename scalar_t>
void register_benchmark(blas_benchmark::Args& args,
                        rocblas_handle& rb_handle_ptr, bool* success) {
  auto blas1_params = blas_benchmark::utils::get_blas1_params(args);

  for (auto size : blas1_params) {
    auto BM_lambda = [&](benchmark::State& st, rocblas_handle rb_handle_ptr,
                         index_t size, bool* success) {
      run<scalar_t>(st, rb_handle_ptr, size, success);
    };
    benchmark::RegisterBenchmark(get_name<scalar_t>(size).c_str(), BM_lambda,
                                 rb_handle_ptr, size, success);
  }
}

namespace blas_benchmark {
void create_benchmark(blas_benchmark::Args& args, rocblas_handle& rb_handle_ptr,
                      bool* success) {
  BLAS_REGISTER_BENCHMARK(args, rb_handle_ptr, success);
}
}  // namespace blas_benchmark