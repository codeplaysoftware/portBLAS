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
 *  portBLAS: BLAS implementation using SYCL
 *
 *  @filename iamin.cpp
 *
 **************************************************************************/

#include "../utils.hpp"

constexpr blas_benchmark::utils::Level1Op benchmark_op =
    blas_benchmark::utils::Level1Op::iamin;

template <typename scalar_t, typename... args_t>
static inline void rocblas_iamin_f(args_t&&... args) {
  if constexpr (std::is_same_v<scalar_t, float>) {
    CHECK_ROCBLAS_STATUS(rocblas_isamin(std::forward<args_t>(args)...));
  } else if constexpr (std::is_same_v<scalar_t, double>) {
    CHECK_ROCBLAS_STATUS(rocblas_idamin(std::forward<args_t>(args)...));
  }
  return;
}

template <typename scalar_t>
void run(benchmark::State& state, rocblas_handle& rb_handle, index_t size,
         bool* success) {
  // initialize the state label
  blas_benchmark::utils::set_benchmark_label<scalar_t>(state);

  // Google-benchmark counters are double.
  blas_benchmark::utils::init_level_1_counters<
      blas_benchmark::utils::Level1Op::iamin, scalar_t>(state, size);

  // Create data
  std::vector<scalar_t> v1 = blas_benchmark::utils::random_data<scalar_t>(size);
  int idx;

  {
    // Device memory allocation & H2D Copy
    blas_benchmark::utils::HIPVector<scalar_t> d_v1(size, v1.data());
    blas_benchmark::utils::HIPScalar<int> d_idx(idx);

#ifdef BLAS_VERIFY_BENCHMARK
    // Reference iamin
    index_t idx_ref =
        static_cast<index_t>(reference_blas::iamin(size, v1.data(), 1));

    // Rocblas verification iamin
    int idx_temp = -1;

    {
      // Temp result on device
      blas_benchmark::utils::HIPScalar<int, true> d_idx_temp(idx_temp);
      // Rocblas routine call
      rocblas_iamin_f<scalar_t>(rb_handle, size, d_v1, 1, d_idx_temp);
    }

    // xMIN follows 1-based indexing in rocBLAS.
    idx_temp -= 1;

    // Verify result
    if (idx_temp != idx_ref) {
      std::ostringstream err_stream;
      err_stream << "Index mismatch: " << idx_temp << "; expected " << idx_ref;
      const std::string& err_str = err_stream.str();
      state.SkipWithError(err_str.c_str());
      *success = false;
    };
#endif

    auto blas_warmup = [&]() -> void {
      rocblas_iamin_f<scalar_t>(rb_handle, size, d_v1, 1, d_idx);
      return;
    };

    hipEvent_t start, stop;
    CHECK_HIP_ERROR(hipEventCreate(&start));
    CHECK_HIP_ERROR(hipEventCreate(&stop));

    auto blas_method_def = [&]() -> std::vector<hipEvent_t> {
      CHECK_HIP_ERROR(hipEventRecord(start, NULL));
      rocblas_iamin_f<scalar_t>(rb_handle, size, d_v1, 1, d_idx);
      CHECK_HIP_ERROR(hipEventRecord(stop, NULL));
      CHECK_HIP_ERROR(hipEventSynchronize(stop));
      return std::vector{start, stop};
    };

    // Warm up to avoid benchmarking data transfer
    blas_benchmark::utils::warmup(blas_warmup);
    CHECK_HIP_ERROR(hipStreamSynchronize(NULL));

    blas_benchmark::utils::init_counters(state);

    // Measure
    for (auto _ : state) {
      std::tuple<double, double> times =
          blas_benchmark::utils::timef_hip(blas_method_def);

      // Report
      blas_benchmark::utils::update_counters(state, times);
    }

    state.SetItemsProcessed(state.iterations() * state.counters["n_fl_ops"]);
    state.SetBytesProcessed(state.iterations() *
                            state.counters["bytes_processed"]);

    blas_benchmark::utils::calc_avg_counters(state);

    CHECK_HIP_ERROR(hipEventDestroy(start));
    CHECK_HIP_ERROR(hipEventDestroy(stop));
  }  // release device memory via utils::DeviceVector destructors
};

template <typename scalar_t>
void register_benchmark(blas_benchmark::Args& args, rocblas_handle& rb_handle,
                        bool* success) {
  auto iamin_params = blas_benchmark::utils::get_blas1_params(args);

  for (auto size : iamin_params) {
    auto BM_lambda = [&](benchmark::State& st, rocblas_handle rb_handle,
                         index_t size, bool* success) {
      run<scalar_t>(st, rb_handle, size, success);
    };
    benchmark::RegisterBenchmark(
        blas_benchmark::utils::get_name<benchmark_op, scalar_t>(
            size, blas_benchmark::utils::MEM_TYPE_USM)
            .c_str(),
        BM_lambda, rb_handle, size, success)
        ->UseRealTime();
  }
}

namespace blas_benchmark {
void create_benchmark(blas_benchmark::Args& args, rocblas_handle& rb_handle,
                      bool* success) {
  BLAS_REGISTER_BENCHMARK(args, rb_handle, success);
}
}  // namespace blas_benchmark
