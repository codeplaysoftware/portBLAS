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
 *  @filename axpy_batch.cpp
 *
 **************************************************************************/

#include "../utils.hpp"
#include "common/common_utils.hpp"

constexpr blas_benchmark::utils::ExtensionOp benchmark_op =
    blas_benchmark::utils::ExtensionOp::axpy_batch;

template <typename scalar_t, typename... args_t>
static inline void rocblas_axpy_strided_batched_f(args_t&&... args) {
  if constexpr (std::is_same_v<scalar_t, float>) {
    CHECK_ROCBLAS_STATUS(
        rocblas_saxpy_strided_batched(std::forward<args_t>(args)...));
  } else if constexpr (std::is_same_v<scalar_t, double>) {
    CHECK_ROCBLAS_STATUS(
        rocblas_daxpy_strided_batched(std::forward<args_t>(args)...));
  }
  return;
}

template <typename scalar_t>
void run(benchmark::State& state, rocblas_handle& rb_handle, index_t size,
         scalar_t alpha, index_t inc_x, index_t inc_y, index_t stride_x_mul,
         index_t stride_y_mul, index_t batch_size, bool* success) {
  // initialize the state label
  blas_benchmark::utils::set_benchmark_label<scalar_t>(state);

  // Google-benchmark counters are double.
  blas_benchmark::utils::init_extension_counters<benchmark_op, scalar_t>(
      state, size, batch_size);

  const auto stride_x{size * std::abs(inc_x) * stride_x_mul};
  const auto stride_y{size * std::abs(inc_y) * stride_y_mul};

  const index_t size_x{stride_x * batch_size};
  const index_t size_y{stride_y * batch_size};
  // Create data
  std::vector<scalar_t> vx =
      blas_benchmark::utils::random_data<scalar_t>(size_x);
  std::vector<scalar_t> vy =
      blas_benchmark::utils::random_data<scalar_t>(size_y);

  blas_benchmark::utils::HIPVector<scalar_t> inx(size_x, vx.data());
  blas_benchmark::utils::HIPVector<scalar_t> iny(size_y, vy.data());

#ifdef BLAS_VERIFY_BENCHMARK
  // Run a first time with a verification of the results
  std::vector<scalar_t> y_ref = vy;
  for (auto i = 0; i < batch_size; ++i) {
    reference_blas::axpy(size, static_cast<scalar_t>(alpha),
                         vx.data() + i * stride_x, inc_x,
                         y_ref.data() + i * stride_y, inc_y);
  }
  std::vector<scalar_t> y_temp = vy;
  {
    blas_benchmark::utils::HIPVector<scalar_t, true> y_temp_gpu(size_y,
                                                                y_temp.data());
    rocblas_axpy_strided_batched_f<scalar_t>(rb_handle, size, &alpha, inx,
                                             inc_x, stride_x, y_temp_gpu, inc_y,
                                             stride_y, batch_size);
  }

  std::ostringstream err_stream;
  if (!utils::compare_vectors(y_temp, y_ref, err_stream, "")) {
    const std::string& err_str = err_stream.str();
    state.SkipWithError(err_str.c_str());
    *success = false;
  };
#endif

  auto blas_warmup = [&]() -> void {
    rocblas_axpy_strided_batched_f<scalar_t>(rb_handle, size, &alpha, inx,
                                             inc_x, stride_x, iny, inc_y,
                                             stride_y, batch_size);
    return;
  };

  hipEvent_t start, stop;
  CHECK_HIP_ERROR(hipEventCreate(&start));
  CHECK_HIP_ERROR(hipEventCreate(&stop));

  auto blas_method_def = [&]() -> std::vector<hipEvent_t> {
    CHECK_HIP_ERROR(hipEventRecord(start, NULL));
    rocblas_axpy_strided_batched_f<scalar_t>(rb_handle, size, &alpha, inx,
                                             inc_x, stride_x, iny, inc_y,
                                             stride_y, batch_size);
    CHECK_HIP_ERROR(hipEventRecord(stop, NULL));
    CHECK_HIP_ERROR(hipEventSynchronize(stop));
    return std::vector{start, stop};
  };

  // Warmup
  blas_benchmark::utils::warmup(blas_method_def);
  CHECK_HIP_ERROR(hipStreamSynchronize(NULL));

  blas_benchmark::utils::init_counters(state);

  // Measure
  for (auto _ : state) {
    // Run
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
}

template <typename scalar_t>
void register_benchmark(blas_benchmark::Args& args, rocblas_handle& rb_handle,
                        bool* success) {
  auto axpy_batch_params =
      blas_benchmark::utils::get_axpy_batch_params<scalar_t>(args);

  for (auto p : axpy_batch_params) {
    index_t n, inc_x, inc_y, stride_x_mul, stride_y_mul, batch_size;
    scalar_t alpha;
    std::tie(n, alpha, inc_x, inc_y, stride_x_mul, stride_y_mul, batch_size) =
        p;
    auto BM_lambda =
        [&](benchmark::State& st, rocblas_handle rb_handle, index_t size,
            scalar_t alpha, index_t inc_x, index_t inc_y, index_t stride_x_mul,
            index_t stride_y_mul, index_t batch_size, bool* success) {
          run<scalar_t>(st, rb_handle, size, alpha, inc_x, inc_y, stride_x_mul,
                        stride_y_mul, batch_size, success);
        };
    benchmark::RegisterBenchmark(
        blas_benchmark::utils::get_name<benchmark_op, scalar_t, index_t>(
            n, alpha, inc_x, inc_y, stride_x_mul, stride_y_mul, batch_size,
            blas_benchmark::utils::MEM_TYPE_USM)
            .c_str(),
        BM_lambda, rb_handle, n, alpha, inc_x, inc_y, stride_x_mul,
        stride_y_mul, batch_size, success)
        ->UseRealTime();
  }
}

namespace blas_benchmark {
void create_benchmark(blas_benchmark::Args& args, rocblas_handle& rb_handle,
                      bool* success) {
  BLAS_REGISTER_BENCHMARK(args, rb_handle, success);
}
}  // namespace blas_benchmark
