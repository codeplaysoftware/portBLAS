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
 *  @filename rotmg.cpp
 *
 **************************************************************************/

#include "../utils.hpp"

constexpr blas_benchmark::utils::Level1Op benchmark_op =
    blas_benchmark::utils::Level1Op::rotmg;

template <typename scalar_t, typename... args_t>
static inline void rocblas_rotmg_f(args_t&&... args) {
  if constexpr (std::is_same_v<scalar_t, float>) {
    CHECK_ROCBLAS_STATUS(rocblas_srotmg(std::forward<args_t>(args)...));
  } else if constexpr (std::is_same_v<scalar_t, double>) {
    CHECK_ROCBLAS_STATUS(rocblas_drotmg(std::forward<args_t>(args)...));
  }
  return;
}

template <typename scalar_t>
void run(benchmark::State& state, rocblas_handle& rb_handle, bool* success) {
  // initialize the state label
  blas_benchmark::utils::set_benchmark_label<scalar_t>(state);

  // Google-benchmark counters are double.
  blas_benchmark::utils::init_level_1_counters<
      blas_benchmark::utils::Level1Op::rotmg, scalar_t>(state, 1);
  // Create data
  constexpr size_t param_size = 5;
  std::vector<scalar_t> param = std::vector<scalar_t>(param_size);
  scalar_t d1 = blas_benchmark::utils::random_data<scalar_t>(1)[0];
  scalar_t d2 = blas_benchmark::utils::random_data<scalar_t>(1)[0];
  scalar_t x1 = blas_benchmark::utils::random_data<scalar_t>(1)[0];
  scalar_t y1 = blas_benchmark::utils::random_data<scalar_t>(1)[0];

  {
    // Device memory allocation & H2D Copy
    blas_benchmark::utils::HIPScalar<scalar_t> d_d1(d1);
    blas_benchmark::utils::HIPScalar<scalar_t> d_d2(d2);
    blas_benchmark::utils::HIPScalar<scalar_t> d_x1(x1);
    blas_benchmark::utils::HIPScalar<scalar_t> d_y1(y1);
    blas_benchmark::utils::HIPVector<scalar_t> d_param(param_size,
                                                       param.data());

#ifdef BLAS_VERIFY_BENCHMARK
    // Reference rotmg
    scalar_t d1_ref = d1;
    scalar_t d2_ref = d2;
    scalar_t x1_ref = x1;
    scalar_t y1_ref = y1;
    std::vector<scalar_t> param_ref = param;

    reference_blas::rotmg(&d1_ref, &d2_ref, &x1_ref, &y1_ref, param_ref.data());

    // Rocblas verification rotmg
    scalar_t d1_verify = d1;
    scalar_t d2_verify = d2;
    scalar_t x1_verify = x1;
    scalar_t y1_verify = y1;
    std::vector<scalar_t> param_verify = param;

    {
      // Temp result on device (copied back to Host upon destruction)
      blas_benchmark::utils::HIPScalar<scalar_t, true> d_d1_verify(d1_verify);
      blas_benchmark::utils::HIPScalar<scalar_t, true> d_d2_verify(d2_verify);
      blas_benchmark::utils::HIPScalar<scalar_t, true> d_x1_verify(x1_verify);
      blas_benchmark::utils::HIPScalar<scalar_t, true> d_y1_verify(y1_verify);
      blas_benchmark::utils::HIPVector<scalar_t, true> d_param_verify(
          param_size, param_verify.data());
      // RocBLAS function call
      rocblas_rotmg_f<scalar_t>(rb_handle, d_d1_verify, d_d2_verify,
                                d_x1_verify, d_y1_verify, d_param_verify);
    }

    // Verify results
    const bool areAlmostEqual =
        utils::almost_equal(d1_verify, d1_ref) &&
        utils::almost_equal(d2_verify, d2_ref) &&
        utils::almost_equal(x1_verify, x1_ref) &&
        utils::almost_equal(param_verify[0], param_ref[0]);

    if (!areAlmostEqual) {
      std::ostringstream err_stream;
      err_stream << "Value mismatch." << std::endl;
      const std::string& err_str = err_stream.str();
      state.SkipWithError(err_str.c_str());
      *success = false;
    };

#endif

    auto blas_warmup = [&]() -> void {
      rocblas_rotmg_f<scalar_t>(rb_handle, d_d1, d_d2, d_x1, d_y1, d_param);
      return;
    };

    hipEvent_t start, stop;
    CHECK_HIP_ERROR(hipEventCreate(&start));
    CHECK_HIP_ERROR(hipEventCreate(&stop));

    auto blas_method_def = [&]() -> std::vector<hipEvent_t> {
      CHECK_HIP_ERROR(hipEventRecord(start, NULL));
      rocblas_rotmg_f<scalar_t>(rb_handle, d_d1, d_d2, d_x1, d_y1, d_param);
      CHECK_HIP_ERROR(hipEventRecord(stop, NULL));
      CHECK_HIP_ERROR(hipEventSynchronize(stop));
      return std::vector{start, stop};
    };

    // Warmup
    blas_benchmark::utils::warmup(blas_warmup);
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
  }  // release device memory via utils::DeviceVector destructors
};

template <typename scalar_t>
void register_benchmark(blas_benchmark::Args& args, rocblas_handle& rb_handle,
                        bool* success) {
  auto BM_lambda = [&](benchmark::State& st, rocblas_handle rb_handle,
                       bool* success) {
    run<scalar_t>(st, rb_handle, success);
  };
  benchmark::RegisterBenchmark(
      blas_benchmark::utils::get_name<benchmark_op, scalar_t>(
          blas_benchmark::utils::MEM_TYPE_USM)
          .c_str(),
      BM_lambda, rb_handle, success)
      ->UseRealTime();
}

namespace blas_benchmark {
void create_benchmark(blas_benchmark::Args& args, rocblas_handle& rb_handle,
                      bool* success) {
  BLAS_REGISTER_BENCHMARK(args, rb_handle, success);
}
}  // namespace blas_benchmark
