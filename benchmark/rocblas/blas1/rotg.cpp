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
 *  @filename rotg.cpp
 *
 **************************************************************************/

#include "../utils.hpp"

constexpr blas_benchmark::utils::Level1Op benchmark_op =
    blas_benchmark::utils::Level1Op::rotg;

template <typename scalar_t, typename... args_t>
static inline void rocblas_rotg_f(args_t&&... args) {
  if constexpr (std::is_same_v<scalar_t, float>) {
    CHECK_ROCBLAS_STATUS(rocblas_srotg(std::forward<args_t>(args)...));
  } else if constexpr (std::is_same_v<scalar_t, double>) {
    CHECK_ROCBLAS_STATUS(rocblas_drotg(std::forward<args_t>(args)...));
  }
  return;
}

template <typename scalar_t>
void run(benchmark::State& state, rocblas_handle& rb_handle, bool* success) {
  // initialize the state label
  blas_benchmark::utils::set_benchmark_label<scalar_t>(state);

  // Create data
  scalar_t a = blas_benchmark::utils::random_data<scalar_t>(1)[0];
  scalar_t b = blas_benchmark::utils::random_data<scalar_t>(1)[0];
  scalar_t c = blas_benchmark::utils::random_data<scalar_t>(1)[0];
  scalar_t s = blas_benchmark::utils::random_data<scalar_t>(1)[0];

  {
    // Device memory allocation & H2D Copy
    blas_benchmark::utils::HIPScalar<scalar_t> d_a(a);
    blas_benchmark::utils::HIPScalar<scalar_t> d_b(b);
    blas_benchmark::utils::HIPScalar<scalar_t> d_c(c);
    blas_benchmark::utils::HIPScalar<scalar_t> d_s(s);

#ifdef BLAS_VERIFY_BENCHMARK
    // Reference rotg
    scalar_t a_ref = a;
    scalar_t b_ref = b;
    scalar_t c_ref = c;
    scalar_t s_ref = s;

    reference_blas::rotg(&a_ref, &b_ref, &c_ref, &s_ref);

    // Rocblas verification rotg
    scalar_t a_verify = a;
    scalar_t b_verify = b;
    scalar_t c_verify = c;
    scalar_t s_verify = s;

    {
      // Temp results on device (copied back to Host upon destruction)
      blas_benchmark::utils::HIPScalar<scalar_t, true> d_a_verify(a_verify);
      blas_benchmark::utils::HIPScalar<scalar_t, true> d_b_verify(b_verify);
      blas_benchmark::utils::HIPScalar<scalar_t, true> d_c_verify(c_verify);
      blas_benchmark::utils::HIPScalar<scalar_t, true> d_s_verify(s_verify);
      // RocBLAS function call
      rocblas_rotg_f<scalar_t>(rb_handle, d_a_verify, d_b_verify, d_c_verify,
                               d_s_verify);
    }

    // Verify results
    const bool areAlmostEqual =
        utils::almost_equal<scalar_t, scalar_t>(a_verify, a_ref) &&
        utils::almost_equal<scalar_t, scalar_t>(b_verify, b_ref) &&
        utils::almost_equal<scalar_t, scalar_t>(c_verify, c_ref) &&
        utils::almost_equal<scalar_t, scalar_t>(s_verify, s_ref);

    if (!areAlmostEqual) {
      std::ostringstream err_stream;
      err_stream << "Value mismatch: " << std::endl;
      err_stream << "Got: "
                 << "A: " << a_verify << " B: " << b_verify
                 << " C: " << c_verify << " S: " << s_verify << std::endl;
      err_stream << "Expected: "
                 << "A: " << a_ref << " B: " << b_ref << " C: " << c_ref
                 << " S: " << s_ref << std::endl;
      const std::string& err_str = err_stream.str();
      state.SkipWithError(err_str.c_str());
      *success = false;
    };
#endif

    auto blas_warmup = [&]() -> void {
      rocblas_rotg_f<scalar_t>(rb_handle, d_a, d_b, d_c, d_s);
      return;
    };

    hipEvent_t start, stop;
    CHECK_HIP_ERROR(hipEventCreate(&start));
    CHECK_HIP_ERROR(hipEventCreate(&stop));

    auto blas_method_def = [&]() -> std::vector<hipEvent_t> {
      CHECK_HIP_ERROR(hipEventRecord(start, NULL));
      rocblas_rotg_f<scalar_t>(rb_handle, d_a, d_b, d_c, d_s);
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
