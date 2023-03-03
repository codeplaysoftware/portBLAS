/***************************************************************************
 *
 *  @license
 *  Copyright (C) 2016 Codeplay Software Limited
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
 *  @filename nrm2.cpp
 *
 **************************************************************************/

#include "../utils.hpp"

template <typename scalar_t>
std::string get_name(int size) {
  std::ostringstream str{};
  str << "BM_Nrm2<" << blas_benchmark::utils::get_type_name<scalar_t>() << ">/";
  str << size;
  return str.str();
}

template <typename scalar_t, typename... args_t>
static inline void rocblas_nrm2_f(args_t&&... args) {
  if constexpr (std::is_same_v<scalar_t, float>) {
    CHECK_ROCBLAS_STATUS(rocblas_snrm2(std::forward<args_t>(args)...));
  } else if constexpr (std::is_same_v<scalar_t, double>) {
    CHECK_ROCBLAS_STATUS(rocblas_dnrm2(std::forward<args_t>(args)...));
  }
  return;
}

template <typename scalar_t>
void run(benchmark::State& state, rocblas_handle& rb_handle, index_t size,
         bool* success) {
  // Google-benchmark counters are double.
  double size_d = static_cast<double>(size);
  state.counters["size"] = size_d;
  state.counters["n_fl_ops"] = 2 * size_d;
  state.counters["bytes_processed"] = size_d * sizeof(scalar_t);

  // Create data
  std::vector<scalar_t> v1 = blas_benchmark::utils::random_data<scalar_t>(size);
  scalar_t vr;

  {
    // Device memory allocation
    blas_benchmark::utils::DeviceVector<scalar_t> d_v1(size);
    blas_benchmark::utils::DeviceVector<scalar_t> d_vr(1);

    // Copy data (H2D)
    CHECK_HIP_ERROR(hipMemcpy(d_v1, v1.data(), sizeof(scalar_t) * size,
                              hipMemcpyHostToDevice));

#ifdef BLAS_VERIFY_BENCHMARK
    // Run a first time with a verification of the results
    scalar_t vr_ref = reference_blas::nrm2(size, v1.data(), 1);
    scalar_t vr_temp = 0;

    {
      blas_benchmark::utils::DeviceVector<scalar_t> d_vr_temp(1);

      CHECK_HIP_ERROR(
          hipMemcpy(d_vr_temp, &vr_temp, sizeof(int), hipMemcpyHostToDevice));
      rocblas_nrm2_f<scalar_t>(rb_handle, size, d_v1, 1, d_vr_temp);
      CHECK_HIP_ERROR(
          hipMemcpy(&vr_temp, d_vr_temp, sizeof(int), hipMemcpyDeviceToHost));
    }

    if (!utils::almost_equal(vr_temp, vr_ref)) {
      std::ostringstream err_stream;
      err_stream << "Value mismatch: " << vr_temp << "; expected " << vr_ref;
      const std::string& err_str = err_stream.str();
      state.SkipWithError(err_str.c_str());
      *success = false;
    };
#endif

    auto blas_warmup = [&]() -> void {
      rocblas_nrm2_f<scalar_t>(rb_handle, size, d_v1, 1, d_vr);
      CHECK_HIP_ERROR(hipStreamSynchronize(NULL));
      return;
    };

    // Create a utility lambda describing the blas method that we want to run.
    auto blas_method_def = [&]() -> std::vector<hipEvent_t> {
      hipEvent_t start, stop;
      CHECK_HIP_ERROR(hipEventCreate(&start));
      CHECK_HIP_ERROR(hipEventCreate(&stop));

      // Assuming the NULL (default) stream is the only one in use
      CHECK_HIP_ERROR(hipEventRecord(start, NULL));

      rocblas_nrm2_f<scalar_t>(rb_handle, size, d_v1, 1, d_vr);

      CHECK_HIP_ERROR(hipEventRecord(stop, NULL));
      CHECK_HIP_ERROR(hipEventSynchronize(stop));

      return std::vector{start, stop};
    };

    // Warm up to avoid benchmarking data transfer
    blas_benchmark::utils::warmup(blas_warmup);

    blas_benchmark::utils::init_counters(state);

    // Measure
    for (auto _ : state) {
      std::tuple<double, double> times =
          blas_benchmark::utils::timef_hip(blas_method_def);

      // Report
      blas_benchmark::utils::update_counters(state, times);
    }

    blas_benchmark::utils::calc_avg_counters(state);
  }  // release device memory via utils::DeviceVector destructors
};

template <typename scalar_t>
void register_benchmark(blas_benchmark::Args& args, rocblas_handle& rb_handle,
                        bool* success) {
  auto blas1_params = blas_benchmark::utils::get_blas1_params(args);

  for (auto size : blas1_params) {
    auto BM_lambda = [&](benchmark::State& st, rocblas_handle rb_handle,
                         index_t size, bool* success) {
      run<scalar_t>(st, rb_handle, size, success);
    };
    benchmark::RegisterBenchmark(get_name<scalar_t>(size).c_str(), BM_lambda,
                                 rb_handle, size, success);
  }
}

namespace blas_benchmark {
void create_benchmark(blas_benchmark::Args& args, rocblas_handle& rb_handle,
                      bool* success) {
  BLAS_REGISTER_BENCHMARK(args, rb_handle, success);
}
}  // namespace blas_benchmark
