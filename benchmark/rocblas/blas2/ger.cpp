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
 *  @filename ger.cpp
 *
 **************************************************************************/

#include "../utils.hpp"

template <typename scalar_t>
std::string get_name(int m, int n) {
  std::ostringstream str{};
  str << "BM_ger<" << blas_benchmark::utils::get_type_name<scalar_t>() << ">/"
      << m << "/" << n;
  return str.str();
}

template <typename scalar_t, typename... args_t>
static inline void rocblas_ger_f(args_t&&... args) {
  if constexpr (std::is_same_v<scalar_t, float>) {
    CHECK_ROCBLAS_STATUS(rocblas_sger(std::forward<args_t>(args)...));
  } else if constexpr (std::is_same_v<scalar_t, double>) {
    CHECK_ROCBLAS_STATUS(rocblas_dger(std::forward<args_t>(args)...));
  }
  return;
}

template <typename scalar_t>
void run(benchmark::State& state, rocblas_handle& rb_handle, index_t m,
         index_t n, scalar_t alpha, bool* success) {
  // Standard test setup.
  index_t xlen = m;
  index_t ylen = n;

  index_t lda = m;
  index_t incX = 1;
  index_t incY = 1;

  // The counters are double. We convert m and n to double to avoid
  // integer overflows for n_fl_ops and bytes_processed
  double m_d = static_cast<double>(m);
  double n_d = static_cast<double>(n);

  state.counters["n"] = n_d;
  state.counters["m"] = m_d;

  {
    double nflops_timesAlpha = std::min(xlen, ylen);
    double nflops_XtimesY = n_d * m_d;
    double nflops_sum = n_d * m_d;
    state.counters["n_fl_ops"] =
        nflops_timesAlpha + nflops_XtimesY + nflops_sum;
  }
  {
    double mem_readA = n_d * m_d;
    double mem_readX = xlen;
    double mem_readY = ylen;
    double mem_writeA = n_d * m_d;
    state.counters["bytes_processed"] =
        (mem_readA + mem_readX + mem_readY + mem_writeA) * sizeof(scalar_t);
  }

  // Data sizes
  const int m_size = lda * n;
  const int v_x_size = 1 + (xlen - 1) * incX;
  const int v_y_size = 1 + (ylen - 1) * incY;

  // Input matrix/vector, output vector.
  std::vector<scalar_t> m_a =
      blas_benchmark::utils::random_data<scalar_t>(m_size);
  std::vector<scalar_t> v_x =
      blas_benchmark::utils::random_data<scalar_t>(v_x_size);
  std::vector<scalar_t> v_y =
      blas_benchmark::utils::random_data<scalar_t>(v_y_size);

  {
    // Device memory allocation & H2D copy
    blas_benchmark::utils::HIPVector<scalar_t> m_a_gpu(m_size, m_a.data());
    blas_benchmark::utils::HIPVector<scalar_t> v_x_gpu(v_x_size, v_x.data());
    blas_benchmark::utils::HIPVector<scalar_t> v_y_gpu(v_y_size, v_y.data());

    CHECK_ROCBLAS_STATUS(
        rocblas_set_pointer_mode(rb_handle, rocblas_pointer_mode_host));

#ifdef BLAS_VERIFY_BENCHMARK
    // Reference ger
    std::vector<scalar_t> m_a_ref = m_a;
    reference_blas::ger(m, n, alpha, m_a.data(), incX, v_x.data(), incY,
                        v_y.data(), lda);

    // Rocblas verification ger
    std::vector<scalar_t> m_a_temp = m_a;
    {
      // Temp result on device (copied back to Host upon destruction)
      blas_benchmark::utils::HIPVector<scalar_t, true> m_a_temp_gpu(
          m_size, m_a_temp.data());
      // rocBLAS function call
      rocblas_ger_f<scalar_t>(rb_handle, m, n, &alpha, v_x_gpu, incX, v_y_gpu,
                              incY, m_a_temp_gpu, lda);
    }

    std::ostringstream err_stream;
    if (!utils::compare_vectors(m_a_temp, m_a_ref, err_stream, "")) {
      const std::string& err_str = err_stream.str();
      state.SkipWithError(err_str.c_str());
      *success = false;
    };
#endif

    auto blas_warmup = [&]() -> void {
      rocblas_ger_f<scalar_t>(rb_handle, m, n, &alpha, v_x_gpu, incX, v_y_gpu,
                              incY, m_a_gpu, lda);
      CHECK_HIP_ERROR(hipStreamSynchronize(NULL));
      return;
    };

    hipEvent_t start, stop;
    CHECK_HIP_ERROR(hipEventCreate(&start));
    CHECK_HIP_ERROR(hipEventCreate(&stop));

    auto blas_method_def = [&]() -> std::vector<hipEvent_t> {
      CHECK_HIP_ERROR(hipEventRecord(start, NULL));
      rocblas_ger_f<scalar_t>(rb_handle, m, n, &alpha, v_x_gpu, incX, v_y_gpu,
                              incY, m_a_gpu, lda);
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

    CHECK_HIP_ERROR(hipEventDestroy(start));
    CHECK_HIP_ERROR(hipEventDestroy(stop));
  }  // release device memory via utils::DeviceVector destructors
}

template <typename scalar_t>
void register_benchmark(blas_benchmark::Args& args, rocblas_handle& rb_handle,
                        bool* success) {
  auto ger_params = blas_benchmark::utils::get_ger_params<scalar_t>(args);

  for (auto p : ger_params) {
    index_t m, n;
    scalar_t alpha;
    std::tie(m, n, alpha) = p;

    auto BM_lambda = [&](benchmark::State& st, rocblas_handle rb_handle,
                         index_t m, index_t n, scalar_t alpha, bool* success) {
      run<scalar_t>(st, rb_handle, m, n, alpha, success);
    };
    benchmark::RegisterBenchmark(get_name<scalar_t>(m, n).c_str(), BM_lambda,
                                 rb_handle, m, n, alpha, success);
  }
}

namespace blas_benchmark {
void create_benchmark(blas_benchmark::Args& args, rocblas_handle& rb_handle,
                      bool* success) {
  BLAS_REGISTER_BENCHMARK(args, rb_handle, success);
}
}  // namespace blas_benchmark
