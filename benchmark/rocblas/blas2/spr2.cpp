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
 *  @filename spr2.cpp
 *
 **************************************************************************/

#include "../utils.hpp"

constexpr blas_benchmark::utils::Level2Op benchmark_op =
    blas_benchmark::utils::Level2Op::spr2;

template <typename scalar_t, typename... args_t>
static inline void rocblas_spr2_f(args_t&&... args) {
  if constexpr (std::is_same_v<scalar_t, float>) {
    CHECK_ROCBLAS_STATUS(rocblas_sspr2(std::forward<args_t>(args)...));
  } else if constexpr (std::is_same_v<scalar_t, double>) {
    CHECK_ROCBLAS_STATUS(rocblas_dspr2(std::forward<args_t>(args)...));
  }
  return;
}

template <typename scalar_t>
void run(benchmark::State& state, rocblas_handle& rb_handle, char uplo,
         index_t n, scalar_t alpha, index_t incX, index_t incY, bool* success) {
  // initialize the state label
  blas_benchmark::utils::set_benchmark_label<scalar_t>(state);

  // Standard test setup.

  index_t xlen = n;
  index_t ylen = n;

  blas_benchmark::utils::init_level_2_counters<
      blas_benchmark::utils::Level2Op::spr2, scalar_t>(state, "n", 0, 0, n);

  // Matrix options (rocBLAS)
  const rocblas_fill uplo_rb =
      uplo == 'u' ? rocblas_fill_upper : rocblas_fill_lower;

  // Data sizes
  const int m_size = n * n;
  const int v_x_size = 1 + (xlen - 1) * std::abs(incX);
  const int v_y_size = 1 + (ylen - 1) * std::abs(incY);

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
    // Reference spr2
    std::vector<scalar_t> m_a_ref = m_a;
    reference_blas::spr2(&uplo, n, alpha, v_x.data(), incX, v_y.data(), incY,
                         m_a_ref.data());

    // Rocblas verification spr2
    std::vector<scalar_t> m_a_temp = m_a;
    {
      // Temp result on device (copied back to Host upon destruction)
      blas_benchmark::utils::HIPVector<scalar_t, true> m_a_temp_gpu(
          m_size, m_a_temp.data());
      // rocBLAS function call
      rocblas_spr2_f<scalar_t>(rb_handle, uplo_rb, n, &alpha, v_x_gpu, incX,
                               v_y_gpu, incY, m_a_temp_gpu);
    }

    std::ostringstream err_stream;
    if (!utils::compare_vectors(m_a_temp, m_a_ref, err_stream, "")) {
      const std::string& err_str = err_stream.str();
      state.SkipWithError(err_str.c_str());
      *success = false;
    };
#endif

    auto blas_warmup = [&]() -> void {
      rocblas_spr2_f<scalar_t>(rb_handle, uplo_rb, n, &alpha, v_x_gpu, incX,
                               v_y_gpu, incY, m_a_gpu);
      return;
    };

    hipEvent_t start, stop;
    CHECK_HIP_ERROR(hipEventCreate(&start));
    CHECK_HIP_ERROR(hipEventCreate(&stop));

    auto blas_method_def = [&]() -> std::vector<hipEvent_t> {
      CHECK_HIP_ERROR(hipEventRecord(start, NULL));
      rocblas_spr2_f<scalar_t>(rb_handle, uplo_rb, n, &alpha, v_x_gpu, incX,
                               v_y_gpu, incY, m_a_gpu);
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
}

template <typename scalar_t>
void register_benchmark(blas_benchmark::Args& args, rocblas_handle& rb_handle,
                        bool* success) {
  auto spr2_params = blas_benchmark::utils::get_spr2_params<scalar_t>(args);

  for (auto p : spr2_params) {
    std::string uplo;
    index_t n, incX, incY;
    scalar_t alpha;
    std::tie(uplo, n, alpha, incX, incY) = p;

    char uplo_c = uplo[0];

    auto BM_lambda = [&](benchmark::State& st, rocblas_handle rb_handle,
                         char uplo, index_t n, scalar_t alpha, index_t incX,
                         index_t incY, bool* success) {
      run<scalar_t>(st, rb_handle, uplo, n, alpha, incX, incY, success);
    };
    benchmark::RegisterBenchmark(
        blas_benchmark::utils::get_name<benchmark_op, scalar_t>(
            uplo, n, alpha, incX, incY, blas_benchmark::utils::MEM_TYPE_USM)
            .c_str(),
        BM_lambda, rb_handle, uplo_c, n, alpha, incX, incY, success)
        ->UseRealTime();
  }
}

namespace blas_benchmark {
void create_benchmark(blas_benchmark::Args& args, rocblas_handle& rb_handle,
                      bool* success) {
  BLAS_REGISTER_BENCHMARK(args, rb_handle, success);
}
}  // namespace blas_benchmark
