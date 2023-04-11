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
 *  @filename gbmv.cpp
 *
 **************************************************************************/

#include "../utils.hpp"

template <typename scalar_t>
std::string get_name(std::string t, int m, int n, int kl, int ku) {
  std::ostringstream str{};
  str << "BM_Gbmv<" << blas_benchmark::utils::get_type_name<scalar_t>() << ">/"
      << t << "/" << m << "/" << n << "/" << kl << "/" << ku;
  return str.str();
}

template <typename scalar_t, typename... args_t>
static inline void rocblas_gbmv_f(args_t&&... args) {
  if constexpr (std::is_same_v<scalar_t, float>) {
    CHECK_ROCBLAS_STATUS(rocblas_sgbmv(std::forward<args_t>(args)...));
  } else if constexpr (std::is_same_v<scalar_t, double>) {
    CHECK_ROCBLAS_STATUS(rocblas_dgbmv(std::forward<args_t>(args)...));
  }
  return;
}

template <typename scalar_t>
void run(benchmark::State& state, rocblas_handle& rb_handle, int ti, index_t m,
         index_t n, index_t kl, index_t ku, scalar_t alpha, scalar_t beta,
         bool* success) {
  // Standard test setup.
  std::string ts = blas_benchmark::utils::from_transpose_enum(
      static_cast<blas_benchmark::utils::Transposition>(ti));
  const char* t_str = ts.c_str();

  index_t xlen = t_str[0] == 'n' ? n : m;
  index_t ylen = t_str[0] == 'n' ? m : n;

  index_t lda = (kl + ku + 1);
  index_t incX = 1;
  index_t incY = 1;

  // The counters are double. We convert m and n to double to avoid
  // integer overflows for n_fl_ops and bytes_processed
  double m_d = static_cast<double>(m);
  double n_d = static_cast<double>(n);
  double kl_d = static_cast<double>(kl);
  double ku_d = static_cast<double>(ku);

  state.counters["m"] = m_d;
  state.counters["n"] = n_d;
  state.counters["kl"] = kl_d;
  state.counters["ku"] = ku_d;

  // Compute the number of A non-zero elements.
  const double A_validVal =
      (t_str[0] == 'n' ? n_d : m_d) * (kl_d + ku_d + 1.0) -
      0.5 * (kl_d * (kl_d + 1.0)) - 0.5 * (ku_d * (ku_d + 1.0));

  double nflops_AtimesX = 2.0 * A_validVal;
  double nflops_timesAlpha = ylen;
  double nflops_addBetaY = (beta != scalar_t{0}) ? 2 * ylen : 0;
  double nflops_tot = nflops_AtimesX + nflops_timesAlpha + nflops_addBetaY;
  state.counters["n_fl_ops"] = nflops_tot;

  {
    double mem_readA = A_validVal;
    double mem_readX = xlen;
    double mem_writeY = ylen;
    double mem_readY = (beta != scalar_t{0}) ? ylen : 0;
    state.counters["bytes_processed"] =
        (mem_readA + mem_readX + mem_writeY + mem_readY) * sizeof(scalar_t);
  }

  // Matrix options (rocBLAS)
  const rocblas_operation transA =
      t_str[0] == 'n' ? rocblas_operation_none : rocblas_operation_transpose;

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
    // Reference gbmv
    std::vector<scalar_t> v_y_ref = v_y;
    reference_blas::gbmv(t_str, m, n, kl, ku, alpha, m_a.data(), lda,
                         v_x.data(), incX, beta, v_y_ref.data(), incY);

    // Rocblas verification gbmv
    std::vector<scalar_t> v_y_temp = v_y;
    {
      // Temp result on device (copied back to Host upon destruction)
      blas_benchmark::utils::HIPVector<scalar_t, true> y_temp_gpu(
          ylen, v_y_temp.data());
      // rocBLAS function call
      rocblas_gbmv_f<scalar_t>(rb_handle, transA, m, n, kl, ku, &alpha, m_a_gpu,
                               lda, v_x_gpu, incX, &beta, y_temp_gpu, incY);
    }

    std::ostringstream err_stream;
    if (!utils::compare_vectors(v_y_temp, v_y_ref, err_stream, "")) {
      const std::string& err_str = err_stream.str();
      state.SkipWithError(err_str.c_str());
      *success = false;
    };
#endif

    auto blas_warmup = [&]() -> void {
      rocblas_gbmv_f<scalar_t>(rb_handle, transA, m, n, kl, ku, &alpha, m_a_gpu,
                               lda, v_x_gpu, incX, &beta, v_y_gpu, incY);
      return;
    };

    hipEvent_t start, stop;
    CHECK_HIP_ERROR(hipEventCreate(&start));
    CHECK_HIP_ERROR(hipEventCreate(&stop));

    auto blas_method_def = [&]() -> std::vector<hipEvent_t> {
      CHECK_HIP_ERROR(hipEventRecord(start, NULL));
      rocblas_gbmv_f<scalar_t>(rb_handle, transA, m, n, kl, ku, &alpha, m_a_gpu,
                               lda, v_x_gpu, incX, &beta, v_y_gpu, incY);
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

    state.SetItemsProcessed(state.iterations() * nflops_tot);

    blas_benchmark::utils::calc_avg_counters(state);

    CHECK_HIP_ERROR(hipEventDestroy(start));
    CHECK_HIP_ERROR(hipEventDestroy(stop));
  }  // release device memory via utils::DeviceVector destructors
}

template <typename scalar_t>
void register_benchmark(blas_benchmark::Args& args, rocblas_handle& rb_handle,
                        bool* success) {
  auto gbmv_params = blas_benchmark::utils::get_gbmv_params<scalar_t>(args);

  for (auto p : gbmv_params) {
    std::string ts;
    index_t m, n, kl, ku;
    scalar_t alpha, beta;
    std::tie(ts, m, n, kl, ku, alpha, beta) = p;
    int t = static_cast<int>(blas_benchmark::utils::to_transpose_enum(ts));

    auto BM_lambda = [&](benchmark::State& st, rocblas_handle rb_handle, int t,
                         index_t m, index_t n, index_t kl, index_t ku,
                         scalar_t alpha, scalar_t beta, bool* success) {
      run<scalar_t>(st, rb_handle, t, m, n, kl, ku, alpha, beta, success);
    };
    benchmark::RegisterBenchmark(get_name<scalar_t>(ts, m, n, kl, ku).c_str(),
                                 BM_lambda, rb_handle, t, m, n, kl, ku, alpha,
                                 beta, success);
  }
}

namespace blas_benchmark {
void create_benchmark(blas_benchmark::Args& args, rocblas_handle& rb_handle,
                      bool* success) {
  BLAS_REGISTER_BENCHMARK(args, rb_handle, success);
}
}  // namespace blas_benchmark
