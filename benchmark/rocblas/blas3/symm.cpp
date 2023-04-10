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
 *  @filename symm.cpp
 *
 **************************************************************************/

#include "../utils.hpp"

template <typename scalar_t>
std::string get_name(char side, char uplo, int m, int n, scalar_t alpha,
                     scalar_t beta) {
  std::ostringstream str{};
  str << "BM_Symm<" << blas_benchmark::utils::get_type_name<scalar_t>() << ">/"
      << side << "/" << uplo << "/" << m << "/" << n << "/" << alpha << "/"
      << beta;
  return str.str();
}

template <typename scalar_t, typename... args_t>
static inline void rocblas_symm_f(args_t&&... args) {
  if constexpr (std::is_same_v<scalar_t, float>) {
    CHECK_ROCBLAS_STATUS(rocblas_ssymm(std::forward<args_t>(args)...));
  } else if constexpr (std::is_same_v<scalar_t, double>) {
    CHECK_ROCBLAS_STATUS(rocblas_dsymm(std::forward<args_t>(args)...));
  }
  return;
}

template <typename scalar_t>
void run(benchmark::State& state, rocblas_handle& rb_handle, char side,
         char uplo, index_t m, index_t n, scalar_t alpha, scalar_t beta,
         bool* success) {
  // Standard test setup.
  index_t lda = (side == 'l') ? m : n;
  index_t ldb = m;
  index_t ldc = ldb;

  {
    // The counters are double. We convert m, n and k to double to avoid
    // integer overflows for n_fl_ops and bytes_processed
    double m_d = static_cast<double>(m);
    double n_d = static_cast<double>(n);

    state.counters["m"] = m_d;
    state.counters["n"] = n_d;

    const double mem_readB = m_d * n_d;
    const double mem_readC = (beta != scalar_t{0}) ? m_d * n_d : 0;
    const double mem_writeC = m_d * n_d;
    const double mem_readA =
        (side == 'l') ? (m_d * (m_d + 1) / 2) : (n_d * (n_d + 1) / 2);
    const double total_mem =
        (mem_readB + mem_readC + mem_writeC + mem_readA) * sizeof(scalar_t);
    state.counters["bytes_processed"] = total_mem;

    const double nflops_AtimesB =
        (side == 'l') ? (2 * m_d * m_d) * n_d : 2 * n_d * n_d * n_d;
    const double nflops_addBetaC = 2 * m_d * n_d;
    const double total_nflops = nflops_AtimesB + nflops_addBetaC;
    state.counters["n_fl_ops"] = total_nflops;

    state.SetBytesProcessed(state.iterations() * total_mem);
    state.SetItemsProcessed(state.iterations() * total_nflops);
  }

  // Matrix options (rocBLAS)
  const rocblas_side side_rb =
      (side == 'l') ? rocblas_side_left : rocblas_side_right;

  const rocblas_fill uplo_rb =
      (uplo == 'u') ? rocblas_fill_upper : rocblas_fill_lower;

  // Data sizes
  const int a_size = lda * lda;
  const int b_size = ldb * n;
  const int c_size = ldc * n;

  // Matrices
  std::vector<scalar_t> a =
      blas_benchmark::utils::random_data<scalar_t>(a_size);
  std::vector<scalar_t> b =
      blas_benchmark::utils::random_data<scalar_t>(b_size);
  std::vector<scalar_t> c =
      blas_benchmark::utils::random_data<scalar_t>(c_size);

  {
    // Device memory allocation & H2D copy
    blas_benchmark::utils::HIPVector<scalar_t> a_gpu(a_size, a.data());
    blas_benchmark::utils::HIPVector<scalar_t> b_gpu(b_size, b.data());
    blas_benchmark::utils::HIPVector<scalar_t> c_gpu(c_size, c.data());

    CHECK_ROCBLAS_STATUS(
        rocblas_set_pointer_mode(rb_handle, rocblas_pointer_mode_host));

#ifdef BLAS_VERIFY_BENCHMARK
    // Reference symm
    std::vector<scalar_t> c_ref = c;
    reference_blas::symm(&side, &uplo, m, n, alpha, a.data(), lda, b.data(),
                         ldb, beta, c_ref.data(), ldc);

    // Rocblas verification symm
    std::vector<scalar_t> c_temp = c;
    {
      blas_benchmark::utils::HIPVector<scalar_t, true> c_temp_gpu(
          c_size, c_temp.data());
      rocblas_symm_f<scalar_t>(rb_handle, side_rb, uplo_rb, m, n, &alpha, a_gpu,
                               lda, b_gpu, ldb, &beta, c_temp_gpu, ldc);
    }

    std::ostringstream err_stream;
    if (!utils::compare_vectors(c_temp, c_ref, err_stream, "")) {
      const std::string& err_str = err_stream.str();
      state.SkipWithError(err_str.c_str());
      *success = false;
    };
#endif
    auto blas_warmup = [&]() -> void {
      rocblas_symm_f<scalar_t>(rb_handle, side_rb, uplo_rb, m, n, &alpha, a_gpu,
                               lda, b_gpu, ldb, &beta, c_gpu, ldc);
      return;
    };

    hipEvent_t start, stop;
    CHECK_HIP_ERROR(hipEventCreate(&start));
    CHECK_HIP_ERROR(hipEventCreate(&stop));

    auto blas_method_def = [&]() -> std::vector<hipEvent_t> {
      CHECK_HIP_ERROR(hipEventRecord(start, NULL));
      rocblas_symm_f<scalar_t>(rb_handle, side_rb, uplo_rb, m, n, &alpha, a_gpu,
                               lda, b_gpu, ldb, &beta, c_gpu, ldc);
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

    blas_benchmark::utils::calc_avg_counters(state);

    CHECK_HIP_ERROR(hipEventDestroy(start));
    CHECK_HIP_ERROR(hipEventDestroy(stop));
  }  // release device memory via utils::DeviceVector destructors
};

template <typename scalar_t>
void register_benchmark(blas_benchmark::Args& args, rocblas_handle& rb_handle,
                        bool* success) {
  auto symm_params = blas_benchmark::utils::get_symm_params<scalar_t>(args);

  for (auto p : symm_params) {
    char side, uplo;
    index_t m, n;
    scalar_t alpha, beta;
    std::tie(side, uplo, m, n, alpha, beta) = p;

    auto BM_lambda = [&](benchmark::State& st, rocblas_handle rb_handle,
                         char side, char uplo, index_t m, index_t n,
                         scalar_t alpha, scalar_t beta, bool* success) {
      run<scalar_t>(st, rb_handle, side, uplo, m, n, alpha, beta, success);
    };
    benchmark::RegisterBenchmark(
        get_name<scalar_t>(side, uplo, m, n, alpha, beta).c_str(), BM_lambda,
        rb_handle, side, uplo, m, n, alpha, beta, success)
        ->UseRealTime();
  }
}

namespace blas_benchmark {
void create_benchmark(blas_benchmark::Args& args, rocblas_handle& rb_handle,
                      bool* success) {
  BLAS_REGISTER_BENCHMARK(args, rb_handle, success);
}
}  // namespace blas_benchmark
