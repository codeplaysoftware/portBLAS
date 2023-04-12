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
 *  @filename trsm_batched.cpp
 *
 **************************************************************************/

#include "../utils.hpp"

template <typename scalar_t>
std::string get_name(const char side, const char uplo, const char t,
                     const char diag, int m, int n, int batch_size,
                     int batch_type) {
  std::ostringstream str{};
  str << "BM_TrsmBatched<" << blas_benchmark::utils::get_type_name<scalar_t>()
      << ">/" << side << "/" << uplo << "/" << t << "/" << diag << "/" << m
      << "/" << n << "/" << batch_size << "/"
      << blas_benchmark::utils::batch_type_to_str(batch_type);
  return str.str();
}

template <typename scalar_t, typename... args_t>
static inline void rocblas_trsm_batched_f(args_t&&... args) {
  if constexpr (std::is_same_v<scalar_t, float>) {
    CHECK_ROCBLAS_STATUS(rocblas_strsm_batched(std::forward<args_t>(args)...));
  } else if constexpr (std::is_same_v<scalar_t, double>) {
    CHECK_ROCBLAS_STATUS(rocblas_dtrsm_batched(std::forward<args_t>(args)...));
  }
  return;
}

template <typename scalar_t>
void run(benchmark::State& state, rocblas_handle& rb_handle, const char side,
         const char uplo, const char trans, const char diag, index_t m,
         index_t n, scalar_t alpha, index_t batch_size, int batch_type_i,
         bool* success) {
  // Standard test setup.
  index_t lda = side == 'l' ? m : n;
  index_t ldb = m;
  index_t k = side == 'l' ? m : n;

  auto batch_type = static_cast<blas::gemm_batch_type_t>(batch_type_i);

  {
    // The counters are double. We convert m, n, k and batch_size to double to
    // avoid integer overflows for n_fl_ops and bytes_processed
    const double m_d = static_cast<double>(m);
    const double n_d = static_cast<double>(n);
    const double batch_size_d = static_cast<double>(batch_size);
    const double k_d = static_cast<double>(k);

    state.counters["m"] = m_d;
    state.counters["n"] = n_d;
    state.counters["batch_size"] = batch_size_d;

    const double mem_readA = k_d * (k_d + 1) / 2;
    const double mem_readBwriteB = 2 * m_d * n_d;
    const double total_mem =
        ((mem_readA + mem_readBwriteB) * sizeof(scalar_t)) * batch_size;
    state.counters["bytes_processed"] = total_mem;
    state.SetBytesProcessed(state.iterations() * total_mem);

    const double nflops_AtimesB =
        2 * k_d * (k_d + 1) / 2 * (side == 'l' ? n_d : m_d);
    const double nflops_timesAlpha = m_d * n_d;
    const double total_nflops =
        (nflops_AtimesB + nflops_timesAlpha) * batch_size;
    state.counters["n_fl_ops"] = total_nflops;
    state.SetItemsProcessed(state.iterations() * total_nflops);
  }

  // Matrix options (rocBLAS)
  const rocblas_side side_rb =
      (side == 'l') ? rocblas_side_left : rocblas_side_right;

  const rocblas_fill uplo_rb =
      (uplo == 'u') ? rocblas_fill_upper : rocblas_fill_lower;

  const rocblas_operation trans_rb =
      (trans == 'n') ? rocblas_operation_none : rocblas_operation_transpose;

  const rocblas_diagonal diag_rb =
      (diag == 'u') ? rocblas_diagonal_unit : rocblas_diagonal_non_unit;

  // Data sizes
  const int a_size = lda * k;
  const int b_size = ldb * n;

  // Matrices
  std::vector<scalar_t> a(a_size * batch_size);
  {
    // Populate the main input diagonal for each batch.
    std::vector<scalar_t> a_batch(a_size);
    for (int i = 0; i < batch_size; ++i) {
      const scalar_t diagValue =
          diag == 'u' ? scalar_t{1}
                      : blas_benchmark::utils::random_scalar<scalar_t>(
                            scalar_t{1}, scalar_t{10});
      blas_benchmark::utils::fill_trsm_matrix(a_batch, k, lda, uplo, diagValue,
                                              scalar_t{0});

      std::copy(a_batch.begin(), a_batch.end(), a.begin() + i * a_size);
    }
  }
  std::vector<scalar_t> b =
      blas_benchmark::utils::random_data<scalar_t>(b_size * batch_size);

  {
    // Device memory allocation & H2D copy
    blas_benchmark::utils::HIPVectorBatched<scalar_t> a_batched_gpu(
        a_size, batch_size, a.data());
    blas_benchmark::utils::HIPVectorBatched<scalar_t> b_batched_gpu(
        b_size, batch_size, b.data());

#ifdef BLAS_VERIFY_BENCHMARK
    // Reference batched trsm
    std::vector<scalar_t> x_ref = b;
    for (int batch = 0; batch < batch_size; batch++) {
      reference_blas::trsm(&side, &uplo, &trans, &diag, m, n, alpha,
                           a.data() + batch * a_size, lda,
                           x_ref.data() + batch * b_size, ldb);
    }

    // Rocblas verification trsm_batched
    std::vector<scalar_t> x_temp = b;
    {
      blas_benchmark::utils::HIPVectorBatched<scalar_t, true> x_temp_gpu(
          b_size, batch_size, x_temp.data());

      rocblas_trsm_batched_f<scalar_t>(rb_handle, side_rb, uplo_rb, trans_rb,
                                       diag_rb, m, n, &alpha, a_batched_gpu,
                                       lda, x_temp_gpu, ldb, batch_size);
    }

    std::ostringstream err_stream;
    if (!utils::compare_vectors(x_temp, x_ref, err_stream, "")) {
      const std::string& err_str = err_stream.str();
      state.SkipWithError(err_str.c_str());
      *success = false;
    };

#endif

    auto blas_warmup = [&]() -> void {
      rocblas_trsm_batched_f<scalar_t>(rb_handle, side_rb, uplo_rb, trans_rb,
                                       diag_rb, m, n, &alpha, a_batched_gpu,
                                       lda, b_batched_gpu, ldb, batch_size);
      return;
    };

    hipEvent_t start, stop;
    CHECK_HIP_ERROR(hipEventCreate(&start));
    CHECK_HIP_ERROR(hipEventCreate(&stop));

    auto blas_method_def = [&]() -> std::vector<hipEvent_t> {
      CHECK_HIP_ERROR(hipEventRecord(start, NULL));
      rocblas_trsm_batched_f<scalar_t>(rb_handle, side_rb, uplo_rb, trans_rb,
                                       diag_rb, m, n, &alpha, a_batched_gpu,
                                       lda, b_batched_gpu, ldb, batch_size);
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

    blas_benchmark::utils::calc_avg_counters(state);

    CHECK_HIP_ERROR(hipEventDestroy(start));
    CHECK_HIP_ERROR(hipEventDestroy(stop));
  }  // release device memory via utils::DeviceVector destructors
};

template <typename scalar_t>
void register_benchmark(blas_benchmark::Args& args, rocblas_handle& rb_handle,
                        bool* success) {
  auto trsm_batched_params =
      blas_benchmark::utils::get_trsm_batched_params<scalar_t>(args);

  for (auto p : trsm_batched_params) {
    char s_side, s_uplo, s_t, s_diag;
    index_t m, n, batch_size;
    scalar_t alpha;
    int batch_type;
    std::tie(s_side, s_uplo, s_t, s_diag, m, n, alpha, batch_size, batch_type) =
        p;

    auto batch_type_enum = static_cast<blas::gemm_batch_type_t>(batch_type);
    if (batch_type_enum == blas::gemm_batch_type_t::interleaved) {
      std::cerr << "interleaved memory for trsm_batched operator is not "
                   "supported by rocBLAS\n";
      continue;
    }

    auto BM_lambda = [&](benchmark::State& st, rocblas_handle rb_handle,
                         char side, char uplo, char t, char diag, index_t m,
                         index_t n, scalar_t alpha, index_t batch_size,
                         int batch_type, bool* success) {
      run<scalar_t>(st, rb_handle, side, uplo, t, diag, m, n, alpha, batch_size,
                    batch_type, success);
    };
    benchmark::RegisterBenchmark(
        get_name<scalar_t>(s_side, s_uplo, s_t, s_diag, m, n, batch_size,
                           batch_type)
            .c_str(),
        BM_lambda, rb_handle, s_side, s_uplo, s_t, s_diag, m, n, alpha,
        batch_size, batch_type, success)
        ->UseRealTime();
  }
}

namespace blas_benchmark {
void create_benchmark(blas_benchmark::Args& args, rocblas_handle& rb_handle,
                      bool* success) {
  BLAS_REGISTER_BENCHMARK(args, rb_handle, success);
}
}  // namespace blas_benchmark
