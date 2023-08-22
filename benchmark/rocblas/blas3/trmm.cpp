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
 *  @filename trmm.cpp
 *
 **************************************************************************/

#include "../utils.hpp"

constexpr blas_benchmark::utils::Level3Op benchmark_op =
    blas_benchmark::utils::Level3Op::trmm;

template <typename scalar_t, typename... args_t>
static inline void rocblas_trmm_f(args_t&&... args) {
  if constexpr (std::is_same_v<scalar_t, float>) {
    CHECK_ROCBLAS_STATUS(rocblas_strmm(std::forward<args_t>(args)...));
  } else if constexpr (std::is_same_v<scalar_t, double>) {
    CHECK_ROCBLAS_STATUS(rocblas_dtrmm(std::forward<args_t>(args)...));
  }
  return;
}

template <typename scalar_t>
void run(benchmark::State& state, rocblas_handle& rb_handle, char side,
         char uplo, char trans, char diag, index_t m, index_t n, scalar_t alpha,
         bool* success) {
  // initialize the state label
  blas_benchmark::utils::set_benchmark_label<scalar_t>(state);

  // Standard test setup.
  index_t lda = side == 'l' ? m : n;
  index_t ldb = m;

  blas_benchmark::utils::init_level_3_counters<
      blas_benchmark::utils::Level3Op::trmm, scalar_t>(state, 0, m, n, 0, 1,
                                                       side);

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
  const index_t a_dim_2 = (side == 'l') ? m : n;
  const index_t a_size = lda * a_dim_2;
  const index_t b_size = ldb * n;

  // Matrices
  std::vector<scalar_t> a(a_size);
  std::vector<scalar_t> b =
      blas_benchmark::utils::random_data<scalar_t>(b_size);

  // Populate the main input diagonal.
  const scalar_t diag_value =
      diag == 'u' ? scalar_t{1}
                  : blas_benchmark::utils::random_scalar<scalar_t>();

  blas_benchmark::utils::fill_trsm_matrix(a, a_dim_2, lda, uplo, diag_value,
                                          scalar_t{0});

  {
    // Device memory allocation & H2D copy
    blas_benchmark::utils::HIPVector<scalar_t> a_gpu(a_size, a.data());
    blas_benchmark::utils::HIPVector<scalar_t> b_gpu(b_size, b.data());

#ifdef BLAS_VERIFY_BENCHMARK
    // Reference trmm
    std::vector<scalar_t> b_ref = b;
    reference_blas::trmm(&side, &uplo, &trans, &diag, m, n, alpha, a.data(),
                         lda, b_ref.data(), ldb);

    // Rocblas verification trmm
    std::vector<scalar_t> b_temp = b;
    {
      blas_benchmark::utils::HIPVector<scalar_t, true> b_temp_gpu(
          b_size, b_temp.data());
      rocblas_trmm_f<scalar_t>(rb_handle, side_rb, uplo_rb, trans_rb, diag_rb,
                               m, n, &alpha, a_gpu, lda, b_temp_gpu, ldb);
    }

    std::ostringstream err_stream;
    if (!utils::compare_vectors(b_temp, b_ref, err_stream, "")) {
      const std::string& err_str = err_stream.str();
      state.SkipWithError(err_str.c_str());
      *success = false;
    };
#endif

    auto blas_warmup = [&]() -> void {
      rocblas_trmm_f<scalar_t>(rb_handle, side_rb, uplo_rb, trans_rb, diag_rb,
                               m, n, &alpha, a_gpu, lda, b_gpu, ldb);
      return;
    };

    hipEvent_t start, stop;
    CHECK_HIP_ERROR(hipEventCreate(&start));
    CHECK_HIP_ERROR(hipEventCreate(&stop));

    auto blas_method_def = [&]() -> std::vector<hipEvent_t> {
      CHECK_HIP_ERROR(hipEventRecord(start, NULL));
      rocblas_trmm_f<scalar_t>(rb_handle, side_rb, uplo_rb, trans_rb, diag_rb,
                               m, n, &alpha, a_gpu, lda, b_gpu, ldb);
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

    state.SetBytesProcessed(state.iterations() *
                            state.counters["bytes_processed"]);
    state.SetItemsProcessed(state.iterations() * state.counters["n_fl_ops"]);

    blas_benchmark::utils::calc_avg_counters(state);

    CHECK_HIP_ERROR(hipEventDestroy(start));
    CHECK_HIP_ERROR(hipEventDestroy(stop));
  }  // release device memory via utils::DeviceVector destructors
};

template <typename scalar_t>
void register_benchmark(blas_benchmark::Args& args, rocblas_handle& rb_handle,
                        bool* success) {
  auto trmm_params = blas_benchmark::utils::get_trsm_params<scalar_t>(args);

  for (auto p : trmm_params) {
    char side, uplo, trans, diag;
    index_t m, n;
    scalar_t alpha;
    std::tie(side, uplo, trans, diag, m, n, alpha) = p;

    auto BM_lambda = [&](benchmark::State& st, rocblas_handle rb_handle,
                         char side, char uplo, char t, char diag, index_t m,
                         index_t n, scalar_t alpha, bool* success) {
      run<scalar_t>(st, rb_handle, side, uplo, t, diag, m, n, alpha, success);
    };
    benchmark::RegisterBenchmark(
        blas_benchmark::utils::get_name<benchmark_op, scalar_t>(
            side, uplo, trans, diag, m, n, blas_benchmark::utils::MEM_TYPE_USM)
            .c_str(),
        BM_lambda, rb_handle, side, uplo, trans, diag, m, n, alpha, success)
        ->UseRealTime();
  }
}

namespace blas_benchmark {
void create_benchmark(blas_benchmark::Args& args, rocblas_handle& rb_handle,
                      bool* success) {
  BLAS_REGISTER_BENCHMARK(args, rb_handle, success);
}
}  // namespace blas_benchmark
