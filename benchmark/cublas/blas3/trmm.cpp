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
static inline void cublas_routine(args_t&&... args) {
  if constexpr (std::is_same_v<scalar_t, float>) {
    CUBLAS_CHECK(cublasStrmm(std::forward<args_t>(args)...));
  } else if constexpr (std::is_same_v<scalar_t, double>) {
    CUBLAS_CHECK(cublasDtrmm(std::forward<args_t>(args)...));
  }
  return;
}

template <typename scalar_t>
void run(benchmark::State& state, cublasHandle_t* cuda_handle_ptr, char side,
         char uplo, char t, char diag, index_t m, index_t n, scalar_t alpha,
         bool* success) {
  // initialize the state label
  blas_benchmark::utils::set_benchmark_label<scalar_t>(state);

  index_t lda = side == 'l' ? m : n;
  index_t ldb = m;
  index_t ldc = ldb;

  blas_benchmark::utils::init_level_3_counters<
      blas_benchmark::utils::Level3Op::trmm, scalar_t>(state, 0, m, n, 0, 1,
                                                       side);

  cublasHandle_t& cuda_handle = *cuda_handle_ptr;

  const index_t tr_m_size = (side == 'l') ? lda * m : lda * n;

  // Matrices
  std::vector<scalar_t> a(tr_m_size);
  std::vector<scalar_t> b =
      blas_benchmark::utils::random_data<scalar_t>(ldb * n);

  const scalar_t diagValue =
      (diag == 'u') ? scalar_t{1}
                    : blas_benchmark::utils::random_scalar<scalar_t>(
                          scalar_t{1}, scalar_t{10});

  const auto a_other_dim{tr_m_size / lda};
  blas_benchmark::utils::fill_trsm_matrix(a, a_other_dim, lda, uplo, diagValue,
                                          scalar_t{0});

  blas_benchmark::utils::CUDAVector<scalar_t> a_gpu(tr_m_size, a.data());
  blas_benchmark::utils::CUDAVector<scalar_t> b_gpu(ldb * n, b.data());
  blas_benchmark::utils::CUDAVector<scalar_t> c_gpu(ldc * n);

  cublasSideMode_t c_side =
      (side == 'l') ? CUBLAS_SIDE_LEFT : CUBLAS_SIDE_RIGHT;

  cublasFillMode_t c_uplo =
      (uplo == 'u') ? CUBLAS_FILL_MODE_UPPER : CUBLAS_FILL_MODE_LOWER;

  cublasOperation_t c_t = (t == 'n') ? CUBLAS_OP_N : CUBLAS_OP_T;

  cublasDiagType_t c_diag =
      (diag == 'u') ? CUBLAS_DIAG_UNIT : CUBLAS_DIAG_NON_UNIT;

#ifdef BLAS_VERIFY_BENCHMARK
  // Run a first time with a verification of the results
  std::vector<scalar_t> b_ref = b;
  reference_blas::trmm(&side, &uplo, &t, &diag, m, n, alpha, a.data(), lda,
                       b_ref.data(), ldb);
  std::vector<scalar_t> c_temp(ldb * n);
  {
    blas_benchmark::utils::CUDAVector<scalar_t, true> c_temp_gpu(m * n,
                                                                 c_temp.data());
    cublas_routine<scalar_t>(cuda_handle, c_side, c_uplo, c_t, c_diag, m, n,
                             &alpha, a_gpu, lda, b_gpu, ldb, c_temp_gpu, ldc);
  }

  std::ostringstream err_stream;
  if (!utils::compare_vectors(c_temp, b_ref, err_stream, "")) {
    const std::string& err_str = err_stream.str();
    state.SkipWithError(err_str.c_str());
    *success = false;
  };
#endif
  auto blas_warmup = [&]() -> void {
    cublas_routine<scalar_t>(cuda_handle, c_side, c_uplo, c_t, c_diag, m, n,
                             &alpha, a_gpu, lda, b_gpu, ldb, c_gpu, ldc);
    return;
  };

  cudaEvent_t start;
  cudaEvent_t stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));

  auto blas_method_def = [&]() -> std::vector<cudaEvent_t> {
    CUDA_CHECK(cudaEventRecord(start));
    cublas_routine<scalar_t>(cuda_handle, c_side, c_uplo, c_t, c_diag, m, n,
                             &alpha, a_gpu, lda, b_gpu, ldb, c_gpu, ldc);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    return std::vector{start, stop};
  };

  // Warmup
  blas_benchmark::utils::warmup(blas_warmup);
  CUDA_CHECK(cudaStreamSynchronize(NULL));

  blas_benchmark::utils::init_counters(state);

  // Measure
  for (auto _ : state) {
    // Run
    std::tuple<double, double> times =
        blas_benchmark::utils::timef_cuda(blas_method_def);

    // Report
    blas_benchmark::utils::update_counters(state, times);
  }

  state.SetItemsProcessed(state.iterations() * state.counters["n_fl_ops"]);
  state.SetBytesProcessed(state.iterations() *
                          state.counters["bytes_processed"]);

  blas_benchmark::utils::calc_avg_counters(state);

  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
};

template <typename scalar_t>
void register_benchmark(blas_benchmark::Args& args,
                        cublasHandle_t* cuda_handle_ptr, bool* success) {
  // Get params from trsm since they are the same
  auto trmm_params = blas_benchmark::utils::get_trsm_params<scalar_t>(args);

  for (auto p : trmm_params) {
    char s_side, s_uplo, s_t, s_diag;
    index_t m, n;
    scalar_t alpha;
    std::tie(s_side, s_uplo, s_t, s_diag, m, n, alpha) = p;

    auto BM_lambda = [&](benchmark::State& st, cublasHandle_t* cuda_handle_ptr,
                         char side, char uplo, char t, char diag, index_t m,
                         index_t n, scalar_t alpha, bool* success) {
      run<scalar_t>(st, cuda_handle_ptr, side, uplo, t, diag, m, n, alpha,
                    success);
    };
    benchmark::RegisterBenchmark(
        blas_benchmark::utils::get_name<benchmark_op, scalar_t>(
            s_side, s_uplo, s_t, s_diag, m, n,
            blas_benchmark::utils::MEM_TYPE_USM)
            .c_str(),
        BM_lambda, cuda_handle_ptr, s_side, s_uplo, s_t, s_diag, m, n, alpha,
        success)
        ->UseRealTime();
  }
}

namespace blas_benchmark {
void create_benchmark(blas_benchmark::Args& args,
                      cublasHandle_t* cuda_handle_ptr, bool* success) {
  BLAS_REGISTER_BENCHMARK(args, cuda_handle_ptr, success);
}
}  // namespace blas_benchmark
