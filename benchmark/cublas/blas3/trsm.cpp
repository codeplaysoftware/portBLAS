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
 *  @filename trsm.cpp
 *
 **************************************************************************/

#include "../utils.hpp"

constexpr blas_benchmark::utils::Level3Op benchmark_op =
    blas_benchmark::utils::Level3Op::trsm;

template <typename scalar_t, typename... args_t>
static inline void cublas_routine(args_t&&... args) {
  if constexpr (std::is_same_v<scalar_t, float>) {
    CUBLAS_CHECK(cublasStrsm(std::forward<args_t>(args)...));
  } else if constexpr (std::is_same_v<scalar_t, double>) {
    CUBLAS_CHECK(cublasDtrsm(std::forward<args_t>(args)...));
  }
  return;
}

template <typename scalar_t>
void run(benchmark::State& state, cublasHandle_t* cuda_handle_ptr, char side,
         char uplo, char trans, char diag, index_t m, index_t n, scalar_t alpha,
         bool* success) {
  // initialize the state label
  blas_benchmark::utils::set_benchmark_label<scalar_t>(state);

  // Standard test setup.
  index_t lda = side == 'l' ? m : n;
  index_t ldb = m;
  index_t k = side == 'l' ? m : n;

  cublasHandle_t& cuda_handle = *cuda_handle_ptr;

  const int sizeA = k * lda;
  const int sizeB = n * ldb;

  blas_benchmark::utils::init_level_3_counters<
      blas_benchmark::utils::Level3Op::trsm, scalar_t>(state, 0, m, n, 0, 1,
                                                       side);

  // Matrices
  std::vector<scalar_t> a(sizeA);
  std::vector<scalar_t> b = blas_benchmark::utils::random_data<scalar_t>(sizeB);

  const scalar_t diagValue =
      diag == 'u' ? scalar_t{1}
                  : blas_benchmark::utils::random_scalar<scalar_t>(
                        scalar_t{1}, scalar_t{10});

  blas_benchmark::utils::fill_trsm_matrix(a, k, lda, uplo, diagValue,
                                          scalar_t{0});

  blas_benchmark::utils::CUDAVector<scalar_t> a_gpu(sizeA, a.data());
  blas_benchmark::utils::CUDAVector<scalar_t> b_gpu(sizeB, b.data());

  cublasSideMode_t c_side =
      (side == 'l') ? CUBLAS_SIDE_LEFT : CUBLAS_SIDE_RIGHT;

  cublasFillMode_t c_uplo =
      (uplo == 'u') ? CUBLAS_FILL_MODE_UPPER : CUBLAS_FILL_MODE_LOWER;

  cublasOperation_t c_t = (trans == 'n') ? CUBLAS_OP_N : CUBLAS_OP_T;

  cublasDiagType_t c_diag =
      (diag == 'u') ? CUBLAS_DIAG_UNIT : CUBLAS_DIAG_NON_UNIT;

#ifdef BLAS_VERIFY_BENCHMARK
  // Run once verifying the results against the reference blas implementation.
  std::vector<scalar_t> x_ref = b;
  std::vector<scalar_t> b_temp = b;

  reference_blas::trsm(&side, &uplo, &trans, &diag, m, n,
                       static_cast<scalar_t>(alpha), a.data(), lda,
                       x_ref.data(), ldb);

  {
    blas_benchmark::utils::CUDAVector<scalar_t, true> b_temp_gpu(sizeB,
                                                                 b_temp.data());
    cublas_routine<scalar_t>(cuda_handle, c_side, c_uplo, c_t, c_diag, m, n,
                             &alpha, a_gpu, lda, b_temp_gpu, ldb);
  }

  std::ostringstream err_stream;
  if (!utils::compare_vectors(b_temp, x_ref, err_stream, "")) {
    const std::string& err_str = err_stream.str();
    state.SkipWithError(err_str.c_str());
    *success = false;
  };
#endif
  auto blas_warmup = [&]() -> void {
    cublas_routine<scalar_t>(cuda_handle, c_side, c_uplo, c_t, c_diag, m, n,
                             &alpha, a_gpu, lda, b_gpu, ldb);
    return;
  };

  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));

  auto blas_method_def = [&]() -> std::vector<cudaEvent_t> {
    CUDA_CHECK(cudaEventRecord(start, NULL));
    cublas_routine<scalar_t>(cuda_handle, c_side, c_uplo, c_t, c_diag, m, n,
                             &alpha, a_gpu, lda, b_gpu, ldb);
    CUDA_CHECK(cudaEventRecord(stop, NULL));
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
  auto trsm_params = blas_benchmark::utils::get_trsm_params<scalar_t>(args);

  for (auto p : trsm_params) {
    char side, uplo, trans, diag;
    index_t m, n;
    scalar_t alpha;
    std::tie(side, uplo, trans, diag, m, n, alpha) = p;

    auto BM_lambda = [&](benchmark::State& st, cublasHandle_t* cuda_handle_ptr,
                         char side, char uplo, char trans, char diag, index_t m,
                         index_t n, scalar_t alpha, bool* success) {
      run<scalar_t>(st, cuda_handle_ptr, side, uplo, trans, diag, m, n, alpha,
                    success);
    };
    benchmark::RegisterBenchmark(
        blas_benchmark::utils::get_name<benchmark_op, scalar_t>(
            side, uplo, trans, diag, m, n, blas_benchmark::utils::MEM_TYPE_USM)
            .c_str(),
        BM_lambda, cuda_handle_ptr, side, uplo, trans, diag, m, n, alpha,
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
