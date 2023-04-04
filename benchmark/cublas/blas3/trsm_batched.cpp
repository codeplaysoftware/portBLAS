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
                     const char diag, int m, int n, int batch_count,
                     int batch_type) {
  std::ostringstream str{};
  str << "BM_TrsmBatched<" << blas_benchmark::utils::get_type_name<scalar_t>()
      << ">/" << side << "/" << uplo << "/" << t << "/" << diag << "/" << m
      << "/" << n << "/" << batch_count << "/"
      << blas_benchmark::utils::batch_type_to_str(batch_type);
  return str.str();
}

template <typename scalar_t, typename... args_t>
static inline void cublas_routine(args_t&&... args) {
  if constexpr (std::is_same_v<scalar_t, float>) {
    CUBLAS_CHECK(cublasStrsmBatched(std::forward<args_t>(args)...));
  } else if constexpr (std::is_same_v<scalar_t, double>) {
    CUBLAS_CHECK(cublasDtrsmBatched(std::forward<args_t>(args)...));
  }
  return;
}

template <typename scalar_t>
void run(benchmark::State& state, cublasHandle_t* cuda_handle_ptr,
         const char side, const char uplo, const char t, const char diag,
         index_t m, index_t n, scalar_t alpha, index_t batch_count,
         int batch_type_i, bool* success) {
  // Standard test setup.
  index_t lda = side == 'l' ? m : n;
  index_t ldb = m;
  index_t k = side == 'l' ? m : n;

  auto batch_type = static_cast<blas::gemm_batch_type_t>(batch_type_i);

  // The counters are double. We convert m, n, k and batch_count to double to
  // avoid integer overflows for n_fl_ops and bytes_processed
  const double m_d = static_cast<double>(m);
  const double n_d = static_cast<double>(n);
  const double batch_count_d = static_cast<double>(batch_count);
  const double k_d = static_cast<double>(k);

  state.counters["m"] = m_d;
  state.counters["n"] = n_d;
  state.counters["batch_count"] = batch_count_d;

  const double mem_readA = k_d * (k_d + 1) / 2;
  const double mem_readBwriteB = 2 * m_d * n_d;
  const double total_mem =
      ((mem_readA + mem_readBwriteB) * sizeof(scalar_t)) * batch_count;
  state.counters["bytes_processed"] = total_mem;

  const double nflops_AtimesB =
      2 * k_d * (k_d + 1) / 2 * (side == 'l' ? n_d : m_d);
  const double nflops_timesAlpha = m_d * n_d;
  const double nflops_tot = (nflops_AtimesB + nflops_timesAlpha) * batch_count;
  state.counters["n_fl_ops"] = nflops_tot;

  cublasHandle_t& cuda_handle = *cuda_handle_ptr;

  const int a_m_size = (side == 'l') ? lda * m : lda * n;

  // Matrices
  std::vector<scalar_t> a(a_m_size * batch_count);
  std::vector<scalar_t> b =
      blas_benchmark::utils::random_data<scalar_t>(ldb * n * batch_count);

  {
    auto copy_vector = [&a, a_m_size](std::vector<scalar_t>& v,
                                      const int curr_batch) -> void {
      for (int i = 0; i < a_m_size; ++i) {
        a[curr_batch * a_m_size + i] = v[i];
      }
      return;
    };

    std::vector<scalar_t> temp_a(a_m_size);
    for (int i = 0; i < batch_count; ++i) {
      const scalar_t diagValue =
          diag == 'u' ? scalar_t{1}
                      : blas_benchmark::utils::random_scalar<scalar_t>(
                            scalar_t{1}, scalar_t{10});

      blas_benchmark::utils::fill_trsm_matrix(temp_a, (a_m_size / lda), lda,
                                              uplo, diagValue, scalar_t{0});

      copy_vector(temp_a, i);
      temp_a.clear();
    }
  }

  blas_benchmark::utils::CUDAVectorBatched<scalar_t> d_A_array(a_m_size,
                                                               batch_count, a);
  blas_benchmark::utils::CUDAVectorBatched<scalar_t> d_B_array(ldb * n,
                                                               batch_count, b);

  cublasSideMode_t c_side =
      (side == 'l') ? CUBLAS_SIDE_LEFT : CUBLAS_SIDE_RIGHT;

  cublasOperation_t c_t = (t == 'n') ? CUBLAS_OP_N : CUBLAS_OP_T;

  cublasFillMode_t c_uplo =
      (uplo == 'u') ? CUBLAS_FILL_MODE_UPPER : CUBLAS_FILL_MODE_LOWER;

  cublasDiagType_t c_diag =
      (diag == 'n') ? CUBLAS_DIAG_NON_UNIT : CUBLAS_DIAG_UNIT;

#ifdef BLAS_VERIFY_BENCHMARK
  // Run once verifying the results against the reference blas implementation.
  {
    std::vector<scalar_t> x_ref = b;
    std::vector<scalar_t> b_temp = b;
    auto _base = [=](index_t dim0, index_t dim1, index_t idx) {
      return dim0 * dim1 * idx;
    };
    for (int batch_idx = 0; batch_idx < batch_count; batch_idx++) {
      reference_blas::trsm(&side, &uplo, &t, &diag, m, n,
                           static_cast<scalar_t>(alpha),
                           a.data() + _base(lda, a_m_size / lda, batch_idx),
                           lda, x_ref.data() + _base(m, n, batch_idx), ldb);
    }

    {
      blas_benchmark::utils::CUDAVectorBatched<scalar_t, true> b_temp_gpu(
          n * m, batch_count, b_temp);
      cublas_routine<scalar_t>(cuda_handle, c_side, c_uplo, c_t, c_diag, m, n,
                               &alpha, d_A_array.get_batch_array(), lda,
                               b_temp_gpu.get_batch_array(), ldb, batch_count);
    }

    std::ostringstream err_stream;
    for (int i = 0; i < batch_count; ++i) {
      if (!utils::compare_vectors(b_temp, x_ref, err_stream, "")) {
        const std::string& err_str = err_stream.str();
        state.SkipWithError(err_str.c_str());
        *success = false;
      };
    }

  }  // close scope for verify benchmark
#endif

  auto blas_warmup = [&]() -> void {
    cublas_routine<scalar_t>(cuda_handle, c_side, c_uplo, c_t, c_diag, m, n,
                             &alpha, d_A_array.get_batch_array(), lda,
                             d_B_array.get_batch_array(), ldb, batch_count);
    return;
  };

  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));

  auto blas_method_def = [&]() -> std::vector<cudaEvent_t> {
    CUDA_CHECK(cudaEventRecord(start));
    cublas_routine<scalar_t>(cuda_handle, c_side, c_uplo, c_t, c_diag, m, n,
                             &alpha, d_A_array.get_batch_array(), lda,
                             d_B_array.get_batch_array(), ldb, batch_count);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));
    return std::vector{start, stop};
  };

  // Warmup
  blas_benchmark::utils::warmup(blas_method_def);
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

  state.SetBytesProcessed(state.iterations() * total_mem);
  state.SetItemsProcessed(state.iterations() * nflops_tot);

  blas_benchmark::utils::calc_avg_counters(state);
};

template <typename scalar_t>
void register_benchmark(blas_benchmark::Args& args,
                        cublasHandle_t* cuda_handle_ptr, bool* success) {
  auto trsm_batched_params =
      blas_benchmark::utils::get_trsm_batched_params<scalar_t>(args);

  for (auto p : trsm_batched_params) {
    char s_side, s_uplo, s_t, s_diag;
    index_t m, n, batch_count;
    scalar_t alpha;
    int batch_type;
    std::tie(s_side, s_uplo, s_t, s_diag, m, n, alpha, batch_count,
             batch_type) = p;

    auto batch_type_enum = static_cast<blas::gemm_batch_type_t>(batch_type);
    if (batch_type_enum == blas::gemm_batch_type_t::interleaved) {
      std::cerr << "interleaved memory for trsm_batched operator is not "
                   "supported by cuBLAS\n";
      continue;
    }

    auto BM_lambda = [&](benchmark::State& st, cublasHandle_t* cuda_handle_ptr,
                         char side, char uplo, char t, char diag, index_t m,
                         index_t n, scalar_t alpha, index_t batch_count,
                         int batch_type, bool* success) {
      run<scalar_t>(st, cuda_handle_ptr, side, uplo, t, diag, m, n, alpha,
                    batch_count, batch_type, success);
    };
    benchmark::RegisterBenchmark(
        get_name<scalar_t>(s_side, s_uplo, s_t, s_diag, m, n, batch_count,
                           batch_type)
            .c_str(),
        BM_lambda, cuda_handle_ptr, s_side, s_uplo, s_t, s_diag, m, n, alpha,
        batch_count, batch_type, success)
        ->UseRealTime();
  }
}

namespace blas_benchmark {
void create_benchmark(blas_benchmark::Args& args,
                      cublasHandle_t* cuda_handle_ptr, bool* success) {
  BLAS_REGISTER_BENCHMARK(args, cuda_handle_ptr, success);
}
}  // namespace blas_benchmark
