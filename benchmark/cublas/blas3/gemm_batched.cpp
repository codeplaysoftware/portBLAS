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
 *  @filename gemm_batched.cpp
 *
 **************************************************************************/

#include "../utils.hpp"

constexpr blas_benchmark::utils::Level3Op benchmark_op =
    blas_benchmark::utils::Level3Op::gemm_batched;

template <typename scalar_t, typename... args_t>
static inline void cublas_routine(args_t&&... args) {
  if constexpr (std::is_same_v<scalar_t, float>) {
    CUBLAS_CHECK(cublasSgemmBatched(std::forward<args_t>(args)...));
  } else if constexpr (std::is_same_v<scalar_t, double>) {
    CUBLAS_CHECK(cublasDgemmBatched(std::forward<args_t>(args)...));
  }
  return;
}

template <typename scalar_t>
void run(benchmark::State& state, cublasHandle_t* cuda_handle_ptr, index_t t1,
         index_t t2, index_t m, index_t k, index_t n, scalar_t alpha,
         scalar_t beta, index_t batch_count, int batch_type_i, bool* success) {
  // initialize the state label
  blas_benchmark::utils::set_benchmark_label<scalar_t>(state);

  // Standard setup
  std::string t1s = blas_benchmark::utils::from_transpose_enum(
      static_cast<blas_benchmark::utils::Transposition>(t1));
  std::string t2s = blas_benchmark::utils::from_transpose_enum(
      static_cast<blas_benchmark::utils::Transposition>(t2));
  const char* t_a = t1s.c_str();
  const char* t_b = t2s.c_str();
  auto batch_type = static_cast<blas::gemm_batch_type_t>(batch_type_i);

  index_t lda = t_a[0] == 'n' ? m : k;
  index_t ldb = t_b[0] == 'n' ? k : n;
  index_t ldc = m;

  blas_benchmark::utils::init_level_3_counters<
      blas_benchmark::utils::Level3Op::gemm_batched, scalar_t>(
      state, beta, m, n, k, batch_count);

  cublasHandle_t& cuda_handle = *cuda_handle_ptr;

  // Matrices
  std::vector<scalar_t> a =
      blas_benchmark::utils::random_data<scalar_t>(m * k * batch_count);
  std::vector<scalar_t> b =
      blas_benchmark::utils::random_data<scalar_t>(k * n * batch_count);
  std::vector<scalar_t> c =
      blas_benchmark::utils::const_data<scalar_t>(m * n * batch_count, 0);

  blas_benchmark::utils::CUDAVectorBatched<scalar_t> d_A_array(m * k,
                                                               batch_count, a);
  blas_benchmark::utils::CUDAVectorBatched<scalar_t> d_B_array(k * n,
                                                               batch_count, b);
  blas_benchmark::utils::CUDAVectorBatched<scalar_t> d_C_array(m * n,
                                                               batch_count);

  cublasOperation_t c_t_a = (*t_a == 'n') ? CUBLAS_OP_N : CUBLAS_OP_T;

  cublasOperation_t c_t_b = (*t_b == 'n') ? CUBLAS_OP_N : CUBLAS_OP_T;

#ifdef BLAS_VERIFY_BENCHMARK
  // Run a first time with a verification of the results
  {
    std::vector<scalar_t> c_ref = c;
    auto _base = [=](index_t dim0, index_t dim1, index_t idx) {
      return dim0 * dim1 * idx;
    };
    for (int batch_idx = 0; batch_idx < batch_count; batch_idx++) {
      reference_blas::gemm(t_a, t_b, m, n, k, alpha,
                           a.data() + _base(m, k, batch_idx), lda,
                           b.data() + _base(k, n, batch_idx), ldb, beta,
                           c_ref.data() + _base(m, n, batch_idx), ldc);
    }

    std::vector<scalar_t> c_temp(m * n * batch_count);

    {
      blas_benchmark::utils::CUDAVectorBatched<scalar_t, true> c_temp_gpu(
          n * m, batch_count, c_temp);
      cublas_routine<scalar_t>(cuda_handle, c_t_a, c_t_b, m, n, k, &alpha,
                               d_A_array.get_batch_array(), lda,
                               d_B_array.get_batch_array(), ldb, &beta,
                               c_temp_gpu.get_batch_array(), ldc, batch_count);
    }

    std::ostringstream err_stream;
    for (int i = 0; i < batch_count; ++i) {
      if (!utils::compare_vectors(c_temp, c_ref, err_stream, "")) {
        const std::string& err_str = err_stream.str();
        state.SkipWithError(err_str.c_str());
        *success = false;
      };
    }

  }  // close scope for verify benchmark
#endif

  auto blas_warmup = [&]() -> void {
    cublas_routine<scalar_t>(cuda_handle, c_t_a, c_t_b, m, n, k, &alpha,
                             d_A_array.get_batch_array(), lda,
                             d_B_array.get_batch_array(), ldb, &beta,
                             d_C_array.get_batch_array(), ldc, batch_count);
    return;
  };

  cudaEvent_t start, stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));

  auto blas_method_def = [&]() -> std::vector<cudaEvent_t> {
    CUDA_CHECK(cudaEventRecord(start));
    cublas_routine<scalar_t>(cuda_handle, c_t_a, c_t_b, m, n, k, &alpha,
                             d_A_array.get_batch_array(), lda,
                             d_B_array.get_batch_array(), ldb, &beta,
                             d_C_array.get_batch_array(), ldc, batch_count);
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
  auto gemm_batched_params =
      blas_benchmark::utils::get_gemm_batched_params<scalar_t>(args);

  for (auto p : gemm_batched_params) {
    std::string t1s, t2s;
    index_t m, n, k, batch_count;
    scalar_t alpha, beta;
    int batch_type;
    std::tie(t1s, t2s, m, k, n, alpha, beta, batch_count, batch_type) = p;

    if (batch_type == 1) {
      std::cerr << "interleaved memory for gemm_batched operator is not "
                   "supported by cuBLAS\n";
      continue;
    }

    int t1 = static_cast<int>(blas_benchmark::utils::to_transpose_enum(t1s));
    int t2 = static_cast<int>(blas_benchmark::utils::to_transpose_enum(t2s));

    auto BM_lambda = [&](benchmark::State& st, cublasHandle_t* cuda_handle_ptr,
                         int t1, int t2, index_t m, index_t k, index_t n,
                         scalar_t alpha, scalar_t beta, index_t batch_count,
                         int batch_type, bool* success) {
      run<scalar_t>(st, cuda_handle_ptr, t1, t2, m, k, n, alpha, beta,
                    batch_count, batch_type, success);
    };
    benchmark::RegisterBenchmark(
        blas_benchmark::utils::get_name<benchmark_op, scalar_t>(
            t1s, t2s, m, k, n, batch_count, batch_type,
            blas_benchmark::utils::MEM_TYPE_USM)
            .c_str(),
        BM_lambda, cuda_handle_ptr, t1, t2, m, k, n, alpha, beta, batch_count,
        batch_type, success)
        ->UseRealTime();
  }
}

namespace blas_benchmark {
void create_benchmark(blas_benchmark::Args& args,
                      cublasHandle_t* cuda_handle_ptr, bool* success) {
  BLAS_REGISTER_BENCHMARK(args, cuda_handle_ptr, success);
}
}  // namespace blas_benchmark
