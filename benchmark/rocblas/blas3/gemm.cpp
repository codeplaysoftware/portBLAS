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
 *  @filename gemm.cpp
 *
 **************************************************************************/

#include "../utils.hpp"

constexpr blas_benchmark::utils::Level3Op benchmark_op =
    blas_benchmark::utils::Level3Op::gemm;

template <typename scalar_t, typename... args_t>
static inline void rocblas_gemm_f(args_t&&... args) {
  if constexpr (std::is_same_v<scalar_t, float>) {
    CHECK_ROCBLAS_STATUS(rocblas_sgemm(std::forward<args_t>(args)...));
  } else if constexpr (std::is_same_v<scalar_t, double>) {
    CHECK_ROCBLAS_STATUS(rocblas_dgemm(std::forward<args_t>(args)...));
  }
  return;
}

template <typename scalar_t>
void run(benchmark::State& state, rocblas_handle& rb_handle, int t_a_i,
         int t_b_i, index_t m, index_t k, index_t n, scalar_t alpha,
         scalar_t beta, bool* success) {
  // initialize the state label
  blas_benchmark::utils::set_benchmark_label<scalar_t>(state);

  // Standard test setup.
  std::string t_a = blas_benchmark::utils::from_transpose_enum(
      static_cast<blas_benchmark::utils::Transposition>(t_a_i));
  std::string t_b = blas_benchmark::utils::from_transpose_enum(
      static_cast<blas_benchmark::utils::Transposition>(t_b_i));
  const char* t_a_str = t_a.c_str();
  const char* t_b_str = t_b.c_str();

  index_t lda = t_a_str[0] == 'n' ? m : k;
  index_t ldb = t_b_str[0] == 'n' ? k : n;
  index_t ldc = m;

  blas_benchmark::utils::init_level_3_counters<
      blas_benchmark::utils::Level3Op::gemm, scalar_t>(state, beta, m, n, k);

  // Matrix options (rocBLAS)
  const rocblas_operation trans_a_rb =
      t_a_str[0] == 'n' ? rocblas_operation_none : rocblas_operation_transpose;
  const rocblas_operation trans_b_rb =
      t_b_str[0] == 'n' ? rocblas_operation_none : rocblas_operation_transpose;

  // Data sizes
  const int a_size = m * k;
  const int b_size = k * n;
  const int c_size = m * n;

  // Matrices
  std::vector<scalar_t> a =
      blas_benchmark::utils::random_data<scalar_t>(a_size);
  std::vector<scalar_t> b =
      blas_benchmark::utils::random_data<scalar_t>(b_size);
  std::vector<scalar_t> c =
      blas_benchmark::utils::const_data<scalar_t>(c_size, 0);

  {
    // Device memory allocation & H2D copy
    blas_benchmark::utils::HIPVector<scalar_t> a_gpu(a_size, a.data());
    blas_benchmark::utils::HIPVector<scalar_t> b_gpu(b_size, b.data());
    blas_benchmark::utils::HIPVector<scalar_t> c_gpu(c_size, c.data());

#ifdef BLAS_VERIFY_BENCHMARK
    // Reference gemm
    std::vector<scalar_t> c_ref = c;
    reference_blas::gemm(t_a_str, t_b_str, m, n, k, alpha, a.data(), lda,
                         b.data(), ldb, beta, c_ref.data(), ldc);

    // Rocblas verification gemm
    std::vector<scalar_t> c_temp = c;
    {
      blas_benchmark::utils::HIPVector<scalar_t, true> c_temp_gpu(
          c_size, c_temp.data());
      rocblas_gemm_f<scalar_t>(rb_handle, trans_a_rb, trans_b_rb, m, n, k,
                               &alpha, a_gpu, lda, b_gpu, ldb, &beta,
                               c_temp_gpu, ldc);
    }

    std::ostringstream err_stream;
    if (!utils::compare_vectors(c_temp, c_ref, err_stream, "")) {
      const std::string& err_str = err_stream.str();
      state.SkipWithError(err_str.c_str());
      *success = false;
    };
#endif
    auto blas_warmup = [&]() -> void {
      rocblas_gemm_f<scalar_t>(rb_handle, trans_a_rb, trans_b_rb, m, n, k,
                               &alpha, a_gpu, lda, b_gpu, ldb, &beta, c_gpu,
                               ldc);
      return;
    };

    hipEvent_t start, stop;
    CHECK_HIP_ERROR(hipEventCreate(&start));
    CHECK_HIP_ERROR(hipEventCreate(&stop));

    auto blas_method_def = [&]() -> std::vector<hipEvent_t> {
      CHECK_HIP_ERROR(hipEventRecord(start, NULL));
      rocblas_gemm_f<scalar_t>(rb_handle, trans_a_rb, trans_b_rb, m, n, k,
                               &alpha, a_gpu, lda, b_gpu, ldb, &beta, c_gpu,
                               ldc);
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
  auto gemm_params = blas_benchmark::utils::get_blas3_params<scalar_t>(args);

  for (auto p : gemm_params) {
    std::string t_a, t_b;
    index_t m, n, k;
    scalar_t alpha, beta;
    std::tie(t_a, t_b, m, k, n, alpha, beta) = p;
    int t_a_i = static_cast<int>(blas_benchmark::utils::to_transpose_enum(t_a));
    int t_b_i = static_cast<int>(blas_benchmark::utils::to_transpose_enum(t_b));

    auto BM_lambda = [&](benchmark::State& st, rocblas_handle rb_handle,
                         int t1i, int t2i, index_t m, index_t k, index_t n,
                         scalar_t alpha, scalar_t beta, bool* success) {
      run<scalar_t>(st, rb_handle, t1i, t2i, m, k, n, alpha, beta, success);
    };
    benchmark::RegisterBenchmark(
        blas_benchmark::utils::get_name<benchmark_op, scalar_t>(
            t_a, t_b, m, k, n, blas_benchmark::utils::MEM_TYPE_USM)
            .c_str(),
        BM_lambda, rb_handle, t_a_i, t_b_i, m, k, n, alpha, beta, success)
        ->UseRealTime();
  }
}

namespace blas_benchmark {
void create_benchmark(blas_benchmark::Args& args, rocblas_handle& rb_handle,
                      bool* success) {
  BLAS_REGISTER_BENCHMARK(args, rb_handle, success);
}
}  // namespace blas_benchmark
