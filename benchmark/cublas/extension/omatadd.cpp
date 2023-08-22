/* *************************************************************************
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
 *  @filename omatadd.cpp
 *
 **************************************************************************/

#include "../../../test/unittest/extension/extension_reference.hpp"
#include "../utils.hpp"

template <typename scalar_t>
std::string get_name(std::string ts_a, std::string ts_b, int m, int n,
                     scalar_t alpha, scalar_t beta, index_t lda_mul,
                     index_t ldb_mul, index_t ldc_mul) {
  std::ostringstream str{};
  str << "BM_omatadd<" << blas_benchmark::utils::get_type_name<scalar_t>()
      << ">/" << ts_a << "/" << ts_b << "/" << m << "/" << n << "/" << alpha
      << "/" << beta << "/" << lda_mul << "/" << ldb_mul << "/" << ldc_mul;
  return str.str();
}

template <typename scalar_t, typename... args_t>
static inline void cublas_routine(args_t&&... args) {
  if constexpr (std::is_same_v<scalar_t, float>) {
    CUBLAS_CHECK(cublasSgeam(std::forward<args_t>(args)...));
  } else if constexpr (std::is_same_v<scalar_t, double>) {
    CUBLAS_CHECK(cublasDgeam(std::forward<args_t>(args)...));
  }
  return;
}

template <typename scalar_t>
void run(benchmark::State& state, cublasHandle_t* cuda_handle_ptr, int ti_a,
         int ti_b, index_t m, index_t n, scalar_t alpha, scalar_t beta,
         index_t lda_mul, index_t ldb_mul, index_t ldc_mul, bool* success) {
  // initialize the state label
  blas_benchmark::utils::set_benchmark_label<scalar_t>(state);

  // Standard test setup.
  std::string ts_a = blas_benchmark::utils::from_transpose_enum(
      static_cast<blas_benchmark::utils::Transposition>(ti_a));
  const char* t_str_a = ts_a.c_str();
  std::string ts_b = blas_benchmark::utils::from_transpose_enum(
      static_cast<blas_benchmark::utils::Transposition>(ti_b));
  const char* t_str_b = ts_b.c_str();

  const auto lda = (*t_str_a == 't') ? lda_mul * n : lda_mul * m;
  const auto ldb = (*t_str_b == 't') ? ldb_mul * n : ldb_mul * m;
  const auto ldc = ldc_mul * m;

  const auto size_a = lda * ((*t_str_a == 't') ? m : n);
  const auto size_b = ldb * ((*t_str_b == 't') ? m : n);
  const auto size_c = ldc * n;

  blas_benchmark::utils::init_extension_counters<
      blas_benchmark::utils::ExtensionOP::omatadd, scalar_t>(
      state, t_str_a, t_str_b, m, n, lda_mul, ldb_mul, ldc_mul);

  cublasHandle_t& cuda_handle = *cuda_handle_ptr;

  // Input matrix/vector, output vector.
  std::vector<scalar_t> m_a =
      blas_benchmark::utils::random_data<scalar_t>(size_a);
  std::vector<scalar_t> m_b =
      blas_benchmark::utils::random_data<scalar_t>(size_b);
  std::vector<scalar_t> m_c =
      blas_benchmark::utils::random_data<scalar_t>(size_c);

  blas_benchmark::utils::CUDAVector<scalar_t> m_a_gpu(size_a, m_a.data());
  blas_benchmark::utils::CUDAVector<scalar_t> m_b_gpu(size_b, m_b.data());
  blas_benchmark::utils::CUDAVector<scalar_t> m_c_gpu(size_c, m_c.data());

  cublasOperation_t c_t_a = (*t_str_a == 'n') ? CUBLAS_OP_N : CUBLAS_OP_T;
  cublasOperation_t c_t_b = (*t_str_b == 'n') ? CUBLAS_OP_N : CUBLAS_OP_T;

#ifdef BLAS_VERIFY_BENCHMARK
  // Run a first time with a verification of the results
  std::vector<scalar_t> m_c_ref = m_c;

  reference_blas::ext_omatadd(*t_str_a, *t_str_b, m, n, alpha, m_a, lda, beta,
                              m_b, ldb, m_c_ref, ldc);

  std::vector<scalar_t> m_c_temp = m_c;
  {
    blas_benchmark::utils::CUDAVector<scalar_t, true> m_c_temp_gpu(
        size_c, m_c_temp.data());

    cublas_routine<scalar_t>(cuda_handle, c_t_a, c_t_b, m, n, &alpha, m_a_gpu,
                             lda, &beta, m_b_gpu, ldb, m_c_temp_gpu, ldc);
  }

  std::ostringstream err_stream;
  if (!utils::compare_vectors(m_c_temp, m_c_ref, err_stream, "")) {
    const std::string& err_str = err_stream.str();
    state.SkipWithError(err_str.c_str());
    *success = false;
  };
#endif
  auto blas_warmup = [&]() -> void {
    cublas_routine<scalar_t>(cuda_handle, c_t_a, c_t_b, m, n, &alpha, m_a_gpu,
                             lda, &beta, m_b_gpu, ldb, m_c_gpu, ldc);
    return;
  };

  cudaEvent_t start;
  cudaEvent_t stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));

  auto blas_method_def = [&]() -> std::vector<cudaEvent_t> {
    CUDA_CHECK(cudaEventRecord(start));
    cublas_routine<scalar_t>(cuda_handle, c_t_a, c_t_b, m, n, &alpha, m_a_gpu,
                             lda, &beta, m_b_gpu, ldb, m_c_gpu, ldc);
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
                        cublasHandle_t* cublas_handle_ptr, bool* success) {
  auto omatadd_params =
      blas_benchmark::utils::get_omatadd_params<scalar_t>(args);

  for (auto p : omatadd_params) {
    std::string ts_a, ts_b;
    index_t m, n, lda_mul, ldb_mul, ldc_mul;
    scalar_t alpha, beta;
    std::tie(ts_a, ts_b, m, n, alpha, beta, lda_mul, ldb_mul, ldc_mul) = p;
    int t_a = static_cast<int>(blas_benchmark::utils::to_transpose_enum(ts_a));
    int t_b = static_cast<int>(blas_benchmark::utils::to_transpose_enum(ts_b));

    auto BM_lambda =
        [&](benchmark::State& st, cublasHandle_t* cublas_handle_ptr, int t_a,
            int t_b, index_t m, index_t n, scalar_t alpha, scalar_t beta,
            index_t lda_mul, index_t ldb_mul, index_t ldc_mul, bool* success) {
          run<scalar_t>(st, cublas_handle_ptr, t_a, t_b, m, n, alpha, beta,
                        lda_mul, ldb_mul, ldc_mul, success);
        };
    benchmark::RegisterBenchmark(
        get_name<scalar_t>(ts_a, ts_b, m, n, alpha, beta, lda_mul, ldb_mul,
                           ldc_mul)
            .c_str(),
        BM_lambda, cublas_handle_ptr, t_a, t_b, m, n, alpha, beta, lda_mul,
        ldb_mul, ldc_mul, success)
        ->UseRealTime();
  }
}

namespace blas_benchmark {
void create_benchmark(blas_benchmark::Args& args,
                      cublasHandle_t* cuda_handle_ptr, bool* success) {
  BLAS_REGISTER_BENCHMARK(args, cuda_handle_ptr, success);
}
}  // namespace blas_benchmark
