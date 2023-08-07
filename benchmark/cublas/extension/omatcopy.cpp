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
 *  @filename omatcopy.cpp
 *
 **************************************************************************/

#include "../../../test/unittest/extension/extension_reference.hpp"
#include "../utils.hpp"

template <typename scalar_t>
std::string get_name(std::string ts_a, int m, int n, scalar_t alpha,
                     index_t lda_mul, index_t ldb_mul) {
  std::ostringstream str{};
  str << "BM_omatcopy<" << blas_benchmark::utils::get_type_name<scalar_t>()
      << ">/" << ts_a << "/" << m << "/" << n << "/" << alpha << "/" << lda_mul
      << "/" << ldb_mul;
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
void run(benchmark::State& state, cublasHandle_t* cuda_handle_ptr, int ti,
         index_t m, index_t n, scalar_t alpha, index_t lda_mul, index_t ldb_mul,
         bool* success) {
  // initialize the state label
  blas_benchmark::utils::set_benchmark_label<scalar_t>(state);

  // Standard test setup.
  std::string ts = blas_benchmark::utils::from_transpose_enum(
      static_cast<blas_benchmark::utils::Transposition>(ti));
  const char* t_str = ts.c_str();

  // These arguments follows cublas indication for sizes and leading dimensions
  // instead of following oneMKL specification.
  const auto cuda_lda = (*t_str == 't') ? lda_mul * n : lda_mul * m;
  const auto cuda_ldb = ldb_mul * m;
  const auto cuda_size_a = cuda_lda * ((*t_str == 't') ? m : n);
  const auto cuda_size_b = cuda_ldb * n;

  blas_benchmark::utils::init_extension_counters<
      blas_benchmark::utils::ExtensionOP::omatcopy, scalar_t>(
      state, t_str, m, n, lda_mul, ldb_mul);

  cublasHandle_t& cuda_handle = *cuda_handle_ptr;

  // Input matrix/vector, output vector.
  std::vector<scalar_t> m_a =
      blas_benchmark::utils::random_data<scalar_t>(cuda_size_a);
  std::vector<scalar_t> m_b =
      blas_benchmark::utils::random_data<scalar_t>(cuda_size_b);

  blas_benchmark::utils::CUDAVector<scalar_t> m_a_gpu(cuda_size_a, m_a.data());
  blas_benchmark::utils::CUDAVector<scalar_t> m_b_gpu(cuda_size_b, m_b.data());

  cublasOperation_t c_t_a = (*t_str == 'n') ? CUBLAS_OP_N : CUBLAS_OP_T;

  // beta set to zero to use cublasTgeam properly
  const scalar_t beta = static_cast<scalar_t>(0.0);
  // place holder to for second matrix in cublasTgeam
  cublasOperation_t c_t_b = CUBLAS_OP_N;

#ifdef BLAS_VERIFY_BENCHMARK
  // Run a first time with a verification of the results
  std::vector<scalar_t> m_b_ref = m_b;  // m_b;

  reference_blas::ext_omatcopy<false>(*t_str, m, n, alpha, m_a, cuda_lda,
                                      m_b_ref, cuda_ldb);

  std::vector<scalar_t> m_b_temp = m_b;
  {
    blas_benchmark::utils::CUDAVector<scalar_t, true> m_b_temp_gpu(
        cuda_size_b, m_b_temp.data());

    cublas_routine<scalar_t>(cuda_handle, c_t_a, c_t_b, m, n, &alpha, m_a_gpu,
                             cuda_lda, &beta, nullptr, cuda_ldb, m_b_temp_gpu,
                             cuda_ldb);
  }

  std::ostringstream err_stream;
  if (!utils::compare_vectors(m_b_temp, m_b_ref, err_stream, "")) {
    const std::string& err_str = err_stream.str();
    state.SkipWithError(err_str.c_str());
    *success = false;
  };
#endif
  auto blas_warmup = [&]() -> void {
    cublas_routine<scalar_t>(cuda_handle, c_t_a, c_t_b, m, n, &alpha, m_a_gpu,
                             cuda_lda, &beta, nullptr, cuda_ldb, m_b_gpu,
                             cuda_ldb);
    return;
  };

  cudaEvent_t start;
  cudaEvent_t stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));

  auto blas_method_def = [&]() -> std::vector<cudaEvent_t> {
    CUDA_CHECK(cudaEventRecord(start));
    cublas_routine<scalar_t>(cuda_handle, c_t_a, c_t_b, m, n, &alpha, m_a_gpu,
                             cuda_lda, &beta, nullptr, cuda_ldb, m_b_gpu,
                             cuda_ldb);
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
  auto omatcopy_params =
      blas_benchmark::utils::get_matcopy_params<scalar_t>(args);

  for (auto p : omatcopy_params) {
    std::string ts_a;
    index_t m, n, lda_mul, ldb_mul;
    scalar_t alpha;
    std::tie(ts_a, m, n, alpha, lda_mul, ldb_mul) = p;
    int t_a = static_cast<int>(blas_benchmark::utils::to_transpose_enum(ts_a));

    auto BM_lambda = [&](benchmark::State& st,
                         cublasHandle_t* cublas_handle_ptr, int t_a, index_t m,
                         index_t n, scalar_t alpha, index_t lda_mul,
                         index_t ldb_mul, bool* success) {
      run<scalar_t>(st, cublas_handle_ptr, t_a, m, n, alpha, lda_mul, ldb_mul,
                    success);
    };
    benchmark::RegisterBenchmark(
        get_name<scalar_t>(ts_a, m, n, alpha, lda_mul, ldb_mul).c_str(),
        BM_lambda, cublas_handle_ptr, t_a, m, n, alpha, lda_mul, ldb_mul,
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
