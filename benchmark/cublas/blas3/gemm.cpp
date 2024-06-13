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
static inline void cublas_routine(args_t&&... args) {
  if constexpr (std::is_same_v<scalar_t, float>) {
    CUBLAS_CHECK(cublasSgemm(std::forward<args_t>(args)...));
  } else if constexpr (std::is_same_v<scalar_t, double>) {
    CUBLAS_CHECK(cublasDgemm(std::forward<args_t>(args)...));
  } else if constexpr (std::is_same_v<scalar_t, sycl::half>) {
    CUBLAS_CHECK(cublasHgemm(std::forward<args_t>(args)...));
  }
  return;
}

#ifdef BLAS_ENABLE_COMPLEX
template <typename scalar_t, typename... args_t>
static inline void cublas_cplx_routine(args_t&&... args) {
  if constexpr (std::is_same_v<scalar_t, float>) {
    CUBLAS_CHECK(cublasCgemm(std::forward<args_t>(args)...));
  } else if constexpr (std::is_same_v<scalar_t, double>) {
    CUBLAS_CHECK(cublasZgemm(std::forward<args_t>(args)...));
  }
  return;
}
#endif

template <typename scalar_t>
void run(benchmark::State& state, cublasHandle_t* cuda_handle_ptr, int t1,
         int t2, index_t m, index_t k, index_t n, scalar_t alpha, scalar_t beta,
         bool* success) {
  // scalar_t if scalar_t!=sycl::half, cuda::__half otherwise
  using cuda_scalar_t =
      typename blas_benchmark::utils::CudaType<scalar_t>::type;

  // initialize the state label
  blas_benchmark::utils::set_benchmark_label<scalar_t>(state);

  // Standard test setup.
  std::string t1s = blas_benchmark::utils::from_transpose_enum(
      static_cast<blas_benchmark::utils::Transposition>(t1));
  std::string t2s = blas_benchmark::utils::from_transpose_enum(
      static_cast<blas_benchmark::utils::Transposition>(t2));
  const char* t_a = t1s.c_str();
  const char* t_b = t2s.c_str();

  index_t lda = t_a[0] == 'n' ? m : k;
  index_t ldb = t_b[0] == 'n' ? k : n;
  index_t ldc = m;

  blas_benchmark::utils::init_level_3_counters<
      blas_benchmark::utils::Level3Op::gemm, scalar_t>(state, beta, m, n, k);

  cublasHandle_t& cuda_handle = *cuda_handle_ptr;

  // Matrices
  std::vector<scalar_t> a = blas_benchmark::utils::random_data<scalar_t>(m * k);
  std::vector<scalar_t> b = blas_benchmark::utils::random_data<scalar_t>(k * n);
  std::vector<scalar_t> c =
      blas_benchmark::utils::const_data<scalar_t>(m * n, 0);

  blas_benchmark::utils::CUDAVector<cuda_scalar_t> a_gpu(
      m * k, reinterpret_cast<cuda_scalar_t*>(a.data()));
  blas_benchmark::utils::CUDAVector<cuda_scalar_t> b_gpu(
      k * n, reinterpret_cast<cuda_scalar_t*>(b.data()));
  blas_benchmark::utils::CUDAVector<cuda_scalar_t> c_gpu(
      n * m, reinterpret_cast<cuda_scalar_t*>(c.data()));

  cublasOperation_t c_t_a = (*t_a == 'n') ? CUBLAS_OP_N : CUBLAS_OP_T;
  cublasOperation_t c_t_b = (*t_b == 'n') ? CUBLAS_OP_N : CUBLAS_OP_T;

  cuda_scalar_t alpha_cuda = *reinterpret_cast<cuda_scalar_t*>(&alpha);
  cuda_scalar_t beta_cuda = *reinterpret_cast<cuda_scalar_t*>(&beta);

#ifdef BLAS_VERIFY_BENCHMARK
  // Run a first time with a verification of the results
  std::vector<scalar_t> c_ref = c;
  reference_blas::gemm(t_a, t_b, m, n, k, alpha, a.data(), lda, b.data(), ldb,
                       beta, c_ref.data(), ldc);
  std::vector<scalar_t> c_temp = c;
  {
    blas_benchmark::utils::CUDAVector<cuda_scalar_t, true> c_temp_gpu(
        m * n, reinterpret_cast<cuda_scalar_t*>(c_temp.data()));
    cublas_routine<scalar_t>(cuda_handle, c_t_a, c_t_b, m, n, k, &alpha_cuda,
                             a_gpu, lda, b_gpu, ldb, &beta_cuda, c_temp_gpu,
                             ldc);
  }

  std::ostringstream err_stream;
  if (!utils::compare_vectors(c_temp, c_ref, err_stream, "")) {
    const std::string& err_str = err_stream.str();
    state.SkipWithError(err_str.c_str());
    *success = false;
  };
#endif

  auto blas_warmup = [&]() -> void {
    cublas_routine<scalar_t>(cuda_handle, c_t_a, c_t_b, m, n, k, &alpha_cuda,
                             a_gpu, lda, b_gpu, ldb, &beta_cuda, c_gpu, ldc);
    return;
  };

  cudaEvent_t start;
  cudaEvent_t stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));

  auto blas_method_def = [&]() -> std::vector<cudaEvent_t> {
    CUDA_CHECK(cudaEventRecord(start));
    cublas_routine<scalar_t>(cuda_handle, c_t_a, c_t_b, m, n, k, &alpha_cuda,
                             a_gpu, lda, b_gpu, ldb, &beta_cuda, c_gpu, ldc);
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
  auto gemm_params = blas_benchmark::utils::get_blas3_params<scalar_t>(args);

  for (auto p : gemm_params) {
    std::string t1s, t2s;
    index_t m, n, k;
    scalar_t alpha, beta;
    std::tie(t1s, t2s, m, k, n, alpha, beta) = p;
    int t1 = static_cast<int>(blas_benchmark::utils::to_transpose_enum(t1s));
    int t2 = static_cast<int>(blas_benchmark::utils::to_transpose_enum(t2s));

    auto BM_lambda = [&](benchmark::State& st, cublasHandle_t* cuda_handle_ptr,
                         int t1, int t2, index_t m, index_t k, index_t n,
                         scalar_t alpha, scalar_t beta, bool* success) {
      run<scalar_t>(st, cuda_handle_ptr, t1, t2, m, k, n, alpha, beta, success);
    };
    benchmark::RegisterBenchmark(
        blas_benchmark::utils::get_name<benchmark_op, scalar_t>(
            t1s, t2s, m, k, n, blas_benchmark::utils::MEM_TYPE_USM)
            .c_str(),
        BM_lambda, cuda_handle_ptr, t1, t2, m, k, n, alpha, beta, success)
        ->UseRealTime();
  }
}

#ifdef BLAS_ENABLE_COMPLEX
template <typename scalar_t>
using cudaComplex = typename std::conditional<sizeof(scalar_t) == 8,
                                              cuDoubleComplex, cuComplex>::type;

template <typename scalar_t>
void run(benchmark::State& state, cublasHandle_t* cuda_handle_ptr, int t1,
         int t2, index_t m, index_t k, index_t n, std::complex<scalar_t> alpha,
         std::complex<scalar_t> beta, bool* success) {
  // initialize the state label
  blas_benchmark::utils::set_benchmark_label<std::complex<scalar_t>>(state);

  // Standard test setup.
  std::string t1s = blas_benchmark::utils::from_transpose_enum(
      static_cast<blas_benchmark::utils::Transposition>(t1));
  std::string t2s = blas_benchmark::utils::from_transpose_enum(
      static_cast<blas_benchmark::utils::Transposition>(t2));
  const char* t_a = t1s.c_str();
  const char* t_b = t2s.c_str();

  index_t lda = t_a[0] == 'n' ? m : k;
  index_t ldb = t_b[0] == 'n' ? k : n;
  index_t ldc = m;

  blas_benchmark::utils::init_level_3_cplx_counters<
      blas_benchmark::utils::Level3Op::gemm, scalar_t>(state, beta, m, n, k,
                                                       static_cast<index_t>(1));

  cublasHandle_t& cuda_handle = *cuda_handle_ptr;

  // Matrices
  std::vector<std::complex<scalar_t>> a =
      blas_benchmark::utils::random_cplx_data<scalar_t>(m * k);
  std::vector<std::complex<scalar_t>> b =
      blas_benchmark::utils::random_cplx_data<scalar_t>(k * n);
  std::vector<std::complex<scalar_t>> c =
      blas_benchmark::utils::const_cplx_data<scalar_t>(m * n, 0);

  blas_benchmark::utils::CUDAVector<cudaComplex<scalar_t>> a_gpu(
      m * k, reinterpret_cast<cudaComplex<scalar_t>*>(a.data()));
  blas_benchmark::utils::CUDAVector<cudaComplex<scalar_t>> b_gpu(
      k * n, reinterpret_cast<cudaComplex<scalar_t>*>(b.data()));
  blas_benchmark::utils::CUDAVector<cudaComplex<scalar_t>> c_gpu(
      n * m, reinterpret_cast<cudaComplex<scalar_t>*>(c.data()));

  cublasOperation_t c_t_a = (*t_a == 'n') ? CUBLAS_OP_N : CUBLAS_OP_T;
  cublasOperation_t c_t_b = (*t_b == 'n') ? CUBLAS_OP_N : CUBLAS_OP_T;

  cudaComplex<scalar_t> cuBeta{beta.real(), beta.imag()};
  cudaComplex<scalar_t> cuAlpha{alpha.real(), alpha.imag()};

#ifdef BLAS_VERIFY_BENCHMARK
  // Run a first time with a verification of the results
  std::vector<std::complex<scalar_t>> c_ref = c;

  reference_blas::cgemm<scalar_t>(t_a, t_b, m, n, k,
                                  reinterpret_cast<const void*>(&alpha),
                                  reinterpret_cast<const void*>(a.data()), lda,
                                  reinterpret_cast<const void*>(b.data()), ldb,
                                  reinterpret_cast<const void*>(&beta),
                                  reinterpret_cast<void*>(c_ref.data()), ldc);
  std::vector<std::complex<scalar_t>> c_temp = c;
  {
    blas_benchmark::utils::CUDAVector<cudaComplex<scalar_t>, true> c_temp_gpu(
        m * n, reinterpret_cast<cudaComplex<scalar_t>*>(c_temp.data()));
    cublas_cplx_routine<scalar_t>(cuda_handle, c_t_a, c_t_b, m, n, k, &cuAlpha,
                                  a_gpu, lda, b_gpu, ldb, &cuBeta, c_temp_gpu,
                                  ldc);
  }

  std::ostringstream err_stream;
  if (!utils::compare_vectors(c_temp, c_ref, err_stream, "")) {
    const std::string& err_str = err_stream.str();
    state.SkipWithError(err_str.c_str());
    *success = false;
  };
#endif
  auto blas_warmup = [&]() -> void {
    cublas_cplx_routine<scalar_t>(cuda_handle, c_t_a, c_t_b, m, n, k, &cuAlpha,
                                  a_gpu, lda, b_gpu, ldb, &cuBeta, c_gpu, ldc);
    return;
  };

  cudaEvent_t start;
  cudaEvent_t stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));

  auto blas_method_def = [&]() -> std::vector<cudaEvent_t> {
    CUDA_CHECK(cudaEventRecord(start));
    cublas_cplx_routine<scalar_t>(cuda_handle, c_t_a, c_t_b, m, n, k, &cuAlpha,
                                  a_gpu, lda, b_gpu, ldb, &cuBeta, c_gpu, ldc);
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
void register_cplx_benchmark(blas_benchmark::Args& args,
                             cublasHandle_t* cuda_handle_ptr, bool* success) {
  auto gemm_params =
      blas_benchmark::utils::get_blas3_cplx_params<scalar_t>(args);
  for (auto p : gemm_params) {
    std::string t1s, t2s;
    index_t m, n, k;
    scalar_t alpha_r, alpha_i, beta_r, beta_i;

    std::tie(t1s, t2s, m, k, n, alpha_r, alpha_i, beta_r, beta_i) = p;
    int t1 = static_cast<int>(blas_benchmark::utils::to_transpose_enum(t1s));
    int t2 = static_cast<int>(blas_benchmark::utils::to_transpose_enum(t2s));
    std::complex<scalar_t> alpha{alpha_r, alpha_i};
    std::complex<scalar_t> beta{beta_r, beta_i};

    auto BM_lambda = [&](benchmark::State& st, cublasHandle_t* cuda_handle_ptr,
                         int t1, int t2, index_t m, index_t k, index_t n,
                         std::complex<scalar_t> alpha,
                         std::complex<scalar_t> beta, bool* success) {
      run<scalar_t>(st, cuda_handle_ptr, t1, t2, m, k, n, alpha, beta, success);
    };
    benchmark::RegisterBenchmark(
        blas_benchmark::utils::get_name<benchmark_op, std::complex<scalar_t>>(
            t1s, t2s, m, k, n, blas_benchmark::utils::MEM_TYPE_USM)
            .c_str(),
        BM_lambda, cuda_handle_ptr, t1, t2, m, k, n, alpha, beta, success)
        ->UseRealTime();
  }
}

#endif

namespace blas_benchmark {
void create_benchmark(blas_benchmark::Args& args,
                      cublasHandle_t* cuda_handle_ptr, bool* success) {
  BLAS_REGISTER_BENCHMARK(args, cuda_handle_ptr, success);
}
}  // namespace blas_benchmark
