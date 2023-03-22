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
 *  @filename spr2.cpp
 *
 **************************************************************************/

#include "../utils.hpp"

template <typename scalar_t>
std::string get_name(char uplo, int size, scalar_t alpha, int incX) {
  std::ostringstream str{};
  str << "BM_Spr2<" << blas_benchmark::utils::get_type_name<scalar_t>() << ">/"
      << uplo << "/" << size << "/" << alpha << "/" << incX;
  return str.str();
}

template <typename scalar_t, typename... args_t>
static inline void cublas_routine(args_t&&... args) {
  if constexpr (std::is_same_v<scalar_t, float>) {
    CUBLAS_CHECK(cublasSspr2(std::forward<args_t>(args)...));
  } else if constexpr (std::is_same_v<scalar_t, double>) {
    CUBLAS_CHECK(cublasDspr2(std::forward<args_t>(args)...));
  }
  return;
}

template <typename scalar_t>
void run(benchmark::State& state, cublasHandle_t* cuda_handle_ptr, char uplo,
         int size, scalar_t alpha, int incX, bool* success) {
  // The counters are double. We convert size to double to avoid
  // integer overflows for n_fl_ops and bytes_processed
  double n_d = static_cast<double>(size * (size + 1) / 2);

  // Temporarely set incY to 1 for all benchmarks
  index_t incY = 1;

  state.counters["n_d"] = n_d;
  state.counters["alpha"] = static_cast<double>(alpha);
  state.counters["incX"] = incX;
  state.counters["incY"] = incY;

  const double nflops_XtimesX = 2.0 * n_d;
  const double nflops_tot = size + nflops_XtimesX;
  state.counters["n_fl_ops"] = nflops_tot;

  const double mem_readA = n_d;
  const double mem_readX = static_cast<double>(size * std::abs(incX));
  state.counters["bytes_processed"] =
      (mem_readA + mem_readX) * sizeof(scalar_t);

  cublasHandle_t& cuda_handle = *cuda_handle_ptr;

  const int m_size = size * size;
  const int v_size = 1 + (size - 1) * std::abs(incX);

  // Input matrix/vector, output vector.
  std::vector<scalar_t> m_a =
      blas_benchmark::utils::random_data<scalar_t>(m_size);
  std::vector<scalar_t> v_x =
      blas_benchmark::utils::random_data<scalar_t>(v_size);
  std::vector<scalar_t> v_y =
      blas_benchmark::utils::random_data<scalar_t>(v_size);

  blas_benchmark::utils::CUDAVector<scalar_t> m_a_gpu(m_size, m_a.data());
  blas_benchmark::utils::CUDAVector<scalar_t> v_x_gpu(v_size, v_x.data());
  blas_benchmark::utils::CUDAVector<scalar_t> v_y_gpu(v_size, v_y.data());

  cublasFillMode_t c_uplo =
      (uplo == 'u') ? CUBLAS_FILL_MODE_UPPER : CUBLAS_FILL_MODE_LOWER;

#ifdef BLAS_VERIFY_BENCHMARK
  // Run a first time with a verification of the results
  std::vector<scalar_t> m_a_ref = m_a;
  reference_blas::spr2<scalar_t>(&uplo, size, alpha, v_x.data(), incX,
                                 v_y.data(), incY, m_a_ref.data());

  std::vector<scalar_t> m_a_temp = m_a;
  {
    blas_benchmark::utils::CUDAVector<scalar_t, true> m_a_temp_gpu(
        m_size, m_a_temp.data());
    cublas_routine<scalar_t>(cuda_handle, c_uplo, size, &alpha, v_x_gpu, incX,
                             v_y_gpu, incY, m_a_temp_gpu);
  }

  std::ostringstream err_stream;
  if (!utils::compare_vectors(m_a_temp, m_a_ref, err_stream, "")) {
    const std::string& err_str = err_stream.str();
    state.SkipWithError(err_str.c_str());
    *success = false;
  };
#endif

  auto blas_warmup = [&]() -> void {
    cublas_routine<scalar_t>(cuda_handle, c_uplo, size, &alpha, v_x_gpu, incX,
                             v_y_gpu, incY, m_a_gpu);
    return;
  };

  cudaEvent_t start;
  cudaEvent_t stop;
  CUDA_CHECK(cudaEventCreate(&start));
  CUDA_CHECK(cudaEventCreate(&stop));

  auto blas_method_def = [&]() -> std::vector<cudaEvent_t> {
    CUDA_CHECK(cudaEventRecord(start));
    cublas_routine<scalar_t>(cuda_handle, c_uplo, size, &alpha, v_x_gpu, incX,
                             v_y_gpu, incY, m_a_gpu);
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

  state.SetItemsProcessed(state.iterations() * nflops_tot);

  blas_benchmark::utils::calc_avg_counters(state);

  CUDA_CHECK(cudaEventDestroy(start));
  CUDA_CHECK(cudaEventDestroy(stop));
}

template <typename scalar_t>
void register_benchmark(blas_benchmark::Args& args,
                        cublasHandle_t* cuda_handle_ptr, bool* success) {
  // spr2 uses the same parameters as spr
  auto spr2_params = blas_benchmark::utils::get_spr_params<scalar_t>(args);

  for (auto p : spr2_params) {
    int n, incX;
    std::string uplo;
    scalar_t alpha;
    std::tie(uplo, n, alpha, incX) = p;

    char uplo_c = uplo[0];

    auto BM_lambda_col =
        [&](benchmark::State& st, cublasHandle_t* cuda_handle_ptr, char uplo,
            int size, scalar_t alpha, int incX, bool* success) {
          run<scalar_t>(st, cuda_handle_ptr, uplo, size, alpha, incX, success);
        };
    benchmark::RegisterBenchmark(
        get_name<scalar_t>(uplo_c, n, alpha, incX).c_str(), BM_lambda_col,
        cuda_handle_ptr, uplo_c, n, alpha, incX, success);
  }
}

namespace blas_benchmark {
void create_benchmark(blas_benchmark::Args& args,
                      cublasHandle_t* cuda_handle_ptr, bool* success) {
  BLAS_REGISTER_BENCHMARK(args, cuda_handle_ptr, success);
}
}  // namespace blas_benchmark
