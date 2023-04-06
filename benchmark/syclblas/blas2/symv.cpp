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
 *  @filename symv.cpp
 *
 **************************************************************************/

#include "../utils.hpp"

template <typename scalar_t>
std::string get_name(std::string uplo, int n, scalar_t alpha, scalar_t beta) {
  std::ostringstream str{};
  str << "BM_Symv<" << blas_benchmark::utils::get_type_name<scalar_t>() << ">/"
      << uplo << "/" << n << "/" << alpha << "/" << beta;
  return str.str();
}

template <typename scalar_t>
void run(benchmark::State& state, blas::SB_Handle* sb_handle_ptr,
         std::string uplo, index_t n, scalar_t alpha, scalar_t beta,
         bool* success) {
  // Standard test setup.
  const char* uplo_str = uplo.c_str();

  index_t incX = 1;
  index_t incY = 1;

  index_t lda = n;
  index_t xlen = n;
  index_t ylen = n;

  // The counters are double. We convert n to double to avoid
  // integer overflows for n_fl_ops and bytes_processed
  double n_d = static_cast<double>(n);

  state.counters["n"] = n_d;

  // Compute the number of A non-zero elements.
  const double A_validVal = (n_d * (n_d + 1) / 2);

  const double nflops_AtimesX = 2 * n_d * n_d;
  const double nflops_timesAlpha = xlen;
  const double nflops_addBetaY = (beta != scalar_t{0}) ? 2 * ylen : 0;
  const double nflops_tot =
      nflops_AtimesX + nflops_timesAlpha + nflops_addBetaY;
  state.counters["n_fl_ops"] = nflops_tot;

  const double mem_readA = A_validVal;
  const double mem_readX = xlen;
  const double mem_writeY = ylen;
  const double mem_readY = (beta != scalar_t{0}) ? ylen : 0;
  const double tot_mem_processed =
      (mem_readA + mem_readX + mem_writeY + mem_readY) * sizeof(scalar_t);
  state.counters["bytes_processed"] = tot_mem_processed;

  blas::SB_Handle& sb_handle = *sb_handle_ptr;

  // Input matrix/vector, output vector.
  std::vector<scalar_t> m_a =
      blas_benchmark::utils::random_data<scalar_t>(lda * n);
  std::vector<scalar_t> v_x =
      blas_benchmark::utils::random_data<scalar_t>(xlen);
  std::vector<scalar_t> v_y =
      blas_benchmark::utils::random_data<scalar_t>(ylen);

  auto m_a_gpu = blas::make_sycl_iterator_buffer<scalar_t>(m_a, n * n);
  auto v_x_gpu = blas::make_sycl_iterator_buffer<scalar_t>(v_x, xlen);
  auto v_y_gpu = blas::make_sycl_iterator_buffer<scalar_t>(v_y, ylen);

#ifdef BLAS_VERIFY_BENCHMARK
  // Run a first time with a verification of the results
  std::vector<scalar_t> v_y_ref = v_y;
  reference_blas::symv(uplo_str, n, alpha, m_a.data(), lda, v_x.data(), incX,
                       beta, v_y_ref.data(), incY);
  std::vector<scalar_t> v_y_temp(v_y);
  {
    auto v_y_temp_gpu =
        blas::make_sycl_iterator_buffer<scalar_t>(v_y_temp, ylen);
    auto event = _symv(sb_handle, *uplo_str, n, alpha, m_a_gpu, lda, v_x_gpu,
                       incX, beta, v_y_temp_gpu, incY);
    sb_handle.wait();
  }

  std::ostringstream err_stream;
  if (!utils::compare_vectors(v_y_temp, v_y_ref, err_stream, "")) {
    const std::string& err_str = err_stream.str();
    state.SkipWithError(err_str.c_str());
    *success = false;
  };
#endif

  auto blas_method_def = [&]() -> std::vector<cl::sycl::event> {
    auto event = _symv(sb_handle, *uplo_str, n, alpha, m_a_gpu, lda, v_x_gpu,
                       incX, beta, v_y_gpu, incY);
    sb_handle.wait();
    return event;
  };

  // Warmup
  blas_benchmark::utils::warmup(blas_method_def);
  sb_handle.wait();

  blas_benchmark::utils::init_counters(state);

  // Measure
  for (auto _ : state) {
    // Run
    std::tuple<double, double> times =
        blas_benchmark::utils::timef(blas_method_def);

    // Report
    blas_benchmark::utils::update_counters(state, times);
  }

  state.SetBytesProcessed(state.iterations() * tot_mem_processed);
  state.SetItemsProcessed(state.iterations() * nflops_tot);

  blas_benchmark::utils::calc_avg_counters(state);
}

template <typename scalar_t>
void register_benchmark(blas_benchmark::Args& args,
                        blas::SB_Handle* sb_handle_ptr, bool* success) {
  auto symv_params = blas_benchmark::utils::get_symv_params<scalar_t>(args);

  for (auto p : symv_params) {
    std::string uplos;
    index_t n;
    scalar_t alpha, beta;
    std::tie(uplos, n, alpha, beta) = p;

    auto BM_lambda = [&](benchmark::State& st, blas::SB_Handle* sb_handle_ptr,
                         std::string uplos, index_t n, scalar_t alpha,
                         scalar_t beta, bool* success) {
      run<scalar_t>(st, sb_handle_ptr, uplos, n, alpha, beta, success);
    };
    benchmark::RegisterBenchmark(
        get_name<scalar_t>(uplos, n, alpha, beta).c_str(), BM_lambda,
        sb_handle_ptr, uplos, n, alpha, beta, success);
  }
}

namespace blas_benchmark {
void create_benchmark(blas_benchmark::Args& args,
                      blas::SB_Handle* sb_handle_ptr, bool* success) {
  BLAS_REGISTER_BENCHMARK(args, sb_handle_ptr, success);
}
}  // namespace blas_benchmark
