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
 *  @filename symm.cpp
 *
 **************************************************************************/

#include "../utils.hpp"

template <typename scalar_t>
std::string get_name(char side, char uplo, int m, int n) {
  std::ostringstream str{};
  str << "BM_Symm<" << blas_benchmark::utils::get_type_name<scalar_t>() << ">/"
      << side << "/" << uplo << "/" << m << "/" << n;
  return str.str();
}

template <typename scalar_t>
void run(benchmark::State& state, blas::SB_Handle* sb_handle_ptr, char side,
         char uplo, index_t m, index_t n, scalar_t alpha, scalar_t beta,
         bool* success) {
  const index_t k = side == 'l' ? m : n;

  index_t lda = k;
  index_t ldb = m;
  index_t ldc = m;

  blas::SB_Handle& sb_handle = *sb_handle_ptr;

  // Matrices
  std::vector<scalar_t> a = blas_benchmark::utils::random_data<scalar_t>(k * k);
  std::vector<scalar_t> b = blas_benchmark::utils::random_data<scalar_t>(m * n);
  std::vector<scalar_t> c =
      blas_benchmark::utils::const_data<scalar_t>(m * n, 0);

  auto a_gpu = blas::make_sycl_iterator_buffer<scalar_t>(a, k * k);
  auto b_gpu = blas::make_sycl_iterator_buffer<scalar_t>(b, m * n);
  auto c_gpu = blas::make_sycl_iterator_buffer<scalar_t>(c, m * n);

#ifdef BLAS_VERIFY_BENCHMARK
  // Run a first time with a verification of the results
  std::vector<scalar_t> c_ref = c;
  const char side_str[2] = {side, '\0'};
  const char uplo_str[2] = {uplo, '\0'};
  reference_blas::symm(side_str, uplo_str, m, n, alpha, a.data(), lda, b.data(),
                       ldb, beta, c_ref.data(), ldc);
  std::vector<scalar_t> c_temp = c;
  {
    auto c_temp_gpu = blas::make_sycl_iterator_buffer<scalar_t>(c_temp, m * n);
    auto event = _symm(sb_handle, side, uplo, m, n, alpha, a_gpu, lda, b_gpu,
                       ldb, beta, c_temp_gpu, ldc);
    sb_handle.wait(event);
  }

  std::ostringstream err_stream;
  if (!utils::compare_vectors(c_temp, c_ref, err_stream, "")) {
    const std::string& err_str = err_stream.str();
    state.SkipWithError(err_str.c_str());
    *success = false;
  };
#endif

  auto blas_method_def = [&]() -> std::vector<cl::sycl::event> {
    auto event = _symm(sb_handle, side, uplo, m, n, alpha, a_gpu, lda, b_gpu,
                       ldb, beta, c_gpu, ldc);
    sb_handle.wait(event);
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

  {
    // The counters are double. We convert m, n and k to double to avoid
    // integer overflows for n_fl_ops and bytes_processed
        // The counters are double. We convert m, n and k to double to avoid
    // integer overflows for n_fl_ops and bytes_processed
    double m_d = static_cast<double>(m);
    double n_d = static_cast<double>(n);

    state.counters["m"] = m_d;
    state.counters["n"] = n_d;

    const double mem_readBreadC =
        (beta != scalar_t{0}) ? 2 * m_d * n_d : m_d * n_d;
    const double mem_writeC = m_d * n_d;
    const double mem_readA =
        (side == 'l') ? (m_d * (m_d + 1) / 2) : (n_d * (n_d + 1) / 2);
    const double total_mem =
        (mem_readBreadC + mem_writeC + mem_readA) * sizeof(scalar_t);
    state.counters["bytes_processed"] = total_mem;

    const double nflops_AtimesB =
        (side == 'l') ? (2 * m_d * m_d) * n_d : 2 * n_d * n_d * n_d;
    const double nflops_addBetaC = 2 * m_d * n_d;
    const double nflops = nflops_AtimesB + nflops_addBetaC;
    state.counters["n_fl_ops"] = nflops;

    state.SetBytesProcessed(state.iterations() * total_mem);
    state.SetItemsProcessed(state.iterations() * nflops);
  }

  blas_benchmark::utils::calc_avg_counters(state);
};

template <typename scalar_t>
void register_benchmark(blas_benchmark::Args& args,
                        blas::SB_Handle* sb_handle_ptr, bool* success) {
  auto symm_params = blas_benchmark::utils::get_symm_params<scalar_t>(args);

  for (auto p : symm_params) {
    char side, uplo;
    index_t m, n;
    scalar_t alpha, beta;
    std::tie(side, uplo, m, n, alpha, beta) = p;

    auto BM_lambda = [&](benchmark::State& st, blas::SB_Handle* sb_handle_ptr,
                         char side, char uplo, index_t m, index_t n,
                         scalar_t alpha, scalar_t beta, bool* success) {
      run<scalar_t>(st, sb_handle_ptr, side, uplo, m, n, alpha, beta, success);
    };
    benchmark::RegisterBenchmark(get_name<scalar_t>(side, uplo, m, n).c_str(),
                                 BM_lambda, sb_handle_ptr, side, uplo, m, n,
                                 alpha, beta, success)
        ->UseRealTime();
  }
}

namespace blas_benchmark {
void create_benchmark(blas_benchmark::Args& args,
                      blas::SB_Handle* sb_handle_ptr, bool* success) {
  BLAS_REGISTER_BENCHMARK(args, sb_handle_ptr, success);
}
}  // namespace blas_benchmark
