/**************************************************************************
 *
 *  @license
 *  Copyright (C) 2021 Codeplay Software Limited
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
 *  @filename trsm.cpp
 *
 **************************************************************************/

#include "../utils.hpp"

template <typename scalar_t>
std::string get_name(char side, char uplo, char trans, char diag, index_t m,
                     index_t n) {
  std::ostringstream str{};
  str << "BM_Trsm<" << blas_benchmark::utils::get_type_name<scalar_t>() << ">/"
      << side << "/" << uplo << "/" << trans << "/" << diag << "/" << m << "/"
      << n;
  return str.str();
}

template <typename scalar_t>
void run(benchmark::State& state, blas::SB_Handle* sb_handle_ptr, char side,
         char uplo, char trans, char diag, index_t m, index_t n, scalar_t alpha,
         bool* success) {
  // Standard test setup.
  index_t lda = side == 'l' ? m : n;
  index_t ldb = m;
  index_t k = side == 'l' ? m : n;

  blas::SB_Handle& sb_handle = *sb_handle_ptr;

  const int sizeA = k * lda;
  const int sizeB = n * ldb;

  // Matrices
  std::vector<scalar_t> a(sizeA);
  std::vector<scalar_t> b = blas_benchmark::utils::random_data<scalar_t>(sizeB);

  const scalar_t diagValue =
      diag == 'u' ? scalar_t{1}
                  : blas_benchmark::utils::random_scalar<scalar_t>(
                        scalar_t{1}, scalar_t{10});

  blas_benchmark::utils::fill_trsm_matrix(a, k, lda, uplo, diagValue,
                                          scalar_t{0});

  auto a_gpu = blas::make_sycl_iterator_buffer<scalar_t>(a, sizeA);
  auto b_gpu = blas::make_sycl_iterator_buffer<scalar_t>(b, sizeB);

  a_gpu.get_buffer().set_final_data(nullptr);
  b_gpu.get_buffer().set_final_data(nullptr);

#ifdef BLAS_VERIFY_BENCHMARK
  // Run once verifying the results against the reference blas implementation.
  std::vector<scalar_t> x_ref = b;
  std::vector<scalar_t> b_temp = b;

  reference_blas::trsm(&side, &uplo, &trans, &diag, m, n,
                       static_cast<scalar_t>(alpha), a.data(), lda,
                       x_ref.data(), ldb);

  {
    auto b_temp_gpu = blas::make_sycl_iterator_buffer<scalar_t>(b_temp, sizeB);
    _trsm(sb_handle, side, uplo, trans, diag, m, n, alpha, a_gpu, lda,
          b_temp_gpu, ldb);
    auto event = blas::helper::copy_to_host(sb_handle.get_queue(), b_temp_gpu,
                                            b_temp.data(), sizeB);
    sb_handle.wait(event);
  }

  std::ostringstream err_stream;
  if (!utils::compare_vectors(b_temp, x_ref, err_stream, "")) {
    const std::string& err_str = err_stream.str();
    state.SkipWithError(err_str.c_str());
    *success = false;
  };
#endif

  auto blas_method_def = [&]() -> std::vector<cl::sycl::event> {
    auto event = _trsm(sb_handle, side, uplo, trans, diag, m, n, alpha, a_gpu,
                       lda, b_gpu, ldb);
    sb_handle.wait(event);
    return event;
  };

  // Warmup
  blas_benchmark::utils::warmup(blas_method_def);
  sb_handle.wait();

  blas_benchmark::utils::init_counters(state);

  // Measure
  for (auto _ : state) {
    auto start = std::chrono::system_clock::now();
    auto events = blas_method_def();
    auto end = std::chrono::system_clock::now();
    double overall_time = (end - start).count();

    double fillTime =
        static_cast<double>(blas_benchmark::utils::time_event(events[0]));
    double inversionTime =
        static_cast<double>(blas_benchmark::utils::time_event(events[1]));
    double copyTime =
        static_cast<double>(blas_benchmark::utils::time_event(events[2]));
    double gemmTime = 0.0;
    for (size_t i = 3; i <= events.size() - 2; ++i) {
      gemmTime +=
          static_cast<double>(blas_benchmark::utils::time_event(events[i]));
    }
    copyTime += static_cast<double>(
        blas_benchmark::utils::time_event(events[events.size() - 1]));

    double event_time =
        static_cast<double>(blas_benchmark::utils::time_events(events));

    std::tuple<double, double> times =
        std::make_tuple(overall_time, event_time);

    state.counters["fill_time"] = fillTime;
    state.counters["inversion_time"] = inversionTime;
    state.counters["gemm_time"] = gemmTime;
    state.counters["copy_time"] = copyTime;

    // Report
    blas_benchmark::utils::update_counters(state, times);
  }

  {
    // The counters are double. We convert m, n and k to double to avoid
    // integer overflows for n_fl_ops and bytes_processed
    double m_d = static_cast<double>(m);
    double n_d = static_cast<double>(n);
    double k_d = static_cast<double>(k);

    state.counters["m"] = m_d;
    state.counters["k"] = k_d;
    state.counters["n"] = n_d;

    double mem_read = k_d * (k_d + 1) / 2;
    double mem_write = m_d * n_d;

    double total_mem = (mem_read * mem_write) * sizeof(scalar_t);
    state.counters["bytes_processed"] = total_mem;
    state.SetBytesProcessed(state.iterations() * total_mem);

    double nflops_AtimesB = 2 * k_d * (k_d + 1) / 2 * (side == 'l' ? n_d : m_d);
    double nflops_timesAlpha = m_d * n_d;
    double nflops = nflops_AtimesB + nflops_timesAlpha;
    state.counters["n_fl_ops"] = nflops;
    state.SetItemsProcessed(state.iterations() * nflops);
  }

  blas_benchmark::utils::calc_avg_counters(state);
};

template <typename scalar_t>
void register_benchmark(blas_benchmark::Args& args, blas::SB_Handle* sb_handle_ptr,
                        bool* success) {
  auto trsm_params = blas_benchmark::utils::get_trsm_params<scalar_t>(args);

  for (auto p : trsm_params) {
    char side, uplo, trans, diag;
    index_t m, n;
    scalar_t alpha;
    std::tie(side, uplo, trans, diag, m, n, alpha) = p;

    auto BM_lambda = [&](benchmark::State& st, blas::SB_Handle* sb_handle_ptr,
                         char side, char uplo, char trans, char diag, index_t m,
                         index_t n, scalar_t alpha, bool* success) {
      run<scalar_t>(st, sb_handle_ptr, side, uplo, trans, diag, m, n, alpha, success);
    };
    benchmark::RegisterBenchmark(
        get_name<scalar_t>(side, uplo, trans, diag, m, n).c_str(), BM_lambda,
        sb_handle_ptr, side, uplo, trans, diag, m, n, alpha, success)
        ->UseRealTime();
  }
}

namespace blas_benchmark {
void create_benchmark(blas_benchmark::Args& args, blas::SB_Handle* sb_handle_ptr,
                      bool* success) {
  BLAS_REGISTER_BENCHMARK(args, sb_handle_ptr, success);
}
}  // namespace blas_benchmark
