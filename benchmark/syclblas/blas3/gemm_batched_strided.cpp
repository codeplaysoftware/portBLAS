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
 *  @filename gemm_batched_strided.cpp
 *
 **************************************************************************/

#include "../utils.hpp"

template <typename scalar_t>
std::string get_name(std::string t1, std::string t2, int m, int k, int n,
                     int batch_size) {
  std::ostringstream str{};
  str << "BM_GemmBatchedStrided<"
      << blas_benchmark::utils::get_type_name<scalar_t>() << ">/" << t1 << "/"
      << t2 << "/" << m << "/" << k << "/" << n << "/" << batch_size;
  return str.str();
}

template <typename scalar_t>
void run(benchmark::State& state, blas::SB_Handle* sb_handle_ptr, int t1,
         int t2, index_t m, index_t k, index_t n, index_t stride_a,
         index_t stride_b, index_t stride_c, scalar_t alpha, scalar_t beta,
         index_t batch_size, bool* success) {
  // Standard test setup.
  std::string t1s = blas_benchmark::utils::from_transpose_enum(
      static_cast<blas_benchmark::utils::Transposition>(t1));
  std::string t2s = blas_benchmark::utils::from_transpose_enum(
      static_cast<blas_benchmark::utils::Transposition>(t2));
  const char* t_a = t1s.c_str();
  const char* t_b = t2s.c_str();

  const bool trA = t_a[0] != 'n';
  const bool trB = t_b[0] != 'n';

  index_t lda = trA ? k : m;
  index_t ldb = trB ? n : k;
  index_t ldc = m;

  // The counters are double. We convert m, n, k and batch_size to double to
  // avoid integer overflows for n_fl_ops and bytes_processed
  {
    double m_d = static_cast<double>(m);
    double n_d = static_cast<double>(n);
    double k_d = static_cast<double>(k);
    double batch_size_d = static_cast<double>(batch_size);
    state.counters["m"] = m_d;
    state.counters["n"] = n_d;
    state.counters["k"] = k_d;
    state.counters["batch_size"] = batch_size_d;
    const double nflops_AtimesB = 2 * k_d * m_d * n_d;
    double nflops_timesAlpha = m_d * n_d;
    const double nflops_addBetaC = (beta != scalar_t{0}) ? 2 * m_d * n_d : 0;
    const double nflops_tot =
        (nflops_AtimesB + nflops_timesAlpha + nflops_addBetaC) * batch_size_d;
    state.counters["n_fl_ops"] = nflops_tot;

    const double mem_readA = m_d * k_d;
    const double mem_readB = k_d * n_d;
    const double mem_writeC = m_d * n_d;
    const double mem_readC = (beta != scalar_t{0}) ? m_d * n_d : 0;
    const double total_mem = (mem_readA + mem_readB + mem_readC + mem_writeC) *
                             batch_size_d * sizeof(scalar_t);
    state.counters["bytes_processed"] = total_mem;
  }

  blas::SB_Handle& sb_handle = *sb_handle_ptr;
  auto q = sb_handle.get_queue();

  // Matrices sizes
  const index_t a_size = m * k;
  const index_t b_size = k * n;
  const index_t c_size = m * n;

  // Matrices (Total size is equal to matrix size x batch_size since we're using
  // default striding values)
  std::vector<scalar_t> a =
      blas_benchmark::utils::random_data<scalar_t>(a_size * batch_size);
  std::vector<scalar_t> b =
      blas_benchmark::utils::random_data<scalar_t>(b_size * batch_size);
  std::vector<scalar_t> c =
      blas_benchmark::utils::const_data<scalar_t>(c_size * batch_size, 0);

#ifdef BLAS_VERIFY_BENCHMARK
  // Run a first time with a verification of the results
  std::vector<scalar_t> c_ref = c;
  auto _base = [=](index_t dim0, index_t dim1, index_t idx) {
    return dim0 * dim1 * idx;
  };
  for (int batch_idx = 0; batch_idx < batch_size; batch_idx++) {
    reference_blas::gemm(t_a, t_b, m, n, k, alpha,
                         a.data() + _base(m, k, batch_idx), lda,
                         b.data() + _base(k, n, batch_idx), ldb, beta,
                         c_ref.data() + _base(m, n, batch_idx), ldc);
  }

#endif

  auto a_gpu =
      blas::make_sycl_iterator_buffer<scalar_t>(a, a_size * batch_size);
  auto b_gpu =
      blas::make_sycl_iterator_buffer<scalar_t>(b, b_size * batch_size);
  auto c_gpu =
      blas::make_sycl_iterator_buffer<scalar_t>(c, c_size * batch_size);

#ifdef BLAS_VERIFY_BENCHMARK
  std::vector<scalar_t> c_temp = c;
  {
    auto c_temp_gpu =
        blas::make_sycl_iterator_buffer<scalar_t>(c_temp, c_size * batch_size);
    auto event = _gemm_strided_batched(
        sb_handle, *t_a, *t_b, m, n, k, alpha, a_gpu, lda, stride_a, b_gpu, ldb,
        stride_b, beta, c_temp_gpu, ldc, stride_c, batch_size);
    sb_handle.wait(event);
  }

  std::ostringstream err_stream;
  if (!utils::compare_vectors_strided(c_temp, c_ref, stride_c, c_size,
                                      err_stream, "")) {
    const std::string& err_str = err_stream.str();
    state.SkipWithError(err_str.c_str());
    *success = false;
  };
#endif

  auto blas_method_def = [&]() -> std::vector<cl::sycl::event> {
    auto event = _gemm_strided_batched(
        sb_handle, *t_a, *t_b, m, n, k, alpha, a_gpu, lda, stride_a, b_gpu, ldb,
        stride_b, beta, c_gpu, ldc, stride_c, batch_size);
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

  state.SetItemsProcessed(state.iterations() * state.counters["n_fl_ops"]);
  state.SetBytesProcessed(state.iterations() *
                          state.counters["bytes_processed"]);

  blas_benchmark::utils::calc_avg_counters(state);
};

template <typename scalar_t>
void register_benchmark(blas_benchmark::Args& args,
                        blas::SB_Handle* sb_handle_ptr, bool* success) {
  auto gemm_batched_strided_params =
      blas_benchmark::utils::get_gemm_batched_params<scalar_t>(args);

  for (auto p : gemm_batched_strided_params) {
    std::string t1s, t2s;
    index_t m, n, k, batch_size;
    scalar_t alpha, beta;
    int batch_type;
    std::tie(t1s, t2s, m, k, n, alpha, beta, batch_size, batch_type) = p;
    int t1 = static_cast<int>(blas_benchmark::utils::to_transpose_enum(t1s));
    int t2 = static_cast<int>(blas_benchmark::utils::to_transpose_enum(t2s));

    index_t stride_a = m * k;
    index_t stride_b = k * n;
    index_t stride_c = m * n;

    auto BM_lambda = [&](benchmark::State& st, blas::SB_Handle* sb_handle_ptr,
                         int t1, int t2, index_t m, index_t k, index_t n,
                         scalar_t alpha, scalar_t beta, index_t batch_size,
                         bool* success) {
      run<scalar_t>(st, sb_handle_ptr, t1, t2, m, k, n, stride_a, stride_b,
                    stride_c, alpha, beta, batch_size, success);
    };
    benchmark::RegisterBenchmark(
        get_name<scalar_t>(t1s, t2s, m, k, n, batch_size).c_str(), BM_lambda,
        sb_handle_ptr, t1, t2, m, k, n, alpha, beta, batch_size, success)
        ->UseRealTime();
  }
}

namespace blas_benchmark {
void create_benchmark(blas_benchmark::Args& args,
                      blas::SB_Handle* sb_handle_ptr, bool* success) {
  BLAS_REGISTER_BENCHMARK(args, sb_handle_ptr, success);
}
}  // namespace blas_benchmark
