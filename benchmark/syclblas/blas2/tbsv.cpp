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
 *  @filename tbsv.cpp
 *
 **************************************************************************/

#include "../utils.hpp"

constexpr blas_benchmark::utils::Level2Op benchmark_op =
    blas_benchmark::utils::Level2Op::tbsv;

template <typename scalar_t>
void run(benchmark::State& state, blas::SB_Handle* sb_handle_ptr,
         std::string uplo, std::string t, std::string diag, index_t n,
         index_t k, bool* success) {
  // initialize the state label
  blas_benchmark::utils::set_benchmark_label<scalar_t>(
      state, sb_handle_ptr->get_queue());

  // Standard test setup.
  const char* uplo_str = uplo.c_str();
  const char* t_str = t.c_str();
  const char* diag_str = diag.c_str();

  index_t incX = 1;
  index_t xlen = 1 + (n - 1) * incX;
  index_t lda = (k + 1);

  blas_benchmark::utils::init_level_2_counters<
      blas_benchmark::utils::Level2Op::tbsv, scalar_t>(state, "n", 0, 0, n, k);

  blas::SB_Handle& sb_handle = *sb_handle_ptr;

  // Input matrix/vector, output vector.
  std::vector<scalar_t> m_a(lda * n);
  std::vector<scalar_t> v_x =
      blas_benchmark::utils::random_data<scalar_t>(xlen);

  // Populate the main diagonal with larger values.
  const index_t main_diag = (uplo_str[0] == 'u') ? k : 0;
  for (index_t j = 0; j < n; ++j)
    for (index_t i = 0; i < lda; ++i)
      m_a[i + lda * j] =
          (i == main_diag)
              ? blas_benchmark::utils::random_scalar(scalar_t{9}, scalar_t{11})
              : (blas_benchmark::utils::random_scalar(scalar_t{-10},
                                                      scalar_t{10}) /
                 scalar_t(n));

  auto m_a_gpu = blas::make_sycl_iterator_buffer<scalar_t>(m_a, lda * n);
  auto v_x_gpu = blas::make_sycl_iterator_buffer<scalar_t>(v_x, xlen);

#ifdef BLAS_VERIFY_BENCHMARK
  // Run a first time with a verification of the results
  std::vector<scalar_t> v_x_ref = v_x;
  reference_blas::tbsv(uplo_str, t_str, diag_str, n, k, m_a.data(), lda,
                       v_x_ref.data(), incX);
  std::vector<scalar_t> v_x_temp = v_x;
  {
    auto v_x_temp_gpu =
        blas::make_sycl_iterator_buffer<scalar_t>(v_x_temp, xlen);
    auto event = _tbsv(sb_handle, *uplo_str, *t_str, *diag_str, n, k, m_a_gpu,
                       lda, v_x_temp_gpu, incX);
    sb_handle.wait();
  }

  std::ostringstream err_stream;
  if (!utils::compare_vectors(v_x_temp, v_x_ref, err_stream, "")) {
    const std::string& err_str = err_stream.str();
    state.SkipWithError(err_str.c_str());
    *success = false;
  };
#endif

  auto blas_method_def = [&]() -> std::vector<cl::sycl::event> {
    auto event = _tbsv(sb_handle, *uplo_str, *t_str, *diag_str, n, k, m_a_gpu,
                       lda, v_x_gpu, incX);
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
}

template <typename scalar_t>
void register_benchmark(blas_benchmark::Args& args,
                        blas::SB_Handle* sb_handle_ptr, bool* success) {
  auto tbsv_params = blas_benchmark::utils::get_tbmv_params(args);

  for (auto p : tbsv_params) {
    std::string uplos;
    std::string ts;
    std::string diags;
    index_t n;
    index_t k;
    std::tie(uplos, ts, diags, n, k) = p;

    auto BM_lambda = [&](benchmark::State& st, blas::SB_Handle* sb_handle_ptr,
                         std::string uplos, std::string ts, std::string diags,
                         index_t n, index_t k, bool* success) {
      run<scalar_t>(st, sb_handle_ptr, uplos, ts, diags, n, k, success);
    };
    benchmark::RegisterBenchmark(
        blas_benchmark::utils::get_name<benchmark_op, scalar_t>(
            uplos, ts, diags, n, k, blas_benchmark::utils::MEM_TYPE_BUFFER)
            .c_str(),
        BM_lambda, sb_handle_ptr, uplos, ts, diags, n, k, success)
        ->UseRealTime();
  }
}

namespace blas_benchmark {
void create_benchmark(blas_benchmark::Args& args,
                      blas::SB_Handle* sb_handle_ptr, bool* success) {
  BLAS_REGISTER_BENCHMARK(args, sb_handle_ptr, success);
}
}  // namespace blas_benchmark
