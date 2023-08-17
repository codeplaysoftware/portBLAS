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
 *  @filename omatcopy.cpp
 *
 **************************************************************************/

#include "../../../test/unittest/extension/extension_reference.hpp"
#include "../utils.hpp"

template <typename scalar_t>
void run(benchmark::State& state, blas::SB_Handle* sb_handle_ptr, int ti,
         index_t m, index_t n, scalar_t alpha, index_t lda_mul, index_t ldb_mul,
         bool* success) {
  // initialize the state label
  blas_benchmark::utils::set_benchmark_label<scalar_t>(
      state, sb_handle_ptr->get_queue());

  // Standard test setup.
  std::string ts = blas_benchmark::utils::from_transpose_enum(
      static_cast<blas_benchmark::utils::Transposition>(ti));
  const char* t_str = ts.c_str();

  const auto lda = lda_mul * m;
  const auto ldb = (*t_str == 't') ? ldb_mul * n : ldb_mul * m;

  const auto size_a = lda * n;
  const auto size_b = ldb * ((*t_str == 't') ? m : n);

  blas_benchmark::utils::init_extension_counters<
      blas_benchmark::utils::ExtensionOp::omatcopy, scalar_t>(
      state, t_str, m, n, lda_mul, ldb_mul);

  blas::SB_Handle& sb_handle = *sb_handle_ptr;

  // Input matrix/vector, output vector.
  std::vector<scalar_t> m_a =
      blas_benchmark::utils::random_data<scalar_t>(size_a);
  std::vector<scalar_t> m_b =
      blas_benchmark::utils::random_data<scalar_t>(size_b);

  auto m_a_gpu = blas::make_sycl_iterator_buffer<scalar_t>(m_a, size_a);
  auto m_b_gpu = blas::make_sycl_iterator_buffer(m_b, size_b);

#ifdef BLAS_VERIFY_BENCHMARK
  // Run a first time with a verification of the results
  std::vector<scalar_t> m_b_ref = m_b;

  reference_blas::ext_omatcopy(*t_str, m, n, alpha, m_a.data(), lda,
                               m_b_ref.data(), ldb);

  std::vector<scalar_t> m_b_temp = m_b;
  {
    auto m_b_temp_gpu =
        blas::make_sycl_iterator_buffer<scalar_t>(m_b_temp, size_b);

    auto event = blas::_omatcopy(sb_handle, *t_str, m, n, alpha, m_a_gpu, lda,
                                 m_b_temp_gpu, ldb);

    sb_handle.wait();
  }

  std::ostringstream err_stream;
  if (!utils::compare_vectors(m_b_temp, m_b_ref, err_stream, "")) {
    const std::string& err_str = err_stream.str();
    state.SkipWithError(err_str.c_str());
    *success = false;
  };
#endif

  auto blas_method_def = [&]() -> std::vector<cl::sycl::event> {
    auto event = blas::_omatcopy(sb_handle, *t_str, m, n, alpha, m_a_gpu, lda,
                                 m_b_gpu, ldb);
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
  auto omatcopy_params =
      blas_benchmark::utils::get_matcopy_params<scalar_t>(args);

  for (auto p : omatcopy_params) {
    std::string ts;
    index_t m, n, lda_mul, ldb_mul;
    scalar_t alpha;
    std::tie(ts, m, n, alpha, lda_mul, ldb_mul) = p;
    int t = static_cast<int>(blas_benchmark::utils::to_transpose_enum(ts));

    auto BM_lambda = [&](benchmark::State& st, blas::SB_Handle* sb_handle_ptr,
                         int t, index_t m, index_t n, scalar_t alpha,
                         index_t lda_mul, index_t ldb_mul, bool* success) {
      run<scalar_t>(st, sb_handle_ptr, t, m, n, alpha, lda_mul, ldb_mul,
                    success);
    };
    benchmark::RegisterBenchmark(
        blas_benchmark::utils::get_name<
            blas_benchmark::utils::ExtensionOp::omatcopy, scalar_t>(
            ts, m, n, alpha, lda_mul, ldb_mul)
            .c_str(),
        BM_lambda, sb_handle_ptr, t, m, n, alpha, lda_mul, ldb_mul, success)
        ->UseRealTime();
  }
}

namespace blas_benchmark {
void create_benchmark(blas_benchmark::Args& args,
                      blas::SB_Handle* sb_handle_ptr, bool* success) {
  BLAS_REGISTER_BENCHMARK(args, sb_handle_ptr, success);
}
}  // namespace blas_benchmark
