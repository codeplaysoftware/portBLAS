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
 *  @filename omatcopy_batched.cpp
 *
 **************************************************************************/

#include "../../../test/unittest/extension/extension_reference.hpp"
#include "../utils.hpp"

template <typename scalar_t>
std::string get_name(std::string t, int m, int n, scalar_t alpha,
                     index_t lda_mul, index_t ldb_mul, index_t stride_a_mul,
                     index_t stride_b_mul, index_t batch_size) {
  std::ostringstream str{};
  str << "BM_omatcopy_batched<"
      << blas_benchmark::utils::get_type_name<scalar_t>() << ">/" << t << "/"
      << m << "/" << n << "/" << alpha << "/" << lda_mul << "/" << ldb_mul
      << "/" << stride_a_mul << "/" << stride_b_mul << "/" << batch_size;
  return str.str();
}

template <typename scalar_t>
void run(benchmark::State& state, blas::SB_Handle* sb_handle_ptr, int ti,
         index_t m, index_t n, scalar_t alpha, index_t lda_mul, index_t ldb_mul,
         index_t stride_a_mul, index_t stride_b_mul, index_t batch_size,
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

  const auto stride_a = lda * n * stride_a_mul;
  const auto stride_b = ((*t_str == 't') ? ldb * m : ldb * n) * stride_b_mul;

  const auto size_a = stride_a * batch_size;
  const auto size_b = stride_b * batch_size;

  blas_benchmark::utils::init_extension_counters<
      blas_benchmark::utils::ExtensionOP::omatcopy_batch, scalar_t>(
      state, t_str, m, n, lda_mul, ldb_mul, stride_a_mul, stride_b_mul,
      batch_size);

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

  for (int i = 0; i < batch_size; ++i) {
    reference_blas::omatcopy_ref(*t_str, m, n, alpha, m_a.data() + i * stride_a,
                                 lda, m_b_ref.data() + i * stride_b, ldb);
  }

  std::vector<scalar_t> m_b_temp = m_b;
  {
    auto m_b_temp_gpu =
        blas::make_sycl_iterator_buffer<scalar_t>(m_b_temp, size_b);

    auto event = blas::extension::_omatcopy_batch(sb_handle, *t_str, m, n, alpha,
                                            m_a_gpu, lda,stride_a, m_b_temp_gpu, ldb, stride_b, batch_size);

    sb_handle.wait();
  }

  std::ostringstream err_stream;
  if (!utils::compare_vectors(m_b_temp, m_b_ref, err_stream, "")) {
    const std::string& err_str = err_stream.str();
    state.SkipWithError(err_str.c_str());
    *success = false;
  };
#endif

  auto blas_warmup_method_def = [&]() -> void {
    auto event = blas::extension::_omatcopy_batch(
        sb_handle, *t_str, m, n, alpha, m_a_gpu, lda, stride_a, m_b_gpu, ldb,
        stride_b, batch_size);
    return;
  };

  auto blas_method_def = [&]() -> std::vector<cl::sycl::event> {
    auto event = blas::extension::_omatcopy_batch(
        sb_handle, *t_str, m, n, alpha, m_a_gpu, lda, stride_a, m_b_gpu, ldb,
        stride_b, batch_size);
    sb_handle.wait(event);
    return event;
  };

  // Warmup
  blas_benchmark::utils::warmup(blas_warmup_method_def);
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
  auto omatcopy_batch_params =
      blas_benchmark::utils::get_matcopy_batch_params<scalar_t>(args);

  for (auto p : omatcopy_batch_params) {
    std::string ts;
    index_t m, n, lda_mul, ldb_mul, stride_a_mul, stride_b_mul, batch_size;
    scalar_t alpha;
    std::tie(ts, m, n, alpha, lda_mul, ldb_mul, stride_a_mul, stride_b_mul,
             batch_size) = p;
    int t = static_cast<int>(blas_benchmark::utils::to_transpose_enum(ts));

    auto BM_lambda = [&](benchmark::State& st, blas::SB_Handle* sb_handle_ptr,
                         int t, index_t m, index_t n, scalar_t alpha,
                         index_t lda_mul, index_t ldb_mul, index_t stride_a_mul,
                         index_t stride_b_mul, index_t batch_size,
                         bool* success) {
      run<scalar_t>(st, sb_handle_ptr, t, m, n, alpha, lda_mul, ldb_mul,
                    stride_a_mul, stride_b_mul, batch_size, success);
    };
    benchmark::RegisterBenchmark(
        get_name<scalar_t>(ts, m, n, alpha, lda_mul, ldb_mul, stride_a_mul,
                           stride_b_mul, batch_size)
            .c_str(),
        BM_lambda, sb_handle_ptr, t, m, n, alpha, lda_mul, ldb_mul,
        stride_a_mul, stride_b_mul, batch_size, success)
        ->UseRealTime();
  }
}

namespace blas_benchmark {
void create_benchmark(blas_benchmark::Args& args,
                      blas::SB_Handle* sb_handle_ptr, bool* success) {
  BLAS_REGISTER_BENCHMARK(args, sb_handle_ptr, success);
}
}  // namespace blas_benchmark
