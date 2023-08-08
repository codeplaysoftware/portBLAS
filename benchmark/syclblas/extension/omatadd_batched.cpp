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
 *  @filename omatadd_batched.cpp
 *
 **************************************************************************/

#include "../../../test/unittest/extension/extension_reference.hpp"
#include "../utils.hpp"

template <typename scalar_t>
std::string get_name(std::string ts_a, std::string ts_b, int m, int n,
                     scalar_t alpha, scalar_t beta, index_t lda_mul,
                     index_t ldb_mul, index_t ldc_mul, index_t stride_a_mul,
                     index_t stride_b_mul, index_t stride_c_mul,
                     index_t batch_size) {
  std::ostringstream str{};
  str << "BM_omatadd_batch<" << blas_benchmark::utils::get_type_name<scalar_t>()
      << ">/" << ts_a << "/" << ts_b << "/" << m << "/" << n << "/" << alpha
      << "/" << beta << "/" << lda_mul << "/" << ldb_mul << "/" << ldc_mul
      << "/" << stride_a_mul << "/" << stride_b_mul << "/" << stride_c_mul
      << "/" << batch_size;
  return str.str();
}

template <typename scalar_t>
void run(benchmark::State& state, blas::SB_Handle* sb_handle_ptr, int ti_a,
         int ti_b, index_t m, index_t n, scalar_t alpha, scalar_t beta,
         index_t lda_mul, index_t ldb_mul, index_t ldc_mul,
         index_t stride_a_mul, index_t stride_b_mul, index_t stride_c_mul,
         index_t batch_size, bool* success) {
  // initiliaze the state label
  blas_benchmark::utils::set_benchmark_label<scalar_t>(
      state, sb_handle_ptr->get_queue());

  // Standard test setup.
  std::string ts_a = blas_benchmark::utils::from_transpose_enum(
      static_cast<blas_benchmark::utils::Transposition>(ti_a));
  const char* t_str_a = ts_a.c_str();
  std::string ts_b = blas_benchmark::utils::from_transpose_enum(
      static_cast<blas_benchmark::utils::Transposition>(ti_b));
  const char* t_str_b = ts_b.c_str();

  const auto lda = (*t_str_b == 't') ? lda_mul * n : lda_mul * m;
  const auto ldb = (*t_str_b == 't') ? ldb_mul * n : ldb_mul * m;
  const auto ldc = ldc_mul * m;

  const auto stride_a = lda * ((*t_str_a == 't') ? m : n);
  const auto stride_b = ldb * ((*t_str_b == 't') ? m : n);
  const auto stride_c = ldc * n;

  const auto size_a = stride_a * stride_a_mul * batch_size;
  const auto size_b = stride_b * stride_b_mul * batch_size;
  const auto size_c = stride_c * stride_c_mul * batch_size;

  blas_benchmark::utils::init_extension_counters<
      blas_benchmark::utils::ExtensionOP::omatadd_batch, scalar_t>(
      state, t_str_a, t_str_b, m, n, lda_mul, ldb_mul, ldc_mul, stride_a_mul,
      stride_b_mul, stride_c_mul, batch_size);

  blas::SB_Handle& sb_handle = *sb_handle_ptr;

  // Input matrix/vector, output vector.
  std::vector<scalar_t> m_a =
      blas_benchmark::utils::random_data<scalar_t>(size_a);
  std::vector<scalar_t> m_b =
      blas_benchmark::utils::random_data<scalar_t>(size_b);
  std::vector<scalar_t> m_c =
      blas_benchmark::utils::random_data<scalar_t>(size_c);

  auto m_a_gpu = blas::make_sycl_iterator_buffer<scalar_t>(m_a, size_a);
  auto m_b_gpu = blas::make_sycl_iterator_buffer<scalar_t>(m_b, size_b);
  auto m_c_gpu = blas::make_sycl_iterator_buffer<scalar_t>(m_c, size_c);

#ifdef BLAS_VERIFY_BENCHMARK
  // Run a first time with a verification of the results
  std::vector<scalar_t> m_c_ref = m_c;

  for (int i = 0; i < batch_size; ++i) {
    reference_blas::ext_omatadd(
        *t_str_a, *t_str_b, m, n, alpha, m_a.data() + i * stride_a, lda, beta,
        m_b.data() + i * stride_b, ldb, m_c_ref.data() + i * stride_c, ldc);
  }

  std::vector<scalar_t> m_c_temp = m_c;
  {
    auto m_c_temp_gpu =
        blas::make_sycl_iterator_buffer<scalar_t>(m_c_temp, size_c);

    auto event = blas::_omatadd_batch(
        sb_handle, *t_str_a, *t_str_b, m, n, alpha, m_a_gpu, lda, stride_a,
        beta, m_b_gpu, ldb, stride_b, m_c_temp_gpu, ldc, stride_c, batch_size);

    sb_handle.wait();
  }

  std::ostringstream err_stream;
  if (!utils::compare_vectors(m_c_temp, m_c_ref, err_stream, "")) {
    const std::string& err_str = err_stream.str();
    state.SkipWithError(err_str.c_str());
    *success = false;
  };
#endif

  auto blas_method_warmup_def = [&]() -> void {
    auto event = blas::_omatadd_batch(
        sb_handle, *t_str_a, *t_str_b, m, n, alpha, m_a_gpu, lda, stride_a,
        beta, m_b_gpu, ldb, stride_b, m_c_gpu, ldc, stride_c, batch_size);
    return;
  };

  auto blas_method_def = [&]() -> std::vector<cl::sycl::event> {
    auto event = blas::_omatadd_batch(
        sb_handle, *t_str_a, *t_str_b, m, n, alpha, m_a_gpu, lda, stride_a,
        beta, m_b_gpu, ldb, stride_b, m_c_gpu, ldc, stride_c, batch_size);
    sb_handle.wait(event);
    return event;
  };

  // Warmup
  blas_benchmark::utils::warmup(blas_method_warmup_def);
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
  auto omatadd_batch_params =
      blas_benchmark::utils::get_omatadd_batch_params<scalar_t>(args);

  for (auto p : omatadd_batch_params) {
    std::string ts_a, ts_b;
    index_t m, n, lda_mul, ldb_mul, ldc_mul, stride_a_mul, stride_b_mul,
        stride_c_mul, batch_size;
    scalar_t alpha, beta;
    std::tie(ts_a, ts_b, m, n, alpha, beta, lda_mul, ldb_mul, ldc_mul,
             stride_a_mul, stride_b_mul, stride_c_mul, batch_size) = p;
    int t_a = static_cast<int>(blas_benchmark::utils::to_transpose_enum(ts_a));
    int t_b = static_cast<int>(blas_benchmark::utils::to_transpose_enum(ts_b));

    auto BM_lambda = [&](benchmark::State& st, blas::SB_Handle* sb_handle_ptr,
                         int t_a, int t_b, index_t m, index_t n, scalar_t alpha,
                         scalar_t beta, index_t lda_mul, index_t ldb_mul,
                         index_t ldc_mul, index_t stride_a_mul,
                         index_t stride_b_mul, index_t stride_c_mul,
                         index_t batch_size, bool* success) {
      run<scalar_t>(st, sb_handle_ptr, t_a, t_b, m, n, alpha, beta, lda_mul,
                    ldb_mul, ldc_mul, stride_a_mul, stride_b_mul, stride_c_mul,
                    batch_size, success);
    };
    benchmark::RegisterBenchmark(
        get_name<scalar_t>(ts_a, ts_b, m, n, alpha, beta, lda_mul, ldb_mul,
                           ldc_mul, stride_a_mul, stride_b_mul, stride_c_mul,
                           batch_size)
            .c_str(),
        BM_lambda, sb_handle_ptr, t_a, t_b, m, n, alpha, beta, lda_mul, ldb_mul,
        ldc_mul, stride_a_mul, stride_b_mul, stride_c_mul, batch_size, success)
        ->UseRealTime();
  }
}

namespace blas_benchmark {
void create_benchmark(blas_benchmark::Args& args,
                      blas::SB_Handle* sb_handle_ptr, bool* success) {
  BLAS_REGISTER_BENCHMARK(args, sb_handle_ptr, success);
}
}  // namespace blas_benchmark
