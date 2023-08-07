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
 *  @filename gemm_batched_strided.cpp
 *
 **************************************************************************/

#include "../utils.hpp"

constexpr blas_benchmark::utils::Level3Op benchmark_op =
    blas_benchmark::utils::Level3Op::gemm_batched_strided;

template <typename scalar_t>
void run(benchmark::State& state, blas::SB_Handle* sb_handle_ptr, int t1,
         int t2, index_t m, index_t k, index_t n, scalar_t alpha, scalar_t beta,
         index_t batch_size, index_t stride_a_mul, index_t stride_b_mul,
         index_t stride_c_mul, bool* success) {
  // initialize the state label
  blas_benchmark::utils::set_benchmark_label<scalar_t>(
      state, sb_handle_ptr->get_queue());

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

  blas_benchmark::utils::init_level_3_counters<
      blas_benchmark::utils::Level3Op::gemm_batched_strided, scalar_t>(
      state, beta, m, n, k, batch_size, stride_a_mul, stride_b_mul,
      stride_c_mul);

  blas::SB_Handle& sb_handle = *sb_handle_ptr;
  auto q = sb_handle.get_queue();

  // Data sizes
  // Elementary matrices
  const index_t a_size = m * k;
  const index_t b_size = k * n;
  const index_t c_size = m * n;
  // Strides
  const index_t stride_a = stride_a_mul * a_size;
  const index_t stride_b = stride_b_mul * b_size;
  const index_t stride_c = stride_c_mul * c_size;
  // Batched matrices
  const int size_a_batch = a_size + (batch_size - 1) * stride_a;
  const int size_b_batch = b_size + (batch_size - 1) * stride_b;
  const int size_c_batch = c_size + (batch_size - 1) * stride_c;

  // Matrices
  std::vector<scalar_t> a =
      blas_benchmark::utils::random_data<scalar_t>(size_a_batch);
  std::vector<scalar_t> b =
      blas_benchmark::utils::random_data<scalar_t>(size_b_batch);
  std::vector<scalar_t> c =
      blas_benchmark::utils::const_data<scalar_t>(size_c_batch, 0);

#ifdef BLAS_VERIFY_BENCHMARK
  // Run a first time with a verification of the results
  std::vector<scalar_t> c_ref = c;
  for (int batch_idx = 0; batch_idx < batch_size; batch_idx++) {
    reference_blas::gemm(t_a, t_b, m, n, k, alpha,
                         a.data() + batch_idx * stride_a, lda,
                         b.data() + batch_idx * stride_b, ldb, beta,
                         c_ref.data() + batch_idx * stride_c, ldc);
  }

#endif

  auto a_gpu = blas::make_sycl_iterator_buffer<scalar_t>(a, size_a_batch);
  auto b_gpu = blas::make_sycl_iterator_buffer<scalar_t>(b, size_b_batch);
  auto c_gpu = blas::make_sycl_iterator_buffer<scalar_t>(c, size_c_batch);

#ifdef BLAS_VERIFY_BENCHMARK
  std::vector<scalar_t> c_temp = c;
  {
    auto c_temp_gpu =
        blas::make_sycl_iterator_buffer<scalar_t>(c_temp, size_c_batch);
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
      blas_benchmark::utils::get_gemm_batched_strided_params<scalar_t>(args);

  for (auto p : gemm_batched_strided_params) {
    std::string t1s, t2s;
    index_t m, n, k, batch_size, stride_a_mul, stride_b_mul, stride_c_mul;
    scalar_t alpha, beta;
    std::tie(t1s, t2s, m, k, n, alpha, beta, batch_size, stride_a_mul,
             stride_b_mul, stride_c_mul) = p;
    int t1 = static_cast<int>(blas_benchmark::utils::to_transpose_enum(t1s));
    int t2 = static_cast<int>(blas_benchmark::utils::to_transpose_enum(t2s));

    auto BM_lambda = [&](benchmark::State& st, blas::SB_Handle* sb_handle_ptr,
                         int t1, int t2, index_t m, index_t k, index_t n,
                         scalar_t alpha, scalar_t beta, index_t batch_size,
                         index_t strd_a_mul, index_t strd_b_mul,
                         index_t strd_c_mul, bool* success) {
      run<scalar_t>(st, sb_handle_ptr, t1, t2, m, k, n, alpha, beta, batch_size,
                    strd_a_mul, strd_b_mul, strd_c_mul, success);
    };
    benchmark::RegisterBenchmark(
        blas_benchmark::utils::get_name<benchmark_op, scalar_t>(
            t1s, t2s, m, k, n, batch_size, stride_a_mul, stride_b_mul,
            stride_c_mul, blas_benchmark::utils::MEM_TYPE_BUFFER)
            .c_str(),
        BM_lambda, sb_handle_ptr, t1, t2, m, k, n, alpha, beta, batch_size,
        stride_a_mul, stride_b_mul, stride_c_mul, success)
        ->UseRealTime();
  }
}

namespace blas_benchmark {
void create_benchmark(blas_benchmark::Args& args,
                      blas::SB_Handle* sb_handle_ptr, bool* success) {
  BLAS_REGISTER_BENCHMARK(args, sb_handle_ptr, success);
}
}  // namespace blas_benchmark
