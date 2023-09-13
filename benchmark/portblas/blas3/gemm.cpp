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
 *  @filename gemm.cpp
 *
 **************************************************************************/

#include "../utils.hpp"

constexpr blas_benchmark::utils::Level3Op benchmark_op =
    blas_benchmark::utils::Level3Op::gemm;

template <typename scalar_t, blas::helper::AllocType mem_alloc>
void run(benchmark::State& state, blas::SB_Handle* sb_handle_ptr, int t1,
         int t2, index_t m, index_t k, index_t n, scalar_t alpha, scalar_t beta,
         bool* success) {
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

  index_t lda = t_a[0] == 'n' ? m : k;
  index_t ldb = t_b[0] == 'n' ? k : n;
  index_t ldc = m;

  blas_benchmark::utils::init_level_3_counters<
      blas_benchmark::utils::Level3Op::gemm, scalar_t>(state, beta, m, n, k,
                                                       static_cast<index_t>(1));

  blas::SB_Handle& sb_handle = *sb_handle_ptr;
  auto q = sb_handle.get_queue();

  // Matrices
  std::vector<scalar_t> a = blas_benchmark::utils::random_data<scalar_t>(m * k);
  std::vector<scalar_t> b = blas_benchmark::utils::random_data<scalar_t>(k * n);
  std::vector<scalar_t> c =
      blas_benchmark::utils::const_data<scalar_t>(m * n, 0);

  auto a_gpu = blas::helper::allocate<mem_alloc, scalar_t>(m * k, q);
  auto b_gpu = blas::helper::allocate<mem_alloc, scalar_t>(k * n, q);
  auto c_gpu = blas::helper::allocate<mem_alloc, scalar_t>(m * n, q);

  auto copy_a =
      blas::helper::copy_to_device<scalar_t>(q, a.data(), a_gpu, m * k);
  auto copy_b =
      blas::helper::copy_to_device<scalar_t>(q, b.data(), b_gpu, n * k);
  auto copy_c =
      blas::helper::copy_to_device<scalar_t>(q, c.data(), c_gpu, m * n);

  sb_handle.wait({copy_a, copy_b, copy_c});

#ifdef BLAS_VERIFY_BENCHMARK
  // Run a first time with a verification of the results
  std::vector<scalar_t> c_ref = c;
  reference_blas::gemm(t_a, t_b, m, n, k, alpha, a.data(), lda, b.data(), ldb,
                       beta, c_ref.data(), ldc);
  std::vector<scalar_t> c_temp = c;
  {
    auto c_temp_gpu = blas::helper::allocate<mem_alloc, scalar_t>(m * n, q);
    auto copy_temp = blas::helper::copy_to_device<scalar_t>(q, c_temp.data(),
                                                            c_temp_gpu, m * n);
    sb_handle.wait(copy_temp);
    auto gemm_event = _gemm(sb_handle, *t_a, *t_b, m, n, k, alpha, a_gpu, lda,
                            b_gpu, ldb, beta, c_temp_gpu, ldc);
    sb_handle.wait(gemm_event);
    auto copy_out = blas::helper::copy_to_host<scalar_t>(q, c_temp_gpu,
                                                         c_temp.data(), m * n);
    sb_handle.wait(copy_out);

    blas::helper::deallocate<mem_alloc>(c_temp_gpu, q);
  }

  std::ostringstream err_stream;
  if (!utils::compare_vectors(c_temp, c_ref, err_stream, "")) {
    const std::string& err_str = err_stream.str();
    state.SkipWithError(err_str.c_str());
    *success = false;
  };
#endif

  auto blas_method_def = [&]() -> std::vector<cl::sycl::event> {
    auto event = _gemm(sb_handle, *t_a, *t_b, m, n, k, alpha, a_gpu, lda, b_gpu,
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

  state.SetItemsProcessed(state.iterations() * state.counters["n_fl_ops"]);
  state.SetBytesProcessed(state.iterations() *
                          state.counters["bytes_processed"]);

  blas_benchmark::utils::calc_avg_counters(state);

  blas::helper::deallocate<mem_alloc>(a_gpu, q);
  blas::helper::deallocate<mem_alloc>(b_gpu, q);
  blas::helper::deallocate<mem_alloc>(c_gpu, q);
};

template <typename scalar_t, blas::helper::AllocType mem_alloc>
void register_benchmark(blas::SB_Handle* sb_handle_ptr, bool* success,
                        std::string mem_type,
                        std::vector<blas3_param_t<scalar_t>> params) {
  for (auto p : params) {
    std::string t1s, t2s;
    index_t m, n, k;
    scalar_t alpha, beta;
    std::tie(t1s, t2s, m, k, n, alpha, beta) = p;
    int t1 = static_cast<int>(blas_benchmark::utils::to_transpose_enum(t1s));
    int t2 = static_cast<int>(blas_benchmark::utils::to_transpose_enum(t2s));

    auto BM_lambda = [&](benchmark::State& st, blas::SB_Handle* sb_handle_ptr,
                         int t1, int t2, index_t m, index_t k, index_t n,
                         scalar_t alpha, scalar_t beta, bool* success) {
      run<scalar_t, mem_alloc>(st, sb_handle_ptr, t1, t2, m, k, n, alpha, beta,
                               success);
    };
    benchmark::RegisterBenchmark(
        blas_benchmark::utils::get_name<benchmark_op, scalar_t>(t1s, t2s, m, k,
                                                                n, mem_type)
            .c_str(),
        BM_lambda, sb_handle_ptr, t1, t2, m, k, n, alpha, beta, success)
        ->UseRealTime();
  }
}

template <typename scalar_t>
void register_benchmark(blas_benchmark::Args& args,
                        blas::SB_Handle* sb_handle_ptr, bool* success) {
  auto gemm_params = blas_benchmark::utils::get_blas3_params<scalar_t>(args);
  register_benchmark<scalar_t, blas::helper::AllocType::buffer>(
      sb_handle_ptr, success, blas_benchmark::utils::MEM_TYPE_BUFFER,
      gemm_params);
#ifdef SB_ENABLE_USM
  register_benchmark<scalar_t, blas::helper::AllocType::usm>(
      sb_handle_ptr, success, blas_benchmark::utils::MEM_TYPE_USM, gemm_params);
#endif
}

namespace blas_benchmark {
void create_benchmark(blas_benchmark::Args& args,
                      blas::SB_Handle* sb_handle_ptr, bool* success) {
  BLAS_REGISTER_BENCHMARK(args, sb_handle_ptr, success);
}
}  // namespace blas_benchmark
