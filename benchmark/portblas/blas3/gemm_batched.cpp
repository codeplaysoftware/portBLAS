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
 *  @filename gemm_batched.cpp
 *
 **************************************************************************/

#include "../utils.hpp"

constexpr blas_benchmark::utils::Level3Op benchmark_op =
    blas_benchmark::utils::Level3Op::gemm_batched;

// Convert batch_type=strided to interleaved on the host
template <typename scalar_t>
std::vector<scalar_t> strided_to_interleaved(const std::vector<scalar_t>& input,
                                             int offset, int ld_rows,
                                             int ld_cols, int batchs) {
  std::vector<scalar_t> output(input.size());
  for (int c = 0; c < ld_cols; ++c) {
    for (int r = 0; r < ld_rows; ++r) {
      for (int b = 0; b < batchs; ++b) {
        output[c * ld_rows * batchs + r * batchs + b + offset] =
            input[b * ld_cols * ld_rows + c * ld_rows + r + offset];
      }
    }
  }
  return output;
}

// Convert batch_type=interleaved to strided on the host
template <typename scalar_t>
std::vector<scalar_t> interleaved_to_strided(const std::vector<scalar_t>& input,
                                             int offset, int ld_rows,
                                             int ld_cols, int batchs) {
  std::vector<scalar_t> output(input.size());
  for (int b = 0; b < batchs; ++b) {
    for (int c = 0; c < ld_cols; ++c) {
      for (int r = 0; r < ld_rows; ++r) {
        output[b * ld_cols * ld_rows + c * ld_rows + r + offset] =
            input[c * ld_rows * batchs + r * batchs + b + offset];
      }
    }
  }
  return output;
}

template <typename scalar_t, blas::helper::AllocType mem_alloc>
void run(benchmark::State& state, blas::SB_Handle* sb_handle_ptr, int t1,
         int t2, index_t m, index_t k, index_t n, scalar_t alpha, scalar_t beta,
         index_t batch_size, int batch_type_i, bool* success) {
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
  auto batch_type = static_cast<blas::gemm_batch_type_t>(batch_type_i);

  index_t lda = t_a[0] == 'n' ? m : k;
  index_t ldb = t_b[0] == 'n' ? k : n;
  index_t ldc = m;

  blas_benchmark::utils::init_level_3_counters<
      blas_benchmark::utils::Level3Op::gemm_batched, scalar_t>(
      state, beta, m, n, k, batch_size);

  blas::SB_Handle& sb_handle = *sb_handle_ptr;
  auto q = sb_handle.get_queue();

  // Matrices
  std::vector<scalar_t> a =
      blas_benchmark::utils::random_data<scalar_t>(m * k * batch_size);
  std::vector<scalar_t> b =
      blas_benchmark::utils::random_data<scalar_t>(k * n * batch_size);
  std::vector<scalar_t> c =
      blas_benchmark::utils::const_data<scalar_t>(m * n * batch_size, 0);

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

  if (batch_type == blas::gemm_batch_type_t::interleaved) {
    constexpr int offset = 0;
    a = strided_to_interleaved(a, offset, lda, t_a[0] == 't' ? m : k,
                               batch_size);
    b = strided_to_interleaved(b, offset, ldb, t_b[0] == 't' ? k : n,
                               batch_size);
    c = strided_to_interleaved(c, offset, ldc, n, batch_size);
  }
#endif

  auto a_gpu =
      blas::helper::allocate<mem_alloc, scalar_t>(m * k * batch_size, q);
  auto b_gpu =
      blas::helper::allocate<mem_alloc, scalar_t>(k * n * batch_size, q);
  auto c_gpu =
      blas::helper::allocate<mem_alloc, scalar_t>(m * n * batch_size, q);

  auto copy_a = blas::helper::copy_to_device<scalar_t>(q, a.data(), a_gpu,
                                                       m * k * batch_size);
  auto copy_b = blas::helper::copy_to_device<scalar_t>(q, b.data(), b_gpu,
                                                       n * k * batch_size);
  auto copy_c = blas::helper::copy_to_device<scalar_t>(q, c.data(), c_gpu,
                                                       m * n * batch_size);

  sb_handle.wait({copy_a, copy_b, copy_c});

#ifdef BLAS_VERIFY_BENCHMARK
  std::vector<scalar_t> c_temp = c;
  {
    auto c_temp_gpu =
        blas::helper::allocate<mem_alloc, scalar_t>(m * n * batch_size, q);
    auto copy_temp = blas::helper::copy_to_device<scalar_t>(
        q, c_temp.data(), c_temp_gpu, m * n * batch_size);
    sb_handle.wait(copy_temp);
    auto gemm_batched_event =
        _gemm_batched(sb_handle, *t_a, *t_b, m, n, k, alpha, a_gpu, lda, b_gpu,
                      ldb, beta, c_temp_gpu, ldc, batch_size, batch_type);
    sb_handle.wait(gemm_batched_event);
    auto copy_out = blas::helper::copy_to_host<scalar_t>(
        q, c_temp_gpu, c_temp.data(), m * n * batch_size);
    sb_handle.wait(copy_out);

    blas::helper::deallocate<mem_alloc>(c_temp_gpu, q);
  }
  if (batch_type == blas::gemm_batch_type_t::interleaved) {
    constexpr int offset = 0;
    c_temp = interleaved_to_strided(c_temp, offset, ldc, n, batch_size);
  }

  std::ostringstream err_stream;
  if (!utils::compare_vectors(c_temp, c_ref, err_stream, "")) {
    const std::string& err_str = err_stream.str();
    state.SkipWithError(err_str.c_str());
    *success = false;
  };
#endif

  auto blas_method_def = [&]() -> std::vector<cl::sycl::event> {
    auto event =
        _gemm_batched(sb_handle, *t_a, *t_b, m, n, k, alpha, a_gpu, lda, b_gpu,
                      ldb, beta, c_gpu, ldc, batch_size, batch_type);
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
                        std::vector<gemm_batched_param_t<scalar_t>> params) {
  for (auto p : params) {
    std::string t1s, t2s;
    index_t m, n, k, batch_size;
    scalar_t alpha, beta;
    int batch_type;
    std::tie(t1s, t2s, m, k, n, alpha, beta, batch_size, batch_type) = p;
    int t1 = static_cast<int>(blas_benchmark::utils::to_transpose_enum(t1s));
    int t2 = static_cast<int>(blas_benchmark::utils::to_transpose_enum(t2s));

    auto BM_lambda = [&](benchmark::State& st, blas::SB_Handle* sb_handle_ptr,
                         int t1, int t2, index_t m, index_t k, index_t n,
                         scalar_t alpha, scalar_t beta, index_t batch_size,
                         int batch_type, bool* success) {
      run<scalar_t, mem_alloc>(st, sb_handle_ptr, t1, t2, m, k, n, alpha, beta,
                               batch_size, batch_type, success);
    };
    benchmark::RegisterBenchmark(
        blas_benchmark::utils::get_name<benchmark_op, scalar_t>(
            t1s, t2s, m, k, n, batch_size, batch_type, mem_type)
            .c_str(),
        BM_lambda, sb_handle_ptr, t1, t2, m, k, n, alpha, beta, batch_size,
        batch_type, success)
        ->UseRealTime();
  }
}

template <typename scalar_t>
void register_benchmark(blas_benchmark::Args& args,
                        blas::SB_Handle* sb_handle_ptr, bool* success) {
  auto gemm_batched_params =
      blas_benchmark::utils::get_gemm_batched_params<scalar_t>(args);
  register_benchmark<scalar_t, blas::helper::AllocType::buffer>(
      sb_handle_ptr, success, blas_benchmark::utils::MEM_TYPE_BUFFER,
      gemm_batched_params);
#ifdef SB_ENABLE_USM
  register_benchmark<scalar_t, blas::helper::AllocType::usm>(
      sb_handle_ptr, success, blas_benchmark::utils::MEM_TYPE_USM,
      gemm_batched_params);
#endif
}

#ifdef BLAS_ENABLE_COMPLEX
template <typename scalar_t, blas::helper::AllocType mem_alloc>
void run(benchmark::State& state, blas::SB_Handle* sb_handle_ptr, int t1,
         int t2, index_t m, index_t k, index_t n, std::complex<scalar_t> alpha,
         std::complex<scalar_t> beta, index_t batch_size, int batch_type_i,
         bool* success) {
  // initialize the state label
  blas_benchmark::utils::set_benchmark_label<std::complex<scalar_t>>(
      state, sb_handle_ptr->get_queue());

  // Standard test setup.
  std::string t1s = blas_benchmark::utils::from_transpose_enum(
      static_cast<blas_benchmark::utils::Transposition>(t1));
  std::string t2s = blas_benchmark::utils::from_transpose_enum(
      static_cast<blas_benchmark::utils::Transposition>(t2));
  const char* t_a = t1s.c_str();
  const char* t_b = t2s.c_str();
  auto batch_type = static_cast<blas::gemm_batch_type_t>(batch_type_i);

  index_t lda = t_a[0] == 'n' ? m : k;
  index_t ldb = t_b[0] == 'n' ? k : n;
  index_t ldc = m;

  blas_benchmark::utils::init_level_3_cplx_counters<
      blas_benchmark::utils::Level3Op::gemm_batched, scalar_t>(
      state, beta, m, n, k, batch_size);

  blas::SB_Handle& sb_handle = *sb_handle_ptr;
  auto q = sb_handle.get_queue();

  // Matrices
  std::vector<std::complex<scalar_t>> a =
      blas_benchmark::utils::random_cplx_data<scalar_t>(m * k * batch_size);
  std::vector<std::complex<scalar_t>> b =
      blas_benchmark::utils::random_cplx_data<scalar_t>(k * n * batch_size);
  std::vector<std::complex<scalar_t>> c =
      blas_benchmark::utils::const_cplx_data<scalar_t>(m * n * batch_size,
                                                       scalar_t(0));

#ifdef BLAS_VERIFY_BENCHMARK
  // Run a first time with a verification of the results
  std::vector<std::complex<scalar_t>> c_ref = c;
  auto _base = [=](index_t dim0, index_t dim1, index_t idx) {
    return dim0 * dim1 * idx;
  };
  for (int batch_idx = 0; batch_idx < batch_size; batch_idx++) {
    reference_blas::cgemm<scalar_t>(
        t_a, t_b, m, n, k, reinterpret_cast<const void*>(&alpha),
        reinterpret_cast<const void*>(a.data() + _base(m, k, batch_idx)), lda,
        reinterpret_cast<const void*>(b.data() + _base(k, n, batch_idx)), ldb,
        reinterpret_cast<const void*>(&beta),
        reinterpret_cast<void*>(c_ref.data() + _base(m, n, batch_idx)), ldc);
  }

#endif  // BLAS_VERIFY_BENCHMARK

  auto a_gpu = blas::helper::allocate<mem_alloc, blas::complex_sycl<scalar_t>>(
      m * k * batch_size, q);
  auto b_gpu = blas::helper::allocate<mem_alloc, blas::complex_sycl<scalar_t>>(
      k * n * batch_size, q);
  auto c_gpu = blas::helper::allocate<mem_alloc, blas::complex_sycl<scalar_t>>(
      m * n * batch_size, q);

  auto copy_a = blas::helper::copy_to_device(
      q, reinterpret_cast<blas::complex_sycl<scalar_t>*>(a.data()), a_gpu,
      m * k * batch_size);
  auto copy_b = blas::helper::copy_to_device(
      q, reinterpret_cast<blas::complex_sycl<scalar_t>*>(b.data()), b_gpu,
      n * k * batch_size);
  auto copy_c = blas::helper::copy_to_device(
      q, reinterpret_cast<blas::complex_sycl<scalar_t>*>(c.data()), c_gpu,
      m * n * batch_size);

  sb_handle.wait({copy_a, copy_b, copy_c});

  // Kernel expects sycl::complex and not std::complex data
  blas::complex_sycl<scalar_t> alpha_sycl(alpha);
  blas::complex_sycl<scalar_t> beta_sycl(beta);

#ifdef BLAS_VERIFY_BENCHMARK
  std::vector<std::complex<scalar_t>> c_temp = c;
  {
    auto c_temp_gpu =
        blas::helper::allocate<mem_alloc, blas::complex_sycl<scalar_t>>(
            m * n * batch_size, q);
    auto copy_temp = blas::helper::copy_to_device(
        q, reinterpret_cast<blas::complex_sycl<scalar_t>*>(c_temp.data()),
        c_temp_gpu, m * n * batch_size);
    sb_handle.wait(copy_temp);
    auto gemm_batched_event = _gemm_batched(
        sb_handle, *t_a, *t_b, m, n, k, alpha_sycl, a_gpu, lda, b_gpu, ldb,
        beta_sycl, c_temp_gpu, ldc, batch_size, batch_type);
    sb_handle.wait(gemm_batched_event);
    auto copy_out = blas::helper::copy_to_host(
        q, c_temp_gpu,
        reinterpret_cast<blas::complex_sycl<scalar_t>*>(c_temp.data()),
        m * n * batch_size);
    sb_handle.wait(copy_out);

    blas::helper::deallocate<mem_alloc>(c_temp_gpu, q);
  }

  std::ostringstream err_stream;
  if (!utils::compare_vectors<scalar_t>(c_temp, c_ref, err_stream, "")) {
    const std::string& err_str = err_stream.str();
    state.SkipWithError(err_str.c_str());
    *success = false;
  };
#endif  // BLAS_VERIFY_BENCHMARK

  auto blas_method_def = [&]() -> std::vector<cl::sycl::event> {
    auto event = _gemm_batched(sb_handle, *t_a, *t_b, m, n, k, alpha_sycl,
                               a_gpu, lda, b_gpu, ldb, beta_sycl, c_gpu, ldc,
                               batch_size, batch_type);
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

/*! @brief Register & run benchmark of complex data types gemm batched.
 * Function is similar to register_benchmark
 *
 * @tparam scalar_t element data type of underlying complex (float or double)
 * @tparam mem_alloc USM or Buffer memory allocation
 */
template <typename scalar_t, blas::helper::AllocType mem_alloc>
void register_cplx_benchmark(
    blas::SB_Handle* sb_handle_ptr, bool* success, std::string mem_type,
    std::vector<gemm_batched_cplx_param_t<scalar_t>> params) {
  for (auto p : params) {
    std::string t1s, t2s;
    index_t m, n, k, batch_size;
    scalar_t alpha_r, alpha_i, beta_r, beta_i;
    int batch_type;
    std::tie(t1s, t2s, m, k, n, alpha_r, alpha_i, beta_r, beta_i, batch_size,
             batch_type) = p;
    // Only batch_type == strided is supported with Complex data
    if (batch_type == 1) {
      std::cerr << "Interleaved memory for gemm_batched operator is not "
                   "supported whith complex data type\n";
      continue;
    }
    int t1 = static_cast<int>(blas_benchmark::utils::to_transpose_enum(t1s));
    int t2 = static_cast<int>(blas_benchmark::utils::to_transpose_enum(t2s));
    std::complex<scalar_t> alpha{alpha_r, alpha_i};
    std::complex<scalar_t> beta{beta_r, beta_i};

    auto BM_lambda = [&](benchmark::State& st, blas::SB_Handle* sb_handle_ptr,
                         int t1, int t2, index_t m, index_t k, index_t n,
                         std::complex<scalar_t> alpha,
                         std::complex<scalar_t> beta, index_t batch_size,
                         int batch_type, bool* success) {
      run<scalar_t, mem_alloc>(st, sb_handle_ptr, t1, t2, m, k, n, alpha, beta,
                               batch_size, batch_type, success);
    };
    benchmark::RegisterBenchmark(
        blas_benchmark::utils::get_name<benchmark_op, std::complex<scalar_t>>(
            t1s, t2s, m, k, n, batch_size, batch_type, mem_type)
            .c_str(),
        BM_lambda, sb_handle_ptr, t1, t2, m, k, n, alpha, beta, batch_size,
        batch_type, success)
        ->UseRealTime();
  }
}

template <typename scalar_t>
void register_cplx_benchmark(blas_benchmark::Args& args,
                             blas::SB_Handle* sb_handle_ptr, bool* success) {
  auto gemm_batched_params =
      blas_benchmark::utils::get_gemm_cplx_batched_params<scalar_t>(args);
  register_cplx_benchmark<scalar_t, blas::helper::AllocType::buffer>(
      sb_handle_ptr, success, blas_benchmark::utils::MEM_TYPE_BUFFER,
      gemm_batched_params);
#ifdef SB_ENABLE_USM
  register_cplx_benchmark<scalar_t, blas::helper::AllocType::usm>(
      sb_handle_ptr, success, blas_benchmark::utils::MEM_TYPE_USM,
      gemm_batched_params);
#endif
}
#endif

namespace blas_benchmark {
void create_benchmark(blas_benchmark::Args& args,
                      blas::SB_Handle* sb_handle_ptr, bool* success) {
  BLAS_REGISTER_BENCHMARK(args, sb_handle_ptr, success);
}
}  // namespace blas_benchmark
