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
 *  @filename tpsv.cpp
 *
 **************************************************************************/

#include "../utils.hpp"

constexpr blas_benchmark::utils::Level2Op benchmark_op =
    blas_benchmark::utils::Level2Op::tpsv;

template <typename scalar_t, blas::helper::AllocType mem_alloc>
void run(benchmark::State& state, blas::SB_Handle* sb_handle_ptr,
         std::string uplo, std::string t, std::string diag, index_t n,
         bool* success) {
  // initialize the state label
  blas_benchmark::utils::set_benchmark_label<scalar_t>(
      state, sb_handle_ptr->get_queue());

  // Standard test setup.
  const char* uplo_str = uplo.c_str();
  const char* t_str = t.c_str();
  const char* diag_str = diag.c_str();

  index_t incX = 1;
  index_t xlen = 1 + (n - 1) * incX;

  blas_benchmark::utils::init_level_2_counters<benchmark_op, scalar_t>(
      state, "n", 0, 0, n);

  blas::SB_Handle& sb_handle = *sb_handle_ptr;
  auto q = sb_handle.get_queue();

  index_t m_size = ((n + 1) * n) / 2;

  // Input matrix/vector, output vector.
  std::vector<scalar_t> m_a(m_size);
  std::vector<scalar_t> v_x =
      blas_benchmark::utils::random_data<scalar_t>(xlen);

  // Populate the main diagonal with larger values.
  {
    index_t d_idx = 0;
    for (index_t i = 0; i < n; ++i) {
      m_a[d_idx] =
          blas_benchmark::utils::random_scalar(scalar_t{50}, scalar_t{100});
      d_idx += (*uplo_str == 'u') ? 2 + i : n - i;
    }
  }

  auto m_a_gpu = blas::helper::allocate<mem_alloc, scalar_t>(m_size, q);
  auto v_x_gpu = blas::helper::allocate<mem_alloc, scalar_t>(xlen, q);

  auto copy_m =
      blas::helper::copy_to_device<scalar_t>(q, m_a.data(), m_a_gpu, m_size);
  auto copy_v =
      blas::helper::copy_to_device<scalar_t>(q, v_x.data(), v_x_gpu, xlen);

  sb_handle.wait({copy_m, copy_v});

#ifdef BLAS_VERIFY_BENCHMARK
  // Run a first time with a verification of the results
  std::vector<scalar_t> v_x_ref = v_x;
  reference_blas::tpsv(uplo_str, t_str, diag_str, n, m_a.data(), v_x_ref.data(),
                       incX);
  std::vector<scalar_t> v_x_temp = v_x;
  {
    auto v_x_temp_gpu = blas::helper::allocate<mem_alloc, scalar_t>(xlen, q);
    auto copy_temp = blas::helper::copy_to_device<scalar_t>(q, v_x_temp.data(),
                                                            v_x_temp_gpu, xlen);
    sb_handle.wait(copy_temp);
    auto event = _tpsv(sb_handle, *uplo_str, *t_str, *diag_str, n, m_a_gpu,
                       v_x_temp_gpu, incX);
    sb_handle.wait();
    auto copy_out = blas::helper::copy_to_host<scalar_t>(q, v_x_temp_gpu,
                                                         v_x_temp.data(), xlen);
    sb_handle.wait(copy_out);

    blas::helper::deallocate<mem_alloc>(v_x_temp_gpu, q);
  }

  std::ostringstream err_stream;
  if (!utils::compare_vectors(v_x_temp, v_x_ref, err_stream, "")) {
    const std::string& err_str = err_stream.str();
    state.SkipWithError(err_str.c_str());
    *success = false;
  };
#endif

  auto blas_method_def = [&]() -> std::vector<cl::sycl::event> {
    auto event = _tpsv(sb_handle, *uplo_str, *t_str, *diag_str, n, m_a_gpu,
                       v_x_gpu, incX);
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

  blas::helper::deallocate<mem_alloc>(m_a_gpu, q);
  blas::helper::deallocate<mem_alloc>(v_x_gpu, q);
}

template <typename scalar_t, blas::helper::AllocType mem_alloc>
void register_benchmark(blas::SB_Handle* sb_handle_ptr, bool* success,
                        std::string mem_type,
                        std::vector<trsv_param_t> params) {
  for (auto p : params) {
    std::string uplos;
    std::string ts;
    std::string diags;
    index_t n;
    std::tie(uplos, ts, diags, n) = p;

    auto BM_lambda = [&](benchmark::State& st, blas::SB_Handle* sb_handle_ptr,
                         std::string uplos, std::string ts, std::string diags,
                         index_t n, bool* success) {
      run<scalar_t, mem_alloc>(st, sb_handle_ptr, uplos, ts, diags, n, success);
    };
    benchmark::RegisterBenchmark(
        blas_benchmark::utils::get_name<benchmark_op, scalar_t>(
            uplos, ts, diags, n, mem_type)
            .c_str(),
        BM_lambda, sb_handle_ptr, uplos, ts, diags, n, success)
        ->UseRealTime();
  }
}

template <typename scalar_t>
void register_benchmark(blas_benchmark::Args& args,
                        blas::SB_Handle* sb_handle_ptr, bool* success) {
  // tpsv uses the same parameters as trsv
  auto tpsv_params = blas_benchmark::utils::get_trsv_params(args);

  register_benchmark<scalar_t, blas::helper::AllocType::buffer>(
      sb_handle_ptr, success, blas_benchmark::utils::MEM_TYPE_BUFFER,
      tpsv_params);
#ifdef SB_ENABLE_USM
  register_benchmark<scalar_t, blas::helper::AllocType::usm>(
      sb_handle_ptr, success, blas_benchmark::utils::MEM_TYPE_USM, tpsv_params);
#endif
}

namespace blas_benchmark {
void create_benchmark(blas_benchmark::Args& args,
                      blas::SB_Handle* sb_handle_ptr, bool* success) {
  BLAS_REGISTER_BENCHMARK(args, sb_handle_ptr, success);
}
}  // namespace blas_benchmark
