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
 *  @filename syr2.cpp
 *
 **************************************************************************/

#include "../utils.hpp"

constexpr blas_benchmark::utils::Level2Op benchmark_op =
    blas_benchmark::utils::Level2Op::syr2;

template <typename scalar_t, blas::helper::AllocType mem_alloc>
void run(benchmark::State& state, blas::SB_Handle* sb_handle_ptr,
         std::string uplo, index_t n, scalar_t alpha, bool* success) {
  // initialize the state label
  blas_benchmark::utils::set_benchmark_label<scalar_t>(
      state, sb_handle_ptr->get_queue());

  // Standard test setup.
  const char* uplo_str = uplo.c_str();

  index_t lda = n;
  index_t incX = 1;
  index_t incY = 1;

  blas_benchmark::utils::init_level_2_counters<
      blas_benchmark::utils::Level2Op::syr2, scalar_t>(state, "n", 0, 0, n);

  blas::SB_Handle& sb_handle = *sb_handle_ptr;
  auto q = sb_handle.get_queue();

  const int v_x_size = 1 + (n - 1) * std::abs(incX);
  const int v_y_size = 1 + (n - 1) * std::abs(incY);

  // Input matrix/vector, output vector.
  std::vector<scalar_t> m_a =
      blas_benchmark::utils::random_data<scalar_t>(n * n);
  std::vector<scalar_t> v_x =
      blas_benchmark::utils::random_data<scalar_t>(v_x_size);
  std::vector<scalar_t> v_y =
      blas_benchmark::utils::random_data<scalar_t>(v_y_size);

  auto m_a_gpu = blas::helper::allocate<mem_alloc, scalar_t>(n * n, q);
  auto v_x_gpu = blas::helper::allocate<mem_alloc, scalar_t>(v_x_size, q);
  auto v_y_gpu = blas::helper::allocate<mem_alloc, scalar_t>(v_y_size, q);

  auto copy_m =
      blas::helper::copy_to_device<scalar_t>(q, m_a.data(), m_a_gpu, n * n);
  auto copy_x =
      blas::helper::copy_to_device<scalar_t>(q, v_x.data(), v_x_gpu, v_x_size);
  auto copy_y =
      blas::helper::copy_to_device<scalar_t>(q, v_y.data(), v_y_gpu, v_y_size);

  sb_handle.wait({copy_m, copy_x, copy_y});

#ifdef BLAS_VERIFY_BENCHMARK
  // Run a first time with a verification of the results
  std::vector<scalar_t> m_a_ref = m_a;
  reference_blas::syr2(uplo_str, n, alpha, v_x.data(), incX, v_y.data(), incY,
                       m_a_ref.data(), lda);
  std::vector<scalar_t> m_a_temp(m_a);

  {
    auto m_a_temp_gpu = blas::helper::allocate<mem_alloc, scalar_t>(n * n, q);
    auto copy_temp = blas::helper::copy_to_device<scalar_t>(
        q, m_a_temp.data(), m_a_temp_gpu, n * n);
    sb_handle.wait(copy_temp);
    auto syr2_event = _syr2(sb_handle, *uplo_str, n, alpha, v_x_gpu, incX,
                            v_y_gpu, incY, m_a_temp_gpu, lda);
    sb_handle.wait(syr2_event);
    auto copy_out = blas::helper::copy_to_host<scalar_t>(
        q, m_a_temp_gpu, m_a_temp.data(), n * n);
    sb_handle.wait(copy_out);

    blas::helper::deallocate<mem_alloc>(m_a_temp_gpu, q);
  }

  std::ostringstream err_stream;
  if (!utils::compare_vectors(m_a_temp, m_a_ref, err_stream, "")) {
    const std::string& err_str = err_stream.str();
    state.SkipWithError(err_str.c_str());
    *success = false;
  };

#endif

  auto blas_method_def = [&]() -> std::vector<cl::sycl::event> {
    auto event = _syr2(sb_handle, *uplo_str, n, alpha, v_x_gpu, incX, v_y_gpu,
                       incY, m_a_gpu, lda);
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
  blas::helper::deallocate<mem_alloc>(v_y_gpu, q);
}

template <typename scalar_t, blas::helper::AllocType mem_alloc>
void register_benchmark(blas::SB_Handle* sb_handle_ptr, bool* success,
                        std::string mem_type,
                        std::vector<syr_param_t<scalar_t>> params) {
  for (auto p : params) {
    std::string uplo;
    index_t n;
    scalar_t alpha;
    std::tie(uplo, n, alpha) = p;

    auto BM_lambda = [&](benchmark::State& st, blas::SB_Handle* sb_handle_ptr,
                         std::string uplo, index_t n, scalar_t alpha,
                         bool* success) {
      run<scalar_t, mem_alloc>(st, sb_handle_ptr, uplo, n, alpha, success);
    };
    benchmark::RegisterBenchmark(
        blas_benchmark::utils::get_name<benchmark_op, scalar_t>(
            uplo, n, alpha, mem_type).c_str(),
        BM_lambda, sb_handle_ptr, uplo, n, alpha, success)
        ->UseRealTime();
  }
}

template <typename scalar_t>
void register_benchmark(blas_benchmark::Args& args,
                        blas::SB_Handle* sb_handle_ptr, bool* success) {
  // syr2 use the same parameters so reuse syr function
  auto syr2_params = blas_benchmark::utils::get_syr_params<scalar_t>(args);
  register_benchmark<scalar_t, blas::helper::AllocType::buffer>(
      sb_handle_ptr, success, blas_benchmark::utils::MEM_TYPE_BUFFER, syr2_params);
#ifdef SB_ENABLE_USM
  register_benchmark<scalar_t, blas::helper::AllocType::usm>(
      sb_handle_ptr, success, blas_benchmark::utils::MEM_TYPE_USM, syr2_params);
#endif
}

namespace blas_benchmark {
void create_benchmark(blas_benchmark::Args& args,
                      blas::SB_Handle* sb_handle_ptr, bool* success) {
  BLAS_REGISTER_BENCHMARK(args, sb_handle_ptr, success);
}
}  // namespace blas_benchmark
