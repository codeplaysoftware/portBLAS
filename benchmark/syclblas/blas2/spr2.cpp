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
 *  @filename spr2.cpp
 *
 **************************************************************************/

#include "../utils.hpp"

constexpr blas_benchmark::utils::Level2Op benchmark_op =
    blas_benchmark::utils::Level2Op::spr2;

template <typename scalar_t, blas::helper::AllocType mem_alloc>
void run(benchmark::State& state, blas::SB_Handle* sb_handle_ptr, char uplo,
         index_t n, scalar_t alpha, index_t incX, index_t incY, bool* success) {
  // initialize the state label
  blas_benchmark::utils::set_benchmark_label<scalar_t>(
      state, sb_handle_ptr->get_queue());

  blas_benchmark::utils::init_level_2_counters<
      blas_benchmark::utils::Level2Op::spr2, scalar_t>(state, "n", 0, 0, n);

  blas::SB_Handle& sb_handle = *sb_handle_ptr;
  auto q = sb_handle.get_queue();

  const index_t m_size = n * n;
  const index_t vx_size = 1 + (n - 1) * std::abs(incX);
  const index_t vy_size = 1 + (n - 1) * std::abs(incY);

  // Input matrix/vector, output vector.
  std::vector<scalar_t> m_a =
      blas_benchmark::utils::random_data<scalar_t>(m_size);
  std::vector<scalar_t> v_x =
      blas_benchmark::utils::random_data<scalar_t>(vx_size);
  std::vector<scalar_t> v_y =
      blas_benchmark::utils::random_data<scalar_t>(vy_size);

  auto m_a_gpu = blas::helper::allocate<mem_alloc, scalar_t>(m_size, q);
  auto v_x_gpu = blas::helper::allocate<mem_alloc, scalar_t>(vx_size, q);
  auto v_y_gpu = blas::helper::allocate<mem_alloc, scalar_t>(vy_size, q);

  auto copy_a =
      blas::helper::copy_to_device<scalar_t>(q, m_a.data(), m_a_gpu, m_size);
  auto copy_x =
      blas::helper::copy_to_device<scalar_t>(q, v_x.data(), v_x_gpu, vx_size);
  auto copy_y =
      blas::helper::copy_to_device<scalar_t>(q, v_y.data(), v_y_gpu, vy_size);

  sb_handle.wait({copy_a, copy_x, copy_y});

#ifdef BLAS_VERIFY_BENCHMARK
  // Run a first time with a verification of the results
  std::vector<scalar_t> x_ref = v_x;
  std::vector<scalar_t> y_ref = v_y;
  std::vector<scalar_t> m_a_ref = m_a;
  reference_blas::spr2<scalar_t>(&uplo, n, alpha, x_ref.data(), incX,
                                 y_ref.data(), incY, m_a_ref.data());

  std::vector<scalar_t> m_a_temp = m_a;
  {
    auto m_a_temp_gpu = blas::helper::allocate<mem_alloc, scalar_t>(m_size, q);
    auto copy_temp = blas::helper::copy_to_device<scalar_t>(
        q, m_a_temp.data(), m_a_temp_gpu, m_size);
    sb_handle.wait({copy_temp});

    auto spr2_event = blas::_spr2(sb_handle, uplo, n, alpha, v_x_gpu, incX,
                                  v_y_gpu, incY, m_a_temp_gpu);
    sb_handle.wait(spr2_event);
    auto copy_out = blas::helper::copy_to_host<scalar_t>(
        q, m_a_temp_gpu, m_a_temp.data(), m_size);
    sb_handle.wait({copy_out});

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
    auto event = blas::_spr2(sb_handle, uplo, n, alpha, v_x_gpu, incX, v_y_gpu,
                             incY, m_a_gpu);
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
                        std::vector<spr2_param_t<scalar_t>> params) {
  for (auto p : params) {
    index_t n, incX, incY;
    std::string uplo;
    scalar_t alpha;
    std::tie(uplo, n, alpha, incX, incY) = p;

    char uplo_c = uplo[0];

    auto BM_lambda_col =
        [&](benchmark::State& st, blas::SB_Handle* sb_handle_ptr, char uplo,
            index_t n, scalar_t alpha, index_t incX, index_t incY,
            bool* success) {
          run<scalar_t, mem_alloc>(st, sb_handle_ptr, uplo, n, alpha, incX,
                                   incY, success);
        };
    benchmark::RegisterBenchmark(
        blas_benchmark::utils::get_name<benchmark_op, scalar_t>(
            uplo, n, alpha, incX, incY, mem_type).c_str(),
        BM_lambda_col, sb_handle_ptr, uplo_c, n, alpha, incX, incY, success)
        ->UseRealTime();
  }
}

template <typename scalar_t>
void register_benchmark(blas_benchmark::Args& args,
                        blas::SB_Handle* sb_handle_ptr, bool* success) {
  auto spr2_params = blas_benchmark::utils::get_spr2_params<scalar_t>(args);

  register_benchmark<scalar_t, blas::helper::AllocType::buffer>(
      sb_handle_ptr, success, blas_benchmark::utils::MEM_TYPE_BUFFER, spr2_params);
#ifdef SB_ENABLE_USM
  register_benchmark<scalar_t, blas::helper::AllocType::usm>(
      sb_handle_ptr, success, blas_benchmark::utils::MEM_TYPE_USM, spr2_params);
#endif
}

namespace blas_benchmark {
void create_benchmark(blas_benchmark::Args& args,
                      blas::SB_Handle* sb_handle_ptr, bool* success) {
  BLAS_REGISTER_BENCHMARK(args, sb_handle_ptr, success);
}
}  // namespace blas_benchmark
