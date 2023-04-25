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
 *  @filename symv.cpp
 *
 **************************************************************************/

#include "../utils.hpp"

template <typename scalar_t>
std::string get_name(std::string uplo, int n, scalar_t alpha, scalar_t beta,
                     std::string mem_type) {
  std::ostringstream str{};
  str << "BM_Symv<" << blas_benchmark::utils::get_type_name<scalar_t>() << ">/"
      << uplo << "/" << n << "/" << alpha << "/" << beta;
  str << "/" << mem_type;
  return str.str();
}

template <typename scalar_t, blas::helper::AllocType mem_alloc>
void run(benchmark::State& state, blas::SB_Handle* sb_handle_ptr,
         std::string uplo, index_t n, scalar_t alpha, scalar_t beta,
         bool* success) {
  // Standard test setup.
  const char* uplo_str = uplo.c_str();

  index_t incX = 1;
  index_t incY = 1;

  index_t lda = n;
  index_t xlen = n;
  index_t ylen = n;

  blas_benchmark::utils::init_level_2_counters<
      blas_benchmark::utils::Level2Op::symv, scalar_t>(state, "n", beta, 0, n);

  blas::SB_Handle& sb_handle = *sb_handle_ptr;
  auto q = sb_handle.get_queue();

  // Input matrix/vector, output vector.
  std::vector<scalar_t> m_a =
      blas_benchmark::utils::random_data<scalar_t>(lda * n);
  std::vector<scalar_t> v_x =
      blas_benchmark::utils::random_data<scalar_t>(xlen);
  std::vector<scalar_t> v_y =
      blas_benchmark::utils::random_data<scalar_t>(ylen);

  auto m_a_gpu = blas::helper::allocate<mem_alloc, scalar_t>(n * n, q);
  auto v_x_gpu = blas::helper::allocate<mem_alloc, scalar_t>(xlen, q);
  auto v_y_gpu = blas::helper::allocate<mem_alloc, scalar_t>(ylen, q);

  auto copy_m =
      blas::helper::copy_to_device<scalar_t>(q, m_a.data(), m_a_gpu, n * n);
  auto copy_x =
      blas::helper::copy_to_device<scalar_t>(q, v_x.data(), v_x_gpu, xlen);
  auto copy_y =
      blas::helper::copy_to_device<scalar_t>(q, v_y.data(), v_y_gpu, ylen);

  sb_handle.wait({copy_m, copy_x, copy_y});

#ifdef BLAS_VERIFY_BENCHMARK
  // Run a first time with a verification of the results
  std::vector<scalar_t> v_y_ref = v_y;
  reference_blas::symv(uplo_str, n, alpha, m_a.data(), lda, v_x.data(), incX,
                       beta, v_y_ref.data(), incY);
  std::vector<scalar_t> v_y_temp(v_y);
  {
    auto v_y_temp_gpu = blas::helper::allocate<mem_alloc, scalar_t>(ylen, q);
    auto copy_temp = blas::helper::copy_to_device<scalar_t>(q, v_y_temp.data(),
                                                            v_y_temp_gpu, ylen);
    sb_handle.wait(copy_temp);
    auto symv_event = _symv(sb_handle, *uplo_str, n, alpha, m_a_gpu, lda,
                            v_x_gpu, incX, beta, v_y_temp_gpu, incY);
    sb_handle.wait(symv_event);
    auto copy_out = blas::helper::copy_to_host<scalar_t>(q, v_y_temp_gpu,
                                                         v_y_temp.data(), ylen);
    sb_handle.wait(copy_out);

    blas::helper::deallocate<mem_alloc>(v_y_temp_gpu, q);
  }

  std::ostringstream err_stream;
  if (!utils::compare_vectors(v_y_temp, v_y_ref, err_stream, "")) {
    const std::string& err_str = err_stream.str();
    state.SkipWithError(err_str.c_str());
    *success = false;
  };
#endif

  auto blas_method_def = [&]() -> std::vector<cl::sycl::event> {
    auto event = _symv(sb_handle, *uplo_str, n, alpha, m_a_gpu, lda, v_x_gpu,
                       incX, beta, v_y_gpu, incY);
    sb_handle.wait();
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
                        std::vector<symv_param_t<scalar_t>> params) {
  for (auto p : params) {
    std::string uplos;
    index_t n;
    scalar_t alpha, beta;
    std::tie(uplos, n, alpha, beta) = p;

    auto BM_lambda = [&](benchmark::State& st, blas::SB_Handle* sb_handle_ptr,
                         std::string uplos, index_t n, scalar_t alpha,
                         scalar_t beta, bool* success) {
      run<scalar_t, mem_alloc>(st, sb_handle_ptr, uplos, n, alpha, beta,
                               success);
    };
    benchmark::RegisterBenchmark(
        get_name<scalar_t>(uplos, n, alpha, beta, mem_type).c_str(), BM_lambda,
        sb_handle_ptr, uplos, n, alpha, beta, success)
        ->UseRealTime();
  }
}

template <typename scalar_t>
void register_benchmark(blas_benchmark::Args& args,
                        blas::SB_Handle* sb_handle_ptr, bool* success) {
  auto symv_params = blas_benchmark::utils::get_symv_params<scalar_t>(args);
  register_benchmark<scalar_t, blas::helper::AllocType::buffer>(
      sb_handle_ptr, success, "buffer", symv_params);
#ifdef SB_ENABLE_USM
  register_benchmark<scalar_t, blas::helper::AllocType::usm>(
      sb_handle_ptr, success, "usm", symv_params);
#endif
}

namespace blas_benchmark {
void create_benchmark(blas_benchmark::Args& args,
                      blas::SB_Handle* sb_handle_ptr, bool* success) {
  BLAS_REGISTER_BENCHMARK(args, sb_handle_ptr, success);
}
}  // namespace blas_benchmark
