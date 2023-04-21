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
 *  @filename gbmv.cpp
 *
 **************************************************************************/

#include "../utils.hpp"

template <typename scalar_t>
std::string get_name(std::string t, int m, int n, int kl, int ku) {
  std::ostringstream str{};
  str << "BM_Gbmv<" << blas_benchmark::utils::get_type_name<scalar_t>() << ">/"
      << t << "/" << m << "/" << n << "/" << kl << "/" << ku;
  return str.str();
}

template <typename scalar_t>
void run(benchmark::State& state, blas::SB_Handle* sb_handle_ptr, int ti,
         index_t m, index_t n, index_t kl, index_t ku, scalar_t alpha,
         scalar_t beta, bool* success) {
  // Standard test setup.
  std::string ts = blas_benchmark::utils::from_transpose_enum(
      static_cast<blas_benchmark::utils::Transposition>(ti));
  const char* t_str = ts.c_str();

  index_t xlen = t_str[0] == 'n' ? n : m;
  index_t ylen = t_str[0] == 'n' ? m : n;

  index_t lda = (kl + ku + 1);
  index_t incX = 1;
  index_t incY = 1;

  blas_benchmark::utils::init_level_2_counters<
      blas_benchmark::utils::Level2Op::gbmv, scalar_t>(state, t_str, beta, m, n,
                                                       0, ku, kl);

  blas::SB_Handle& sb_handle = *sb_handle_ptr;

  // Input matrix/vector, output vector.
  std::vector<scalar_t> m_a =
      blas_benchmark::utils::random_data<scalar_t>(lda * n);
  std::vector<scalar_t> v_x =
      blas_benchmark::utils::random_data<scalar_t>(xlen);
  std::vector<scalar_t> v_y =
      blas_benchmark::utils::random_data<scalar_t>(ylen);

  auto m_a_gpu = blas::make_sycl_iterator_buffer<scalar_t>(m_a, lda * n);
  auto v_x_gpu = blas::make_sycl_iterator_buffer<scalar_t>(v_x, xlen);
  auto v_y_gpu = blas::make_sycl_iterator_buffer<scalar_t>(v_y, ylen);

#ifdef BLAS_VERIFY_BENCHMARK
  // Run a first time with a verification of the results
  std::vector<scalar_t> v_y_ref = v_y;
  reference_blas::gbmv(t_str, m, n, kl, ku, alpha, m_a.data(), lda, v_x.data(),
                       incX, beta, v_y_ref.data(), incY);
  std::vector<scalar_t> v_y_temp = v_y;
  {
    auto v_y_temp_gpu =
        blas::make_sycl_iterator_buffer<scalar_t>(v_y_temp, ylen);
    auto event = _gbmv(sb_handle, *t_str, m, n, kl, ku, alpha, m_a_gpu, lda,
                       v_x_gpu, incX, beta, v_y_temp_gpu, incY);
    sb_handle.wait();
  }

  std::ostringstream err_stream;
  if (!utils::compare_vectors(v_y_temp, v_y_ref, err_stream, "")) {
    const std::string& err_str = err_stream.str();
    state.SkipWithError(err_str.c_str());
    *success = false;
  };
#endif

  auto blas_method_def = [&]() -> std::vector<cl::sycl::event> {
    auto event = _gbmv(sb_handle, *t_str, m, n, kl, ku, alpha, m_a_gpu, lda,
                       v_x_gpu, incX, beta, v_y_gpu, incY);
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
  auto gbmv_params = blas_benchmark::utils::get_gbmv_params<scalar_t>(args);

  for (auto p : gbmv_params) {
    std::string ts;
    index_t m, n, kl, ku;
    scalar_t alpha, beta;
    std::tie(ts, m, n, kl, ku, alpha, beta) = p;
    int t = static_cast<int>(blas_benchmark::utils::to_transpose_enum(ts));

    auto BM_lambda = [&](benchmark::State& st, blas::SB_Handle* sb_handle_ptr,
                         int t, index_t m, index_t n, index_t kl, index_t ku,
                         scalar_t alpha, scalar_t beta, bool* success) {
      run<scalar_t>(st, sb_handle_ptr, t, m, n, kl, ku, alpha, beta, success);
    };
    benchmark::RegisterBenchmark(get_name<scalar_t>(ts, m, n, kl, ku).c_str(),
                                 BM_lambda, sb_handle_ptr, t, m, n, kl, ku,
                                 alpha, beta, success)
        ->UseRealTime();
  }
}

namespace blas_benchmark {
void create_benchmark(blas_benchmark::Args& args,
                      blas::SB_Handle* sb_handle_ptr, bool* success) {
  BLAS_REGISTER_BENCHMARK(args, sb_handle_ptr, success);
}
}  // namespace blas_benchmark
