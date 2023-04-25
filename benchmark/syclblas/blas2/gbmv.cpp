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
std::string get_name(std::string t, int m, int n, int kl, int ku,
                     std::string mem_type) {
  std::ostringstream str{};
  str << "BM_Gbmv<" << blas_benchmark::utils::get_type_name<scalar_t>() << ">/"
      << t << "/" << m << "/" << n << "/" << kl << "/" << ku;
  str << "/" << mem_type;
  return str.str();
}

template <typename scalar_t, blas::helper::AllocType mem_alloc>
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

  // The counters are double. We convert m and n to double to avoid
  // integer overflows for n_fl_ops and bytes_processed
  double m_d = static_cast<double>(m);
  double n_d = static_cast<double>(n);
  double kl_d = static_cast<double>(kl);
  double ku_d = static_cast<double>(ku);

  state.counters["m"] = m_d;
  state.counters["n"] = n_d;
  state.counters["kl"] = kl_d;
  state.counters["ku"] = ku_d;

  // Compute the number of A non-zero elements.
  const double A_validVal =
      (t_str[0] == 'n' ? n_d : m_d) * (kl_d + ku_d + 1.0) -
      0.5 * (kl_d * (kl_d + 1.0)) - 0.5 * (ku_d * (ku_d + 1.0));

  {
    double nflops_AtimesX = 2.0 * A_validVal;
    double nflops_timesAlpha = ylen;
    double nflops_addBetaY = (beta != scalar_t{0}) ? 2 * ylen : 0;
    state.counters["n_fl_ops"] =
        nflops_AtimesX + nflops_timesAlpha + nflops_addBetaY;
  }
  {
    double mem_readA = A_validVal;
    double mem_readX = xlen;
    double mem_writeY = ylen;
    double mem_readY = (beta != scalar_t{0}) ? ylen : 0;
    state.counters["bytes_processed"] =
        (mem_readA + mem_readX + mem_writeY + mem_readY) * sizeof(scalar_t);
  }

  blas::SB_Handle& sb_handle = *sb_handle_ptr;
  auto q = sb_handle.get_queue();

  // Input matrix/vector, output vector.
  std::vector<scalar_t> m_a =
      blas_benchmark::utils::random_data<scalar_t>(lda * n);
  std::vector<scalar_t> v_x =
      blas_benchmark::utils::random_data<scalar_t>(xlen);
  std::vector<scalar_t> v_y =
      blas_benchmark::utils::random_data<scalar_t>(ylen);

  auto m_a_gpu = blas::helper::allocate<mem_alloc, scalar_t>(lda * n, q);
  auto v_x_gpu = blas::helper::allocate<mem_alloc, scalar_t>(xlen, q);
  auto v_y_gpu = blas::helper::allocate<mem_alloc, scalar_t>(ylen, q);

  auto copy_a =
      blas::helper::copy_to_device<scalar_t>(q, m_a.data(), m_a_gpu, lda * n);
  auto copy_x =
      blas::helper::copy_to_device<scalar_t>(q, v_x.data(), v_x_gpu, xlen);
  auto copy_y =
      blas::helper::copy_to_device<scalar_t>(q, v_y.data(), v_y_gpu, ylen);

  sb_handle.wait({copy_a, copy_x, copy_y});
#ifdef BLAS_VERIFY_BENCHMARK
  // Run a first time with a verification of the results
  std::vector<scalar_t> v_y_ref = v_y;
  reference_blas::gbmv(t_str, m, n, kl, ku, alpha, m_a.data(), lda, v_x.data(),
                       incX, beta, v_y_ref.data(), incY);
  std::vector<scalar_t> v_y_temp = v_y;
  {
    auto v_y_temp_gpu = blas::helper::allocate<mem_alloc, scalar_t>(ylen, q);
    auto copy_temp = blas::helper::copy_to_device<scalar_t>(q, v_y_temp.data(),
                                                            v_y_temp_gpu, ylen);
    sb_handle.wait({copy_temp});
    auto gbmv_event = _gbmv(sb_handle, *t_str, m, n, kl, ku, alpha, m_a_gpu,
                            lda, v_x_gpu, incX, beta, v_y_temp_gpu, incY);
    sb_handle.wait({gbmv_event});
    auto copy_out = blas::helper::copy_to_host<scalar_t>(q, v_y_temp_gpu,
                                                         v_y_temp.data(), ylen);
    sb_handle.wait({copy_out});

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

  blas_benchmark::utils::calc_avg_counters(state);

  blas::helper::deallocate<mem_alloc>(m_a_gpu, q);
  blas::helper::deallocate<mem_alloc>(v_x_gpu, q);
  blas::helper::deallocate<mem_alloc>(v_y_gpu, q);
}

template <typename scalar_t, blas::helper::AllocType mem_alloc>
void register_benchmark(blas::SB_Handle* sb_handle_ptr, bool* success,
                        std::string mem_type,
                        std::vector<gbmv_param_t<scalar_t>> params) {
  for (auto p : params) {
    std::string ts;
    index_t m, n, kl, ku;
    scalar_t alpha, beta;
    std::tie(ts, m, n, kl, ku, alpha, beta) = p;
    int t = static_cast<int>(blas_benchmark::utils::to_transpose_enum(ts));

    auto BM_lambda = [&](benchmark::State& st, blas::SB_Handle* sb_handle_ptr,
                         int t, index_t m, index_t n, index_t kl, index_t ku,
                         scalar_t alpha, scalar_t beta, bool* success) {
      run<scalar_t, mem_alloc>(st, sb_handle_ptr, t, m, n, kl, ku, alpha, beta,
                               success);
    };
    benchmark::RegisterBenchmark(
        get_name<scalar_t>(ts, m, n, kl, ku, mem_type).c_str(), BM_lambda,
        sb_handle_ptr, t, m, n, kl, ku, alpha, beta, success);
  }
}

template <typename scalar_t>
void register_benchmark(blas_benchmark::Args& args,
                        blas::SB_Handle* sb_handle_ptr, bool* success) {
  auto gbmv_params = blas_benchmark::utils::get_gbmv_params<scalar_t>(args);

  register_benchmark<scalar_t, blas::helper::AllocType::buffer>(
      sb_handle_ptr, success, "buffer", gbmv_params);
#ifdef SB_ENABLE_USM
  register_benchmark<scalar_t, blas::helper::AllocType::usm>(
      sb_handle_ptr, success, "usm", gbmv_params);
#endif
}

namespace blas_benchmark {
void create_benchmark(blas_benchmark::Args& args,
                      blas::SB_Handle* sb_handle_ptr, bool* success) {
  BLAS_REGISTER_BENCHMARK(args, sb_handle_ptr, success);
}
}  // namespace blas_benchmark
