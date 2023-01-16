/**************************************************************************
 *
 *  @license
 *  Copyright (C) 2016 Codeplay Software Limited
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
 *  @filename spr.cpp
 *
 **************************************************************************/

#include "../utils.hpp"

template <typename scalar_t>
std::string get_name(std::string layout, char uplo, int size) {
  std::ostringstream str{};
  str << "BM_Spr<" << blas_benchmark::utils::get_type_name<scalar_t>() << ">/"
      << layout << "/" << uplo << "/" << size;
  return str.str();
}

template <typename scalar_t, typename layout>
void run(benchmark::State& state, blas::SB_Handle* sb_handle_ptr, char uplo,
         int size, scalar_t alpha, int lda, bool* success) {
  using index_t = int32_t;
  index_t incX = 1;
  // The counters are double. We convert size to double to avoid
  // integer overflows for n_fl_ops and bytes_processed
  double size_d = static_cast<double>(size);

  state.counters["size_d"] = size_d;
  state.counters["alpha"] = alpha;

  constexpr bool isColMajor = std::is_same<layout, blas::col_major>::value;

  {
    double nflops_AtimesX = 2.0 * size_d;
    double nflops_timesAlpha = size_d;
    state.counters["n_fl_ops"] = nflops_AtimesX + nflops_timesAlpha;
  }
  {
    double mem_readA = size_d * size_d;
    double mem_readX = size_d;
    double mem_writeY = size_d;
    state.counters["bytes_processed"] =
        (mem_readA + mem_readX + mem_writeY) * sizeof(scalar_t);
  }

  blas::SB_Handle& sb_handle = *sb_handle_ptr;

  // Input matrix/vector, output vector.
  std::vector<scalar_t> m_a =
      blas_benchmark::utils::random_data<scalar_t>(size * (size + 1) / 2);
  std::vector<scalar_t> v_x =
      blas_benchmark::utils::random_data<scalar_t>(size);

  auto m_a_gpu =
      blas::make_sycl_iterator_buffer<scalar_t>(m_a, size * (size + 1) / 2);
  auto v_x_gpu = blas::make_sycl_iterator_buffer<scalar_t>(v_x, size);

#ifdef BLAS_VERIFY_BENCHMARK
  // Run a first time with a verification of the results
  std::vector<scalar_t> x_ref = v_x;
  std::vector<scalar_t> m_a_ref = m_a;
  reference_blas::spr<scalar_t, isColMajor>(&uplo, size, alpha, x_ref.data(),
                                            incX, m_a_ref.data(), lda);

  std::vector<scalar_t> m_a_temp = m_a;
  {
    auto m_a_temp_gpu = blas::make_sycl_iterator_buffer<scalar_t>(
        m_a_temp, size * (size + 1) / 2);

    blas::_spr<blas::SB_Handle, index_t, scalar_t, decltype(v_x_gpu), index_t,
               decltype(m_a_gpu), layout>(sb_handle, uplo, size, alpha, v_x_gpu,
                                          incX, m_a_temp_gpu, lda);
    sb_handle.wait();
  }

  std::ostringstream err_stream;
  if (!utils::compare_vectors(m_a_temp, m_a_ref, err_stream, "")) {
    const std::string& err_str = err_stream.str();
    state.SkipWithError(err_str.c_str());
    *success = false;
  };
#endif

  auto blas_method_def = [&]() -> std::vector<cl::sycl::event> {
    auto event =
        blas::_spr<blas::SB_Handle, index_t, scalar_t, decltype(v_x_gpu),
                   index_t, decltype(m_a_gpu), layout>(
            sb_handle, uplo, size, alpha, v_x_gpu, incX, m_a_gpu, lda);
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
}

template <typename scalar_t>
void register_benchmark(blas_benchmark::Args& args,
                        blas::SB_Handle* sb_handle_ptr, bool* success) {
  auto gemm_params = blas_benchmark::utils::get_spr_params<scalar_t>(args);

  for (auto p : gemm_params) {
    int n, lda;
    std::string uplo;
    scalar_t alpha;
    std::tie(uplo, n, alpha) = p;
    lda = n;

    char uplo_c = uplo[0];

    auto BM_lambda_row = [&](benchmark::State& st,
                             blas::SB_Handle* sb_handle_ptr, char uplo,
                             int size, scalar_t alpha, int lda, bool* success) {
      run<scalar_t, blas::row_major>(st, sb_handle_ptr, uplo, size, alpha, lda,
                                     success);
    };
    benchmark::RegisterBenchmark(get_name<scalar_t>("row", uplo_c, n).c_str(),
                                 BM_lambda_row, sb_handle_ptr, uplo_c, n, alpha,
                                 lda, success);

    auto BM_lambda_col = [&](benchmark::State& st,
                             blas::SB_Handle* sb_handle_ptr, char uplo,
                             int size, scalar_t alpha, int lda, bool* success) {
      run<scalar_t, blas::col_major>(st, sb_handle_ptr, uplo, size, alpha, lda,
                                     success);
    };
    benchmark::RegisterBenchmark(get_name<scalar_t>("col", uplo_c, n).c_str(),
                                 BM_lambda_col, sb_handle_ptr, uplo_c, n, alpha,
                                 lda, success);
  }
}

namespace blas_benchmark {
void create_benchmark(blas_benchmark::Args& args,
                      blas::SB_Handle* sb_handle_ptr, bool* success) {
  BLAS_REGISTER_BENCHMARK(args, sb_handle_ptr, success);
}
}  // namespace blas_benchmark