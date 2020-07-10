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
 *  @filename gemm_batched.cpp
 *
 **************************************************************************/

#include "../utils.hpp"

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

template <typename scalar_t>
std::string get_name(std::string t1, std::string t2, int m, int k, int n,
                     int batch_size, int batch_type) {
  std::ostringstream str{};
  str << "BM_GemmBatched<" << blas_benchmark::utils::get_type_name<scalar_t>()
      << ">/" << t1 << "/" << t2 << "/" << m << "/" << k << "/" << n << "/"
      << batch_size << "/"
      << blas_benchmark::utils::batch_type_to_str(batch_type);
  return str.str();
}

template <typename scalar_t>
void run(benchmark::State& state, ExecutorType* executorPtr, int t1, int t2,
         index_t m, index_t k, index_t n, scalar_t alpha, scalar_t beta,
         index_t batch_size, int batch_type_i, bool* success) {
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

  // The counters are double. We convert m, n, k and batch_size to double to
  // avoid integer overflows for n_fl_ops and bytes_processed
  double m_d = static_cast<double>(m);
  double n_d = static_cast<double>(n);
  double k_d = static_cast<double>(k);
  double batch_size_d = static_cast<double>(batch_size);

  state.counters["m"] = m_d;
  state.counters["k"] = k_d;
  state.counters["n"] = n_d;
  state.counters["batch_size"] = batch_size_d;

  {
    double nflops_AtimesB = (2 * k_d - 1) * m_d * n_d;
    double nflops_timesAlpha = m_d * n_d;
    double nflops_addBetaC = (beta != scalar_t{0}) ? 2 * m_d * n_d : 0;
    state.counters["n_fl_ops"] =
        (nflops_AtimesB + nflops_timesAlpha + nflops_addBetaC) * batch_size_d;
  }
  {
    double mem_readA = m_d * k_d;
    double mem_readB = k_d * n_d;
    double mem_writeC = m_d * n_d;
    double mem_readC = (beta != scalar_t{0}) ? m_d * n_d : 0;
    state.counters["bytes_processed"] =
        (mem_readA + mem_readB + mem_readC + mem_writeC) * batch_size_d *
        sizeof(scalar_t);
  }

  ExecutorType& ex = *executorPtr;

  using data_t = utils::data_storage_t<scalar_t>;

  // Matrices
  std::vector<data_t> a =
      blas_benchmark::utils::random_data<data_t>(m * k * batch_size);
  std::vector<data_t> b =
      blas_benchmark::utils::random_data<data_t>(k * n * batch_size);
  std::vector<data_t> c =
      blas_benchmark::utils::const_data<data_t>(m * n * batch_size, 0);

#ifdef BLAS_VERIFY_BENCHMARK
  // Run a first time with a verification of the results
  std::vector<data_t> c_ref = c;
  auto _base = [=](index_t dim0, index_t dim1, index_t idx) {
    return dim0 * dim1 * idx;
  };
  for (int batch_idx = 0; batch_idx < batch_size; batch_idx++) {
    reference_blas::gemm(t_a, t_b, m, n, k, static_cast<data_t>(alpha),
                         a.data() + _base(m, k, batch_idx), lda,
                         b.data() + _base(k, n, batch_idx), ldb,
                         static_cast<data_t>(beta),
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

  auto a_gpu = utils::make_quantized_buffer<scalar_t>(ex, a);
  auto b_gpu = utils::make_quantized_buffer<scalar_t>(ex, b);
  auto c_gpu = utils::make_quantized_buffer<scalar_t>(ex, c);

#ifdef BLAS_VERIFY_BENCHMARK
  std::vector<data_t> c_temp = c;
  {
    auto c_temp_gpu = utils::make_quantized_buffer<scalar_t>(ex, c_temp);
    _gemm_batched(ex, *t_a, *t_b, m, n, k, alpha, a_gpu, lda, b_gpu, ldb, beta,
                  c_temp_gpu, ldc, batch_size, batch_type);
    auto event =
        utils::quantized_copy_to_host<scalar_t>(ex, c_temp_gpu, c_temp);
    ex.get_policy_handler().wait(event);
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
        _gemm_batched(ex, *t_a, *t_b, m, n, k, alpha, a_gpu, lda, b_gpu, ldb,
                      beta, c_gpu, ldc, batch_size, batch_type);
    ex.get_policy_handler().wait(event);
    return event;
  };

  // Warmup
  blas_benchmark::utils::warmup(blas_method_def);
  ex.get_policy_handler().wait();

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
};

template <typename scalar_t>
void register_benchmark(blas_benchmark::Args& args, ExecutorType* exPtr,
                        bool* success) {
  auto gemm_params =
      blas_benchmark::utils::get_gemm_batched_params<scalar_t>(args);

  for (auto p : gemm_params) {
    std::string t1s, t2s;
    index_t m, n, k, batch_size;
    scalar_t alpha, beta;
    int batch_type;
    std::tie(t1s, t2s, m, k, n, alpha, beta, batch_size, batch_type) = p;
    int t1 = static_cast<int>(blas_benchmark::utils::to_transpose_enum(t1s));
    int t2 = static_cast<int>(blas_benchmark::utils::to_transpose_enum(t2s));

    auto BM_lambda = [&](benchmark::State& st, ExecutorType* exPtr, int t1,
                         int t2, index_t m, index_t k, index_t n,
                         scalar_t alpha, scalar_t beta, index_t batch_size,
                         int batch_type, bool* success) {
      run<scalar_t>(st, exPtr, t1, t2, m, k, n, alpha, beta, batch_size,
                    batch_type, success);
    };
    benchmark::RegisterBenchmark(
        get_name<scalar_t>(t1s, t2s, m, k, n, batch_size, batch_type).c_str(),
        BM_lambda, exPtr, t1, t2, m, k, n, alpha, beta, batch_size, batch_type,
        success);
  }
}

namespace blas_benchmark {
void create_benchmark(blas_benchmark::Args& args, ExecutorType* exPtr,
                      bool* success) {
  BLAS_REGISTER_BENCHMARK(args, exPtr, success);
}
}  // namespace blas_benchmark
