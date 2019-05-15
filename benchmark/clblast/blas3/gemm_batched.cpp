/***************************************************************************
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
 *  @filename gemm.cpp
 *
 **************************************************************************/

#include "utils.hpp"

template <typename scalar_t>
std::string get_name(std::string t1, std::string t2, int m, int k, int n, int batch_size) {
  std::ostringstream str{};
  str << "BM_Gemm<" << blas_benchmark::utils::get_type_name<scalar_t>() << ">/"
      << t1 << "/" << t2 << "/" << m << "/" << k << "/" << n << "/" << batch_size;
  return str.str();
}

template <typename scalar_t>
void run(benchmark::State& state, ExecutorType* executorPtr, int t1, int t2,
         index_t m, index_t k, index_t n, scalar_t alpha, scalar_t beta, index_t batch_size) {
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

  // The counters are double. We convert m, n and k to double to avoid
  // integer overflows and write them in the counters
  double m_d = static_cast<double>(m);
  double n_d = static_cast<double>(n);
  double k_d = static_cast<double>(k);
  double batch_size_d = static_cast<double>(batch_size);

  state.counters["m"] = m_d;
  state.counters["k"] = k_d;
  state.counters["n"] = n_d;
  state.counters["batch_size"] = batch_size_d;

  state.counters["n_fl_ops"] = (2 * (m_d * n_d * k_d) + 3 * (m_d * n_d)) * batch_size_d;
  state.counters["bytes_processed"] =
      (m_d * k_d + k_d * n_d + 2 * m_d * n_d) * batch_size_d * sizeof(scalar_t);
  if (beta == 0.0) {
    // not adding beta * C
    state.counters["n_fl_ops"] -= 2 * m_d * n_d * batch_size_d;
    // not reading C
    state.counters["bytes_processed"] -= m_d * n_d * batch_size_d * sizeof(scalar_t);
  }

  // Matrices
  std::vector<scalar_t> a = blas_benchmark::utils::random_data<scalar_t>(m * k * batch_size);
  std::vector<scalar_t> b = blas_benchmark::utils::random_data<scalar_t>(k * n * batch_size);
  std::vector<scalar_t> c =
      blas_benchmark::utils::const_data<scalar_t>(m * n * batch_size, 0);

  // Specify the transpositions
  clblast::Transpose a_tr = blas_benchmark::utils::translate_transposition(t_a);
  clblast::Transpose b_tr = blas_benchmark::utils::translate_transposition(t_b);

  // Specify the layout. As with GEMV, this needs to be kColMajor, and results
  // in errors otherwise. It may be that this is incorrect (especially for
  // performance reasons), so may need to be revisited.
  auto layout = clblast::Layout::kColMajor;

  // Device matrices
  MemBuffer<scalar_t> a_gpu(executorPtr, a.data(), static_cast<size_t>(m * k * batch_size));
  MemBuffer<scalar_t> b_gpu(executorPtr, b.data(), static_cast<size_t>(k * n * batch_size));
  MemBuffer<scalar_t> c_gpu(executorPtr, c.data(), static_cast<size_t>(m * n * batch_size));

  // Alphas and betas
  std::vector<scalar_t> alphas;
  std::vector<scalar_t> betas;
  alphas.resize(batch_size, alpha);
  betas.resize(batch_size, beta);

  // Offsets
  std::vector<size_t> a_offsets;
  std::vector<size_t> b_offsets;
  std::vector<size_t> c_offsets;
  a_offsets.resize(batch_size);
  b_offsets.resize(batch_size);
  c_offsets.resize(batch_size);
  auto _base = [=](index_t dim0, index_t dim1, index_t idx) {
    return dim0 * dim1 * idx;
  };
  for (int batch_idx = 0; batch_idx < batch_size; batch_idx++) {
    a_offsets[batch_idx] = _base(m, k, batch_idx);
    b_offsets[batch_idx] = _base(k, n, batch_idx);
    c_offsets[batch_idx] = _base(m, n, batch_idx);
  }

#ifdef BLAS_VERIFY_BENCHMARK
  // Run a first time with a verification of the results
  std::vector<scalar_t> c_ref = c;
  for (int batch_idx = 0; batch_idx < batch_size; batch_idx++) {
    reference_blas::gemm(t_a, t_b, m, n, k, alpha,
                         a.data() + a_offsets[batch_idx], lda,
                         b.data() + b_offsets[batch_idx], ldb, beta,
                         c_ref.data() + c_offsets[batch_idx], ldc);
  }
  std::vector<scalar_t> c_temp = c;
  {
    MemBuffer<scalar_t> c_temp_gpu(executorPtr, c_temp.data(), static_cast<size_t>(m * n * batch_size));
    cl_event event;
    clblast::GemmBatched<scalar_t>(layout, a_tr, b_tr, m, n, k, alphas.data(), a_gpu.dev(), a_offsets.data(),
                                   lda, b_gpu.dev(), b_offsets.data(), ldb, betas.data(), c_temp_gpu.dev(), c_offsets.data(), ldc,
                                   batch_size,
                                   executorPtr->_queue(), &event);
    CLEventHandler::wait(event);
  }

  if (!utils::compare_vectors<scalar_t>(c_temp, c_ref)) {
    exit(1);
  };
#endif

  // Create a utility lambda describing the blas method that we want to run.
  auto blas_method_def = [&]() -> std::vector<cl_event> {
    cl_event event;
    clblast::GemmBatched<scalar_t>(layout, a_tr, b_tr, m, n, k, alphas.data(), a_gpu.dev(), a_offsets.data(),
                                   lda, b_gpu.dev(), b_offsets.data(), ldb, betas.data(), c_gpu.dev(), c_offsets.data(), ldc,
                                   batch_size,
                                   executorPtr->_queue(), &event);
    CLEventHandler::wait(event);
    return {event};
  };

  // Warm up to avoid benchmarking data transfer
  blas_benchmark::utils::warmup(blas_method_def);

  blas_benchmark::utils::init_counters(state);

  // Measure
  for (auto _ : state) {
    std::tuple<double, double> times =
        blas_benchmark::utils::timef(blas_method_def);

    // Report
    blas_benchmark::utils::update_counters(state, times);
  }

  blas_benchmark::utils::calc_avg_counters(state);
};

template <typename scalar_t>
void register_benchmark(blas_benchmark::Args& args, ExecutorType* exPtr) {
  auto gemm_params =
      blas_benchmark::utils::get_gemm_batched_params<scalar_t>(args);

  for (auto p : gemm_params) {
    std::string t1s, t2s;
    index_t m, n, k, batch_size;
    scalar_t alpha, beta;
    std::tie(t1s, t2s, m, k, n, alpha, beta, batch_size) = p;
    int t1 = static_cast<int>(blas_benchmark::utils::to_transpose_enum(t1s));
    int t2 = static_cast<int>(blas_benchmark::utils::to_transpose_enum(t2s));

    auto BM_lambda = [&](benchmark::State& st, ExecutorType* exPtr, int t1,
                         int t2, index_t m, index_t k, index_t n,
                         scalar_t alpha, scalar_t beta, index_t batch_size) {
      run<scalar_t>(st, exPtr, t1, t2, m, k, n, alpha, beta, batch_size);
    };
    benchmark::RegisterBenchmark(
        get_name<scalar_t>(t1s, t2s, m, k, n, batch_size).c_str(), BM_lambda,
        exPtr, t1, t2, m, k, n, alpha, beta, batch_size);
  }
}

namespace blas_benchmark {
void create_benchmark(blas_benchmark::Args& args, ExecutorType* exPtr) {
  register_benchmark<float>(args, exPtr);
#ifdef DOUBLE_SUPPORT
  register_benchmark<double>(args, exPtr);
#endif
}
}  // namespace blas_benchmark
