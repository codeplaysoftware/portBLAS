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

#ifdef ACL_BACKEND_OPENCL
#ifndef ARM_COMPUTE_CL
#define ARM_COMPUTE_CL
#endif /*ACL_BACKEND_OPENCL */
#endif /* ARM_COMPUTE_CL */

#include "../utils.hpp"

std::string get_name(std::string t1, std::string t2, int m, int k, int n) {
  std::ostringstream str{};
  str << "BM_Gemm<float>/" << t1 << "/" << t2 << "/" << m << "/" << k << "/"
      << n;
  return str.str();
}

void run(benchmark::State& state, int t1, int t2, index_t m, index_t k,
         index_t n, float alpha, float beta, bool* success) {
  // Standard test setup.
  std::string t1s = blas_benchmark::utils::from_transpose_enum(
      static_cast<blas_benchmark::utils::Transposition>(t1));
  std::string t2s = blas_benchmark::utils::from_transpose_enum(
      static_cast<blas_benchmark::utils::Transposition>(t2));
  if (t1s != "n" || t2s != "n") {
    state.SkipWithError("Transposed matrices not supported in ACL benchmarks");
    return;
  }
  const char* t_a = "n";
  const char* t_b = "n";

  index_t lda = m;
  index_t ldb = k;
  index_t ldc = m;

  // The counters are double. We convert m, n and k to double to avoid
  // integer overflows and write them in the counters
  double m_d = static_cast<double>(m);
  double n_d = static_cast<double>(n);
  double k_d = static_cast<double>(k);

  state.counters["m"] = m_d;
  state.counters["k"] = k_d;
  state.counters["n"] = n_d;

  {
    double nflops_AtimesB = (2 * k_d - 1) * m_d * n_d;
    double nflops_timesAlpha = m_d * n_d;
    double nflops_addBetaC = (beta != 0) ? 2 * m_d * n_d : 0;
    state.counters["n_fl_ops"] =
        nflops_AtimesB + nflops_timesAlpha + nflops_addBetaC;
  }
  {
    double mem_readA = m_d * k_d;
    double mem_readB = k_d * n_d;
    double mem_writeC = m_d * n_d;
    double mem_readC = (beta != 0) ? m_d * n_d : 0;
    state.counters["bytes_processed"] =
        (mem_readA + mem_readB + mem_readC + mem_writeC) * sizeof(float);
  }

  // Matrices
  std::vector<float> a = blas_benchmark::utils::random_data<float>(m * k);
  std::vector<float> b = blas_benchmark::utils::random_data<float>(k * n);
  std::vector<float> c = blas_benchmark::utils::const_data<float>(m * n, 0);

  // Device matrices
  const arm_compute::TensorShape shape_a(k, m), shape_b(n, k), shape_c(n, m);
#ifdef ACL_BACKEND_NEON
  arm_compute::Tensor arm_a, arm_b, arm_c;
#else
  arm_compute::CLScheduler::get().default_init();
  arm_compute::CLTensor arm_a, arm_b, arm_c;
#endif
  arm_a.allocator()->init(
      arm_compute::TensorInfo(shape_a, 1, arm_compute::DataType::F32));
  arm_b.allocator()->init(
      arm_compute::TensorInfo(shape_b, 1, arm_compute::DataType::F32));
  arm_c.allocator()->init(
      arm_compute::TensorInfo(shape_c, 1, arm_compute::DataType::F32));
  arm_a.allocator()->allocate();
  arm_b.allocator()->allocate();
  arm_c.allocator()->allocate();
  blas_benchmark::utils::fill_tensor(arm_a, a);
  blas_benchmark::utils::fill_tensor(arm_b, b);
  blas_benchmark::utils::fill_tensor(arm_c, c);

  // Configure the BLAS routine
#ifdef ACL_BACKEND_NEON
  arm_compute::NEGEMM arm_gemm;
#else
  arm_compute::CLGEMM arm_gemm;
#endif
  arm_gemm.configure(&arm_a, &arm_b, &arm_c, &arm_c, alpha, beta);

  // Create a utility lambda describing the blas method that we want to run.
  auto blas_method_def = [&]() -> std::vector<void*> {
    arm_gemm.run();
#ifdef ACL_BACKEND_OPENCL
    arm_compute::CLScheduler::get().sync();
#endif
    return {nullptr};
  };

#ifdef BLAS_VERIFY_BENCHMARK
  // Run a first time with a verification of the results
  std::vector<float> c_ref = c;
  reference_blas::gemm(t_a, t_b, m, n, k, alpha, a.data(), lda, b.data(), ldb,
                       beta, c_ref.data(), ldc);
  blas_method_def();
  std::vector<float> c_temp(m * n);
  blas_benchmark::utils::extract_tensor(arm_c, c_temp);

  std::ostringstream err_stream;
  if (!utils::compare_vectors<float>(c_temp, c_ref, err_stream, "")) {
    const std::string& err_str = err_stream.str();
    state.SkipWithError(err_str.c_str());
    *success = false;
  };
#endif

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

  arm_a.allocator()->free();
  arm_b.allocator()->free();
  arm_c.allocator()->free();
};

void register_benchmark(blas_benchmark::Args& args, bool* success) {
  auto gemm_params = blas_benchmark::utils::get_blas3_params<float>(args);

  for (auto p : gemm_params) {
    std::string t1s, t2s;
    index_t m, n, k;
    float alpha, beta;
    std::tie(t1s, t2s, m, k, n, alpha, beta) = p;
    int t1 = static_cast<int>(blas_benchmark::utils::to_transpose_enum(t1s));
    int t2 = static_cast<int>(blas_benchmark::utils::to_transpose_enum(t2s));

    auto BM_lambda = [&](benchmark::State& st, int t1, int t2, index_t m,
                         index_t k, index_t n, float alpha, float beta,
                         bool* success) {
      run(st, t1, t2, m, k, n, alpha, beta, success);
    };
    benchmark::RegisterBenchmark(get_name(t1s, t2s, m, k, n).c_str(), BM_lambda,
                                 t1, t2, m, k, n, alpha, beta, success);
  }
}

namespace blas_benchmark {
void create_benchmark(blas_benchmark::Args& args, bool* success) {
  register_benchmark(args, success);
}
}  // namespace blas_benchmark
