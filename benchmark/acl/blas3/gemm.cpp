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

#include "utils.hpp"

std::string get_name(std::string t1, std::string t2, int m, int k, int n) {
  std::ostringstream str{};
  str << "BM_Gemm<float>/" << t1 << "/" << t2 << "/" << m << "/" << k << "/"
      << n;
  return str.str();
}

void run(benchmark::State& state, int t1, int t2,
         index_t m, index_t k, index_t n, float alpha, float beta) {
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

  state.counters["m"] = m;
  state.counters["k"] = k;
  state.counters["n"] = n;

  // The counters are double. We convert m, n and k to double to avoid
  // integer overflows and write them in the counters
  double m_d = static_cast<double>(m);
  double n_d = static_cast<double>(n);
  double k_d = static_cast<double>(k);

  state.counters["n_fl_ops"] = 2 * (m_d * n_d * k_d) + 3 * (m_d * n_d);
  state.counters["bytes_processed"] =
      (m_d * k_d + k_d * n_d + 2 * m_d * n_d) * sizeof(float);
  if (beta == 0.0) {
    // not adding beta * C
    state.counters["n_fl_ops"] -= 2 * m_d * n_d;
    // not reading C
    state.counters["bytes_processed"] -= m_d * n_d * sizeof(float);
  }

  // Matrices
  std::vector<float> a = blas_benchmark::utils::random_data<float>(m * k);
  std::vector<float> b = blas_benchmark::utils::random_data<float>(k * n);
  std::vector<float> c =
      blas_benchmark::utils::const_data<float>(m * n, 0);

  // Device matrices
  const arm_compute::TensorShape shape_a(k, m), shape_b(n, k), shape_c(n, m);
#ifdef ACL_BACKEND_NEON
  arm_compute::Tensor arm_a, arm_b, arm_c;
#else
  arm_compute::CLScheduler::get().default_init();
  arm_compute::CLTensor arm_a, arm_b, arm_c;
#endif
  arm_a.allocator()->init(arm_compute::TensorInfo(shape_a, 1, arm_compute::DataType::F32));
  arm_b.allocator()->init(arm_compute::TensorInfo(shape_b, 1, arm_compute::DataType::F32));
  arm_c.allocator()->init(arm_compute::TensorInfo(shape_c, 1, arm_compute::DataType::F32));
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
  arm_gemm.configure(&arm_a, &arm_b, nullptr, &arm_c, alpha, beta);

  // Create a utility lambda describing the blas method that we want to run.
  auto blas_method_def = [&]() -> std::vector<void*> {
    arm_gemm.run();
#ifdef ACL_BACKEND_OPENCL
    arm_compute::CLScheduler::get().sync();
#endif
    return {nullptr};
  };

  // Warm up to avoid benchmarking data transfer
  // blas_benchmark::utils::warmup(blas_method_def);

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

void register_benchmark(blas_benchmark::Args& args) {
  auto gemm_params = blas_benchmark::utils::get_blas3_params<float>(args);

  for (auto p : gemm_params) {
    std::string t1s, t2s;
    index_t m, n, k;
    float alpha, beta;
    std::tie(t1s, t2s, m, k, n, alpha, beta) = p;
    int t1 = static_cast<int>(blas_benchmark::utils::to_transpose_enum(t1s));
    int t2 = static_cast<int>(blas_benchmark::utils::to_transpose_enum(t2s));

    auto BM_lambda = [&](benchmark::State& st, int t1,
                         int t2, index_t m, index_t k, index_t n,
                         float alpha, float beta) {
      run(st, t1, t2, m, k, n, alpha, beta);
    };
    benchmark::RegisterBenchmark(get_name(t1s, t2s, m, k, n).c_str(),
                                 BM_lambda, t1, t2, m, k, n, alpha,
                                 beta);
  }
}

namespace blas_benchmark {
void create_benchmark(blas_benchmark::Args& args) {
  register_benchmark(args);
}
}  // namespace blas_benchmark
