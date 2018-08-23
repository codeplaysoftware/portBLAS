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
 *  @filename syclblas_benchmark.cpp
 *
 **************************************************************************/

#include "blas_benchmark.hpp"

#include <interface/blas3_interface.hpp>

using namespace blas;

template <typename ExecutorType = SYCL>
class SyclBlasBenchmarker {
  cl::sycl::queue q;
  Executor<ExecutorType> ex;

 public:
  SyclBlasBenchmarker()
      : q(cl::sycl::default_selector(),
          [=](cl::sycl::exception_list eL) {
            for (auto &e : eL) {
              try {
                std::rethrow_exception(e);
              } catch (cl::sycl::exception &e) {
                std::cout << " E " << e.what() << std::endl;
              } catch (...) {
                std::cout << " An exception " << std::endl;
              }
            }
          }),
        ex(q) {}

  BENCHMARK_FUNCTION(gemm_nn) {
    using ScalarT = TypeParam;

    size_t m_size = size * size;

    auto lda = size;
    auto ldb = size;
    auto ldc = size;

    auto m = size;
    auto n = size;
    auto k = size;

    char const *t_a = "n";
    char const *t_b = "n";

    ScalarT alpha = ScalarT(1);
    ScalarT beta = ScalarT(1);
    // make two square matrices of size N * N
    std::vector<ScalarT> a = random_data<ScalarT>(m_size);
    std::vector<ScalarT> b = random_data<ScalarT>(m_size);
    std::vector<ScalarT> c = const_data<ScalarT>(m_size, 0);

    auto a_gpu = ex.template allocate<ScalarT>(m_size);
    auto b_gpu = ex.template allocate<ScalarT>(m_size);
    auto c_gpu = ex.template allocate<ScalarT>(m_size);

    ex.copy_to_device(a.data(), a_gpu, m_size);
    ex.copy_to_device(b.data(), b_gpu, m_size);
    ex.copy_to_device(c.data(), c_gpu, m_size);

    double flops = benchmark<>::measure(no_reps, m_size * 4, [&]() {
      auto event = _gemm(ex, *t_a, *t_b, m, n, k, alpha, a_gpu, lda, b_gpu, ldb,
                         beta, c_gpu, ldc);
      ex.wait(event);
    });

    auto event = ex.copy_to_host(c_gpu, c.data(), m_size);

    ex.wait(event);

    ex.template deallocate<ScalarT>(a_gpu);
    ex.template deallocate<ScalarT>(b_gpu);
    ex.template deallocate<ScalarT>(c_gpu);

    return flops;
  }

  BENCHMARK_FUNCTION(gemm_tn) {
    using ScalarT = TypeParam;

    size_t m_size = size * size;

    auto lda = size;
    auto ldb = size;
    auto ldc = size;

    auto m = size;
    auto n = size;
    auto k = size;

    char const *t_a = "t";
    char const *t_b = "n";

    ScalarT alpha = ScalarT(1);
    ScalarT beta = ScalarT(1);
    // make two square matrices of size N * N
    std::vector<ScalarT> a = random_data<ScalarT>(m_size);
    std::vector<ScalarT> b = random_data<ScalarT>(m_size);
    std::vector<ScalarT> c = const_data<ScalarT>(m_size, 0);

    auto a_gpu = ex.template allocate<ScalarT>(m_size);
    auto b_gpu = ex.template allocate<ScalarT>(m_size);
    auto c_gpu = ex.template allocate<ScalarT>(m_size);

    ex.copy_to_device(a.data(), a_gpu, m_size);
    ex.copy_to_device(b.data(), b_gpu, m_size);
    ex.copy_to_device(c.data(), c_gpu, m_size);

    double flops = benchmark<>::measure(no_reps, m_size * 4, [&]() {
      auto event = _gemm(ex, *t_a, *t_b, m, n, k, alpha, a_gpu, lda, b_gpu, ldb,
                         beta, c_gpu, ldc);
      ex.wait(event);
    });

    auto event = ex.copy_to_host(c_gpu, c.data(), m_size);

    ex.wait(event);

    ex.template deallocate<ScalarT>(a_gpu);
    ex.template deallocate<ScalarT>(b_gpu);
    ex.template deallocate<ScalarT>(c_gpu);

    return flops;
  }

  BENCHMARK_FUNCTION(gemm_nt) {
    using ScalarT = TypeParam;

    size_t m_size = size * size;

    auto lda = size;
    auto ldb = size;
    auto ldc = size;

    auto m = size;
    auto n = size;
    auto k = size;

    char const *t_a = "n";
    char const *t_b = "t";

    ScalarT alpha = ScalarT(1);
    ScalarT beta = ScalarT(1);
    // make two square matrices of size N * N
    std::vector<ScalarT> a = random_data<ScalarT>(m_size);
    std::vector<ScalarT> b = random_data<ScalarT>(m_size);
    std::vector<ScalarT> c = const_data<ScalarT>(m_size, 0);

    auto a_gpu = ex.template allocate<ScalarT>(m_size);
    auto b_gpu = ex.template allocate<ScalarT>(m_size);
    auto c_gpu = ex.template allocate<ScalarT>(m_size);

    ex.copy_to_device(a.data(), a_gpu, m_size);
    ex.copy_to_device(b.data(), b_gpu, m_size);
    ex.copy_to_device(c.data(), c_gpu, m_size);

    double flops = benchmark<>::measure(no_reps, m_size * 4, [&]() {
      auto event = _gemm(ex, *t_a, *t_b, m, n, k, alpha, a_gpu, lda, b_gpu, ldb,
                         beta, c_gpu, ldc);
      ex.wait(event);
    });

    auto event = ex.copy_to_host(c_gpu, c.data(), m_size);

    ex.wait(event);

    ex.template deallocate<ScalarT>(a_gpu);
    ex.template deallocate<ScalarT>(b_gpu);
    ex.template deallocate<ScalarT>(c_gpu);

    return flops;
  }
};

BENCHMARK_MAIN_BEGIN(1 << 1, 1 << 13, 10);
SyclBlasBenchmarker<SYCL> blasbenchmark;

BENCHMARK_REGISTER_FUNCTION("gemm_nn_float", gemm_nn<float>);
BENCHMARK_REGISTER_FUNCTION("gemm_tn_float", gemm_tn<float>);
BENCHMARK_REGISTER_FUNCTION("gemm_nt_float", gemm_nt<float>);

#ifndef NO_DOUBLE_SUPPORT
BENCHMARK_REGISTER_FUNCTION("gemm_nn_double", gemm_nn<double>);
BENCHMARK_REGISTER_FUNCTION("gemm_tn_double", gemm_tn<double>);
BENCHMARK_REGISTER_FUNCTION("gemm_nt_double", gemm_nt<double>);
#endif
BENCHMARK_MAIN_END();
