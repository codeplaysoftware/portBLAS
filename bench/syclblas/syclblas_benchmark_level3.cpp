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

// #include "../blas_benchmark.hpp"
#include "../blas_benchmark2.hpp"

#include <interface/blas1_interface.hpp>
#include <interface/blas3_interface.hpp>

using namespace blas;

BENCHMARK(gemm) {
  using ScalarT = ElemT;

  size_t m_size =
      std::get<0>(params) * std::get<1>(params) * std::get<2>(params);

  size_t m = std::get<0>(params);
  size_t k = std::get<1>(params);
  size_t n = std::get<2>(params);
  
  std::cout << "Got sizes: (" << m << "," << k << ") (" << k<< "," << n << ") (" << m << "," <<n << ")" << std::endl;

  char const *t_a = "n";
  char const *t_b = "n";

  size_t lda = m;
  size_t ldb = n; 
  size_t ldc = m; 

  ScalarT alpha = ScalarT(1);
  ScalarT beta = ScalarT(1);
  // // make two square matrices of size m*n and n*k
  std::vector<ScalarT> a = random_data<ScalarT>(m * k);
  std::vector<ScalarT> b = random_data<ScalarT>(k * n);
  std::vector<ScalarT> c = const_data<ScalarT>(m * n, 0);

  auto a_gpu = ex.template allocate<ScalarT>(m * k);
  auto b_gpu = ex.template allocate<ScalarT>(k * n);
  auto c_gpu = ex.template allocate<ScalarT>(m * n);

  ex.copy_to_device(a.data(), a_gpu, m * k);
  ex.copy_to_device(b.data(), b_gpu, k * n);
  ex.copy_to_device(c.data(), c_gpu, m * n);

  double flops = benchmark<>::measure(reps, m_size * 4, [&]() {
    auto event = _gemm(ex, *t_a, *t_b, m, n, k, alpha, a_gpu, lda, b_gpu, ldb,
                       beta, c_gpu, lda);
    ex.wait(event);
  });

  std::cout << " Got : " << flops << " Flops! " << std::endl;

  auto event = ex.copy_to_host(c_gpu, c.data(), m_size);

  ex.wait(event);

  ex.template deallocate<ScalarT>(a_gpu);
  ex.template deallocate<ScalarT>(b_gpu);
  ex.template deallocate<ScalarT>(c_gpu);

  return flops;
}

SUITE(ADD(gemm))

auto three_d_range =
    nd_range(size_range(10, 1000, 10), size_range(10, 1000, 10),
             size_range(10, 1000, 10));

BENCHMARK_MAIN(three_d_range, 10)