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

BENCHMARK_NAME_FORMAT(blas_level_3) {
  std::ostringstream fname;
  fname << name() << "_" << std::get<0>(params) << "_" << std::get<1>(params)
        << "_" << std::get<2>(params) << "_" << std::get<3>(params) << "_"
        << std::get<4>(params);
  return fname.str();
}

BENCHMARK(gemm, blas_level_3) {
  using ScalarT = ElemT;

  const size_t m = std::get<0>(params);
  const size_t k = std::get<1>(params);
  const size_t n = std::get<2>(params);
  char const *t_a = std::get<3>(params);
  char const *t_b = std::get<4>(params);

  size_t n_fl_ops = (2 * m * n * k); 

  size_t lda = t_a[0] == 'n' ? m : k;
  size_t ldb = t_b[0] == 'n' ? k : n;
  size_t ldc = m;

  ScalarT alpha = ScalarT(1);
  ScalarT beta = ScalarT(1);

  std::vector<ScalarT> a = benchmark<>::random_data<ScalarT>(m * k);
  std::vector<ScalarT> b = benchmark<>::random_data<ScalarT>(k * n);
  std::vector<ScalarT> c = benchmark<>::const_data<ScalarT>(m * n, 0);

  auto a_gpu = ex.template allocate<ScalarT>(m * k);
  auto b_gpu = ex.template allocate<ScalarT>(k * n);
  auto c_gpu = ex.template allocate<ScalarT>(m * n);

  ex.copy_to_device(a.data(), a_gpu, m * k);
  ex.copy_to_device(b.data(), b_gpu, k * n);
  ex.copy_to_device(c.data(), c_gpu, m * n);

  benchmark<>::flops_units_t flops =
      benchmark<>::measure(reps, n_fl_ops, [&]() {
        auto event = _gemm(ex, *t_a, *t_b, m, n, k, alpha, a_gpu, lda, b_gpu,
                           ldb, beta, c_gpu, lda);
        ex.wait(event);
      });

  auto event = ex.copy_to_host(c_gpu, c.data(), m * n);

  ex.wait(event);

  ex.template deallocate<ScalarT>(a_gpu);
  ex.template deallocate<ScalarT>(b_gpu);
  ex.template deallocate<ScalarT>(c_gpu);

  return flops;
}

SUITE(ADD(gemm))

auto level_3_ranges = nd_range(size_range(2, 1024, 2), size_range(2, 1024, 2),
                               size_range(2, 1024, 2), value_range({"n"}),
                               value_range({"n", "t", "c"}));

BENCHMARK_MAIN(level_3_ranges, 10)