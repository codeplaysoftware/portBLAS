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

#include "../common/blas_benchmark.hpp"

#include "sycl_blas.h"

using namespace blas;

BENCHMARK_NAME_FORMAT(syclblas_level_3) {
  std::ostringstream fname;
  fname << benchmark<>::typestr<ElemT>() << "_" << name() << "_"
        << std::get<0>(params) << "_" << std::get<1>(params) << "_"
        << std::get<2>(params) << "_" << std::get<3>(params) << "_"
        << std::get<4>(params);
  return fname.str();
}

BENCHMARK(gemm, syclblas_level_3) {
  using scalar_t = ElemT;
  using index_t = int;

  char const *t_a = std::get<0>(params);
  char const *t_b = std::get<1>(params);
  const index_t m = std::get<2>(params);
  const index_t k = std::get<3>(params);
  const index_t n = std::get<4>(params);
  const index_t batched_size = 16;

  size_t n_fl_ops = (static_cast<size_t>(2) * static_cast<size_t>(m) *
                     static_cast<size_t>(n) * static_cast<size_t>(k) *
                     static_cast<size_t>(batched_size));

  index_t lda = t_a[0] == 'n' ? m : k;
  index_t ldb = t_b[0] == 'n' ? k : n;
  index_t ldc = m;
  scalar_t alpha = benchmark<>::random_scalar<scalar_t>();
  scalar_t beta = benchmark<>::random_scalar<scalar_t>();

  std::vector<scalar_t> a =
      benchmark<>::random_data<scalar_t>(m * k * batched_size);
  std::vector<scalar_t> b =
      benchmark<>::random_data<scalar_t>(k * n * batched_size);
  std::vector<scalar_t> c =
      benchmark<>::const_data<scalar_t>(m * n * batched_size, 0);

  auto a_gpu =
      ex.get_policy_handler().template allocate<scalar_t>(m * k * batched_size);
  auto b_gpu =
      ex.get_policy_handler().template allocate<scalar_t>(k * n * batched_size);
  auto c_gpu =
      ex.get_policy_handler().template allocate<scalar_t>(m * n * batched_size);

  ex.get_policy_handler().copy_to_device(a.data(), a_gpu, m * k * batched_size);
  ex.get_policy_handler().copy_to_device(b.data(), b_gpu, k * n * batched_size);
  ex.get_policy_handler().copy_to_device(c.data(), c_gpu, m * n * batched_size);

  benchmark<>::datapoint_t flops = benchmark<>::measure(
      reps, n_fl_ops, [&]() -> std::vector<cl::sycl::event> {
        auto event = _gemm_batched(ex, *t_a, *t_b, m, n, k, alpha, a_gpu, lda,
                                   b_gpu, ldb, beta, c_gpu, ldc, batched_size);
        ex.get_policy_handler().wait(event);
        return event;
      });

  auto event = ex.get_policy_handler().copy_to_host(c_gpu, c.data(),
                                                    m * n * batched_size);

  ex.get_policy_handler().wait(event);

  ex.get_policy_handler().template deallocate<scalar_t>(a_gpu);
  ex.get_policy_handler().template deallocate<scalar_t>(b_gpu);
  ex.get_policy_handler().template deallocate<scalar_t>(c_gpu);

  return flops;
}

SUITE(ADD(gemm))

SYCL_BENCHMARK_MAIN(default_ranges::level_3, 10)
