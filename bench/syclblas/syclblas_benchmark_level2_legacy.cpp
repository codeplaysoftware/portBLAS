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

BENCHMARK_NAME_FORMAT(syclblas_level_2_legacy) {
  std::ostringstream fname;
  fname << benchmark<>::typestr<ElemT>() << "_" << name() << "_"
        << std::get<0>(params) << "_" << std::get<1>(params) << "_"
        << std::get<2>(params);
  return fname.str();
}

BENCHMARK(gemv, syclblas_level_2_legacy) {
  using scalar_t = ElemT;
  using index_t = int;

  const char* t_str = std::get<0>(params);
  const index_t m = std::get<1>(params);
  const index_t n = std::get<2>(params);

  index_t vlen = t_str[0] == 'n' ? n : m;
  index_t rlen = t_str[0] == 'n' ? m : n;

  size_t n_fl_ops = m * n * 2;

  index_t lda = m;
  index_t incX = 1;
  index_t incY = 1;

  scalar_t alpha = benchmark<>::random_scalar<scalar_t>();
  scalar_t beta = benchmark<>::random_scalar<scalar_t>();

  // Input matrix
  std::vector<scalar_t> a_m = benchmark<>::random_data<scalar_t>(m * n);
  // Input Vector
  std::vector<scalar_t> b_v = benchmark<>::random_data<scalar_t>(vlen);
  // output Vector
  std::vector<scalar_t> c_v_gpu_result =
      benchmark<>::const_data<scalar_t>(rlen, 0);

  auto m_a_gpu = ex.get_policy_handler().template allocate<scalar_t>(m * n);
  auto v_b_gpu = ex.get_policy_handler().template allocate<scalar_t>(vlen);
  auto v_c_gpu = ex.get_policy_handler().template allocate<scalar_t>(rlen);
  ex.get_policy_handler().copy_to_device(a_m.data(), m_a_gpu, m * n);
  ex.get_policy_handler().copy_to_device(b_v.data(), v_b_gpu, vlen);
  ex.get_policy_handler().copy_to_device(c_v_gpu_result.data(), v_c_gpu, rlen);

  benchmark<>::datapoint_t flops = benchmark<>::measure(
      reps, n_fl_ops, [&]() -> std::vector<cl::sycl::event> {
        auto event = _gemv_legacy(ex, *t_str, m, n, alpha, m_a_gpu, m, v_b_gpu,
                                  incX, beta, v_c_gpu, incY);
        ex.get_policy_handler().wait(event);
        return event;
      });

  auto event = ex.get_policy_handler().copy_to_host(
      v_c_gpu, c_v_gpu_result.data(), rlen);

  ex.get_policy_handler().wait(event);

  ex.get_policy_handler().template deallocate<scalar_t>(m_a_gpu);
  ex.get_policy_handler().template deallocate<scalar_t>(v_b_gpu);
  ex.get_policy_handler().template deallocate<scalar_t>(v_c_gpu);

  return flops;
}

SUITE(ADD(gemv))

SYCL_BENCHMARK_MAIN(default_ranges::level_2, 10)
