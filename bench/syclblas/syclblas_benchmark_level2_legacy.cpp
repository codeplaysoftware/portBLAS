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

#include <interface/blas2_interface.hpp>

using namespace blas;

BENCHMARK_NAME_FORMAT(syclblas_level_2_legacy) {
  std::ostringstream fname;
  fname << benchmark<>::typestr<ElemT>() << "_" << name() << "_"
        << std::get<0>(params) << "_" << std::get<1>(params) << "_"
        << std::get<2>(params);
  return fname.str();
}

BENCHMARK(gemv, syclblas_level_2_legacy) {
  using ScalarT = ElemT;
  using IndexType = unsigned int;

  const char* t_str = std::get<0>(params);
  const IndexType m = std::get<1>(params);
  const IndexType n = std::get<2>(params);

  IndexType vlen = t_str[0] == 'n' ? n : m;
  IndexType rlen = t_str[0] == 'n' ? m : n;

  size_t n_fl_ops = m * n * 2;

  IndexType lda = m;
  long incX = 1;
  long incY = 1;

  ScalarT alpha = benchmark<>::random_scalar<ScalarT>();
  ScalarT beta = benchmark<>::random_scalar<ScalarT>();

  // Input matrix
  std::vector<ScalarT> a_m = benchmark<>::random_data<ScalarT>(m * n);
  // Input Vector
  std::vector<ScalarT> b_v = benchmark<>::random_data<ScalarT>(vlen);
  // output Vector
  std::vector<ScalarT> c_v_gpu_result =
      benchmark<>::const_data<ScalarT>(rlen, 0);

  auto m_a_gpu = ex.template allocate<ScalarT>(m * n);
  auto v_b_gpu = ex.template allocate<ScalarT>(vlen);
  auto v_c_gpu = ex.template allocate<ScalarT>(rlen);
  ex.copy_to_device(a_m.data(), m_a_gpu, m * n);
  ex.copy_to_device(b_v.data(), v_b_gpu, vlen);
  ex.copy_to_device(c_v_gpu_result.data(), v_c_gpu, rlen);

  benchmark<>::flops_units_t flops =
      benchmark<>::measure(reps, n_fl_ops, [&]() {
        auto event = _gemv_legacy(ex, *t_str, m, n, alpha, m_a_gpu, m, v_b_gpu,
                                  incX, beta, v_c_gpu, incY);
        ex.wait(event);
      });

  auto event = ex.copy_to_host(v_c_gpu, c_v_gpu_result.data(), rlen);

  ex.wait(event);

  ex.template deallocate<ScalarT>(m_a_gpu);
  ex.template deallocate<ScalarT>(v_b_gpu);
  ex.template deallocate<ScalarT>(v_c_gpu);

  return flops;
}

SUITE(ADD(gemv))

SYCL_BENCHMARK_MAIN(default_ranges::level_2, 10)
