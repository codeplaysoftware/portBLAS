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
 *  @filename clblast_benchmark.cpp
 *
 **************************************************************************/

#include <complex>
#include <vector>

#include <clblast.h>

#include "../blas_benchmark.hpp"

BENCHMARK_NAME_FORMAT(clblast_level_2) {
  std::ostringstream fname;
  fname << typeid(ElemT).name() << "_" << name() << "_" << std::get<0>(params)
        << "_" << std::get<1>(params) << "_" << std::get<2>(params);
  return fname.str();

  return fname.str();
}

BENCHMARK(gemv, clblast_level_2) {
  using ScalarT = ElemT;

  size_t m = std::get<0>(params);
  size_t n = std::get<1>(params);
  const char* t_str = std::get<2>(params);

  size_t vlen = t_str[0] == 'n' ? n : m;
  size_t rlen = t_str[0] == 'n' ? m : n;

  size_t n_fl_ops = m * n * 2;

  size_t lda = m;
  long incX = 1;
  long incY = 1;

  // decide on the layout.
  // To be consistent with the sycl-blas, we should be:
  // Column major if t_str == 'n'
  // Row major otherwise
  // We may need to change more parameters here..
  auto layout = clblast::Layout::kColMajor;
    // t_str[0] == 'n' ? clblast::Layout::kColMajor : clblast::Layout::kRowMajor;

  // specify the transposition.
  clblast::Transpose a_transpose;
  if(t_str[0] == 'n') { 
    a_transpose = clblast::Transpose::kNo;
  } else if(t_str[0] == 't') { 
    a_transpose = clblast::Transpose::kYes;
  } else if(t_str[0] == 'c') { 
    a_transpose = clblast::Transpose::kConjugate;
  } else {
    throw std::runtime_error("Got invalid transpose parameter!");
  }

  ScalarT alpha = ScalarT(1);
  ScalarT beta = ScalarT(1);
  // Input matrix
  size_t msize = m * n;
  std::vector<ScalarT> a_m_host = benchmark<>::random_data<ScalarT>(msize);
  MemBuffer<ScalarT> a_m(*ex, a_m_host.data(), msize);
  // Input Vector
  std::vector<ScalarT> b_v_host = benchmark<>::random_data<ScalarT>(vlen);
  MemBuffer<ScalarT> b_v(*ex, b_v_host.data(), vlen);
  // output Vector
  std::vector<ScalarT> c_v_host = benchmark<>::const_data<ScalarT>(rlen, 0);
  MemBuffer<ScalarT> c_v(*ex, c_v_host.data(), rlen);

  Event event;
  benchmark<>::flops_units_t flops =
      benchmark<>::measure(reps, n_fl_ops, [&]() {
        clblast::Gemv<ScalarT>(layout, a_transpose, m, n, alpha, a_m.dev(), 0,
                               lda, b_v.dev(), 0, incX, beta, c_v.dev(), 0, incY,
                               (*ex)._queue(), &event._cl());
        event.wait();
        event.release();
      });

  return flops;
}

SUITE(ADD(gemv))

CLBLAST_BENCHMARK_MAIN(default_ranges::level_2, 10)