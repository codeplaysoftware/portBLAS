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

#include "../common/blas_benchmark.hpp"

BENCHMARK_NAME_FORMAT(clblast_level_3) {
  std::ostringstream fname;
  fname << benchmark<>::typestr<ElemT>() << "_" << name() << "_"
        << std::get<0>(params) << "_" << std::get<1>(params) << "_"
        << std::get<2>(params) << "_" << std::get<3>(params) << "_"
        << std::get<4>(params);
  return fname.str();
}

// Helper function to translate transposition information from netlib blas style
// strings into clblast types.
clblast::Transpose translate_transposition(const char *t_str) {
  if (t_str[0] == 'n') {
    return clblast::Transpose::kNo;
  } else if (t_str[0] == 't') {
    return clblast::Transpose::kYes;
  } else if (t_str[0] == 'c') {
    return clblast::Transpose::kConjugate;
  } else {
    throw std::runtime_error("Got invalid transpose parameter!");
  }
}

BENCHMARK(gemm, clblast_level_3) {
  using ScalarT = ElemT;

  char const *t_a = std::get<0>(params);
  char const *t_b = std::get<1>(params);
  const size_t m = std::get<2>(params);
  const size_t k = std::get<3>(params);
  const size_t n = std::get<4>(params);

  size_t n_fl_ops = (static_cast<size_t>(2) * static_cast<size_t>(m) *
                     static_cast<size_t>(n) * static_cast<size_t>(k));

  size_t lda = t_a[0] == 'n' ? m : k;
  size_t ldb = t_b[0] == 'n' ? k : n;
  size_t ldc = m;

  // Specify the layout. As with GEMV, this needs to be kColMajor, and results
  // in errors otherwise. It may be that this is incorrect (especially for
  // performance reasons), so may need to be revisited.
  auto layout = clblast::Layout::kColMajor;

  // specify the transpositions.
  clblast::Transpose a_transpose = translate_transposition(t_a);
  clblast::Transpose b_transpose = translate_transposition(t_b);

  ScalarT alpha = benchmark<>::random_scalar<ScalarT>();
  ScalarT beta = benchmark<>::random_scalar<ScalarT>();

  size_t a_size = m * k;
  std::vector<ScalarT> a_host = benchmark<>::random_data<ScalarT>(a_size);
  MemBuffer<ScalarT> a(*ex, a_host.data(), a_size);

  size_t b_size = k * n;
  std::vector<ScalarT> b_host = benchmark<>::random_data<ScalarT>(b_size);
  MemBuffer<ScalarT> b(*ex, b_host.data(), b_size);

  size_t c_size = m * n;
  std::vector<ScalarT> c_host = benchmark<>::const_data<ScalarT>(c_size, 0);
  MemBuffer<ScalarT> c(*ex, c_host.data(), c_size);

  Event event;
  benchmark<>::datapoint_t flops = benchmark<>::measure(reps, n_fl_ops, [&]() {
    clblast::Gemm<ScalarT>(layout, a_transpose, b_transpose, m, n, k, alpha,
                           a.dev(), 0, lda, b.dev(), 0, ldb, beta, c.dev(), 0,
                           ldc, (*ex)._queue(), &event._cl());
    event.wait();
    return event;
  });

  return flops;
}

SUITE(ADD(gemm))

CLBLAST_BENCHMARK_MAIN(default_ranges::level_3, 10)