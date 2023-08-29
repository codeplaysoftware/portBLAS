/***************************************************************************
 *
 *  @license
 *  Copyright (C) Codeplay Software Limited
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
 *  portBLAS: BLAS implementation using SYCL
 *
 *  @filename blas3_state_counters.hpp
 *
 **************************************************************************/

#ifndef COMMON_BLAS3_STATE_COUNTERS
#define COMMON_BLAS3_STATE_COUNTERS

#include "benchmark_identifier.hpp"

namespace blas_benchmark {
namespace utils {

template <Level3Op op, typename scalar_t, typename index_t>
inline typename std::enable_if<op == Level3Op::gemm_batched_strided ||
                               op == Level3Op::gemm_batched ||
                               op == Level3Op::gemm>::type
init_level_3_counters(benchmark::State& state, scalar_t beta = 0, index_t m = 0,
                      index_t n = 0, index_t k = 0, index_t batch_size = 1,
                      index_t stride_a_mul = 1, index_t stride_b_mul = 1,
                      index_t stride_c_mul = 1) {
  // Google-benchmark counters are double.
  double beta_d = static_cast<double>(beta);
  double m_d = static_cast<double>(m);
  double n_d = static_cast<double>(n);
  double k_d = static_cast<double>(k);
  double batch_size_d = static_cast<double>(batch_size);
  state.counters["beta"] = beta_d;
  state.counters["m"] = m_d;
  state.counters["n"] = n_d;
  state.counters["k"] = k_d;
  state.counters["batch_size"] = batch_size_d;
  if constexpr (op == Level3Op::gemm_batched_strided) {
    double stride_a_mul_d = static_cast<double>(stride_a_mul);
    double stride_b_mul_d = static_cast<double>(stride_b_mul);
    double stride_c_mul_d = static_cast<double>(stride_c_mul);

    state.counters["stride_a_mul"] = stride_a_mul_d;
    state.counters["stride_b_mul"] = stride_b_mul_d;
    state.counters["stride_c_mul"] = stride_c_mul_d;
  }
  const double nflops_AtimesB = 2 * k_d * m_d * n_d;
  double nflops_timesAlpha = m_d * n_d;
  const double nflops_addBetaC = (beta != scalar_t{0}) ? 2 * m_d * n_d : 0;
  const double nflops_tot =
      (nflops_AtimesB + nflops_timesAlpha + nflops_addBetaC) * batch_size_d;
  state.counters["n_fl_ops"] = nflops_tot;

  const double mem_readA = m_d * k_d;
  const double mem_readB = k_d * n_d;
  const double mem_writeC = m_d * n_d;
  const double mem_readC = (beta != scalar_t{0}) ? m_d * n_d : 0;
  const double total_mem = (mem_readA + mem_readB + mem_readC + mem_writeC) *
                           batch_size_d * sizeof(scalar_t);
  state.counters["bytes_processed"] = total_mem;
  return;
}

template <Level3Op op, typename scalar_t>
inline typename std::enable_if<op == Level3Op::symm>::type
init_level_3_counters(benchmark::State& state, scalar_t beta = 0, index_t m = 0,
                      index_t n = 0, index_t k = 0, index_t batch_size = 1,
                      char side = 'l') {
  // Google-benchmark counters are double.
  double beta_d = static_cast<double>(beta);
  double m_d = static_cast<double>(m);
  double n_d = static_cast<double>(n);
  state.counters["beta"] = beta_d;
  state.counters["m"] = m_d;
  state.counters["n"] = n_d;

  const double nflops_AtimesB =
      (side == 'l') ? 2 * m_d * m_d * n_d : 2 * n_d * n_d * n_d;
  const double nflops_addBetaC = beta != scalar_t{0} ? 2 * m_d * n_d : 0;
  const double nflops = nflops_AtimesB + nflops_addBetaC;
  state.counters["n_fl_ops"] = nflops;

  const double mem_readBreadC = (beta != scalar_t{0} ? 2 : 1) * m_d * n_d;
  const double mem_writeC = m_d * n_d;
  const double mem_readA =
      (side == 'l') ? (m_d * (m_d + 1) / 2) : (n_d * (n_d + 1) / 2);
  const double total_mem =
      (mem_readBreadC + mem_writeC + mem_readA) * sizeof(scalar_t);
  state.counters["bytes_processed"] = total_mem;

  return;
}

template <Level3Op op, typename scalar_t>
inline typename std::enable_if<op == Level3Op::syr2k>::type
init_level_3_counters(benchmark::State& state, scalar_t beta = 0, index_t m = 0,
                      index_t n = 0, index_t k = 0, index_t batch_size = 1,
                      char side = 'l') {
  // Google-benchmark counters are double.
  double beta_d = static_cast<double>(beta);
  double k_d = static_cast<double>(k);
  double n_d = static_cast<double>(n);
  state.counters["beta"] = beta_d;
  state.counters["k"] = k_d;
  state.counters["n"] = n_d;

  const double nflops_AtimesBtwice = 2 * n_d * (n_d + 1) * k_d;
  const double nflops_timesAlpha = n_d * (n_d + 1) / 2.;
  const double nflops_addBetaC =
      (beta != scalar_t{0}) ? (2 * n_d * (n_d + 1) / 2.) : 0;
  const double nflops =
      nflops_AtimesBtwice + nflops_timesAlpha + nflops_addBetaC;
  state.counters["n_fl_ops"] = nflops;

  const double mem_readAreadB = 2 * n_d * k_d;
  const double mem_readWriteC =
      (beta != scalar_t{0} ? 2 : 1) * n_d * (n_d + 1) / 2.;
  const double total_mem = (mem_readAreadB + mem_readWriteC) * sizeof(scalar_t);

  state.counters["bytes_processed"] = total_mem;
  return;
}

template <Level3Op op, typename scalar_t>
inline typename std::enable_if<op == Level3Op::syrk>::type
init_level_3_counters(benchmark::State& state, scalar_t beta = 0, index_t m = 0,
                      index_t n = 0, index_t k = 0, index_t batch_size = 1,
                      char side = 'l') {
  // Google-benchmark counters are double.
  double beta_d = static_cast<double>(beta);
  double k_d = static_cast<double>(k);
  double n_d = static_cast<double>(n);
  state.counters["beta"] = beta_d;
  state.counters["k"] = k_d;
  state.counters["n"] = n_d;

  const double nflops_AtimesA = n_d * (n_d + 1) * k_d;
  const double nflops_addBetaC = (beta != scalar_t{0}) ? n_d * (n_d + 1) : 0;
  const double nflops = nflops_AtimesA + nflops_addBetaC;
  state.counters["n_fl_ops"] = nflops;

  const double mem_readAreadB = 2 * n_d * k_d;
  const double mem_readWriteC =
      (beta != scalar_t{0} ? 2 : 1) * n_d * (n_d + 1) / 2.;
  const double total_mem = (mem_readAreadB + mem_readWriteC) * sizeof(scalar_t);

  state.counters["bytes_processed"] = total_mem;
  return;
}

template <Level3Op op, typename scalar_t>
inline typename std::enable_if<op == Level3Op::trmm || op == Level3Op::trsm ||
                               op == Level3Op::trsm_batched>::type
init_level_3_counters(benchmark::State& state, scalar_t beta = 0, index_t m = 0,
                      index_t n = 0, index_t k = 0, index_t batch_size = 1,
                      char side = 'l') {
  // Google-benchmark counters are double.
  double beta_d = static_cast<double>(beta);
  double m_d = static_cast<double>(m);
  double n_d = static_cast<double>(n);
  state.counters["beta"] = beta_d;
  state.counters["m"] = m_d;
  state.counters["n"] = n_d;

  const double nflops_AtimesB = side == 'l' ? 2 * m_d * (m_d + 1) / 2 * n_d
                                            : 2 * n_d * (n_d + 1) / 2 * m_d;
  const double nflops_timesAlpha = m_d * n_d;
  const double nflops = nflops_AtimesB + nflops_timesAlpha;
  state.counters["n_fl_ops"] = nflops;

  const double mem_readA = side == 'l' ? m_d * (m_d + 1) / 2 : m * n;
  const double mem_readB = side == 'r' ? n_d * (n_d + 1) / 2 : m * n;
  const double mem_writeB = m * n;
  const double total_mem =
      (mem_readA + mem_readB + mem_writeB) * sizeof(scalar_t);

  state.counters["bytes_processed"] = total_mem;
  return;
}

}  // namespace utils
}  // namespace blas_benchmark

#endif  // COMMON_BLAS3_STATE_COUNTERS
