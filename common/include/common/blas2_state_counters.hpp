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
 *  @filename blas2_state_counters.hpp
 *
 **************************************************************************/

#ifndef COMMON_BLAS2_STATE_COUNTERS
#define COMMON_BLAS2_STATE_COUNTERS

#include "benchmark_identifier.hpp"

namespace blas_benchmark {
namespace utils {

template <Level2Op op, typename scalar_t, typename index_t>
inline typename std::enable_if<op == Level2Op::gbmv>::type
init_level_2_counters(benchmark::State& state, const char* t_str,
                      scalar_t beta = scalar_t{0}, index_t m = 0, index_t n = 0,
                      index_t k = 0, index_t ku = 0, index_t kl = 0) {
  // Google-benchmark counters are double.
  double beta_d = static_cast<double>(beta);
  double m_d = static_cast<double>(m);
  double n_d = static_cast<double>(n);
  double kl_d = static_cast<double>(kl);
  double ku_d = static_cast<double>(ku);
  double xlen = t_str[0] == 'n' ? n_d : m_d;
  double ylen = t_str[0] == 'n' ? m_d : n_d;
  state.counters["beta"] = beta_d;
  state.counters["m"] = m_d;
  state.counters["n"] = n_d;
  state.counters["kl"] = kl_d;
  state.counters["ku"] = ku_d;
  const double A_validVal = xlen * (kl_d + ku_d + 1.0) -
                            0.5 * (kl_d * (kl_d + 1.0)) -
                            0.5 * (ku_d * (ku_d + 1.0));

  const double nflops_AtimesX = 2.0 * A_validVal;
  const double nflops_timesAlpha = ylen;
  const double nflops_addBetaY = (beta != scalar_t{0}) ? 2 * ylen : 0;
  const double nflops_tot =
      nflops_AtimesX + nflops_timesAlpha + nflops_addBetaY;
  state.counters["n_fl_ops"] = nflops_tot;

  const double mem_readA = A_validVal;
  const double mem_readX = xlen;
  const double mem_writeY = ylen;
  const double mem_readY = (beta != scalar_t{0}) ? ylen : 0;
  state.counters["bytes_processed"] =
      (mem_readA + mem_readX + mem_writeY + mem_readY) * sizeof(scalar_t);
  return;
}

template <Level2Op op, typename scalar_t>
inline typename std::enable_if<op == Level2Op::gemv>::type
init_level_2_counters(benchmark::State& state, const char* t_str,
                      scalar_t beta = scalar_t{0}, index_t m = 0, index_t n = 0,
                      index_t k = 0, index_t ku = 0, index_t kl = 0) {
  // Google-benchmark counters are double.
  double beta_d = static_cast<double>(beta);
  double m_d = static_cast<double>(m);
  double n_d = static_cast<double>(n);
  double xlen = t_str[0] == 'n' ? n_d : m_d;
  double ylen = t_str[0] == 'n' ? m_d : n_d;
  state.counters["beta"] = beta_d;
  state.counters["m"] = m_d;
  state.counters["n"] = n_d;

  const double nflops_AtimesX = 2.0 * m_d * n_d;
  const double nflops_timesAlpha = xlen;
  const double nflops_addBetaY = (beta != scalar_t{0}) ? 2 * ylen : 0;
  const double nflops_tot =
      nflops_AtimesX + nflops_timesAlpha + nflops_addBetaY;
  state.counters["n_fl_ops"] = nflops_tot;

  const double mem_readA = m_d * n_d;
  const double mem_readX = xlen;
  const double mem_writeY = ylen;
  const double mem_readY = (beta != scalar_t{0}) ? ylen : 0;
  state.counters["bytes_processed"] =
      (mem_readA + mem_readX + mem_writeY + mem_readY) * sizeof(scalar_t);
  return;
}

template <Level2Op op, typename scalar_t>
inline typename std::enable_if<op == Level2Op::ger>::type init_level_2_counters(
    benchmark::State& state, const char* t_str, scalar_t beta = scalar_t{0},
    index_t m = 0, index_t n = 0, index_t k = 0, index_t ku = 0,
    index_t kl = 0) {
  // Google-benchmark counters are double.
  double m_d = static_cast<double>(m);
  double n_d = static_cast<double>(n);
  state.counters["m"] = m_d;
  state.counters["n"] = n_d;
  state.counters["n_fl_ops"] = 2 * m_d * n_d + std::min(n_d, m_d);
  state.counters["bytes_processed"] =
      (2 * m_d * n_d + m_d + n_d) * sizeof(scalar_t);
  return;
}

template <Level2Op op, typename scalar_t>
inline typename std::enable_if<op == Level2Op::sbmv>::type
init_level_2_counters(benchmark::State& state, const char* t_str,
                      scalar_t beta = scalar_t{0}, index_t m = 0, index_t n = 0,
                      index_t k = 0, index_t ku = 0, index_t kl = 0) {
  // Google-benchmark counters are double.
  double beta_d = static_cast<double>(beta);
  double k_d = static_cast<double>(k);
  double n_d = static_cast<double>(n);
  state.counters["beta"] = beta_d;
  state.counters["k"] = k_d;
  state.counters["n"] = n_d;

  // Compute the number of A non-zero elements.
  const double A_validVal = (n_d * (2.0 * k_d + 1.0)) - (k_d * (k_d + 1.0));

  const double nflops_AtimesX = 2.0 * A_validVal;
  const double nflops_timesAlpha = n_d;
  const double nflops_addBetaY = (beta != scalar_t{0}) ? 2 * n_d : 0;
  const double nflops_tot =
      nflops_AtimesX + nflops_timesAlpha + nflops_addBetaY;
  state.counters["n_fl_ops"] = nflops_tot;

  const double mem_readA = A_validVal;
  const double mem_readX = n_d;
  const double mem_writeY = n_d;
  const double mem_readY = (beta != scalar_t{0}) ? n_d : 0;
  state.counters["bytes_processed"] =
      (mem_readA + mem_readX + mem_writeY + mem_readY) * sizeof(scalar_t);
  return;
}

template <Level2Op op, typename scalar_t>
inline
    typename std::enable_if<op == Level2Op::spmv || op == Level2Op::symv>::type
    init_level_2_counters(benchmark::State& state, const char* t_str,
                          scalar_t beta = scalar_t{0}, index_t m = 0,
                          index_t n = 0, index_t k = 0, index_t ku = 0,
                          index_t kl = 0) {
  // Google-benchmark counters are double.
  double beta_d = static_cast<double>(beta);
  double n_d = static_cast<double>(n);
  state.counters["beta"] = beta_d;
  state.counters["n"] = n_d;
  // Compute the number of A non-zero elements.
  const double A_validVal = (n_d * (n_d + 1) / 2);

  const double nflops_AtimesX = 2 * n_d * n_d;
  const double nflops_timesAlpha = n_d;
  const double nflops_addBetaY = (beta != scalar_t{0}) ? 2 * n_d : 0;
  const double nflops_tot =
      nflops_AtimesX + nflops_timesAlpha + nflops_addBetaY;
  state.counters["n_fl_ops"] = nflops_tot;

  const double mem_readA = A_validVal;
  const double mem_readX = n_d;
  const double mem_writeY = n_d;
  const double mem_readY = (beta != scalar_t{0}) ? n_d : 0;
  state.counters["bytes_processed"] =
      (mem_readA + mem_readX + mem_writeY + mem_readY) * sizeof(scalar_t);
  return;
}

template <Level2Op op, typename scalar_t>
inline typename std::enable_if<op == Level2Op::spr || op == Level2Op::syr>::type
init_level_2_counters(benchmark::State& state, const char* t_str,
                      scalar_t beta = scalar_t{0}, index_t m = 0, index_t n = 0,
                      index_t k = 0, index_t ku = 0, index_t kl = 0) {
  // Google-benchmark counters are double.
  double n_d = static_cast<double>(n);
  state.counters["n"] = n_d;
  state.counters["n_fl_ops"] = 2 * n * ((n + 1) / 2) + n;
  state.counters["bytes_processed"] =
      ((2 * ((n + 1) / 2) + 1) * n) * sizeof(scalar_t);
  return;
}

template <Level2Op op, typename scalar_t>
inline
    typename std::enable_if<op == Level2Op::spr2 || op == Level2Op::syr2>::type
    init_level_2_counters(benchmark::State& state, const char* t_str,
                          scalar_t beta = scalar_t{0}, index_t m = 0,
                          index_t n = 0, index_t k = 0, index_t ku = 0,
                          index_t kl = 0) {
  // Google-benchmark counters are double.
  double n_d = static_cast<double>(n);
  state.counters["n"] = n_d;
  state.counters["n_fl_ops"] = 4 * n * ((n + 1) / 2) + n;
  state.counters["bytes_processed"] =
      ((2 * ((n + 1) / 2) + 2) * n) * sizeof(scalar_t);
  return;
}

template <Level2Op op, typename scalar_t>
inline
    typename std::enable_if<op == Level2Op::tbmv || op == Level2Op::tbsv>::type
    init_level_2_counters(benchmark::State& state, const char* t_str,
                          scalar_t beta = scalar_t{0}, index_t m = 0,
                          index_t n = 0, index_t k = 0, index_t ku = 0,
                          index_t kl = 0) {
  // Google-benchmark counters are double.
  double k_d = static_cast<double>(k);
  double n_d = static_cast<double>(n);
  state.counters["n"] = n_d;
  state.counters["k"] = k_d;
  // Compute the number of A non-zero elements.
  const double A_validVal = (n_d * (k_d + 1.0)) - (0.5 * (k_d * (k_d + 1.0)));

  const double nflops_AtimesX = 2.0 * A_validVal;
  state.counters["n_fl_ops"] = nflops_AtimesX;

  const double mem_readA = A_validVal;
  const double mem_readX = n_d;
  const double mem_writeX = n_d;
  state.counters["bytes_processed"] =
      (mem_readA + mem_readX + mem_writeX) * sizeof(scalar_t);
  return;
}

template <Level2Op op, typename scalar_t>
inline
    typename std::enable_if<op == Level2Op::tpmv || op == Level2Op::trmv ||
                            op == Level2Op::trsv || op == Level2Op::tpsv>::type
    init_level_2_counters(benchmark::State& state, const char* t_str,
                          scalar_t beta = scalar_t{0}, index_t m = 0,
                          index_t n = 0, index_t k = 0, index_t ku = 0,
                          index_t kl = 0) {
  // Google-benchmark counters are double.
  double n_d = static_cast<double>(n);
  state.counters["n"] = n_d;
  state.counters["n_fl_ops"] = 2 * n_d * ((n_d + 1) / 2);
  state.counters["bytes_processed"] =
      ((((n_d + 1) / 2) + 2) * n_d) * sizeof(scalar_t);
  return;
}

}  // namespace utils
}  // namespace blas_benchmark

#endif  // COMMON_BLAS2_STATE_COUNTERS
