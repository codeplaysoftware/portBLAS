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
 *  SYCL-BLAS: BLAS implementation using SYCL
 *
 *  @filename state_counters.hpp
 *
 **************************************************************************/

#ifndef COMMON_STATE_COUNTERS
#define COMMON_STATE_COUNTERS

namespace blas_benchmark {
namespace utils {

enum class Level2Op : int {
  gbmv = 0,
  gemv = 1,
  ger = 2,
  sbmv = 3,
  spmv = 4,
  spr = 5,
  spr2 = 6,
  symv = 7,
  syr = 8,
  syr2 = 9,
  tbmv = 10,
  tbsv = 11,
  tpmv = 12,
  tpsv = 13,
  trmv = 14,
  trsv = 15
};

template <Level2Op op, typename scalar_t, typename index_t>
inline typename std::enable_if<op == Level2Op::gbmv>::type
init_level_2_counters(benchmark::State& state, scalar_t beta = 0, index_t m = 0, index_t n = 0,
                      index_t k = 0, index_t ku = 0, index_t kl = 0) {
  // Google-benchmark counters are double.
  double m_d = static_cast<double>(m);
  double n_d = static_cast<double>(n);
  double kl_d = static_cast<double>(kl);
  double ku_d = static_cast<double>(ku);
  state.counters["m"] = m_d;
  state.counters["n"] = n_d;
  state.counters["kl"] = kl_d;
  state.counters["ku"] = ku_d;
  state.counters["n_fl_ops"] = 2 * (kl + ku + 1) * std::min(m, n) + m;
  state.counters["bytes_processed"] =
      ((kl + ku + 1) * std::min(m, n) + 2 * m + n) * sizeof(scalar_t);
  return;
}

template <Level2Op op, typename scalar_t, typename index_t>
inline typename std::enable_if<op == Level2Op::gemv>::type
init_level_2_counters(benchmark::State& state, scalar_t beta = 0, index_t m = 0, index_t n = 0,
                      index_t k = 0, index_t ku = 0, index_t kl = 0) {
  // Google-benchmark counters are double.
  double m_d = static_cast<double>(m);
  double n_d = static_cast<double>(n);
  state.counters["m"] = m_d;
  state.counters["n"] = n_d;
  state.counters["n_fl_ops"] = 2 * m * n + m + std::min(n, m);
  state.counters["bytes_processed"] = (m * n + 2 * m + n) * sizeof(scalar_t);
  return;
}

template <Level2Op op, typename scalar_t, typename index_t>
inline typename std::enable_if<op == Level2Op::ger>::type init_level_2_counters(
init_level_2_counters(benchmark::State& state, scalar_t beta = 0, index_t m = 0, index_t n = 0,
                      index_t k = 0, index_t ku = 0, index_t kl = 0) {
  // Google-benchmark counters are double.
  double m_d = static_cast<double>(m);
  double n_d = static_cast<double>(n);
  state.counters["m"] = m_d;
  state.counters["n"] = n_d;
  state.counters["n_fl_ops"] = 2 * m * n + std::min(n, m);
  state.counters["bytes_processed"] = (2 * m * n + m + n) * sizeof(scalar_t);
  return;
}

template <Level2Op op, typename scalar_t, typename index_t>
inline typename std::enable_if<op == Level2Op::sbmv>::type
init_level_2_counters(benchmark::State& state, scalar_t beta = 0, index_t m = 0, index_t n = 0,
                      index_t k = 0, index_t ku = 0, index_t kl = 0) {
  // Google-benchmark counters are double.
  double k_d = static_cast<double>(k);
  double n_d = static_cast<double>(n);
  state.counters["k"] = k_d;
  state.counters["n"] = n_d;
  state.counters["n_fl_ops"] = 2 * (2 * k + 1) * n + n;
  state.counters["bytes_processed"] = (((k + 1) + 3) * n) * sizeof(scalar_t);
  return;
}

template <Level2Op op, typename scalar_t, typename index_t>
inline
    typename std::enable_if<op == Level2Op::spmv || op == Level2Op::symv>::type
init_level_2_counters(benchmark::State& state, scalar_t beta = 0, index_t m = 0, index_t n = 0,
                      index_t k = 0, index_t ku = 0, index_t kl = 0) {
  // Google-benchmark counters are double.
  double n_d = static_cast<double>(n);
  state.counters["n"] = n_d;
  state.counters["n_fl_ops"] = 2 * n * n + n;
  state.counters["bytes_processed"] =
      ((((n + 1) / 2) + 3) * n) * sizeof(scalar_t);
  return;
}

template <Level2Op op, typename scalar_t, typename index_t>
inline typename std::enable_if<op == Level2Op::spr || op == Level2Op::syr>::type
init_level_2_counters(benchmark::State& state, scalar_t beta = 0, index_t m = 0, index_t n = 0,
                      index_t k = 0, index_t ku = 0, index_t kl = 0) {
  // Google-benchmark counters are double.
  double n_d = static_cast<double>(n);
  state.counters["n"] = n_d;
  state.counters["n_fl_ops"] = 2 * n * ((n + 1) / 2) + n;
  state.counters["bytes_processed"] =
      ((2 * ((n + 1) / 2) + 1) * n) * sizeof(scalar_t);
  return;
}

template <Level2Op op, typename scalar_t, typename index_t>
inline
    typename std::enable_if<op == Level2Op::spr2 || op == Level2Op::syr2>::type
init_level_2_counters(benchmark::State& state, scalar_t beta = 0, index_t m = 0, index_t n = 0,
                      index_t k = 0, index_t ku = 0, index_t kl = 0) {
  // Google-benchmark counters are double.
  double n_d = static_cast<double>(n);
  state.counters["n"] = n_d;
  state.counters["n_fl_ops"] = 4 * n * ((n + 1) / 2) + n;
  state.counters["bytes_processed"] =
      ((2 * ((n + 1) / 2) + 2) * n) * sizeof(scalar_t);
  return;
}

template <Level2Op op, typename scalar_t, typename index_t>
inline
    typename std::enable_if<op == Level2Op::tbmv || op == Level2Op::tbsv>::type
init_level_2_counters(benchmark::State& state, scalar_t beta = 0, index_t m = 0, index_t n = 0,
                      index_t k = 0, index_t ku = 0, index_t kl = 0) {
  // Google-benchmark counters are double.
  double k_d = static_cast<double>(k);
  double n_d = static_cast<double>(n);
  state.counters["n"] = n_d;
  state.counters["k"] = k_d;
  state.counters["n_fl_ops"] = 2 * (k + 1) * n;
  state.counters["bytes_processed"] = (((k + 1) + 2) * n) * sizeof(scalar_t);
  return;
}

template <Level2Op op, typename scalar_t, typename index_t>
inline
    typename std::enable_if<op == Level2Op::tpmv || op == Level2Op::trmv ||
                            op == Level2Op::trsv || op == Level2Op::tpsv>::type
init_level_2_counters(benchmark::State& state, scalar_t beta = 0, index_t m = 0, index_t n = 0,
                      index_t k = 0, index_t ku = 0, index_t kl = 0) {
  // Google-benchmark counters are double.
  double n_d = static_cast<double>(n);
  state.counters["n"] = n_d;
  state.counters["n_fl_ops"] = 2 * n * ((n + 1) / 2);
  state.counters["bytes_processed"] =
      ((((n + 1) / 2) + 2) * n) * sizeof(scalar_t);
  return;
}

}  // namespace utils
}  // namespace blas_benchmark

#endif  // COMMON_STATE_COUNTERS