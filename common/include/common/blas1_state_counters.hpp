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
 *  @filename blas1_state_counters.hpp
 *
 **************************************************************************/

#ifndef COMMON_BLAS1_STATE_COUNTERS
#define COMMON_BLAS1_STATE_COUNTERS

#include "benchmark_identifier.hpp"

namespace blas_benchmark {
namespace utils {

template <Level1Op op, typename scalar_t, typename index_t>
inline typename std::enable_if<op == Level1Op::asum || op == Level1Op::iamax ||
                               op == Level1Op::iamin>::type
init_level_1_counters(benchmark::State& state, index_t size) {
  // Google-benchmark counters are double.
  double size_d = static_cast<double>(size);
  state.counters["size"] = size_d;
  state.counters["n_fl_ops"] = size_d;
  state.counters["bytes_processed"] = (size_d + 1) * sizeof(scalar_t);
  return;
}

template <Level1Op op, typename scalar_t>
inline typename std::enable_if<op == Level1Op::axpy>::type
init_level_1_counters(benchmark::State& state, index_t size) {
  // Google-benchmark counters are double.
  double size_d = static_cast<double>(size);
  state.counters["size"] = size_d;
  state.counters["n_fl_ops"] = 2.0 * size_d;
  state.counters["bytes_processed"] = 3 * size_d * sizeof(scalar_t);
  return;
}

template <Level1Op op, typename scalar_t>
inline typename std::enable_if<op == Level1Op::dot>::type init_level_1_counters(
    benchmark::State& state, index_t size) {
  // Google-benchmark counters are double.
  double size_d = static_cast<double>(size);
  state.counters["size"] = size_d;
  state.counters["n_fl_ops"] = 2 * size_d;
  state.counters["bytes_processed"] = (2 * size_d + 1) * sizeof(scalar_t);
  return;
}

template <Level1Op op, typename scalar_t>
inline typename std::enable_if<op == Level1Op::nrm2>::type
init_level_1_counters(benchmark::State& state, index_t size) {
  // Google-benchmark counters are double.
  double size_d = static_cast<double>(size);
  state.counters["size"] = size_d;
  state.counters["n_fl_ops"] = 2 * size_d;
  state.counters["bytes_processed"] = (size_d + 1) * sizeof(scalar_t);
  return;
}

template <Level1Op op, typename scalar_t>
inline typename std::enable_if<op == Level1Op::rotm>::type
init_level_1_counters(benchmark::State& state, index_t size) {
  // Google-benchmark counters are double.
  double size_d = static_cast<double>(size);
  state.counters["size"] = size_d;
  state.counters["n_fl_ops"] = 6 * size_d;
  state.counters["bytes_processed"] = 4 * size_d * sizeof(scalar_t);
  return;
}

template <Level1Op op, typename scalar_t>
inline typename std::enable_if<op == Level1Op::rotmg>::type
init_level_1_counters(benchmark::State& state, index_t size) {
  // Google-benchmark counters are double.
  double size_d = static_cast<double>(size);
  state.counters["size"] = size_d;
  state.counters["n_fl_ops"] = 6;
  state.counters["bytes_processed"] = 4 * sizeof(scalar_t);
  return;
}

template <Level1Op op, typename scalar_t>
inline typename std::enable_if<op == Level1Op::scal>::type
init_level_1_counters(benchmark::State& state, index_t size) {
  // Google-benchmark counters are double.
  double size_d = static_cast<double>(size);
  state.counters["size"] = size_d;
  state.counters["n_fl_ops"] = size_d;
  state.counters["bytes_processed"] = 2 * size_d * sizeof(scalar_t);
  return;
}

template <Level1Op op, typename scalar_t>
inline typename std::enable_if<op == Level1Op::sdsdot>::type
init_level_1_counters(benchmark::State& state, index_t size) {
  // Google-benchmark counters are double.
  double size_d = static_cast<double>(size);
  state.counters["size"] = size_d;
  state.counters["n_fl_ops"] = 2 * size_d;
  state.counters["bytes_processed"] = 2 * size_d * sizeof(scalar_t);
  return;
}

template <Level1Op op, typename scalar_t>
inline typename std::enable_if<op == Level1Op::copy>::type
init_level_1_counters(benchmark::State& state, index_t size) {
  // Google-benchmark counters are double.
  double size_d = static_cast<double>(size);
  state.counters["size"] = size_d;
  state.counters["bytes_processed"] = 2 * size_d * sizeof(scalar_t);
  return;
}

}  // namespace utils
}  // namespace blas_benchmark

#endif  // COMMON_BLAS1_STATE_COUNTERS
