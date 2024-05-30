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
 *  @filename blas_extension_state_counters.hpp
 *
 **************************************************************************/

#ifndef COMMON_BLAS_EXTENSION_STATE_COUNTERS
#define COMMON_BLAS_EXTENSION_STATE_COUNTERS

#include "benchmark_identifier.hpp"

namespace blas_benchmark {
namespace utils {

template <ExtensionOp op, typename scalar_t, typename index_t>
inline typename std::enable_if<
    op == ExtensionOp::omatcopy || op == ExtensionOp::imatcopy ||
    op == ExtensionOp::omatcopy2 || op == ExtensionOp::omatcopy_batch ||
    op == ExtensionOp::imatcopy_batch>::type
init_extension_counters(benchmark::State& state, const char* trans, index_t m,
                        index_t n, index_t lda_mul, index_t ldb_mul,
                        index_t inc_a = 1, index_t inc_b = 1,
                        index_t stride_a_mul = 1, index_t stride_b_mul = 1,
                        index_t batch_size = 1) {
  // Google-benchmark counters are double.
  double size_d = static_cast<double>(m * n);
  state.counters["m"] = static_cast<double>(m);
  state.counters["n"] = static_cast<double>(n);
  state.counters["n_fl_ops"] = size_d * batch_size;
  state.counters["lda_m"] = static_cast<double>(lda_mul);
  state.counters["ldb_m"] = static_cast<double>(ldb_mul);
  state.counters["trans"] = static_cast<double>((*trans == 't') ? 1 : 0);
  state.counters["bytes_processed"] =
      (2 * size_d + 1) * sizeof(scalar_t) * batch_size;
  if constexpr (op == ExtensionOp::omatcopy_batch ||
                op == ExtensionOp::imatcopy_batch) {
    state.counters["stride_a_mul"] = static_cast<double>(stride_a_mul);
    state.counters["stride_b_mul"] = static_cast<double>(stride_b_mul);
    state.counters["batch_size"] = static_cast<double>(batch_size);
  }
  if constexpr (op == ExtensionOp::omatcopy2) {
    state.counters["inc_a"] = static_cast<double>(inc_a);
    state.counters["inc_b"] = static_cast<double>(inc_b);
  }
  return;
}

template <ExtensionOp op, typename scalar_t, typename index_t>
inline typename std::enable_if<op == ExtensionOp::omatadd ||
                               op == ExtensionOp::omatadd_batch>::type
init_extension_counters(benchmark::State& state, const char* t_a,
                        const char* t_b, index_t m, index_t n, index_t lda_mul,
                        index_t ldb_mul, index_t ldc_mul,
                        index_t stride_a_mul = 1, index_t stride_b_mul = 1,
                        index_t stride_c_mul = 1, index_t batch_size = 1) {
  // Google-benchmark counters are double.
  double size_d = static_cast<double>(m * n);
  state.counters["m"] = static_cast<double>(m);
  state.counters["n"] = static_cast<double>(n);
  state.counters["n_fl_ops"] = 3 * static_cast<double>(m * n);
  state.counters["lda_m"] = static_cast<double>(lda_mul);
  state.counters["ldb_m"] = static_cast<double>(ldb_mul);
  state.counters["ldc_m"] = static_cast<double>(ldc_mul);
  state.counters["trans_a"] = static_cast<double>((*t_a == 't') ? 1 : 0);
  state.counters["trans_b"] = static_cast<double>((*t_b == 't') ? 1 : 0);
  state.counters["bytes_processed"] = (3 * size_d + 1) * sizeof(scalar_t);
  if constexpr (op == ExtensionOp::omatadd_batch) {
    state.counters["stride_a_mul"] = static_cast<double>(stride_a_mul);
    state.counters["stride_b_mul"] = static_cast<double>(stride_b_mul);
    state.counters["stride_c_mul"] = static_cast<double>(stride_c_mul);
    state.counters["batch_size"] = static_cast<double>(batch_size);
  }
  return;
}

template <ExtensionOp op, typename scalar_t, typename index_t>
inline typename std::enable_if<op == ExtensionOp::axpy_batch>::type
init_extension_counters(benchmark::State& state, index_t n,
                        index_t batch_size) {
  // The way counters are computed are the same as axpy but multiplied
  // by the batch_size
  // Google-benchmark counters are double.
  double size_d = static_cast<double>(n);
  state.counters["size"] = size_d * batch_size;
  state.counters["n_fl_ops"] = 2.0 * size_d * batch_size;
  state.counters["bytes_processed"] =
      3 * size_d * sizeof(scalar_t) * batch_size;
  return;
}
}  // namespace utils
}  // namespace blas_benchmark

#endif  // COMMON_BLAS_EXTENSION_STATE_COUNTERS
