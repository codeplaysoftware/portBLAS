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

namespace blas_benchmark {
namespace utils {

enum class ExtensionOP : int {
  omatcopy = 0,
  imatcopy = 1,
  omatadd = 2,
  omatcopy_batch = 3,
  imatcopy_batch = 4,
  omatadd_batch = 5,
  omatcopy2 = 6
};

template <ExtensionOP op, typename scalar_t, typename index_t>
inline typename std::enable_if<op == ExtensionOP::omatcopy ||
                               op == ExtensionOP::imatcopy ||
                               op == ExtensionOP::omatcopy2>::type
init_extension_counters(benchmark::State& state, const char* trans, index_t m,
                        index_t n, index_t lda_mul, index_t ldb_mul,
                        index_t inc_a = 1, index_t inc_b = 1) {
  // Google-benchmark counters are double.
  double size_d = static_cast<double>(m * n);
  state.counters["m"] = static_cast<double>(m);
  state.counters["n"] = static_cast<double>(n);
  state.counters["n_fl_ops"] = size_d;
  state.counters["lda_m"] = static_cast<double>(lda_mul);
  state.counters["ldb_m"] = static_cast<double>(ldb_mul);
  state.counters["trans"] = static_cast<double>((*trans == 't') ? 1 : 0);
  state.counters["bytes_processed"] = (2 * size_d + 1) * sizeof(scalar_t);
  if constexpr (op == ExtensionOP::omatcopy2) {
    state.counters["inc_a"] = static_cast<double>(inc_a);
    state.counters["inc_b"] = static_cast<double>(inc_b);
  }
  return;
}

template <ExtensionOP op, typename scalar_t, typename index_t>
inline typename std::enable_if<op == ExtensionOP::omatadd>::type
init_extension_counters(benchmark::State& state, const char* t_a,
                        const char* t_b, index_t m, index_t n, index_t lda_mul,
                        index_t ldb_mul, index_t ldc_mul) {
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
  return;
}
}  // namespace utils
}  // namespace blas_benchmark

#endif  // COMMON_BLAS_EXTENSION_STATE_COUNTERS
