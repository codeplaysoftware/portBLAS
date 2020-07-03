/***************************************************************************
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
 *  @filename gemm_common.hpp
 *
 **************************************************************************/

#ifndef SYCL_BLAS_BLAS3_GEMM_COMMON_HPP
#define SYCL_BLAS_BLAS3_GEMM_COMMON_HPP

#include "operations/blas3_trees.h"
#include "views/view.h"
#include <CL/sycl.hpp>
#include <string>
#include <type_traits>

namespace blas {

template <typename T>
struct type_string {
  static const char *get_value() { return "unknown"; }
};

#define ENABLE_TYPE_STRING(_type)                     \
  template <>                                         \
  struct type_string<_type> {                         \
    static const char *get_value() { return #_type; } \
  };

ENABLE_TYPE_STRING(float)
ENABLE_TYPE_STRING(double)

#undef ENABLE_TYPE_STRING

/*!
 * Returns a human-readable description of a tile type.
 *
 * @return The type string as a std::string
 *
 * @note See the struct definition in include/operations/blas3_trees.h for more
 *       info about the tiling configuration of gemm
 */
template <int ItemRows, int ItemCols, int WgRows, int WgCols, int TlRows,
          int TlCols, int ItemBatchs, int WgBatchs>
SYCL_BLAS_INLINE std::string
Tile<ItemRows, ItemCols, WgRows, WgCols, TlRows, TlCols, ItemBatchs,
     WgBatchs>::get_type_string() noexcept {
  std::ostringstream str{};
  str << "Tile<" << item_rows << ", " << item_cols << ", " << wg_rows << ", "
      << wg_cols << ", " << tl_rows << ", " << tl_cols << ", " << item_batchs
      << ", " << wg_batchs << ">";
  return str.str();
}

/*!
 * Optionally avoid evaluating the expression given as input.
 *
 * @return If the template parameter is true, return the value of expression
 *         given by cond, otherwise return true.
 *
 * @note This function can be used to hint the compiler that a boolean
 *       expression does not have to be evaluated in certain situations.
 */
template <bool>
SYCL_BLAS_INLINE bool do_check(bool cond) {
  return cond;
}
template <>
SYCL_BLAS_INLINE bool do_check<false>(bool) {
  return true;
}

}  // namespace blas

#endif  // SYCL_BLAS_BLAS3_GEMM_COMMON_HPP
