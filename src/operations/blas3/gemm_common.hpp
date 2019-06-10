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
 * @brief The Tile structure determines the tiling configuration of a gemm
 *        implementation.
 *
 * The structure defines a hierarchical mapping of work items to matrix blocks,
 * and the Tile parameter have the largest impact on performance.
 * The blocking is done in 3 layers.
 *
 * The largest, top-level layer groups multiple consecutive work groups into a
 * top-level tile. The size of the tile (expressed in the number of work groups
 * in a tile) is determined by TlRows and TlColumns template parameters.
 * Different settings of the top-level layer do not have any impact on the
 * amount of required resources, but can impact data locality between
 * neighboring work groups, which can improve cache hit rates of high-level
 * caches if they are shared between multiple work groups.
 *
 * The second, block-level layer groups multiple work items into a block-level
 * tile. One block-level tile is assigned to one work-group, and hence
 * determines the number of work items within a work group.
 * It impacts local memory requirement (larger tile size requires more
 * local memory). A larger tile will also increase global data reuse
 * (average number of arithmetic operations performed per each data element
 * fetched from global memory). Generally, the larger block-level tile the
 * better, but its size is restricted by the maximal work-group size, and by
 * the available amount of shared memory.
 *
 * The last, item-level layer determines the size of the item-level tile,
 * which represents the size of the matrix block processed by a single work
 * item. A larger tile results in higher global data reuse, as well as local
 * data reuse (average number of arithmetic operations performed per each data
 * element fetched from local). However, larger tiles require more
 * register space, as well as more local memory.
 *
 * @note Square, or close-to-square tiles achieve the highest data reuse rate
 *       among all tiles that use the same amount of local / register
 *       space.
 *
 * @tparam ItemRows  the number of rows processed by each work item
 * @tparam ItemCols  the number of columns processed by each work item
 * @tparam WgRows  the number of item-level tiles within each column of
 *                 block-level tile
 * @tparam WgCols  the number of item-level tiles within each row of
 *                 block-level tile
 * @tparam TlRows  the number of block-level tiles within each column of
 *                 top-level tile
 * @tparam TlCols  the number of block-level tiles within each row of
 *                 top-level tile
 *
 * @see GemmFactory
 */
template <int ItemRows, int ItemCols, int WgRows, int WgCols, int TlRows,
          int TlCols>
SYCL_BLAS_INLINE std::string Tile<ItemRows, ItemCols, WgRows, WgCols, TlRows,
                                  TlCols>::get_type_string() noexcept {
  std::ostringstream str{};
  str << "Tile<" << item_rows << ", " << item_cols << ", " << wg_rows << ", "
      << wg_cols << ", " << tl_rows << ", " << tl_cols << ">";
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
