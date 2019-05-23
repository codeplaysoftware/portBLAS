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
 *  @filename blas3_trees.h
 *
 **************************************************************************/

#ifndef SYCL_BLAS_BLAS3_TREES_GEMM_H
#define SYCL_BLAS_BLAS3_TREES_GEMM_H

#include <CL/sycl.hpp>
#include <string>
#include <type_traits>

namespace blas {
/*
 * @brief Determines the type of the GEMM kernel.
 * It can be either a naive kernel; a kernel uses local memory or a kernel that
 * does not use local memory
 */
enum class Gemm_t : int { naive = 0, local_memory = 1, no_local_memory = 2 };

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
template <int ItemRows = 8, int ItemCols = 8, int WgRows = 16, int WgCols = 16,
          int TlRows = 1, int TlCols = 1>
struct Tile {
  static constexpr int item_rows = ItemRows;
  static constexpr int item_cols = ItemCols;
  static constexpr int wg_rows = WgRows;
  static constexpr int wg_cols = WgCols;
  static constexpr int tl_rows = TlRows;
  static constexpr int tl_cols = TlCols;
  /*!
   * @brief Get tile type as human readable string.
   */
  static std::string get_type_string() noexcept;
};
/*!
 * @brief GemmFactory is a template class whose instantiations provide
 *        different implementations of the GEMM device function. It also support
 * batch GEMM
 *
 * To use the function, each item of a kernel launched with nd_range given by
 * GemmFactory::get_nd_range() should call GemmFactory::run(). The size of
 * local memory required per work group can be queried with
 * GemmFactory::local_memory_size.
 *
 * @tparam DoubleBuffer  iff true,  enables the use of double buffering
 *                       (doubles the amount of consumed local memory,
 *                        but halves the number of required local barriers)
 * @tparam NbcA  iff true, avoids bank conflicts when accessing blocks of
 *               matrix A in local memory (slightly increases local
 *               memory consumption) - may be useful in combination with TranA
 * @tparam NbcB  iff true, avoids bank conflicts when accessing blocks of
 *               matrix B in local memory (slightly increases local
 *               memory consumption) - may be useful in combination with TranB
 * @tparam ClSize  the size of the cache line of the architecture
 *                 (If the value passed in is smaller than the actual cache
 *                 line size, some values fetched will be wasted, which can
 *                 significantly reduce performance. It can be set to a
 *                 multiple of the physical cache line size. In this case, it
 *                 will significantly increase local memory usage, but
 *                 will result in fewer local barriers.)
 * @tparam TileType  determines the size of the local, work group, and top
 *                   level tiles to use, see Tile
 * @tparam TransA  iff true, matrix A will be transposed on the fly
 * @tparam TransB  iff true, matrix B will be transposed on the fly
 * @tparam element_t  type of matrix elements
 * @param a_ the lhs_t matrix
 * @param b_ the rhs_t matrix
 * @param c_ the output matrix
 * @param alpha_ specifies the scalar alpha
 * @param beta_ specifies the scalar beta
 * @param m  the number  of rows  of the  matrix _C
 * @param n  the number  of column  of the  matrix _C
 * @param k the contracting dimension between a_ and b_
 * @param lda the leading dimension of the matrix a_
 * @param ldb the leading dimension of the matrix b_
 * @param ldc the leading dimension of the matrix _C
 * @param batch_size_ the number batches of matrices of a_ b_ _C
 */
template <typename input_t, typename output_t, bool DoubleBuffer, bool NbcA,
          bool NbcB, int ClSize, typename tile_type, bool TransA, bool TransB,
          typename element_t, bool is_beta_zero, int Gemm_type>
class Gemm {
 public:
  using value_t = element_t;
  using index_t = typename std::make_signed<typename input_t::index_t>::type;
  static constexpr int type = Gemm_type;
  static constexpr int wg_size = tile_type::wg_rows * tile_type::wg_cols;
  static constexpr bool trans_a = TransA;
  static constexpr bool trans_b = TransB;
  static constexpr int local_memory_size = 0;
  input_t a_;
  input_t b_;
  output_t c_;
  element_t alpha_;
  element_t beta_;
  index_t m_;
  index_t n_;
  index_t k_;
  index_t lda_;
  index_t ldb_;
  index_t ldc_;
  index_t batch_size_;
  Gemm(input_t A, input_t B, output_t C, element_t alpha, element_t beta,
       index_t batch_size);
  static std::string get_type_string() noexcept;
  static index_t get_workgroup_cluster(index_t m, index_t n) noexcept;
  static index_t get_num_workgroup_cluster(index_t m, index_t n,
                                           index_t compute_units) noexcept;
  static cl::sycl::nd_range<1> get_nd_range(index_t m, index_t n,
                                            index_t compute_units) noexcept;
  index_t get_size() const;
  bool valid_thread(cl::sycl::nd_item<1> ndItem) const;
  void eval(cl::sycl::nd_item<1> id) noexcept;
  void bind(cl::sycl::handler &h);
};

/*
 * @brief a helper function used for constructing the GEMM
 *  see GEMM for the parammeters passed here.
 */
template <bool DoubleBuffer, bool ConflictA, bool ConflictB, int ClSize,
          typename TileType, bool TransA, bool TransB, int Gemm_type,
          bool is_beta_zero, typename input_t, typename output_t,
          typename element_t, typename index_t>
inline Gemm<input_t, output_t, DoubleBuffer, ConflictA, ConflictB, ClSize,
            TileType, TransA, TransB, element_t, is_beta_zero, Gemm_type>
make_gemm(input_t buffer_a, input_t buffer_b, output_t buffer_c,
          element_t alpha, element_t beta, index_t batch_size) {
  return Gemm<input_t, output_t, DoubleBuffer, ConflictA, ConflictB, ClSize,
              TileType, TransA, TransB, element_t, is_beta_zero, Gemm_type>(
      buffer_a, buffer_b, buffer_c, alpha, beta, batch_size);
}

}  // namespace blas

#endif  // BLAS3_TREES_GEMM_H
