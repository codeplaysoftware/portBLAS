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
 *  @filename blas3_trees.hpp
 *
 **************************************************************************/

#ifndef SYCL_BLAS_BLAS3_TREES_GEMM_HPP
#define SYCL_BLAS_BLAS3_TREES_GEMM_HPP

#include "operations/blas3_trees.h"
#include "views/view.h"
#include <CL/sycl.hpp>
#include <string>
#include <type_traits>

namespace blas {

template <typename element_t>
struct type_string {
  static SYCL_BLAS_INLINE const char *get_value() { return "unknown"; }
};

#define ENABLE_TYPE_STRING(_type)                                      \
  template <>                                                          \
  struct type_string<_type> {                                          \
    static SYCL_BLAS_INLINE const char *get_value() { return #_type; } \
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
 * @brief This factory generates reference GEMM implementations.
 *
 * These implementations use a naive approach of mapping one value of the
 * output matrix to each work item, and are highly memory bound.
 * They should only be used as a reference in performance testing, or to check
 * correctness of other implementations.
 * Refer to GemmFactory for details about how to use this. Note that there is
 * no local_memory value, as these functions do not use local memory.
 *
 * @tparam WgSize  the number of items in a work group
 * @tparam TransA  iff true, A will be transposed on the fly
 * @tparam TransB  iff true, B will be transposed on the fly
 * @tparam element_t  the type of matrix elements
 */
template <typename input_t, typename output_t, bool DoubleBuffer, bool NbcA,
          bool NbcB, int ClSize, typename tile_type, bool TransA, bool TransB,
          typename element_t, bool is_beta_zero, int Gemm_type>
SYCL_BLAS_INLINE
Gemm<input_t, output_t, DoubleBuffer, NbcA, NbcB, ClSize, tile_type, TransA,
     TransB, element_t, is_beta_zero, Gemm_type>::
    Gemm(input_t A, input_t B, output_t C, element_t alpha, element_t beta,
         typename std::make_signed<typename input_t::index_t>::type batch_size)
    : a_(A),
      b_(B),
      c_(C),
      alpha_(alpha),
      beta_(beta),
      m_(a_.get_size_row()),
      n_(b_.get_size_col()),
      k_(a_.get_size_col()),
      lda_(a_.getSizeL()),
      ldb_(b_.getSizeL()),
      ldc_(c_.getSizeL()),
      batch_size_(batch_size) {}
template <typename input_t, typename output_t, bool DoubleBuffer, bool NbcA,
          bool NbcB, int ClSize, typename tile_type, bool TransA, bool TransB,
          typename element_t, bool is_beta_zero, int Gemm_type>
SYCL_BLAS_INLINE std::string
Gemm<input_t, output_t, DoubleBuffer, NbcA, NbcB, ClSize, tile_type, TransA,
     TransB, element_t, is_beta_zero, Gemm_type>::get_type_string() noexcept {
  std::ostringstream str{};
  str << "ReferenceGemmFactory<" << wg_size << ", "
      << type_string<value_t>::get_value() << ">";
  return str.str();
}
/*!
 *@brief gt_workgroup_cluster. This function is used to find the optimum
 *number of work_group required to execute each GEMM.
 *
 */
template <typename input_t, typename output_t, bool DoubleBuffer, bool NbcA,
          bool NbcB, int ClSize, typename tile_type, bool TransA, bool TransB,
          typename element_t, bool is_beta_zero, int Gemm_type>
SYCL_BLAS_INLINE typename Gemm<input_t, output_t, DoubleBuffer, NbcA, NbcB,
                               ClSize, tile_type, TransA, TransB, element_t,
                               is_beta_zero, Gemm_type>::index_t
Gemm<input_t, output_t, DoubleBuffer, NbcA, NbcB, ClSize, tile_type, TransA,
     TransB, element_t, is_beta_zero,
     Gemm_type>::get_workgroup_cluster(index_t m, index_t n) noexcept {
  return ((m * n - 1) / wg_size + 1);
}
/*!
 *@brief get_num_workgroup_cluster. This function is used to extend the number
 *of work_group cluster, in order to make sure that atleast 4
 *gemm operations is available per work group. The number 4
 *is used based on empirical research.
 *
 */
template <typename input_t, typename output_t, bool DoubleBuffer, bool NbcA,
          bool NbcB, int ClSize, typename tile_type, bool TransA, bool TransB,
          typename element_t, bool is_beta_zero, int Gemm_type>
SYCL_BLAS_INLINE typename Gemm<input_t, output_t, DoubleBuffer, NbcA, NbcB,
                               ClSize, tile_type, TransA, TransB, element_t,
                               is_beta_zero, Gemm_type>::index_t
Gemm<input_t, output_t, DoubleBuffer, NbcA, NbcB, ClSize, tile_type, TransA,
     TransB, element_t, is_beta_zero,
     Gemm_type>::get_num_workgroup_cluster(index_t m, index_t n,
                                           index_t compute_units) noexcept {
  constexpr index_t num_gemm_per_compute_units = 4;
  return ((num_gemm_per_compute_units * compute_units - 1) /
              Gemm<input_t, output_t, DoubleBuffer, NbcA, NbcB, ClSize,
                   tile_type, TransA, TransB, element_t, is_beta_zero,
                   Gemm_type>::get_workgroup_cluster(m, n) +
          1);
}

template <typename input_t, typename output_t, bool DoubleBuffer, bool NbcA,
          bool NbcB, int ClSize, typename tile_type, bool TransA, bool TransB,
          typename element_t, bool is_beta_zero, int Gemm_type>
SYCL_BLAS_INLINE cl::sycl::nd_range<1>
Gemm<input_t, output_t, DoubleBuffer, NbcA, NbcB, ClSize, tile_type, TransA,
     TransB, element_t, is_beta_zero,
     Gemm_type>::get_nd_range(index_t m, index_t n,
                              index_t compute_units) noexcept {
  const cl::sycl::range<1> nwg(
      Gemm<input_t, output_t, DoubleBuffer, NbcA, NbcB, ClSize, tile_type,
           TransA, TransB, element_t, is_beta_zero,
           Gemm_type>::get_workgroup_cluster(m, n) *
      Gemm<input_t, output_t, DoubleBuffer, NbcA, NbcB, ClSize, tile_type,
           TransA, TransB, element_t, is_beta_zero,
           Gemm_type>::get_num_workgroup_cluster(m, n, compute_units));
  const cl::sycl::range<1> wgs(wg_size);
  return cl::sycl::nd_range<1>(nwg * wgs, wgs);
}
template <typename input_t, typename output_t, bool DoubleBuffer, bool NbcA,
          bool NbcB, int ClSize, typename tile_type, bool TransA, bool TransB,
          typename element_t, bool is_beta_zero, int Gemm_type>
SYCL_BLAS_INLINE typename Gemm<input_t, output_t, DoubleBuffer, NbcA, NbcB,
                               ClSize, tile_type, TransA, TransB, element_t,
                               is_beta_zero, Gemm_type>::index_t
Gemm<input_t, output_t, DoubleBuffer, NbcA, NbcB, ClSize, tile_type, TransA,
     TransB, element_t, is_beta_zero, Gemm_type>::get_size() const {
  return m_ * n_;
}

template <typename input_t, typename output_t, bool DoubleBuffer, bool NbcA,
          bool NbcB, int ClSize, typename tile_type, bool TransA, bool TransB,
          typename element_t, bool is_beta_zero, int Gemm_type>
SYCL_BLAS_INLINE bool Gemm<input_t, output_t, DoubleBuffer, NbcA, NbcB, ClSize,
                           tile_type, TransA, TransB, element_t, is_beta_zero,
                           Gemm_type>::valid_thread(cl::sycl::nd_item<1> ndItem)
    const {
  return true;
}

template <typename input_t, typename output_t, bool DoubleBuffer, bool NbcA,
          bool NbcB, int ClSize, typename tile_type, bool TransA, bool TransB,
          typename element_t, bool is_beta_zero, int Gemm_type>
SYCL_BLAS_INLINE void Gemm<input_t, output_t, DoubleBuffer, NbcA, NbcB, ClSize,
                           tile_type, TransA, TransB, element_t, is_beta_zero,
                           Gemm_type>::eval(cl::sycl::nd_item<1> id) noexcept {
  const index_t wg_batch_id = id.get_group(0) / get_workgroup_cluster(m_, n_);
  // This will disable all workgroups that dont have any batch to work on
  if (wg_batch_id >= batch_size_) {
    return;
  }
  const index_t batch_stride =
      id.get_group_range(0) / get_workgroup_cluster(m_, n_);

  const index_t a_size = trans_a ? m_ * lda_ : k_ * lda_;
  const index_t b_size = trans_b ? ldb_ * k_ : n_ * ldb_;
  const index_t c_size = ldc_ * n_;

  auto orig_A = a_.get_data().get_pointer().get() +
                a_.get_access_displacement() + (wg_batch_id * a_size);
  auto orig_B = b_.get_data().get_pointer().get() +
                b_.get_access_displacement() + (wg_batch_id * b_size);
  auto orig_C = c_.get_data().get_pointer().get() +
                c_.get_access_displacement() + (wg_batch_id * c_size);

  index_t item_id = (id.get_group(0) % get_workgroup_cluster(m_, n_)) *
                        (id.get_local_range(0)) +
                    id.get_local_id(0);
  if (item_id >= m_ * n_) {
    return;
  }

  const index_t row = item_id % m_;
  const index_t col = item_id / m_;

  orig_A = orig_A + row * (trans_a ? lda_ : 1);
  orig_B = orig_B + col * (trans_b ? 1 : ldb_);
  orig_C = orig_C + row + col * ldc_;

  do {
    auto A = orig_A;
    auto B = orig_B;
    auto C = orig_C;
    value_t reg_res = {};
    while (k_ > 0) {
      reg_res = cl::sycl::mad(A[0], B[0], reg_res);
      --k_;
      A = A + (trans_a ? 1 : lda_);
      B = B + (trans_b ? ldb_ : 1);
    }
    // when C is uninitialized the element of the C can be NaN, and Nan*0
    // will be NaN
    if (is_beta_zero) {
      C[0] = alpha_ * reg_res;
    } else {
      C[0] = alpha_ * reg_res + beta_ * C[0];
    }

    orig_A += (a_size * batch_stride);
    orig_B += (b_size * batch_stride);
    orig_C += (c_size * batch_stride);
    k_ = a_.get_size_col();
    // batch_size_ must be signed as the negative value has meaning here.
    batch_size_ -= batch_stride;
  } while (batch_size_ > wg_batch_id);
}

template <typename input_t, typename output_t, bool DoubleBuffer, bool NbcA,
          bool NbcB, int ClSize, typename tile_type, bool TransA, bool TransB,
          typename element_t, bool is_beta_zero, int Gemm_type>
SYCL_BLAS_INLINE void
Gemm<input_t, output_t, DoubleBuffer, NbcA, NbcB, ClSize, tile_type, TransA,
     TransB, element_t, is_beta_zero, Gemm_type>::bind(cl::sycl::handler &h) {
  a_.bind(h);
  b_.bind(h);
  c_.bind(h);
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

/*!
 * @brief NoLocalGemmFactory is a template class whose instantiations provide
 *        different implementations of the GEMM kernel where the is no
 * local memory available on the device.
 *
 * To use the function, each item of a kernel dispatched with an nd_range given
 * by NoLocalGemmFactory::get_nd_range() should call eval().
 *
 * @tparam ClSize  the size of the cache line of the architecture
 *                 This parameter has been reserved for further optimisation
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
 */
template <typename input_t, typename output_t, bool DoubleBuffer, bool NbcA,
          bool NbcB, int ClSize, typename tile_type, bool TransA, bool TransB,
          typename element_t, bool is_beta_zero>
class Gemm<input_t, output_t, DoubleBuffer, NbcA, NbcB, ClSize, tile_type,
           TransA, TransB, element_t, is_beta_zero,
           static_cast<int>(Gemm_t::no_local_memory)> {
 public:
  using value_t = element_t;
  using index_t = typename std::make_signed<typename input_t::index_t>::type;
  static constexpr int type = static_cast<int>(Gemm_t::no_local_memory);
  static constexpr int local_memory_size = 0;
  /*! @brief The number of rows processed by each work item */
  static constexpr index_t item_rows = tile_type::item_rows;
  /*! @brief The number of cols processed by each work item */
  static constexpr index_t item_cols = tile_type::item_cols;
  /*! @brief The number of work items in each row of work group */
  static constexpr index_t wg_rows = tile_type::wg_rows;
  /*! @brief The number of work items in each column of work group */
  static constexpr index_t wg_cols = tile_type::wg_cols;
  /*! @brief Number of rows within a work-group level tile */
  static constexpr index_t block_rows = wg_rows * item_rows;
  /*! @brief Number of columns within a work-group level tile */
  static constexpr index_t block_cols = wg_cols * item_cols;
  /*! @brief The size of tile processed by a work-group */
  static constexpr index_t tile_size = block_rows * block_cols;
  /*! @brief A boolean parameter represents wheather or not matrix A is
   * transposed */
  static constexpr bool trans_a = TransA;
  /*! @brief A boolean parameter represents wheather or not matrix B is
   * transposed */
  static constexpr bool trans_b = TransB;

  static_assert(wg_cols * item_cols == item_rows * wg_rows,
                "Work group size should be a multiple "
                "of the number of rows in a block\n"
                " --- this is ensured iff: item_rows | wg_cols");

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
  SYCL_BLAS_INLINE Gemm(input_t A, input_t B, output_t C, element_t alpha,
                        element_t beta, index_t batch_size)
      : a_(A),
        b_(B),
        c_(C),
        alpha_(alpha),
        beta_(beta),
        m_(a_.get_size_row()),
        n_(b_.get_size_col()),
        k_(a_.get_size_col()),
        lda_(a_.getSizeL()),
        ldb_(b_.getSizeL()),
        ldc_(c_.getSizeL()),
        batch_size_(batch_size) {}

  /*!
   * @brief Get the type of this NoLocalGemmFactory as a human readable string.
   */
  static SYCL_BLAS_INLINE std::string get_type_string() noexcept {
    std::ostringstream str{};
    str << "NoLocalGemmFactory<" << ClSize << ", "
        << tile_type::get_type_string() << ", "
        << type_string<value_t>::get_value() << ">";
    return str.str();
  }
  /*!
   *@brief gt_workgroup_cluster. This function is used to find the optimum
   *number of work_group required to execute each GEMM.
   *
   */
  static SYCL_BLAS_INLINE index_t get_workgroup_cluster(index_t m,
                                                        index_t n) noexcept {
    return (((m - 1) / (item_rows * wg_rows) + 1) *
            ((n - 1) / (item_cols * wg_cols) + 1));
  }
  /*!
   *@brief get_num_workgroup_cluster. This function is used to extend the number
   *of work_group cluster, in order to make sure that atleast 4 gemm operations
   *is available per work group. The number 4 is used based on empirical
   *research.
   *
   */
  static SYCL_BLAS_INLINE index_t get_num_workgroup_cluster(
      index_t m, index_t n, index_t compute_units) noexcept {
    constexpr index_t num_gemm_per_compute_units = 4;
    return ((num_gemm_per_compute_units * compute_units - 1) /
                get_workgroup_cluster(m, n) +
            1);
  }

  static SYCL_BLAS_INLINE cl::sycl::nd_range<1> get_nd_range(
      index_t m, index_t n, index_t compute_units) noexcept {
    const cl::sycl::range<1> nwg(
        get_workgroup_cluster(m, n) *
        get_num_workgroup_cluster(m, n, compute_units));
    const cl::sycl::range<1> wgs(wg_rows * wg_cols);

    return cl::sycl::nd_range<1>(nwg * wgs, wgs);
  }

  SYCL_BLAS_INLINE index_t get_size() const { return m_ * n_; }

  SYCL_BLAS_INLINE bool valid_thread(cl::sycl::nd_item<1> ndItem) const {
    return true;
  }

  SYCL_BLAS_INLINE void eval(cl::sycl::nd_item<1> id) noexcept {
    // The batch index that each workgroup should start working with
    const index_t wg_batch_id = id.get_group(0) / get_workgroup_cluster(m_, n_);
    // This will disable all workgroups that dont have any batch to work on
    if (wg_batch_id >= batch_size_) {
      return;
    }

    const index_t batch_stride =
        id.get_group_range(0) / get_workgroup_cluster(m_, n_);

    const index_t a_size = trans_a ? m_ * lda_ : k_ * lda_;
    const index_t b_size = trans_b ? ldb_ * k_ : n_ * ldb_;
    const index_t c_size = ldc_ * n_;

    auto orig_A = a_.get_data().get_pointer().get() +
                  a_.get_access_displacement() + (wg_batch_id * a_size);
    auto orig_B = b_.get_data().get_pointer().get() +
                  b_.get_access_displacement() + (wg_batch_id * b_size);
    auto orig_C = c_.get_data().get_pointer().get() +
                  c_.get_access_displacement() + (wg_batch_id * c_size);

    const index_t number_of_block_per_row = ((m_ - 1) / block_rows) + 1;
    /* linear work group id The number of work-group required to executed each
     * batch efficiently*/
    const index_t wg_id = id.get_group(0) % get_workgroup_cluster(m_, n_);
    /*linear work item id*/
    const index_t item_id = id.get_local_id(0);
    /* row tile id  per work group */
    const index_t tile_id_row = wg_id % number_of_block_per_row;
    /* column tile id per work group */
    const index_t tile_id_col = wg_id / number_of_block_per_row;
    /* work item id per row */
    const index_t local_item_id_row = item_id % wg_rows;
    /* work item id per column */
    const index_t local_item_id_col = item_id / wg_rows;
    /* the start position of the tile-row per work group */
    const index_t wg_row = tile_id_row * block_rows;
    /* the start position of the tile-column per work group */
    const index_t wg_col = tile_id_col * block_cols;

    /* Exiting from any threads outside of the m and n boundary */
    const bool out_of_range = ((local_item_id_row + wg_row >= m_) ||
                               (local_item_id_col + wg_col >= n_));
    /*
     * The ma and na are used to adjust the start position of each work-item for
     * A, B and C matrices.
     */
    const index_t dim_m_a_start = (local_item_id_row + wg_row);
    const index_t dim_n_b_start = (local_item_id_col + wg_col);

    /*! @brief Adjusting the start position of A, B , and C */
    orig_A += dim_m_a_start * (trans_a ? lda_ : 1);
    orig_B += dim_n_b_start * (trans_b ? 1 : ldb_);
    orig_C += dim_m_a_start + (dim_n_b_start * ldc_);

    /*!
     * @brief is_internal_block_m and is_internal_block_n is used to distinguish
     * the internal block. Therefore, work items using these blocks dont need to
     * check for boundaries.
     */
    const bool is_internal_block =
        (m_ - wg_row >= block_rows) && (n_ - wg_col >= block_cols);

    /*
     * The following lambdas: boundary_check_m, boundary_check_n, and
     * boundary_check_c  are used to check the A, B , and C boundaries
     * respectively.
     */
    const auto boundary_check_m = [&](index_t dim_m_a_start) {
      return dim_m_a_start < m_;
    };
    const auto boundary_check_n = [&](index_t dim_n_b_start) {
      return dim_n_b_start < n_;
    };
    const auto boundary_check_c = [&](index_t dim_m_c_start,
                                      index_t dim_n_c_start) {
      return (dim_m_c_start < m_ && dim_n_c_start < n_);
    };

    // computing the next element for a and b;
    const index_t A_ptr_index = (trans_a ? lda_ : 1) * wg_rows;
    const index_t B_ptr_index = (trans_b ? 1 : ldb_) * wg_cols;
    /* temporary register array used to prefetch columns of A*/
    value_t reg_a[item_rows];
    /* temporary register used to prefetch elements of B*/
    value_t reg_b[item_cols];
    /*
     * computing the gemm panel
     */
    if ((is_internal_block == true)) {
      compute_gemm_no_shared_pannel<false>(
          orig_A, orig_B, orig_C, a_size, b_size, c_size, a_.get_size_col(), k_,
          dim_m_a_start, dim_n_b_start, A_ptr_index, B_ptr_index,
          boundary_check_m, boundary_check_n, boundary_check_c, reg_a, reg_b,
          out_of_range, batch_stride, wg_batch_id, batch_size_, lda_, ldb_,
          ldc_, alpha_, beta_
#ifdef ARM_GPU
          ,
          id
#endif
      );
    } else {
      compute_gemm_no_shared_pannel<true>(
          orig_A, orig_B, orig_C, a_size, b_size, c_size, a_.get_size_col(), k_,
          dim_m_a_start, dim_n_b_start, A_ptr_index, B_ptr_index,
          boundary_check_m, boundary_check_n, boundary_check_c, reg_a, reg_b,
          out_of_range, batch_stride, wg_batch_id, batch_size_, lda_, ldb_,
          ldc_, alpha_, beta_
#ifdef ARM_GPU
          ,
          id
#endif
      );
    }
  }
  template <bool need_check_boundary, typename A_t, typename B_t, typename C_t,
            typename check_boundary_m_t, typename check_boundary_n_t,
            typename check_boundary_c_t>
  static void SYCL_BLAS_INLINE compute_gemm_no_shared_pannel(
      A_t orig_A, B_t orig_B, C_t orig_C, const index_t &a_size,
      const index_t &b_size, const index_t &c_size, index_t orig_k, index_t k,
      const index_t &dim_m_a_start, const index_t &dim_n_b_start,
      const index_t &A_ptr_index, const index_t &B_ptr_index,
      const check_boundary_m_t &boundary_check_m,
      const check_boundary_n_t &boundary_check_n,
      const check_boundary_c_t &boundary_check_c, element_t (&reg_a)[item_rows],
      element_t (&reg_b)[item_cols], const bool out_of_range,
      const index_t &batch_stride, const index_t &wg_batch_id,
      index_t batch_size, const index_t &lda, const index_t &ldb,
      const index_t &ldc, const element_t &alpha, const element_t &beta
#ifdef ARM_GPU
      ,
      cl::sycl::nd_item<1> id
#endif
      ) noexcept {
    do {
      auto A = orig_A;
      auto B = orig_B;
      auto C = orig_C;

      /* 2D register array used to store the result C*/
      value_t reg_res[item_rows][item_cols] = {};
      while (k > 0) {
        /*
         * Loading a corresponding block of matrix A into reg_a
         */
        load<item_rows, wg_rows, need_check_boundary>(
            A, reg_a, A_ptr_index, dim_m_a_start, boundary_check_m,
            out_of_range);
#ifdef ARM_GPU
        id.barrier(cl::sycl::access::fence_space::local_space);
#endif
        /*
         * Loading a corresponding block of matrix B into reg_b
         */
        load<item_cols, wg_cols, need_check_boundary>(
            B, reg_b, B_ptr_index, dim_n_b_start, boundary_check_n,
            out_of_range);

        /*
         * Computing a the partial GEMM for the loaded block of reg_a andd
         * reg_b and adding the result into reg_res
         */
        compute_block_gemm_no_shared(reg_a, reg_b, reg_res);
        /*
         * Moving forward to the next block
         */
        --k;
        A = A + (trans_a ? 1 : lda);
        B = B + (trans_b ? ldb : 1);
      }
      /*
       *  Storing the reg_res into C matrix
       */
      store<need_check_boundary>(C, reg_res, alpha, beta, dim_m_a_start,
                                 dim_n_b_start, boundary_check_c, out_of_range,
                                 ldc);

      orig_A += (a_size * batch_stride);
      orig_B += (b_size * batch_stride);
      orig_C += (c_size * batch_stride);
      k = orig_k;
      // batch_size_ must be signed as the negative value has meaning here.
      batch_size -= batch_stride;
    } while (batch_size > wg_batch_id);
  }
  /*!
   * @brief binding the placeholder accessors to the SYCL command group
   * handler
   * @param h: SYCL command group handler. */
  void bind(cl::sycl::handler &h) {
    a_.bind(h);
    b_.bind(h);
    c_.bind(h);
  }

 private:
  /*!
   * @brief Following function load a block of row_items/col_items elements from
   * A/B matrix into reg_a/reg_b.
   * @tparam item_size it is the size of private register: either row_items or
   * column_item
   * @tparam next_element : is the stride to acces the next element of A or B
   * matrix. it is either wg_rows or wg_cols.
   * @tparam check_block: determined whether or not the requested block is
   * internal. false means no need to check the boundaries
   * @tparam pointerType: the type of the input matrix
   * @tparam check_boundary: the type of a function used for checking the
   * boundary for blocks of data located at the edge of the input matrix
   * @param ptr : the input matrix, either A or B.
   * @param reg[item_size] the private array containing the input block per
   * work-item: it is either reg_a or reg_b.
   * @param ld : the leading dimension of the input matrix.
   * @param index: the start position of the block of data to be loaded.
   * @param chk_boundary: an instance of the check_boundary function
   */

  template <index_t item_size, index_t next_element, bool check_block,
            typename PointerType, typename check_boundary>
  static SYCL_BLAS_INLINE void load(PointerType ptr,
                                    element_t (&reg)[item_size],
                                    const index_t &ld, index_t index,
                                    const check_boundary &chk_boundary,
                                    const bool out_of_range) noexcept {
    if (out_of_range) {
      return;
    }
#pragma unroll
    for (int i = 0; i < item_size; i++) {
      reg[i] =
          do_check<check_block>(chk_boundary(index)) ? ptr[0] : element_t(0);
      ptr += ld;
      index += next_element;
    }
  }

  /*!
   * @brief The following function compute the partial GEMM for the input block
   * reg_a and reg_b and add the result to the reg_res
   * @param reg_a  temporary register array used to prefetch columns of A
   * @param reg_b  temporary register used to prefetch elements of B
   * @param reg_res  2D register array used to store the result C
   */
  static SYCL_BLAS_INLINE void compute_block_gemm_no_shared(
      element_t (&reg_a)[item_rows], element_t (&reg_b)[item_cols],
      element_t (&reg_res)[item_rows][item_cols]) noexcept {
#pragma unroll
    for (int j = 0; j < item_cols; j++) {
#pragma unroll
      for (int i = 0; i < item_rows; i++) {
        reg_res[i][j] = cl::sycl::mad(reg_a[i], reg_b[j], reg_res[i][j]);
      }
    }
  }

  /*!
   * @brief For each work itemThe following function store the computed block of
   * GEMM reg_res into output matrix C
   * @tparam check_block: determined whether or not the requested block is
   * internal. false means no need to check the boundaries
   * @tparam pointerType: the type of the matrix C
   * @tparam check_boundary: the type of a function used for checking the
   * boundary for blocks of data located at the edge of the input matrix
   * @param c: is the output matrix C
   * @param reg_res  2D register array used to store the result C
   * @param chk_boundary: an instance of the check_boundary function
   * @param ldc is the leading dimension of C
   * @param mc and nc are indices, used to check the boundary of C
   */
  template <bool check_block, typename PointerType, typename check_boundary>
  static SYCL_BLAS_INLINE void store(
      PointerType C, element_t (&reg_res)[item_rows][item_cols],
      const element_t &alpha, const element_t &beta,
      const index_t &dim_m_c_start, const index_t &dim_n_c_start,
      const check_boundary &chk_boundary, const bool out_of_range,
      const index_t &ldc) noexcept {
    if (out_of_range) {
      return;
    }
#pragma unroll
    for (int j = 0; j < item_cols; j++) {
#pragma unroll
      for (int i = 0; i < item_rows; i++) {
        if (do_check<check_block>(chk_boundary(dim_m_c_start + i * wg_rows,
                                               dim_n_c_start + j * wg_cols))) {
          // when C is uninitialized the element of the C can be NaN, and Nan*0
          // will be NaN
          if (is_beta_zero) {
            C[i * wg_rows] = alpha * reg_res[i][j];
          } else {
            C[i * wg_rows] = alpha * reg_res[i][j] + beta * C[i * wg_rows];
          }
        }
      }
      C = C + (wg_cols * ldc);
    }
  }
};  // end class No Local GemmFactory

/*!
 * @brief GemmFactory is a template class whose instantiations provide
 *        different implementations of the GEMM device function.
 *
 * To use the function, each item of a kernel launched with nd_range given by
 * GemmFactory::get_nd_range() should call GemmFactory::run(). The size of
 * local memory required per work group can be queried with
 * GemmFactory::local_memory.
 *
 * @tparam DoubleBuffer  iff true,  enables the use of double buffering
 *                       (doubles the amount of consumed local memory,
 *                        but halves the number of required local barriers)
 * @tparam NbcA  iff true, avoids bank conflicts when accessing blocks of
 *               matrix A in local memory (slightly increases local
 *               memory consumption) - may be useful in combination with TranA
 * @tparam NbcA  iff true, avoids bank conflicts when accessing blocks of
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
 */
template <typename input_t, typename output_t, bool DoubleBuffer, bool NbcA,
          bool NbcB, int ClSize, typename TileType, bool TransA, bool TransB,
          typename element_t, bool is_beta_zero>
class Gemm<input_t, output_t, DoubleBuffer, NbcA, NbcB, ClSize, TileType,
           TransA, TransB, element_t, is_beta_zero,
           static_cast<int>(Gemm_t::local_memory)> {
 public:
  using tile_type = TileType;
  using value_t = element_t;
  using index_t = typename std::make_signed<typename input_t::index_t>::type;
  using local_memory_t =
      cl::sycl::accessor<element_t, 1, cl::sycl::access::mode::read_write,
                         cl::sycl::access::target::local>;

  static constexpr int type = static_cast<int>(Gemm_t::local_memory);

  // enable easier access to tile dimensions
  static constexpr index_t item_rows = tile_type::item_rows;
  static constexpr index_t item_cols = tile_type::item_cols;
  static constexpr index_t wg_rows = tile_type::wg_rows;
  static constexpr index_t wg_cols = tile_type::wg_cols;
  static constexpr index_t tl_rows = tile_type::tl_rows;
  static constexpr index_t tl_cols = tile_type::tl_cols;

  static constexpr bool double_buffer = DoubleBuffer;
  static constexpr bool nbc_a = NbcA;
  static constexpr bool nbc_b = NbcB;
  static constexpr bool trans_a = TransA;
  static constexpr bool trans_b = TransB;

  static constexpr index_t cl_size = ClSize;
  //! @brief Number of elements which fit within a cache line.
  static constexpr index_t cl_elems = cl_size / sizeof(element_t);
  //! @brief Number of work items within a work group
  static constexpr index_t wg_size = wg_rows * wg_cols;
  //! @brief Number of rows within a work-group level tile
  static constexpr index_t block_rows = wg_rows * item_rows;
  //! @brief Number of columns within a work-group level tile
  static constexpr index_t block_cols = wg_cols * item_cols;
  //! @brief Number of rows within a top-level tile
  static constexpr index_t big_tile_rows = tl_rows * block_rows;
  //! @brief Number of columns within a top-level tile
  static constexpr index_t big_tile_cols = tl_cols * block_cols;

  static_assert(wg_size % cl_elems == 0,
                "Work group size should be a multiple "
                "of elements in a cache line\n"
                " --- this is ensured iff:"
                " cl_size | sizeof(element_t) * wg_rows * wg_cols");

  static_assert(wg_size % block_rows == 0,
                "Work group size should be a multiple "
                "of the number of rows in a block\n"
                " --- this is ensured iff: item_rows | wg_cols");

  static_assert(wg_size % block_cols == 0,
                "Work group size should be a multiple "
                "of the number of columns in a block\n"
                " --- this is ensured iff: item_cols | wg_rows");

  //! @brief leading dimension of block of A in local
  static constexpr index_t ldsa = block_rows + nbc_a;
  //! @brief leading dimension of block of B in local
  static constexpr index_t ldsb = cl_elems + nbc_b;
  //! @brief size (in elements) of local (local) memory required by each
  //         work group
  static constexpr index_t local_memory_size =
      (double_buffer + 1) * (ldsa * cl_elems + ldsb * block_cols);

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

  SYCL_BLAS_INLINE Gemm(input_t A, input_t B, output_t C, element_t alpha,
                        element_t beta, index_t batch_size)
      : a_(A),
        b_(B),
        c_(C),
        alpha_(alpha),
        beta_(beta),
        m_(a_.get_size_row()),
        n_(b_.get_size_col()),
        k_(a_.get_size_col()),
        lda_(a_.getSizeL()),
        ldb_(b_.getSizeL()),
        ldc_(c_.getSizeL()),
        batch_size_(batch_size) {}

  /*!
   * @brief Get the type of this GemmFactory as a human readable string.
   */
  static SYCL_BLAS_INLINE std::string get_type_string() noexcept {
    std::ostringstream str{};
    str << "GemmFactory<" << double_buffer << ", " << nbc_a << ", " << nbc_b
        << ", " << cl_size << ", " << tile_type::get_type_string() << ", "
        << type_string<value_t>::get_value() << ">";
    return str.str();
  }

  /*!
   *@brief gt_workgroup_cluster. This function is used to find the optimum
   *number of work_group required to execute each GEMM.
   *
   */
  static SYCL_BLAS_INLINE index_t get_workgroup_cluster(index_t m,
                                                        index_t n) noexcept {
    return (((m - 1) / big_tile_rows + 1) * ((n - 1) / big_tile_cols + 1) *
            tl_rows * tl_cols);
  }
  /*!
   *@brief get_num_workgroup_cluster. This function is used to extend the number
   *of work_group cluster, in order to make sure that atleast 4 gemm operations
   *is available per work group. The number 4 is used based on empirical
   *research.
   *
   */
  static SYCL_BLAS_INLINE index_t get_num_workgroup_cluster(
      index_t m, index_t n, index_t compute_units) noexcept {
    return ((4 * compute_units - 1) / get_workgroup_cluster(m, n) + 1);
  }

  /*!
   * @brief Get the nd_range value which has to be used for kernels that
   *        intend to call GemmFactory::run().
   *
   * @note This requirement can be alleviated a bit, by calling multiple
   * instances of GemmFactory::run() from a single work-group, but with a
   * different wg_id parameter (the only requirement is that GemmFactory::run()
   * is called with a full set of wg_id values. Similarly, the kernel can be
   * invoked with a larger local range, and mapping each large physical work
   * group to multiple work groups with size as expected by GemmFactory::run().
   * (This is done by manipulating wg_id and item_id parameters.)
   */
  static SYCL_BLAS_INLINE cl::sycl::nd_range<1> get_nd_range(
      index_t m, index_t n, index_t compute_units) noexcept {
    const cl::sycl::range<1> nwg(
        get_workgroup_cluster(m, n) *
        get_num_workgroup_cluster(m, n, compute_units));
    const cl::sycl::range<1> wgs(wg_size);
#ifdef VERBOSE
    std::cout << " M: " << m << " , N " << n
              << " , big_tile_rows: " << big_tile_rows
              << " , big_tile_cols: " << big_tile_cols
              << " , wg_size: " << wg_size << " , nwg : "
              << ((m - 1) / big_tile_rows + 1) * ((n - 1) / big_tile_cols + 1) *
                     tl_rows * tl_cols
              << std::endl;
#endif
    return cl::sycl::nd_range<1>(nwg * wgs, wgs);
  }

  SYCL_BLAS_INLINE index_t get_size() const { return m_ * n_; }

  /*!
   * @brief Run the generated GEMM device function.
   * @tparam local_memory_t LocalMemory type
   * @param id  nd_item used for calls to local barriers
   * @param scratch local memory
   */
  template <typename local_memory_t>
  SYCL_BLAS_INLINE void eval(local_memory_t scratch_acc,
                             cl::sycl::nd_item<1> id) noexcept {
    // The batch index that each workgroup should start working with
    const index_t wg_batch_id = id.get_group(0) / get_workgroup_cluster(m_, n_);
    // This will disable all workgroups that dont have any batch to work on
    if (wg_batch_id >= batch_size_) {
      return;
    }
    const index_t batch_stride =
        id.get_group_range(0) / get_workgroup_cluster(m_, n_);

    auto scratch = scratch_acc.localAcc.get_pointer().get();
    using ScratchPointerType = decltype(scratch);
    // The number of work-group required to executed each batch efficiently
    const index_t wg_id = id.get_group(0) % get_workgroup_cluster(m_, n_);

    const index_t a_size = trans_a ? m_ * lda_ : k_ * lda_;
    const index_t b_size = trans_b ? ldb_ * k_ : n_ * ldb_;
    const index_t c_size = ldc_ * n_;
    auto orig_A = a_.get_data().get_pointer().get() +
                  a_.get_access_displacement() + (wg_batch_id * a_size);
    auto orig_B = b_.get_data().get_pointer().get() +
                  b_.get_access_displacement() + (wg_batch_id * b_size);
    auto orig_C = c_.get_data().get_pointer().get() +
                  c_.get_access_displacement() + (wg_batch_id * c_size);
    const index_t item_id = id.get_local_id(0);
    const index_t tile_size = tl_rows * tl_cols;
    const index_t tile_id = wg_id / tile_size;
    const index_t tile_local_id = wg_id % tile_size;
    const index_t tiles_per_col = (m_ - 1) / big_tile_rows + 1;
    const index_t tile_row = (tile_id % tiles_per_col) * tl_rows;
    const index_t tile_col = (tile_id / tiles_per_col) * tl_cols;
    const index_t wg_row = (tile_row + tile_local_id % tl_rows) * block_rows;
    const index_t wg_col = (tile_col + tile_local_id / tl_rows) * block_rows;
    const bool out_of_range = (wg_row >= m_ || wg_col >= n_);
    const index_t item_row = item_id % wg_rows;
    const index_t item_col = (item_id / wg_rows) * item_cols;
    const index_t row = wg_row + item_row;
    const index_t col = wg_col + item_col;

    element_t reg_a[item_rows];
    element_t reg_b;

    orig_C = orig_C + row + col * ldc_;
    const index_t mc = m_ - row;
    const index_t nc = n_ - col;

    const bool internal =
        m_ - wg_row >= block_rows && n_ - wg_col >= block_cols;
    orig_B =
        orig_B +
        (trans_b
             ? (item_id / block_cols) * ldb_ + (wg_col + item_id % block_cols)
             : item_id % cl_elems + (wg_col + item_id / cl_elems) * ldb_);
    n_ = n_ - wg_col - (trans_b ? item_id % block_cols : item_id / cl_elems);
    orig_A =
        orig_A +
        (trans_a
             ? (wg_row + item_id / cl_elems) * lda_ + (item_id % cl_elems)
             : (wg_row + item_id % block_rows) + (item_id / block_rows) * lda_);
    m_ = m_ - wg_row - (trans_a ? item_id / cl_elems : item_id % block_rows);

    ScratchPointerType s1 =
        scratch + (trans_b
                       ? item_id / block_cols + (item_id % block_cols) * ldsb
                       : item_id % cl_elems + (item_id / cl_elems) * ldsb);
    ScratchPointerType s2 = scratch + item_col * ldsb;
    const index_t ofs = (double_buffer + 1) * block_cols * ldsb;
    ScratchPointerType s3 =
        scratch + ofs +
        (trans_a ? item_id / cl_elems + (item_id % cl_elems) * ldsa
                 : item_id % block_rows + (item_id / block_rows) * ldsa);
    ScratchPointerType s4 = scratch + ofs + item_row;

    if (internal) {
      compute_panel_gemm<double_buffer, false, false>(
          id, item_id, m_, mc, n_, nc, a_.get_size_col(), k_, a_size, b_size,
          c_size, alpha_, orig_A, lda_, orig_B, ldb_, beta_, orig_C, ldc_, s1,
          s2, s3, s4, reg_a, reg_b, out_of_range, batch_stride, wg_batch_id,
          batch_size_);
    } else {
      compute_panel_gemm<double_buffer, true, true>(
          id, item_id, m_, mc, n_, nc, a_.get_size_col(), k_, a_size, b_size,
          c_size, alpha_, orig_A, lda_, orig_B, ldb_, beta_, orig_C, ldc_, s1,
          s2, s3, s4, reg_a, reg_b, out_of_range, batch_stride, wg_batch_id,
          batch_size_);
    }
  }

  void bind(cl::sycl::handler &h) {
    a_.bind(h);
    b_.bind(h);
    c_.bind(h);
  }
  SYCL_BLAS_INLINE bool valid_thread(cl::sycl::nd_item<1> ndItem) const {
    return true;
  }

 private:
  /*!
   * @brief Compute a GEMM of a block-row of A (transpose) and a block-column
   *        of B (transpose).
   *
   * This is a collective operation between all items in as work-group.
   * This method should actually be a generic lambda (as in C++20), it only
   * forwards parameters from GemmFactory::run().
   *
   * @tparam check_m_limit  iff true, check if row indexes of C are
   *                        out-of-bound
   * @tparam check_n_limit  iff true, check if no indexes of C are
   *                        out-of-bound
   */
  template <bool double_buffer, bool check_m_limit, bool check_n_limit,
            typename InputPointerType, typename OutputPointerType,
            typename ScratchPointerType>
  static SYCL_BLAS_INLINE void compute_panel_gemm(
      cl::sycl::nd_item<1> id, index_t item_id, index_t m, index_t mc,
      index_t n, index_t nc, index_t orig_k, index_t k, index_t a_size,
      index_t b_size, index_t c_size, element_t alpha, InputPointerType orig_A,
      index_t lda, InputPointerType orig_B, index_t ldb, element_t beta,
      OutputPointerType orig_C, index_t ldc, ScratchPointerType s1,
      ScratchPointerType s2, ScratchPointerType s3, ScratchPointerType s4,
      element_t (&reg_a)[item_rows], element_t &reg_b, const bool out_of_range,
      const index_t batch_stride, const index_t wg_batch_id,
      index_t batch_size) noexcept {
    index_t ofs = 1;
    do {
      auto A = orig_A;
      auto B = orig_B;
      auto C = orig_C;
      element_t reg_res[item_rows][item_cols] = {};
      while (k >= cl_elems) {
        extract_input_blocks<check_m_limit, check_n_limit, false>(
            item_id, m, n, k, A, lda, B, ldb, s1, s3, out_of_range);
        id.barrier(cl::sycl::access::fence_space::local_space);
        compute_block_gemm(s2, s4, reg_a, reg_b, reg_res);
        A = A + cl_elems * (trans_a ? 1 : lda);
        B = B + cl_elems * (trans_b ? ldb : 1);
        k -= cl_elems;
        sync_smem<double_buffer, block_cols * ldsb, block_cols * ldsb,
                  ldsa * cl_elems, ldsa * cl_elems>(id, ofs, s1, s2, s3, s4);
      }

      if (k > 0) {
        extract_input_blocks<check_m_limit, check_n_limit, true>(
            item_id, m, n, k, A, lda, B, ldb, s1, s3, out_of_range);
        id.barrier(cl::sycl::access::fence_space::local_space);
        compute_block_gemm(s2, s4, reg_a, reg_b, reg_res);
        sync_smem<double_buffer, block_cols * ldsb, block_cols * ldsb,
                  ldsa * cl_elems, ldsa * cl_elems>(id, ofs, s1, s2, s3, s4);
      }

      // store the output
      store_output_block<check_m_limit, check_n_limit>(
          mc, nc, alpha, beta, C, ldc, reg_res, out_of_range);
      orig_A += (a_size * batch_stride);
      orig_B += (b_size * batch_stride);
      orig_C += (c_size * batch_stride);
      k = orig_k;
      // batch_size_ must be signed as the negative value has meaning here.
      batch_size -= batch_stride;
    } while (batch_size > wg_batch_id);
  }

  /*!
   * @brief Store the computed gemm result to the C matrix
   *
   * @tparam check_m_limit  iff true, check if row indexes of C are
   *                        out-of-bound
   * @tparam check_n_limit  iff true, check if no indexes of C are
   *                        out-of-bound
   * @tparam OutputPointerType the type of C
   * @param mc the computed boundary limit of m in matrix C
   * @param nc the computed boundary limit of n in matrix C
   *  @param alpha  scaling factor of AB
   * @param beta  scaling factor of C
   * @param C  pointer to the first element of C
   * @param ldc  leading dimension of C
   * @param reg_res  2D register array containing the partial resull of C per
   * thread
   */
  template <bool check_m_limit, bool check_n_limit, typename OutputPointerType>
  static SYCL_BLAS_INLINE void store_output_block(
      index_t mc, index_t nc, element_t alpha, element_t beta,
      OutputPointerType C, index_t ldc,
      element_t (&reg_res)[item_rows][item_cols],
      const bool out_of_range) noexcept {
    if (out_of_range) {
      return;
    }
#pragma unroll
    for (index_t i = 0; i < item_cols; ++i) {
#pragma unroll
      for (index_t j = 0; j < item_rows; ++j) {
        const bool in_range = do_check<check_m_limit>(j * wg_rows < mc) &&
                              do_check<check_n_limit>(i < nc);
        if (in_range) {
          // when C is uninitialized the element of the C can be NaN, and
          // Nan*0 will be NaN
          if (is_beta_zero) {
            C[j * wg_rows] = alpha * reg_res[j][i];
          } else {
            C[j * wg_rows] = alpha * reg_res[j][i] + beta * C[j * wg_rows];
          }
        }
      }
      C = C + ldc;
    }
  }

  /*!
   * @brief Extract a block of A, and a conformant block of B.
   *
   * @see GemmFactory::extract_block()
   */
  template <bool check_m_limit, bool check_n_limit, bool check_k_limit,
            typename InputPointerType, typename ScratchPointerType>
  static SYCL_BLAS_INLINE void extract_input_blocks(
      index_t item_id, index_t m, index_t n, index_t k, InputPointerType A,
      index_t lda, InputPointerType B, index_t ldb, ScratchPointerType sB,
      ScratchPointerType sA, const bool out_of_range) noexcept {
    if (out_of_range) {
      return;
    }
    extract_block<check_m_limit, check_k_limit, trans_a, block_rows, cl_elems,
                  ldsa>(
        item_id, A, lda, sA, [&](index_t ir, index_t cr) { return cr < m; },
        [&](index_t ic, index_t cc) { return cc < k - ic; });
    extract_block<check_k_limit, check_n_limit, trans_b, cl_elems, block_cols,
                  ldsb>(
        item_id, B, ldb, sB,
        [&](index_t ir, index_t cr) { return cr < k - ir; },
        [&](index_t ic, index_t cc) { return cc < n; });
  }

  /*!
   * @brief Extract a block of a matrix from global to shared memory, and
   *        optionally transpose it on the fly.
   *
   * This is a collective operation on all items in a work group.
   *
   * @tparam check_row_limit  iff true, check the row out-of-bound condition
   * @tparam check_col_limit  iff true, check the column out-of-bound condition
   * @tparam trans  iff true, transpose the matrix
   * @tparam rows  number of rows in the block
   * @tparam cols  number of columns in the block
   * @tparam lds  leading dimension of the block in shared memory
   * @tparam InputPointerType  pointer type of the input matrix
   * @tparam ScratchPointerType  pointer type of the memory used to store the
   *                             extracted block
   * @tparam RowPredicate  row out-of-bound condition type
   * @tparam ColPredicate  column out-of-bound condition type
   *
   * @param item_id  id of the work item which called this method
   * @param ptr  pointer to the input matrix with proper item-dependent offset,
   *             see GemmFactory::run() for details
   * @param ld  the leading dimension of the input matrix
   * @param scratch  the pointer to memory where the output block is stored,
   *                 with proper item-dependent offset, see GemmFactory::run()
   *                 for details
   * @param in_row  a predicate which checks whether a row index is within
   *                matrix bounds
   * @param in_col  a predicate which checks whether a col index is within
   *                matrix bounds
   */
  template <bool check_row_limit, bool check_col_limit, bool trans,
            index_t rows, index_t cols, index_t lds, typename InputPointerType,
            typename ScratchPointerType, typename RowPredicate,
            typename ColPredicate>
  static SYCL_BLAS_INLINE typename std::enable_if<!trans>::type extract_block(
      index_t item_id, InputPointerType ptr, index_t ld,
      ScratchPointerType scratch, RowPredicate in_row, ColPredicate in_col) {
    const index_t bs = rows * cols;
#pragma unroll
    for (index_t i = 0; i < (bs - 1) / wg_size + 1; ++i) {
      if (!do_check<((bs % wg_size) != 0)>(item_id + i * wg_size < bs))
        continue;
      const index_t col_ofs = i * (wg_size / rows);
      const bool in_range =
          do_check<check_row_limit>(in_row(item_id % rows, 0)) &&
          do_check<check_col_limit>(in_col(item_id / rows, col_ofs));
      scratch[col_ofs * lds] = in_range ? ptr[col_ofs * ld] : element_t(0);
    }
  }

  template <bool check_row_limit, bool check_col_limit, bool trans,
            index_t rows, index_t cols, index_t lds, typename InputPointerType,
            typename ScratchPointerType, typename RowPredicate,
            typename ColPredicate>
  static SYCL_BLAS_INLINE typename std::enable_if<trans>::type extract_block(
      index_t item_id, InputPointerType ptr, index_t ld,
      ScratchPointerType scratch, RowPredicate in_row, ColPredicate in_col) {
    const index_t bs = rows * cols;
#pragma unroll
    for (index_t i = 0; i < (bs - 1) / wg_size + 1; ++i) {
      if (!do_check<((bs % wg_size) != 0)>(item_id + i * wg_size < bs))
        continue;
      const index_t row_ofs = i * (wg_size / cols);
      const bool in_range =
          do_check<check_row_limit>(in_row(item_id / cols, row_ofs)) &&
          do_check<check_col_limit>(in_col(item_id % cols, 0));
      scratch[row_ofs] = in_range ? ptr[row_ofs * ld] : element_t(0);
    }
  }

  /*!
   * @brief Compute a small matrix-matrix product `reg_res += A*B`.
   *
   * @tparam InputPointerType  pointer type for A and B
   *
   * @param B  pointer to matrix A with proper item-dependent offset,
   *           see GemmFactory::run() for details
   * @param A  pointer to matrix B with proper item-dependent offset,
   *           see GemmFactory::run() for details
   * @param reg_a  temporary register array used to prefetch columns of A
   * @param reg_b  temporary register used to prefetch elements of B
   * @param reg_res  2D register array used to store the result C
   */
  template <typename InputPointerType>
  static SYCL_BLAS_INLINE void compute_block_gemm(
      InputPointerType B, InputPointerType A, element_t (&reg_a)[item_rows],
      element_t &reg_b, element_t (&reg_res)[item_rows][item_cols]) noexcept {
    // NOTE: Adding "#pragma unroll" here reduces performance on AMD R9 Nano.
    //       Seems that the small reduction of arithmetic operations does not
    //       amortize the cost of loading the larger kernel binary resulting
    //       from loop unrollment.
    for (index_t i = 0; i < cl_elems; ++i) {
#pragma unroll
      for (index_t j = 0; j < item_rows; ++j) {
        reg_a[j] = A[j * wg_rows];
      }
#pragma unroll
      for (index_t j = 0; j < item_cols; ++j) {
        reg_b = B[j * ldsb];
#pragma unroll
        for (index_t l = 0; l < item_rows; ++l) {
          reg_res[l][j] = cl::sycl::mad(reg_a[l], reg_b, reg_res[l][j]);
        }
      }
      A = A + ldsa;
      B = B + 1;
    }
  }

  /*!
   * @brief Synchronize multiple shared memory blocks using a barrier or
   *        double buffering.
   *
   * @tparam db  if true, use double buffering, otherwise use barrier
   * @tparam o  size of the first memory block
   * @tparam os  sizes of other memory blocks
   * @tparam P  type of first memory block
   * @tparam Ps  types of other memory blocks
   *
   * @param id  nd_item used to call barrier sync
   * @param ofs_sign  if 1, use next block for double buffering, if -1 use
   *                  previous block
   * @param s  pointer to first memory block
   * @param ss  pointers to other memory blocks
   */
  template <bool db, index_t o, index_t... os, typename P, typename... Ps>
  static SYCL_BLAS_INLINE typename std::enable_if<db>::type sync_smem(
      cl::sycl::nd_item<1> id, index_t &ofs_sign, P &s, Ps &... ss) noexcept {
    s = s + ofs_sign * o;
    sync_smem<db, os...>(id, ofs_sign, ss...);
  }

  template <bool db>
  static SYCL_BLAS_INLINE typename std::enable_if<db>::type sync_smem(
      cl::sycl::nd_item<1>, index_t &ofs_sign) noexcept {
    ofs_sign = -ofs_sign;
  }

  template <bool db, index_t..., typename... Ps>
  static SYCL_BLAS_INLINE typename std::enable_if<!db>::type sync_smem(
      cl::sycl::nd_item<1> id, index_t &, Ps &...) noexcept {
    id.barrier(cl::sycl::access::fence_space::local_space);
  }
};  // namespace blas

}  // namespace blas

#endif  // BLAS3_TREES_GEMM_HPP
