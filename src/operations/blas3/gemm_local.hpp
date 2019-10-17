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
 *  @filename gemm_local.hpp
 *
 **************************************************************************/

#ifndef SYCL_BLAS_BLAS3_LOCAL_GEMM_HPP
#define SYCL_BLAS_BLAS3_LOCAL_GEMM_HPP

#include "gemm_common.hpp"

namespace blas {

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
           static_cast<int>(gemm_memory_t::local),
           static_cast<int>(gemm_algorithm_t::standard)> {
 public:
  using tile_type = TileType;
  using value_t = element_t;
  using index_t = typename std::make_signed<typename input_t::index_t>::type;

  // enable easier access to tile dimensions
  static constexpr index_t item_rows = tile_type::item_rows;
  static constexpr index_t item_cols = tile_type::item_cols;
  static constexpr index_t wg_rows = tile_type::wg_rows;
  static constexpr index_t wg_cols = tile_type::wg_cols;
  static constexpr index_t tl_rows = tile_type::tl_rows;
  static constexpr index_t tl_cols = tile_type::tl_cols;
  static constexpr index_t tile_size = tl_rows * tl_cols;

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
        beta_(beta / alpha),
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
    auto orig_A = a_.get_pointer() + (wg_batch_id * a_size);
    auto orig_B = b_.get_pointer() + (wg_batch_id * b_size);
    auto orig_C = c_.get_pointer() + (wg_batch_id * c_size);
    const index_t item_id = id.get_local_id(0);
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
  void adjust_access_displacement() {
    a_.adjust_access_displacement();
    b_.adjust_access_displacement();
    c_.adjust_access_displacement();
  }
  SYCL_BLAS_INLINE bool valid_thread(cl::sycl::nd_item<1> ndItem) const {
    return true;
  }

 private:
  /** @brief If beta is not zero then this function will load in values from C,
  multiply them by the beta value and store them in the results register. If
  beta is zero then this function does nothing. */
  template <bool check_m_limit, bool check_n_limit, typename InputPointerType,
            bool beta_zero = is_beta_zero>
  static SYCL_BLAS_INLINE typename std::enable_if<!beta_zero>::type scaling_c(
      element_t *reg_res, InputPointerType C, const index_t &ldc,
      const index_t &mc, const index_t &nc, const element_t &beta,
      bool out_of_range) {
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
          reg_res[i * item_rows + j] = beta * C[j * wg_rows];
        }
      }
      C = C + ldc;
    }
  }

  template <bool check_m_limit, bool check_n_limit, typename InputPointerType,
            bool beta_zero = is_beta_zero>
  static SYCL_BLAS_INLINE typename std::enable_if<beta_zero>::type scaling_c(
      element_t *reg_res, InputPointerType, const index_t &, const index_t &,
      const index_t &, const element_t &, bool) {
#pragma unroll
    for (index_t i = 0; i < item_cols * item_rows; ++i) {
      reg_res[i] = 0;
    }
  }
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
      element_t *reg_a, element_t &reg_b, const bool out_of_range,
      const index_t batch_stride, const index_t wg_batch_id,
      index_t batch_size) noexcept {
    index_t ofs = 1;
    do {
      auto A = orig_A;
      auto B = orig_B;
      auto C = orig_C;
      element_t reg_res[item_rows * item_cols];
      scaling_c<check_m_limit, check_n_limit>(reg_res, C, ldc, mc, nc, beta,
                                              out_of_range);
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
      OutputPointerType C, index_t ldc, element_t *reg_res,
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
          C[j * wg_rows] = alpha * reg_res[i * item_rows + j];
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
                  ldsa>(item_id, A, lda, sA,
                        [&](index_t ir, index_t cr) { return cr < m; },
                        [&](index_t ic, index_t cc) { return cc < k - ic; });
    extract_block<check_k_limit, check_n_limit, trans_b, cl_elems, block_cols,
                  ldsb>(item_id, B, ldb, sB,
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
  static SYCL_BLAS_INLINE void compute_block_gemm(InputPointerType B,
                                                  InputPointerType A,
                                                  element_t *reg_a,
                                                  element_t &reg_b,
                                                  element_t *reg_res) noexcept {
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
          reg_res[j * item_rows + l] =
              cl::sycl::mad(reg_a[l], reg_b, reg_res[j * item_rows + l]);
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

#endif  // SYCL_BLAS_BLAS3_LOCAL_GEMM_HPP
