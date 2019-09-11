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
 *  @filename gemm_no_local.hpp
 *
 **************************************************************************/

#ifndef SYCL_BLAS_BLAS3_NO_LOCAL_GEMM_HPP
#define SYCL_BLAS_BLAS3_NO_LOCAL_GEMM_HPP

#include "gemm_common.hpp"

namespace blas {

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
          typename element_t, bool is_beta_zero, int VectorSize>
class Gemm<input_t, output_t, DoubleBuffer, NbcA, NbcB, ClSize, tile_type,
           TransA, TransB, element_t, is_beta_zero,
           static_cast<int>(gemm_memory_t::no_local),
           static_cast<int>(gemm_algorithm_t::standard), VectorSize> {
 public:
  using value_t = element_t;
  using index_t = typename std::make_signed<typename input_t::index_t>::type;
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

    auto orig_A = a_.get_pointer() + (wg_batch_id * a_size);
    auto orig_B = b_.get_pointer() + (wg_batch_id * b_size);
    auto orig_C = c_.get_pointer() + (wg_batch_id * c_size);

    const index_t number_of_block_per_row = ((m_ - 1) / block_rows) + 1;
    /* linear work group id The number of work-group required to executed each
     * batch efficiently*/
    const index_t wg_id = id.get_group(0) % get_workgroup_cluster(m_, n_);
    /* linear work item id */
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

  void adjust_access_displacement() {
    a_.adjust_access_displacement();
    b_.adjust_access_displacement();
    c_.adjust_access_displacement();
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

}  // namespace blas

#endif  // SYCL_BLAS_BLAS3_NO_LOCAL_GEMM_HPP
