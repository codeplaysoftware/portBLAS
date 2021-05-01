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
 *  @filename gemm_no_local_full_vec.hpp
 *
 **************************************************************************/

#ifndef SYCL_BLAS_BLAS3_NO_LOCAL_FULL_VEC_GEMM_HPP
#define SYCL_BLAS_BLAS3_NO_LOCAL_FULL_VEC_GEMM_HPP

#include "gemm_common.hpp"
#include "gemm_load_store.hpp"

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
           static_cast<int>(gemm_algorithm_t::standard),
           static_cast<int>(gemm_vectorization_t::full), VectorSize,
           static_cast<int>(gemm_batch_type_t::strided)> {
 public:
  using value_t = element_t;
  using index_t = typename std::make_signed<typename input_t::index_t>::type;
  using address_t = cl::sycl::access::address_space;
  using packetize_t = Packetize<VectorSize, value_t, index_t>;
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
  /*! @brief A boolean parameter represents whether or not matrix A is
   * transposed */
  static constexpr bool trans_a = TransA;
  /*! @brief A boolean parameter represents whether or not matrix B is
   * transposed */
  static constexpr bool trans_b = TransB;

  static_assert(wg_cols * item_cols == item_rows * wg_rows,
                "Block level tile should be square");

  static_assert(item_rows % packetize_t::packet_size == 0,
                "Item rows must be a multiple of the vector packet size");
  static_assert(item_cols % packetize_t::packet_size == 0,
                "Item cols must be a multiple of the vector packet size");
  static_assert(
      packetize_t::template check_size<item_rows>(),
      "If vectorization is enabled item_rows must equal the packet_size");
  static_assert(
      packetize_t::template check_size<item_cols>(),
      "If vectorization is enabled item_cols must equal the packet_size");

  input_t a_;
  input_t b_;
  output_t c_;
  const element_t alpha_;
  const element_t beta_;
  index_t batch_size_;
  SYCL_BLAS_INLINE Gemm(input_t A, input_t B, output_t C, element_t alpha,
                        element_t beta, index_t batch_size)
      : a_(A),
        b_(B),
        c_(C),
        alpha_(alpha),
        beta_(beta / alpha_),
        batch_size_(batch_size) {}

  /*!
   * @brief Get the type of this Gemm as a human readable string.
   */
  static SYCL_BLAS_INLINE std::string get_type_string() noexcept {
    std::ostringstream str{};
    str << "Gemm <" << DoubleBuffer << ", " << NbcA << ", " << NbcB << ", "
        << ClSize << ", " << tile_type::get_type_string() << ", "
        << type_string<value_t>::get_value() << "gemm_memory:no_local, "
        << "gemm_algorithm:standard, "
        << "gemm_vectorization:full, "
        << "vector size" << VectorSize << ", batch_type:strided>";
    return str.str();
  }
  /*!
   *@brief get_workgroup_cluster. This function is used to find the optimum
   *number of work_group required to execute each GEMM.
   *
   */
  SYCL_BLAS_INLINE index_t get_workgroup_cluster() const noexcept {
    return (((a_.get_size_row() - 1) / (item_rows * wg_rows) + 1) *
            ((b_.get_size_col() - 1) / (item_cols * wg_cols) + 1));
  }
  /*!
   *@brief get_num_workgroup_cluster. This function is used to extend the number
   *of work_group cluster, in order to make sure that atleast 4 gemm operations
   *is available per work group. The number 4 is used based on empirical
   *research.
   *
   */
  SYCL_BLAS_INLINE index_t
  get_num_workgroup_cluster(index_t compute_units) const noexcept {
    constexpr index_t num_gemm_per_compute_units = 4;
    return ((num_gemm_per_compute_units * compute_units - 1) /
                get_workgroup_cluster() +
            1);
  }

  SYCL_BLAS_INLINE cl::sycl::nd_range<1> get_nd_range(
      index_t compute_units) const noexcept {
    const cl::sycl::range<1> nwg(get_workgroup_cluster() *
                                 get_num_workgroup_cluster(compute_units));
    const cl::sycl::range<1> wgs(wg_rows * wg_cols);

    return cl::sycl::nd_range<1>(nwg * wgs, wgs);
  }

  SYCL_BLAS_INLINE index_t get_size() const {
    return a_.get_size_row() * b_.get_size_col();
  }

  SYCL_BLAS_INLINE bool valid_thread(const cl::sycl::nd_item<1> &ndItem) const {
    return true;
  }

  SYCL_BLAS_INLINE void eval(cl::sycl::nd_item<1> id) noexcept {
    index_t m = a_.get_size_row();
    index_t n = b_.get_size_col();
    const index_t original_m = m;
    const index_t original_n = n;
    const index_t k = a_.get_size_col();
    const index_t lda = a_.getSizeL();
    const index_t ldb = b_.getSizeL();
    const index_t ldc = c_.getSizeL();

    // The batch index that each workgroup should start working with
    const index_t wg_batch_id = id.get_group(0) / get_workgroup_cluster();
    // This will disable all workgroups that dont have any batch to work on
    if (wg_batch_id >= batch_size_) {
      return;
    }

    const index_t batch_stride =
        id.get_group_range(0) / get_workgroup_cluster();

    const index_t a_size = trans_a ? m * lda : k * lda;
    const index_t b_size = trans_b ? ldb * k : n * ldb;
    const index_t c_size = ldc * n;

    auto orig_A = a_.get_pointer() + (wg_batch_id * a_size);
    auto orig_B = b_.get_pointer() + (wg_batch_id * b_size);
    auto orig_C = c_.get_pointer() + (wg_batch_id * c_size);

    const index_t number_of_block_per_row = ((m - 1) / block_rows) + 1;
    /* linear work group id The number of work-group required to executed each
     * batch efficiently*/
    const index_t wg_id = id.get_group(0) % get_workgroup_cluster();
    /* linear work item id */
    const index_t item_id = id.get_local_id(0);
    /* row tile id  per work group */
    const index_t tile_id_row = wg_id % number_of_block_per_row;
    /* column tile id per work group */
    const index_t tile_id_col = wg_id / number_of_block_per_row;
    /* the start position of the tile-row per work group */
    const index_t wg_row = tile_id_row * block_rows;
    /* the start position of the tile-column per work group */
    const index_t wg_col = tile_id_col * block_cols;
    /*!
     * @brief is_internal_block is used to distinguish
     * the internal block. Therefore, work items using these blocks don't need
     * to check for boundaries. Checking the packet size is a workaround because
     * normally the vector size and item rows/cols must all be equal, but when
     * vectorization is disabled the vector size is always 1 and the algorithm
     * breaks.
     */
    const bool is_internal_block =
        (packetize_t::packet_size !=
         1) &&  // This is a workaround when vectorization is disabled.
        (m - wg_row >= block_rows) &&
        (n - wg_col >= block_cols);

    const index_t vector_ofs = is_internal_block ? packetize_t::packet_size : 1;
    /* work item id per row */
    const index_t local_item_id_row = item_id % wg_rows * vector_ofs;
    /* work item id per column */
    const index_t local_item_id_col =
        (item_id / wg_rows) * (trans_b ? vector_ofs : 1);

    /* Exiting from any threads outside of the m and n boundary */
    const bool out_of_range = ((local_item_id_row + wg_row >= m) ||
                               (local_item_id_col + wg_col >= n));

    m -= local_item_id_row + wg_row;
    n -= local_item_id_col + wg_col;
    /*
     * The dim_m_a_start and dim_n_b_start are used to adjust the start position
     * of each work-item for A, B and C matrices.
     */
    const index_t dim_m_a_start = (local_item_id_row + wg_row);
    const index_t dim_n_b_start = (local_item_id_col + wg_col);

    /*! @brief Adjusting the start position of A, B , and C */
    orig_A += dim_m_a_start * (trans_a ? lda : 1);
    orig_B += dim_n_b_start * (trans_b ? 1 : ldb);
    orig_C += dim_m_a_start + (dim_n_b_start * ldc);

    /*
     * The following lambdas: boundary_check_m, boundary_check_n, and
     * boundary_check_c  are used to check the A, B , and C boundaries
     * respectively.
     */
    const auto boundary_check_m =
        [&](const index_t &idx) SYCL_BLAS_ALWAYS_INLINE { return idx < m; };
    const auto boundary_check_n =
        [&](const index_t &idx) SYCL_BLAS_ALWAYS_INLINE { return idx < n; };
    const auto boundary_check_c =
        [&](const index_t &dim_m_c_start, const index_t &dim_n_c_start)
            SYCL_BLAS_ALWAYS_INLINE {
              return (dim_m_c_start < original_m && dim_n_c_start < original_n);
            };

    // computing the next element for a and b;
    const index_t A_ptr_index = (trans_a ? lda : 1) * wg_rows * vector_ofs;
    const index_t B_ptr_index = (trans_b ? vector_ofs : ldb) * wg_cols;

    /*
     * computing the gemm panel
     */
    if ((is_internal_block == true)) {
      compute_gemm_no_shared_pannel<false, packetize_t::packet_size>(
          orig_A, orig_B, orig_C, a_size, b_size, c_size, a_.get_size_col(), k,
          dim_m_a_start, dim_n_b_start, A_ptr_index, B_ptr_index,
          boundary_check_m, boundary_check_n, boundary_check_c, out_of_range,
          batch_stride, wg_batch_id, batch_size_, lda, ldb, ldc
#ifdef ARM_GPU
          ,
          id
#endif
      );
    } else {
      compute_gemm_no_shared_pannel<true, 1>(
          orig_A, orig_B, orig_C, a_size, b_size, c_size, a_.get_size_col(), k,
          dim_m_a_start, dim_n_b_start, A_ptr_index, B_ptr_index,
          boundary_check_m, boundary_check_n, boundary_check_c, out_of_range,
          batch_stride, wg_batch_id, batch_size_, lda, ldb, ldc
#ifdef ARM_GPU
          ,
          id
#endif
      );
    }
  }
  /** @brief If beta is not zero then this function will load in values from C,
  multiply them by the beta value and store them in the results register. If
  beta is zero then this function sets the values in the results array to 0 */
  template <bool need_check_boundary, index_t packet_size,
            typename InputPointerType, typename CheckBoundaryType,
            bool beta_zero = is_beta_zero>
  SYCL_BLAS_INLINE typename std::enable_if<!beta_zero>::type scaling_c(
      element_t *reg_res, InputPointerType C, const index_t &ldc,
      const index_t &dim_m_c_start, const index_t &dim_n_c_start,
      CheckBoundaryType check_boundary, bool out_of_range) {
    if (out_of_range) {
      return;
    }
#pragma unroll
    for (index_t i = 0; i < item_cols; ++i) {
#pragma unroll
      for (index_t j = 0; j < item_rows / packet_size; ++j) {
        if (do_check<need_check_boundary>(check_boundary(
                dim_m_c_start + j * wg_rows, dim_n_c_start + i * wg_cols))) {
          cl::sycl::vec<element_t, packet_size> out_vec{};

          out_vec.template load<address_t::global_space>(
              0, cl::sycl::multi_ptr<const element_t, address_t::global_space>(
                     C + j * wg_rows * packet_size));
          out_vec *= beta_;

          out_vec.template store<address_t::private_space>(
              0, reg_res + i * item_rows + j * packet_size);
        }
      }
      C += ldc * (need_check_boundary || !trans_b ? wg_cols
                                                  : item_cols / packet_size);
    }
  }

  template <bool need_check_boundary, index_t, typename InputPointerType,
            typename CheckBoundaryType, bool beta_zero = is_beta_zero>
  SYCL_BLAS_INLINE typename std::enable_if<beta_zero>::type scaling_c(
      element_t *reg_res, InputPointerType, const index_t &, const index_t &,
      const index_t &, CheckBoundaryType, bool) {
#pragma unroll
    for (index_t i = 0; i < item_cols * item_rows; ++i) {
      reg_res[i] = 0;
    }
  }

  template <bool need_check_boundary, index_t packet_size, typename A_t,
            typename B_t, typename C_t, typename check_boundary_m_t,
            typename check_boundary_n_t, typename check_boundary_c_t>
  SYCL_BLAS_INLINE void compute_gemm_no_shared_pannel(
      A_t orig_A, B_t orig_B, C_t orig_C, const index_t &a_size,
      const index_t &b_size, const index_t &c_size, index_t orig_k, index_t k,
      const index_t &dim_m_a_start, const index_t &dim_n_b_start,
      const index_t &A_ptr_index, const index_t &B_ptr_index,
      const check_boundary_m_t &boundary_check_m,
      const check_boundary_n_t &boundary_check_n,
      const check_boundary_c_t &boundary_check_c, const bool out_of_range,
      const index_t &batch_stride, const index_t &wg_batch_id,
      index_t batch_size, const index_t &lda, const index_t &ldb,
      const index_t &ldc
#ifdef ARM_GPU
      ,
      const cl::sycl::nd_item<1> &id
#endif
      ) noexcept {
    /* temporary register array used to prefetch columns of A*/
    value_t reg_a[item_rows * packet_size];
    /* temporary register used to prefetch elements of B*/
    value_t reg_b[packet_size];
    do {
      auto A = orig_A;
      auto B = orig_B;
      auto C = orig_C;

      /* register array used to store the result*/
      value_t reg_res[item_rows * item_cols];
      scaling_c<need_check_boundary, packet_size>(
          reg_res, C, ldc, dim_m_a_start, dim_n_b_start, boundary_check_c,
          out_of_range);
      while (k >= packet_size) {
        load_and_compute_block<packet_size, need_check_boundary, false>(
            A, B, boundary_check_m, boundary_check_n, A_ptr_index, B_ptr_index,
            lda, ldb, k, reg_a, reg_b, reg_res, out_of_range
#ifdef ARM_GPU
            ,
            id
#endif
        );
        /*
         * Moving forward to the next block
         */
        k -= packet_size;
        A = A + (trans_a ? 1 : lda) * packet_size;
        B = B + (trans_b ? ldb : 1) * packet_size;
      }
      if (k > 0) {
        load_and_compute_block<packet_size, need_check_boundary, true>(
            A, B, boundary_check_m, boundary_check_n, A_ptr_index, B_ptr_index,
            lda, ldb, k, reg_a, reg_b, reg_res, out_of_range
#ifdef ARM_GPU
            ,
            id
#endif
        );
      }
      /*
       *  Storing the reg_res into C matrix
       */
      store<need_check_boundary, packet_size>(C, reg_res, dim_m_a_start,
                                              dim_n_b_start, boundary_check_c,
                                              out_of_range, ldc);

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
   * @brief Loads a full block of A and multiple partial blocks of B, computing
   * the result for each iteration.
   * @tparam packet_size : Vector size
   * @tparam check_boundary : True if block is external, false if not
   * @tparam check_k : Whether to check in K dimension.
   * @tparam BoundaryCheckM : Type of function for checking boundary in M
   * dimension.
   * @tparam BoundaryCheckN : Type of function for checking boundary in N
   * dimension.
   * @tparam PointerType : Type of the input pointers for A and B matrices.
   * @param A : Input matrix A
   * @param B : Input matrix B
   * @param boundary_check_m : Function which checks boundary in M dimension.
   * @param boundary_check_n : Function which checks boundary in N dimension.
   * @param A_ptr_index : index of next element for A
   * @param B_ptr_index : index of next element for B
   * @param lda : leading dimension of A
   * @param ldb : leading dimension of B
   * @param k : the current value of K from the main loop which calls this
   * method.
   * @param reg_a : Pointer to private register for A.
   * @param reg_b : Pointer to private register for B.
   * @param reg_res : Pointer to private register for result.
   * @param out_of_range : Controls whether to exit some functions early if
   * block is out of range of A or B.*/
  template <index_t packet_size, bool check_boundary, bool check_k,
            typename BoundaryCheckM, typename BoundaryCheckN,
            typename PointerType>
  SYCL_BLAS_INLINE void load_and_compute_block(
      PointerType A, PointerType B, BoundaryCheckM boundary_check_m,
      BoundaryCheckN boundary_check_n, const index_t &A_ptr_index,
      const index_t &B_ptr_index, const index_t &lda, const index_t &ldb,
      const index_t &k, element_t *reg_a, element_t *reg_b, element_t *reg_res,
      bool out_of_range
#ifdef ARM_GPU
      ,
      const cl::sycl::nd_item<1> &id
#endif
  ) {
    /*
     * Loading a corresponding block of matrix A into reg_a
     */
    load_block_a<item_rows, packet_size, wg_rows * packet_size, check_boundary,
                 check_k, packet_size, trans_a>(
        A, reg_a, A_ptr_index, lda, boundary_check_m,
        [=](const index_t &idx) SYCL_BLAS_ALWAYS_INLINE { return idx < k; },
        out_of_range);
#ifdef ARM_GPU
    id.barrier(cl::sycl::access::fence_space::local_space);
#endif

#pragma unroll
    for (int j = 0; j < packet_size; j++) {
      index_t ofs = 0;
      index_t col_ofs = 0;
#pragma unroll
      for (int i = 0; i < item_cols / packet_size; i++) {
        /*
         * Loading a corresponding partial block of matrix B into reg_b
         */
        load_single_b<check_k, check_boundary, packet_size, trans_b>(
            B + ofs, reg_b, j, col_ofs,
            [=](const index_t &idx) SYCL_BLAS_ALWAYS_INLINE { return idx < k; },
            boundary_check_n, out_of_range);

        /*
         * Computing a partial GEMM for the loaded block of reg_a and partial
         * reg_b and adding the result into reg_res
         */

        compute_block_gemm_no_shared<packet_size>(i + j, reg_a, reg_b, reg_res);

        ofs += B_ptr_index;
        col_ofs += (trans_b ? packet_size : 1) * wg_cols;
      }
      B += ldb * (trans_b ? 1 : wg_cols);
    }
  }
  /*!
   * @brief Loads a block of rows x cols from global A into private registers.
   * This version of the function is called when trans == false.
   * @tparam rows : the number of rows to load.
   * @tparam cols : the number of columns to load.
   * @tparam next_element : is the stride to access the next element of the A
   * maxtrix.
   * @tparam check_row : determines whether to perform bounds checking in the
   * row direction.
   * @tparam check_col : determines whether to perform bounds checking in the
   * column direction.
   * @tparam work_per_load : the number of elements loaded at one time, this is
   * also called the packet or vector size.
   * @tparam trans : true if A's representation is transposed i.e. it is row
   * major instead of column.
   * @tparam PointerType: the type of the input matrix.
   * @tparam RowCheckType: the type of a function used for checking the
   * boundary in the row direction for blocks of data located at the edge of the
   * input matrix.
   * @tparam ColCheckType: the type of a function used for checking the
   * boundary in the column direction for blocks of data located at the edge of
   * the input matrix.
   * @param ptr : the input matrix A.
   * @param reg : the private register for A.
   * @param ptr_next: offset for the next value to be loaded.
   * @param ld : the leading dimension of the input matrix.
   * @param is_valid_row : function which checks the boundary in the row
   * direction.
   * @param is_valid_col : function which checks the boundary in the col
   * direction.
   * @param out_of_range: exits the function early if block is out of range.
   */

  template <index_t rows, index_t cols, index_t next_element, bool check_row,
            bool check_col, index_t work_per_load, bool trans,
            typename PointerType, typename RowCheckType, typename ColCheckType>
  SYCL_BLAS_INLINE typename std::enable_if<!trans>::type load_block_a(
      PointerType ptr, element_t *reg, const index_t &ptr_next,
      const index_t &ld, const RowCheckType &is_valid_row,
      const ColCheckType &is_valid_col, const bool out_of_range) noexcept {
    if (out_of_range) {
      return;
    }
#pragma unroll
    for (int i = 0; i < cols; i++) {
#pragma unroll
      for (int j = 0; j < rows / work_per_load; j++) {
        // Check that the last element of the packet loaded is in range
        bool in_range = do_check<check_row>(is_valid_row(work_per_load - 1)) &&
                        do_check<check_col>(is_valid_col(i));

        cl::sycl::vec<element_t, work_per_load> in_vec{};
        if (in_range) {
          // if in range perform a vectorised load
          in_vec.template load<address_t::global_space>(
              0, cl::sycl::multi_ptr<const element_t, address_t::global_space>(
                     ptr + j * ptr_next));
        } else {
          // if not in range perform element-wise load checking boundaries at
          // each load.
#pragma unroll
          for (int l = 0; l < work_per_load; l++) {
            if (do_check<check_row>(is_valid_row(l)) &&
                do_check<check_col>(is_valid_col(i))) {
              reinterpret_cast<element_t *>(&in_vec)[l] =
                  *(ptr + j * ptr_next + l);
            }
          }
        }
        in_vec.template store<address_t::private_space>(0, reg);
        reg += work_per_load;
      }
      ptr += ld;
    }
  }

  /*!
   * @brief Loads a block of rows x cols from global A into private registers.
   * This version of the function is called when trans == true.
   * @tparam rows : the number of rows to load.
   * @tparam cols : the number of columns to load.
   * @tparam next_element : is the stride to access the next element of the A
   * maxtrix.
   * @tparam check_row : determines whether to perform bounds checking in the
   * row direction.
   * @tparam check_col : determines whether to perform bounds checking in the
   * column direction.
   * @tparam work_per_load : the number of elements loaded at one time, this is
   * also called the packet or vector size.
   * @tparam trans : true if A's representation is transposed i.e. it is row
   * major instead of column.
   * @tparam PointerType: the type of the input matrix.
   * @tparam RowCheckType: the type of a function used for checking the
   * boundary in the row direction for blocks of data located at the edge of the
   * input matrix.
   * @tparam ColCheckType: the type of a function used for checking the
   * boundary in the column direction for blocks of data located at the edge of
   * the input matrix.
   * @param ptr : the input matrix A.
   * @param reg : the private register for A.
   * @param ptr_next: offset for the next value to be loaded.
   * @param ld : the leading dimension of the input matrix.
   * @param is_valid_row : function which checks the boundary in the row
   * direction.
   * @param is_valid_col : function which checks the boundary in the col
   * direction.
   * @param out_of_range: exits the function early if block is out of range.
   */
  template <index_t rows, index_t cols, index_t next_element, bool check_row,
            bool check_col, index_t work_per_load, bool trans,
            typename PointerType, typename RowCheckType, typename ColCheckType>
  SYCL_BLAS_INLINE typename std::enable_if<trans>::type load_block_a(
      PointerType ptr, element_t *reg, const index_t &ptr_next,
      const index_t &ld, const RowCheckType &is_valid_row,
      const ColCheckType &is_valid_col, const bool out_of_range) noexcept {
    if (out_of_range) {
      return;
    }
#pragma unroll
    for (int i = 0; i < rows / work_per_load; i++) {
#pragma unroll
      for (int j = 0; j < cols; j++) {
        // Check that the last element of the packet loaded is in range
        bool in_range =
            do_check<check_row>(is_valid_row(i * next_element + j)) &&
            do_check<check_col>(is_valid_col(work_per_load - 1));
        cl::sycl::vec<element_t, work_per_load> in_vec{};
        if (in_range) {
          // if in range perform a vectorised load
          in_vec.template load<address_t::global_space>(
              0, cl::sycl::multi_ptr<const element_t, address_t::global_space>(
                     ptr + j * ld));

        } else {
          // if not in range perform element-wise load checking boundaries at
          // each load.
#pragma unroll
          for (int l = 0; l < work_per_load; l++) {
            if (do_check<check_row>(is_valid_row(i * next_element + j)) &&
                do_check<check_col>(is_valid_col(l))) {
              reinterpret_cast<element_t *>(&in_vec)[l] = *(ptr + j * ld + l);
            }
          }
        }
        // Stores the loaded value in the register while untransposing it.
#pragma unroll
        for (int k = 0; k < work_per_load; k++) {
          reg[j + i * work_per_load + k * rows] =
              reinterpret_cast<element_t *>(&in_vec)[k];
        }
      }
      ptr += next_element * ld;
    }
  }

  /*!
   * @brief Performs a single load from B into private registers, with the
   * amount of elements loaded determined by work_per_load. This version of the
   * function is called if trans == false.
   * @tparam check_row : determines whether to perform bounds checking in the
   * row direction.
   * @tparam check_col : determines whether to perform bounds checking in the
   * column direction.
   * @tparam work_per_load : the number of elements loaded at one time, this is
   * also called the packet or vector size.
   * @tparam trans : true if B's representation is transposed i.e. it is row
   * major instead of column.
   * @tparam PointerType: the type of the input matrix.
   * @tparam RowCheckType: the type of a function used for checking the
   * boundary in the row direction for blocks of data located at the edge of the
   * input matrix.
   * @tparam ColCheckType: the type of a function used for checking the
   * boundary in the column direction for blocks of data located at the edge of
   * the input matrix.
   * @param ptr : the input matrix B.
   * @param reg : the private register for B
   * @param row_ofs: How many rows B has been offset by, used in bounds
   * checking.
   * @param col_ofs : How many columns B has been offset by, used in bounds
   * checking.
   * @param is_valid_row : function which checks the boundary in the row
   * direction.
   * @param is_valid_col : function which checks the boundary in the col
   * direction.
   * @param out_of_range: exits the function early if block is out of range.
   */
  template <bool check_row, bool check_col, index_t work_per_load, bool trans,
            typename PointerType, typename RowCheckType, typename ColCheckType>
  SYCL_BLAS_INLINE typename std::enable_if<!trans>::type load_single_b(
      PointerType ptr, element_t *reg, const index_t &row_ofs,
      const index_t &col_ofs, const RowCheckType &is_valid_row,
      const ColCheckType &is_valid_col, const bool out_of_range) noexcept {
    if (out_of_range) {
      return;
    }

    // Check that the last element of the packet loaded is in range
    bool in_range = do_check<check_row>(is_valid_row(work_per_load - 1)) &&
                    do_check<check_col>(is_valid_col(col_ofs));

    cl::sycl::vec<element_t, work_per_load> in_vec{};
    if (in_range) {
      // If in range perform a vectorised load.
      in_vec.template load<address_t::global_space>(
          0,
          cl::sycl::multi_ptr<const element_t, address_t::global_space>(ptr));
    } else {
      // Otherwise perform an element-wise load, checking boundaries each load.
#pragma unroll
      for (int k = 0; k < work_per_load; k++) {
        if (do_check<check_row>(is_valid_row(k)) &&
            do_check<check_col>(is_valid_col(col_ofs))) {
          reinterpret_cast<element_t *>(&in_vec)[k] = *(ptr + k);
        }
      }
    }
    in_vec.template store<address_t::private_space>(0, reg);
  }

  /*!
   * @brief Performs a single load from B into private registers, with the
   * amount of elements loaded determined by work_per_load. This version of the
   * function is called if trans == true.
   * @tparam check_row : determines whether to perform bounds checking in the
   * row direction.
   * @tparam check_col : determines whether to perform bounds checking in the
   * column direction.
   * @tparam work_per_load : the number of elements loaded at one time, this is
   * also called the packet or vector size.
   * @tparam trans : true if B's representation is transposed i.e. it is row
   * major instead of column.
   * @tparam PointerType: the type of the input matrix.
   * @tparam RowCheckType: the type of a function used for checking the
   * boundary in the row direction for blocks of data located at the edge of the
   * input matrix.
   * @tparam ColCheckType: the type of a function used for checking the
   * boundary in the column direction for blocks of data located at the edge of
   * the input matrix.
   * @param ptr : the input matrix B.
   * @param reg : the private register for B
   * @param ptr_next: offset for the next value to be loaded.
   * @param ld : the leading dimension of the input matrix.
   * @param is_valid_row : function which checks the boundary in the row
   * direction.
   * @param is_valid_col : function which checks the boundary in the col
   * direction.
   * @param out_of_range: exits the function early if block is out of range.
   */
  template <bool check_row, bool check_col, index_t work_per_load, bool trans,
            typename PointerType, typename RowCheckType, typename ColCheckType>
  SYCL_BLAS_INLINE typename std::enable_if<trans>::type load_single_b(
      PointerType ptr, element_t *reg, const index_t &row_ofs,
      const index_t &col_ofs, const RowCheckType &is_valid_row,
      const ColCheckType &is_valid_col, const bool out_of_range) noexcept {
    if (out_of_range) {
      return;
    }

    // Check that the last element of the packet loaded is in range
    bool in_range = do_check<check_row>(is_valid_row(row_ofs)) &&
                    do_check<check_col>(is_valid_col(work_per_load - 1));

    cl::sycl::vec<element_t, work_per_load> in_vec{};
    if (in_range) {
      // If in range perform a vectorised load.
      in_vec.template load<address_t::global_space>(
          0,
          cl::sycl::multi_ptr<const element_t, address_t::global_space>(ptr));
    } else {
      // Otherwise perform an element-wise load, checking boundaries each load.
#pragma unroll
      for (int k = 0; k < work_per_load; k++) {
        if (do_check<check_row>(is_valid_row(row_ofs)) &&
            do_check<check_col>(is_valid_col(k))) {
          reinterpret_cast<element_t *>(&in_vec)[k] = *(ptr + k);
        }
      }
    }
    in_vec.template store<address_t::private_space>(0, reg);
  }
  /*!
   * @brief The following function computes the partial GEMM for the input
   * block reg_a and reg_b and add the result to the reg_res. This version is
   * called if trans == false.
   * @tparam packet_size the packet or vector size
   * @tparam trans if is B transposed or not. Set by default to trans_b and only
   * used for SFINAE.
   * @param iteration the iteration of the outside loop.
   * @param reg_a  temporary register array used to prefetch columns of A
   * @param reg_b  temporary register used to prefetch elements of B
   * @param reg_res  pointer to register used to store the result C
   */
  template <index_t packet_size, bool trans = trans_b>
  SYCL_BLAS_INLINE typename std::enable_if<!trans>::type
  compute_block_gemm_no_shared(index_t iteration, element_t *reg_a,
                               element_t *reg_b, element_t *reg_res) noexcept {
    reg_res += iteration * item_rows;
#pragma unroll
    for (int k = 0; k < packet_size; k++) {
#pragma unroll
      for (int j = 0; j < item_rows; j++) {
        reg_res[j] = cl::sycl::mad(reg_a[j], *reg_b, reg_res[j]);
      }
      reg_a += item_rows;
      reg_b += 1;
    }
  }

  /*!
   * @brief The following function computes the partial GEMM for the input
   * block reg_a and reg_b and add the result to the reg_res. This version is
   * called if trans == true and the packet_size != 1.
   * @tparam packet_size the packet or vector size
   * @tparam trans if is B transposed or not. Set by default to trans_b and only
   * used for SFINAE.
   * @param iteration the iteration of the outside loop.
   * @param reg_a  temporary register array used to prefetch columns of A
   * @param reg_b  temporary register used to prefetch elements of B
   * @param reg_res  pointer to register used to store the result C
   */
  template <index_t packet_size, bool trans = trans_b>
  SYCL_BLAS_INLINE typename std::enable_if<(packet_size != 1 && trans)>::type
  compute_block_gemm_no_shared(index_t iteration, element_t *reg_a,
                               element_t *reg_b, element_t *reg_res) noexcept {
    reg_a += iteration * item_rows;
#pragma unroll
    for (int i = 0; i < packet_size; i++) {
#pragma unroll
      for (int j = 0; j < item_rows; j++) {
        reg_res[i * item_rows + j] =
            cl::sycl::mad(reg_a[j], reg_b[i], reg_res[i * item_rows + j]);
      }
    }
  }
  /*!
   * @brief The following function computes the partial GEMM for the input
   * block reg_a and reg_b and add the result to the reg_res. This version is
   * called if trans == true and the packet_size == 1.
   * @tparam packet_size the packet or vector size
   * @tparam trans if is B transposed or not. Set by default to trans_b and only
   * used for SFINAE.
   * @param iteration the iteration of the outside loop.
   * @param reg_a  temporary register array used to prefetch columns of A
   * @param reg_b  temporary register used to prefetch elements of B
   * @param reg_res  pointer to register used to store the result C
   */
  template <index_t packet_size, bool trans = trans_b>
  SYCL_BLAS_INLINE typename std::enable_if<(packet_size == 1 && trans)>::type
  compute_block_gemm_no_shared(index_t iteration, element_t *reg_a,
                               element_t *reg_b, element_t *reg_res) noexcept {
    reg_res += iteration * item_rows;
#pragma unroll
    for (int j = 0; j < item_rows; j++) {
      reg_res[j] = cl::sycl::mad(reg_a[j], *reg_b, reg_res[j]);
    }
  }

  /*!
   * @brief For each work item the following function stores the computed block
   * of GEMM reg_res into output matrix C
   * @tparam check_block: determined whether or not the requested block is
   * internal. false means no need to check the boundaries
   * @tparam packet_size packet/vector size of A and B
   * @tparam pointerType: the type of the matrix C
   * @tparam check_boundary: the type of a function used for checking the
   * boundary for blocks of data located at the edge of the input matrix
   * @param C: is the output matrix C
   * @param reg_res  registers which store the result
   * @param dim_m_c_start Starting offset in the m dimension used for bounds
   * checking.
   * @param dim_n_c_start Starting offset in the n dimension used for bounds
   * checking.
   * @param chk_boundary: an instance of the check_boundary function
   * @param out_of_range used to exit the function if the current block is out
   * of range of the input matrices.
   * @param ldc is the leading dimension of C
   */
  template <bool check_block, index_t packet_size, typename PointerType,
            typename check_boundary>
  SYCL_BLAS_INLINE void store(PointerType C, element_t *reg_res,
                              const index_t &dim_m_c_start,
                              const index_t &dim_n_c_start,
                              const check_boundary &chk_boundary,
                              const bool out_of_range,
                              const index_t &ldc) noexcept {
    if (out_of_range) {
      return;
    }
#pragma unroll
    for (int i = 0; i < item_cols; i++) {
#pragma unroll
      for (int j = 0; j < item_rows / packet_size; j++) {
        if (do_check<check_block>(chk_boundary(dim_m_c_start + j * wg_rows,
                                               dim_n_c_start + i * wg_cols))) {
          cl::sycl::vec<element_t, packet_size> out_vec{};

          out_vec.template load<address_t::private_space>(
              0, cl::sycl::multi_ptr<const element_t, address_t::private_space>(
                     reg_res + i * item_rows + j * packet_size));
          out_vec *= alpha_;

          out_vec.template store<address_t::global_space>(
              0, C + j * wg_rows * packet_size);
        }
      }
      C += ldc * (check_block || !trans_b ? wg_cols : item_cols / packet_size);
    }
  }
};

}  // namespace blas

#endif  // SYCL_BLAS_BLAS3_NO_LOCAL_FULL_VEC_GEMM_HPP
