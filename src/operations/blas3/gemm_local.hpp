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
 *  portBLAS: BLAS implementation using SYCL
 *
 *  @filename gemm_local.hpp
 *
 **************************************************************************/

#ifndef PORTBLAS_BLAS3_LOCAL_GEMM_HPP
#define PORTBLAS_BLAS3_LOCAL_GEMM_HPP

#include "gemm_common.hpp"
#include "gemm_load_store.hpp"

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
 * @tparam SymmA   whether the matrix A is a symmetric triangular matrix
 * @tparam SymmB   whether the matrix B is a symmetric triangular matrix
 * @tparam element_t  type of matrix elements
 * @tparam is_beta_zero True if beta == 0.
 * @tparam VectorSize The packet size to be used for vectorization.
 * @tparam batch_type the type of batch strideded /interleaved
 * @tparam UseJointMatrix boolean parameter to decide whether to use
 * joint_matrix or not
 */
template <typename input_t, typename output_t, bool DoubleBuffer, bool NbcA,
          bool NbcB, int ClSize, typename TileType, bool TransA, bool TransB,
          bool SymmA, bool SymmB, typename element_t, bool is_beta_zero,
          int VectorSize>
class Gemm<input_t, output_t, DoubleBuffer, NbcA, NbcB, ClSize, TileType,
           TransA, TransB, SymmA, SymmB, element_t, is_beta_zero,
           static_cast<int>(gemm_memory_t::local),
           static_cast<int>(gemm_algorithm_t::standard),
           static_cast<int>(gemm_vectorization_t::full), VectorSize,
           static_cast<int>(gemm_batch_type_t::strided), false> {
 public:
  using tile_type = TileType;
  using value_t = element_t;
  using index_t = typename std::make_signed<typename input_t::index_t>::type;
  using packetize_t = Packetize<VectorSize, value_t, index_t>;
  using vector_t = typename packetize_t::PacketType;
  using address_t = cl::sycl::access::address_space;

  // enable easier access to tile dimensions
  static constexpr index_t item_rows = tile_type::item_rows;
  static constexpr index_t item_cols = tile_type::item_cols;
  static constexpr index_t wg_rows = tile_type::wg_rows;
  static constexpr index_t wg_cols = tile_type::wg_cols;
  static constexpr index_t sg_rows = tile_type::sg_rows;
  static constexpr index_t sg_cols = tile_type::sg_cols;
  static constexpr index_t tl_rows = tile_type::tl_rows;
  static constexpr index_t tl_cols = tile_type::tl_cols;
  static constexpr index_t tile_size = tl_rows * tl_cols;

  static constexpr bool double_buffer = DoubleBuffer;
  static constexpr bool nbc_a = NbcA;
  static constexpr bool nbc_b = NbcB;
  static constexpr bool trans_a = TransA;
  static constexpr bool trans_b = TransB;
  /*! @brief A boolean parameter represents whether the matrix is
   * symmetric */
  static constexpr bool symm_a = SymmA;
  static constexpr bool symm_b = SymmB;

  //! @brief Number of elements which fit within a cache line.
  static constexpr index_t cl_elems = ClSize / sizeof(element_t);
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

  static_assert(item_rows % packetize_t::packet_size == 0,
                "Item rows must be a multiple of the vector packet size");

  static_assert(cl_elems % packetize_t::packet_size == 0,
                "Cache line size must be a multiple of packet_size");

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
  const element_t alpha_;
  const element_t beta_;
  index_t batch_size_;
  index_t stridea_;
  index_t strideb_;
  index_t stridec_;

  PORTBLAS_INLINE Gemm(input_t A, input_t B, output_t C, element_t alpha,
                        element_t beta, index_t batch_size, index_t stride_a,
                        index_t stride_b, index_t stride_c)
      : a_(A),
        b_(B),
        c_(C),
        alpha_(alpha),
        beta_(beta / alpha),
        batch_size_(batch_size),
        stridea_{stride_a},
        strideb_{stride_b},
        stridec_{stride_c} {}

  /*!
   * @brief Get the type of this GemmFactory as a human readable string.
   */
  static PORTBLAS_INLINE std::string get_type_string() noexcept {
    std::ostringstream str{};
    str << "Gemm <" << double_buffer << ", " << nbc_a << ", " << nbc_b << ", "
        << cl_elems * sizeof(element_t) << ", " << tile_type::get_type_string()
        << ", " << type_string<value_t>::get_value() << "gemm_memory:local, "
        << "gemm_algorithm:standard, "
        << "gemm_vectorization:full, "
        << "vector size" << VectorSize << ", batch_type:strided>";
    return str.str();
  }

  /*!
   *@brief gt_workgroup_cluster. This function is used to find the optimum
   *number of work_group required to execute each GEMM.
   *
   */
  PORTBLAS_INLINE index_t get_workgroup_cluster() const noexcept {
    return (((a_.get_size_row() - 1) / big_tile_rows + 1) *
            ((b_.get_size_col() - 1) / big_tile_cols + 1) * tl_rows * tl_cols);
  }
  /*!
   *@brief get_num_workgroup_cluster. This function is used to extend the number
   *of work_group cluster, in order to make sure that atleast 4 gemm operations
   *is available per work group. The number 4 is used based on empirical
   *research.
   *
   */
  PORTBLAS_INLINE index_t
  get_num_workgroup_cluster(index_t compute_units) const noexcept {
    return (batch_size_ > 1)
               ? ((4 * compute_units - 1) / get_workgroup_cluster() + 1)
               : 1;
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
  PORTBLAS_INLINE cl::sycl::nd_range<1> get_nd_range(
      index_t compute_units) const noexcept {
    const cl::sycl::range<1> nwg(get_workgroup_cluster() *
                                 get_num_workgroup_cluster(compute_units));
    const cl::sycl::range<1> wgs(wg_size);
#ifdef VERBOSE
    std::cout << " M: " << a_.get_size_row() << " , N " << b_.get_size_col()
              << " , big_tile_rows: " << big_tile_rows
              << " , big_tile_cols: " << big_tile_cols
              << " , wg_size: " << wg_size
              << " , nwg : " << get_workgroup_cluster() << std::endl;
#endif
    return cl::sycl::nd_range<1>(nwg * wgs, wgs);
  }

  PORTBLAS_INLINE index_t get_size() const {
    return a_.get_size_row() * b_.get_size_col();
  }

  /*!
   * @brief Run the generated GEMM device function.
   * @tparam local_memory_t LocalMemory type
   * @param id  nd_item used for calls to local barriers
   * @param scratch local memory
   */
  template <typename local_memory_t>
  PORTBLAS_INLINE void eval(local_memory_t scratch_acc,
                             const cl::sycl::nd_item<1> &id) noexcept {
    index_t m = a_.get_size_row();
    index_t n = b_.get_size_col();
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

    auto scratch = scratch_acc.localAcc.get_pointer();

    // The number of work-group required to executed each batch efficiently
    const index_t wg_id = id.get_group(0) % get_workgroup_cluster();

    const index_t a_size = trans_a ? m * lda : k * lda;
    const index_t b_size = trans_b ? ldb * k : n * ldb;
    const index_t c_size = ldc * n;

    auto ptr_A = a_.get_pointer() + (wg_batch_id * stridea_);
    auto ptr_B = b_.get_pointer() + (wg_batch_id * strideb_);
    auto ptr_C = c_.get_pointer() + (wg_batch_id * stridec_);

    const index_t item_id = id.get_local_id(0);
    const index_t tile_id = wg_id / tile_size;
    const index_t tile_local_id = wg_id % tile_size;
    const index_t tiles_per_col = (m - 1) / big_tile_rows + 1;
    const index_t tile_row = (tile_id % tiles_per_col) * tl_rows;
    const index_t tile_col = (tile_id / tiles_per_col) * tl_cols;
    const index_t wg_row = (tile_row + tile_local_id % tl_rows) * block_rows;
    const index_t wg_col = (tile_col + tile_local_id / tl_rows) * block_cols;
    const bool out_of_range = (wg_row >= m || wg_col >= n);
    const bool internal = m - wg_row >= block_rows && n - wg_col >= block_cols;
    const index_t vector_offset = internal ? packetize_t::packet_size : 1;
    const index_t item_id_ofs = item_id * vector_offset;
    const index_t row_c = wg_row + item_id % wg_rows * vector_offset;
    const index_t col_c = wg_col + (item_id / wg_rows) * item_cols;

    element_t reg_a[item_rows];
    element_t reg_b;
    ptr_C += row_c + col_c * ldc;

    const index_t mc = m - row_c;
    const index_t nc = n - col_c;

    const index_t row_b =
        trans_b ? wg_col + item_id_ofs % block_cols : item_id_ofs % cl_elems;
    const index_t col_b =
        trans_b ? item_id_ofs / block_cols : wg_col + item_id_ofs / cl_elems;
    ptr_B += col_b * ldb + row_b;

    n = n - wg_col - ((trans_b ? row_b : col_b) - wg_col);
    const index_t row_a =
        trans_a ? item_id_ofs % cl_elems : wg_row + item_id_ofs % block_rows;
    const index_t col_a =
        trans_a ? wg_row + item_id_ofs / cl_elems : item_id_ofs / block_rows;
    ptr_A += col_a * lda + row_a;

    m = m - wg_row - ((trans_a ? col_a : row_a) - wg_row);

    auto s1 =
        scratch +
        (trans_b ? item_id_ofs / block_cols + (item_id_ofs % block_cols) * ldsb
                 : item_id_ofs % cl_elems + (item_id_ofs / cl_elems) * ldsb);
    auto s2 = scratch + (item_id / wg_rows) * item_cols * ldsb;
    index_t ofs = (double_buffer + 1) * block_cols * ldsb;
    auto s3 =
        scratch + ofs +
        (trans_a
             ? item_id_ofs / cl_elems + (item_id_ofs % cl_elems) * ldsa
             : item_id_ofs % block_rows + (item_id_ofs / block_rows) * ldsa);
    auto s4 = scratch + ofs + (item_id % wg_rows * vector_offset);

    if (internal) {
      compute_panel_gemm<double_buffer, false, false>(
          id, item_id, row_a, col_a, row_b, col_b, m, n, k, mc, nc, a_size,
          b_size, c_size, ptr_A, lda, ptr_B, ldb, ptr_C, ldc, s1, s2, s3, s4,
          reg_a, reg_b, out_of_range, batch_stride, wg_batch_id, batch_size_);
    } else {
      compute_panel_gemm<double_buffer, true, true>(
          id, item_id, row_a, col_a, row_b, col_b, m, n, k, mc, nc, a_size,
          b_size, c_size, ptr_A, lda, ptr_B, ldb, ptr_C, ldc, s1, s2, s3, s4,
          reg_a, reg_b, out_of_range, batch_stride, wg_batch_id, batch_size_);
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
  PORTBLAS_INLINE bool valid_thread(const cl::sycl::nd_item<1> &ndItem) const {
    return true;
  }

 private:
  /** @brief If beta is not zero then this function will load in values from C,
  multiply them by the beta value and store them in the results register. If
  beta is zero then this function does nothing. */
  template <bool check_m_limit, bool check_n_limit, typename InputPointerType,
            bool beta_zero = is_beta_zero>
  PORTBLAS_INLINE typename std::enable_if<!beta_zero>::type scaling_c(
      element_t *reg_res, InputPointerType C, const index_t &mc,
      const index_t &nc, const index_t &ldc, const bool out_of_range) {
    if (out_of_range) {
      return;
    }
    constexpr index_t offset =
        (!check_m_limit && !check_n_limit) ? packetize_t::packet_size : 1;
    for (index_t i = 0; i < item_cols; ++i) {
      for (index_t j = 0; j < item_rows / offset; ++j) {
        const bool in_range =
            do_check<check_m_limit>(j * wg_rows * offset < mc) &&
            do_check<check_n_limit>(i < nc);
        if (in_range) {
          for (index_t l = 0; l < offset; ++l) {
            reg_res[i * item_rows + j * offset + l] =
                beta_ * *(C + j * (wg_rows * offset) + l);
          }
        }
      }
      C = C + ldc;
    }
  }

  template <bool check_m_limit, bool check_n_limit, typename InputPointerType,
            bool beta_zero = is_beta_zero>
  PORTBLAS_INLINE typename std::enable_if<beta_zero>::type scaling_c(
      element_t *reg_res, InputPointerType, const index_t &, const index_t &,
      const index_t &, const bool) {
    for (index_t i = 0; i < item_cols * item_rows; ++i) {
      reg_res[i] = 0;
    }
  }

  /*!
   * @brief Compute a GEMM of a block-row of A (transpose) and a
   * block-column of B (transpose).
   *
   * This is a collective operation between all items in as work-group.
   * This method should actually be a generic lambda (as in C++20), it
   * only forwards parameters from GemmFactory::run().
   *
   * @tparam check_m_limit  iff true, check if row indexes of C are
   *                        out-of-bound
   * @tparam check_n_limit  iff true, check if no indexes of C are
   *                        out-of-bound
   */
  template <bool double_buffer, bool check_m_limit, bool check_n_limit,
            typename InputPointerType, typename OutputPointerType,
            typename ScratchPointerType>
  PORTBLAS_INLINE void compute_panel_gemm(
      const cl::sycl::nd_item<1> &id, const index_t &item_id,
      const index_t &row_a, const index_t &col_a, const index_t &row_b,
      const index_t &col_b, const index_t &m, const index_t &n,
      const index_t &orig_k, const index_t &mc, const index_t &nc,
      const index_t &a_size, const index_t &b_size, const index_t &c_size,
      InputPointerType orig_A, const index_t &lda, InputPointerType orig_B,
      const index_t &ldb, OutputPointerType orig_C, const index_t &ldc,
      ScratchPointerType s1, ScratchPointerType s2, ScratchPointerType s3,
      ScratchPointerType s4, element_t *reg_a, element_t &reg_b,
      const bool out_of_range, index_t batch_stride, index_t wg_batch_id,
      index_t batch_size) noexcept {
    index_t ofs = 1;
    do {
      auto A = orig_A;
      auto B = orig_B;
      auto C = orig_C;
      auto k = orig_k;
      index_t ra = row_a;
      index_t ca = col_a;
      index_t rb = row_b;
      index_t cb = col_b;
      element_t reg_res[item_rows * item_cols];
      scaling_c<check_m_limit, check_n_limit>(reg_res, C, mc, nc, ldc,
                                              out_of_range);
      while (k >= cl_elems) {
        extract_input_blocks<check_m_limit, check_n_limit, false, symm_a,
                             symm_b>(item_id, m, n, k, ra, ca, rb, cb, A, lda,
                                     B, ldb, s1, s3, out_of_range);
        id.barrier(cl::sycl::access::fence_space::local_space);
        compute_block_gemm<check_m_limit, check_n_limit>(item_id, s2, s4, reg_a,
                                                         reg_b, reg_res);
        A += cl_elems * (trans_a ? 1 : lda);
        B += cl_elems * (trans_b ? ldb : 1);

        if constexpr (symm_a) {
          if constexpr (trans_a) {
            ra += cl_elems;
          } else {
            ca += cl_elems;
          }
        }
        if constexpr (symm_b) {
          if constexpr (trans_b) {
            cb += cl_elems;
          } else {
            rb += cl_elems;
          }
        }

        sync_smem<double_buffer, block_cols * ldsb, block_cols * ldsb,
                  ldsa * cl_elems, ldsa * cl_elems>(id, ofs, s1, s2, s3, s4);
        k -= cl_elems;
      }

      if (k > 0) {
        if constexpr (symm_a) {
          if constexpr (trans_a) {
            ra = row_a + (orig_k - k);
          } else {
            ca = col_a + (orig_k - k);
          }
        }
        if constexpr (symm_b) {
          if constexpr (trans_b) {
            cb = col_b + (orig_k - k);
          } else {
            rb = row_b + (orig_k - k);
          }
        }

        extract_input_blocks<check_m_limit, check_n_limit, true, symm_a,
                             symm_b>(item_id, m, n, k, ra, ca, rb, cb, A, lda,
                                     B, ldb, s1, s3, out_of_range);
        id.barrier(cl::sycl::access::fence_space::local_space);
        compute_block_gemm<check_m_limit, check_n_limit>(item_id, s2, s4, reg_a,
                                                         reg_b, reg_res);

        sync_smem<double_buffer, block_cols * ldsb, block_cols * ldsb,
                  ldsa * cl_elems, ldsa * cl_elems>(id, ofs, s1, s2, s3, s4);
      }

      // store the output
      store_output_block<check_m_limit, check_n_limit>(item_id, mc, nc, C, ldc,
                                                       reg_res, out_of_range);
      orig_A += (stridea_ * batch_stride);
      orig_B += (strideb_ * batch_stride);
      orig_C += (stridec_ * batch_stride);
      // batch_size_ must be signed as the negative value has meaning here.
      batch_size -= batch_stride;
    } while (batch_size > wg_batch_id);
  }

  template <bool internal, index_t p_size = packetize_t::packet_size,
            typename OutputPointerType>
  PORTBLAS_INLINE typename std::enable_if<!internal>::type store_packet(
      element_t *reg, OutputPointerType out_ptr) {
    *out_ptr = alpha_ * (*reg);
  }

  template <bool internal, index_t p_size = packetize_t::packet_size,
            typename OutputPointerType>
  PORTBLAS_INLINE typename std::enable_if<internal>::type store_packet(
      element_t *reg, OutputPointerType out_ptr) {
    vector_t out_vec{};

    out_vec.template load<address_t::private_space>(
        0, cl::sycl::multi_ptr<const element_t, address_t::private_space>(reg));
    out_vec *= alpha_;

    out_vec.template store<address_t::global_space>(
        0, cl::sycl::multi_ptr<element_t, address_t::global_space>(out_ptr));
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
   * @param reg_res  2D register array containing the partial resull of C
   * per thread
   */

  template <bool check_m_limit, bool check_n_limit, typename OutputPointerType>
  PORTBLAS_INLINE void store_output_block(index_t, index_t mc, index_t nc,
                                           OutputPointerType C, index_t ldc,
                                           element_t *reg_res,
                                           const bool out_of_range) noexcept {
    if (out_of_range) {
      return;
    }
    constexpr index_t offset =
        (!check_m_limit && !check_n_limit) ? packetize_t::packet_size : 1;
    for (index_t i = 0; i < item_cols; ++i) {
      for (index_t j = 0; j < item_rows / offset; j++) {
        const bool in_range =
            do_check<check_m_limit>(j * wg_rows * offset < mc) &&
            do_check<check_n_limit>(i < nc);

        if (in_range) {
          store_packet<!check_m_limit && !check_n_limit>(
              reg_res, C + j * (wg_rows * offset));
        }
        reg_res += offset;
      }
      C += ldc;
    }
  }

  /*!
   * @brief Extract a block of A, and a conformant block of B.
   *
   * @see GemmFactory::extract_block()
   */
  template <bool check_m_limit, bool check_n_limit, bool check_k_limit,
            bool symm_a, bool symm_b, typename InputPointerType,
            typename ScratchPointerType>
  PORTBLAS_INLINE void extract_input_blocks(
      index_t item_id, index_t m, index_t n, index_t k, index_t row_a,
      index_t col_a, index_t row_b, index_t col_b, InputPointerType A,
      index_t lda, InputPointerType B, index_t ldb, ScratchPointerType sB,
      ScratchPointerType sA, const bool out_of_range) noexcept {
    if (out_of_range) {
      return;
    }

    extract_block<!check_m_limit && !check_n_limit, check_m_limit,
                  check_k_limit, trans_a, symm_a, true, block_rows, cl_elems,
                  ldsa>(
        item_id, row_a, col_a, A, lda, sA,
        [&](index_t, index_t cr) PORTBLAS_ALWAYS_INLINE { return cr < m; },
        [&](index_t ic, index_t cc)
            PORTBLAS_ALWAYS_INLINE { return cc < k - ic; });
    extract_block<!check_m_limit && !check_n_limit, check_k_limit,
                  check_n_limit, trans_b, symm_b, false, cl_elems, block_cols,
                  ldsb>(
        item_id, row_b, col_b, B, ldb, sB,
        [&](index_t ir, index_t cr)
            PORTBLAS_ALWAYS_INLINE { return cr < k - ir; },
        [&](index_t, index_t cc) PORTBLAS_ALWAYS_INLINE { return cc < n; });
  }

  /*!
   * @brief Extract a block of a matrix from global to shared memory, and
   *        optionally transpose it on the fly.
   *
   * This is a collective operation on all items in a work group.
   *
   * @tparam check_row_limit  iff true, check the row out-of-bound
   * condition
   * @tparam check_col_limit  iff true, check the column out-of-bound
   * condition
   * @tparam trans  iff true, transpose the matrix
   * @tparam symm  whether the matrix is symmetric
   * @tparam left_side if the matrix is in the left side.
   * @tparam rows  number of rows in the block
   * @tparam cols  number of columns in the block
   * @tparam lds  leading dimension of the block in shared memory
   * @tparam InputPointerType  pointer type of the input matrix
   * @tparam ScratchPointerType  pointer type of the memory used to store
   * the extracted block
   * @tparam RowPredicate  row out-of-bound condition type
   * @tparam ColPredicate  column out-of-bound condition type
   *
   * @param item_id  id of the work item which called this method
   * @param ptr  pointer to the input matrix with proper item-dependent
   * offset, see GemmFactory::run() for details
   * @param ld  the leading dimension of the input matrix
   * @param scratch  the pointer to memory where the output block is
   * stored, with proper item-dependent offset, see GemmFactory::run() for
   * details
   * @param in_row  a predicate which checks whether a row index is within
   *                matrix bounds
   * @param in_col  a predicate which checks whether a col index is within
   *                matrix bounds
   */
  template <bool internal, bool check_row_limit, bool check_col_limit,
            bool trans, bool symm, bool left_side, index_t rows, index_t cols,
            index_t lds, typename InputPointerType, typename ScratchPointerType,
            typename RowPredicate, typename ColPredicate>
  PORTBLAS_INLINE typename std::enable_if<!trans>::type extract_block(
      index_t item_id, index_t row, index_t col, InputPointerType ptr,
      index_t ld, ScratchPointerType scratch, RowPredicate in_row,
      ColPredicate in_col) {
    constexpr index_t bs = rows * cols;
    constexpr index_t multiplier = internal ? packetize_t::packet_size : 1;
#pragma unroll
    for (index_t i = 0; i < (bs - 1) / (wg_size * multiplier) + 1; ++i) {
      if (!do_check<((bs % (wg_size * multiplier)) != 0)>(
              item_id + i * (wg_size * multiplier) < bs))
        continue;
      const index_t col_ofs = i * ((wg_size * multiplier) / rows);
      const bool in_range =
          do_check<check_row_limit>(
              in_row(((item_id * multiplier) % rows), multiplier - 1)) &&
          do_check<check_col_limit>(
              in_col((item_id * multiplier / rows), col_ofs));

      const auto edge_in_range = [&](const index_t &ofs) {
        return in_row((item_id * multiplier) % rows, ofs) &&
               in_col((item_id * multiplier) / rows, col_ofs);
      };
      if constexpr (symm) {
        load_symm<trans, left_side, internal, lds>(
            row, col, col_ofs, ld, in_range, ptr + (col_ofs * ld),
            scratch + col_ofs * lds, edge_in_range);
      } else {
        packetize_t::template load<trans, internal, lds>(
            in_range, ptr + (col_ofs * ld), scratch + col_ofs * lds,
            edge_in_range);
      }
    }
  }
  template <bool internal, bool check_row_limit, bool check_col_limit,
            bool trans, bool symm, bool left_side, index_t rows, index_t cols,
            index_t lds, typename InputPointerType, typename ScratchPointerType,
            typename RowPredicate, typename ColPredicate>
  PORTBLAS_INLINE typename std::enable_if<trans>::type extract_block(
      index_t item_id, index_t row, index_t col, InputPointerType ptr,
      index_t ld, ScratchPointerType scratch, RowPredicate in_row,
      ColPredicate in_col) {
    const index_t bs = rows * cols;
    constexpr index_t multiplier = internal ? packetize_t::packet_size : 1;
#pragma unroll
    for (index_t i = 0; i < (bs - 1) / (wg_size * multiplier) + 1; ++i) {
      if (!do_check<((bs % (wg_size * multiplier)) != 0)>(
              item_id + i * (wg_size * multiplier) < bs))
        continue;
      const index_t row_ofs = i * ((wg_size * multiplier) / cols);
      const bool in_range = do_check<check_row_limit>(in_row(
                                (item_id * multiplier) / cols, row_ofs)) &&
                            do_check<check_col_limit>(in_col(
                                (item_id * multiplier) % cols, multiplier - 1));

      auto edge_in_range = [&](const index_t &ofs) PORTBLAS_ALWAYS_INLINE {
        return in_col((item_id * multiplier) % cols, ofs) &&
               in_row((item_id * multiplier) / cols, row_ofs);
      };

      if constexpr (symm) {
        load_symm<trans, left_side, internal, lds>(
            row, col, row_ofs, ld, in_range, ptr + (row_ofs * ld),
            scratch + row_ofs, edge_in_range);
      } else {
        packetize_t::template load<trans, internal, lds>(
            in_range, ptr + (row_ofs * ld), scratch + row_ofs, edge_in_range);
      }
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
  template <bool check_m_limit, bool check_n_limit, typename InputPointerType>
  PORTBLAS_INLINE void compute_block_gemm(index_t, InputPointerType B,
                                           InputPointerType A, element_t *reg_a,
                                           element_t &reg_b,
                                           element_t *reg_res) noexcept {
    // NOTE: Adding "#pragma unroll" here reduces performance on AMD R9
    // Nano.
    //       Seems that the small reduction of arithmetic operations does
    //       not amortize the cost of loading the larger kernel binary
    //       resulting from loop unrollment.
    constexpr index_t work_per_load =
        !check_m_limit && !check_n_limit ? packetize_t::packet_size : 1;
#if defined NVIDIA_GPU
#pragma unroll
#endif
    for (index_t i = 0; i < cl_elems; ++i) {
#pragma unroll
      for (index_t j = 0; j < item_rows / work_per_load; ++j) {
#pragma unroll
        for (int i = 0; i < work_per_load; i++) {
          reg_a[i + j * work_per_load] =
              *(A + (i + j * wg_rows * work_per_load));
        }
      }
#pragma unroll
      for (index_t j = 0; j < item_cols; ++j) {
        reg_b = *(B + j * ldsb);
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
  static PORTBLAS_INLINE typename std::enable_if<db>::type sync_smem(
      const cl::sycl::nd_item<1> &id, index_t &ofs_sign, P &s,
      Ps &... ss) noexcept {
    s += ofs_sign * o;
    sync_smem<db, os...>(id, ofs_sign, ss...);
  }

  template <bool db>
  static PORTBLAS_INLINE typename std::enable_if<db>::type sync_smem(
      const cl::sycl::nd_item<1> &, index_t &ofs_sign) noexcept {
    ofs_sign = -ofs_sign;
  }

  template <bool db, index_t..., typename... Ps>
  static PORTBLAS_INLINE typename std::enable_if<!db>::type sync_smem(
      const cl::sycl::nd_item<1> &id, index_t &, Ps &...) noexcept {
    id.barrier(cl::sycl::access::fence_space::local_space);
  }

  /**
   * @brief Performs a vectorised load for symmetric matrices.
   */
  template <bool trans, bool left_side, bool internal, index_t lds,
            typename SrcPointerType, typename DestPointerType,
            typename EdgePredicate>
  static PORTBLAS_INLINE typename std::enable_if<internal>::type load_symm(
      index_t row, index_t col, index_t col_ofs, index_t ld,
      const bool in_range, SrcPointerType src, DestPointerType dest,
      EdgePredicate edge_in_range) {
    constexpr index_t packet_size = internal ? packetize_t::packet_size : 1;
    const index_t curr_col = col + col_ofs;
    const bool in_triangle_range =
        in_triangle<trans, left_side>(row, curr_col) &&
        in_triangle<trans, left_side>(row + (packet_size - 1), curr_col);

    if (in_triangle_range) {
      packetize_t::template load<trans, internal, lds>(in_range, src, dest,
                                                       edge_in_range);
    } else {
      vector_t packet{};
      for (index_t i = 0; i < packet_size; ++i) {
        reinterpret_cast<value_t *>(&packet)[i] =
            edge_in_range(i) ? in_triangle<trans, left_side>(row + i, curr_col)
                                   ? *(src + i)
                                   : *((src + i) + ((row + i) - curr_col) * ld +
                                       (curr_col - (row + i)))
                             : value_t{0.0};
      }
      packetize_t::template store<trans, lds>(packet, dest);
    }
  }
  template <bool trans, bool left_side, bool internal, index_t lds,
            typename SrcPointerType, typename DestPointerType,
            typename EdgePredicate>
  static PORTBLAS_INLINE typename std::enable_if<!internal>::type load_symm(
      index_t row, index_t col, index_t col_ofs, index_t ld,
      const bool in_range, SrcPointerType src, DestPointerType dest,
      EdgePredicate edge_in_range) {
    const index_t curr_col = col + col_ofs;
    auto ptr = src;
    if (!in_triangle<trans, left_side>(row, curr_col)) {
      ptr += (row - curr_col) * ld + (curr_col - row);
    }
    packetize_t::template load<trans, internal, lds>(in_range, ptr, dest,
                                                     edge_in_range);
  }

  template <bool trans, bool left_side>
  static PORTBLAS_INLINE bool in_triangle(index_t row, index_t col) {
    // If the matrix is symmetric, the valid values are expected on the lower
    // triangle unless it is transposed, in which case valid values will
    // be on the upper side.
    if constexpr (trans) {
      if constexpr (left_side) {
        return row <= col;
      }
      return row >= col;
    }
    if constexpr (left_side) {
      return row >= col;
    }
    return row <= col;
  };

};  // Gemm

}  // namespace blas

#endif  // PORTBLAS_BLAS3_LOCAL_GEMM_HPP
