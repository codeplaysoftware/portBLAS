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
 *  @filename gemm_local_joint_matrix.hpp
 *
 **************************************************************************/

#ifndef PORTBLAS_BLAS3_LOCAL_GEMM_JOINT_MATRIX_HPP
#define PORTBLAS_BLAS3_LOCAL_GEMM_JOINT_MATRIX_HPP

#ifdef SB_ENABLE_JOINT_MATRIX

#include "gemm_common.hpp"
#include "gemm_load_store_joint_matrix.hpp"

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
           static_cast<int>(gemm_vectorization_t::none), VectorSize,
           static_cast<int>(gemm_batch_type_t::strided), true> {
 public:
  using tile_type = TileType;
  using value_t = element_t;
  using index_t = typename std::make_signed<typename input_t::index_t>::type;
  using packetize_t = PacketizeJointMatrix<VectorSize, value_t, index_t>;
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

  static constexpr index_t sg_size = sg_rows * sg_cols;
  static constexpr index_t jm_row_frags =
      block_rows / tile_type::joint_matrix_M;
  static constexpr index_t jm_col_frags =
      block_cols / tile_type::joint_matrix_N;
  static constexpr index_t num_jm_frags = jm_row_frags * jm_col_frags;
  static constexpr index_t num_sub_groups = wg_size / sg_size;
  static constexpr index_t frags_per_sg = num_jm_frags / num_sub_groups;

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

  static_assert(big_tile_rows == big_tile_cols,
                "Big tile level dimensions should be square, i.e. tl_rows * "
                "block_rows == tl_cols * block_cols");

  static_assert(item_rows % packetize_t::packet_size == 0,
                "Item rows must be a multiple of the vector packet size");

  static_assert(cl_elems % packetize_t::packet_size == 0,
                "Cache line size must be a multiple of packet_size");

  static_assert(sg_size == 32, "Sub_group size must be equal to 32");

  static_assert(std::is_same<value_t, float>::value,
                "This code is only supported for float data type.");

  //! @brief leading dimension of block of A in local
  static constexpr index_t ldsa =
      block_rows + nbc_a * tile_type::joint_matrix_M / sizeof(float) * 2;
  //! @brief leading dimension of block of B in local
  static constexpr index_t ldsb =
      cl_elems + nbc_b * tile_type::joint_matrix_K / sizeof(float) * 2;

  static constexpr index_t ldsc = block_rows + (nbc_a | nbc_b) *
                                                   tile_type::joint_matrix_M /
                                                   sizeof(float) * 2;
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
        beta_(beta),
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
        << "gemm_vectorization:none, "
        << "vector size" << VectorSize << ", batch_type:strided> "
        << "with joint_matrix extension";
    return str.str();
  }

  /*!
   *@brief get_wg_*_cluster. This function is used to find the optimum
   *number of work_group required to execute each GEMM.
   *
   */
  PORTBLAS_INLINE index_t get_wg_x_cluster() const noexcept {
    return ((a_.get_size_row() - 1) / tile_type::joint_matrix_M + 1);
  }

  PORTBLAS_INLINE index_t get_wg_y_cluster() const noexcept {
    return ((b_.get_size_col() - 1) / tile_type::joint_matrix_N + 1);
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
  PORTBLAS_INLINE cl::sycl::nd_range<1> get_nd_range(index_t) const noexcept {
    size_t x_groups =
        static_cast<size_t>((get_wg_x_cluster() - 1) / jm_row_frags + 1);
    size_t y_groups =
        static_cast<size_t>((get_wg_y_cluster() - 1) / jm_col_frags + 1);
#ifdef VERBOSE
    std::cout << " M: " << a_.get_size_row() << " , N " << b_.get_size_col()
              << " , big_tile_rows: " << big_tile_rows
              << " , big_tile_cols: " << big_tile_cols
              << " , wg_size: " << wg_size << " , nwg : " << x_groups * y_groups
              << std::endl;
#endif
    return cl::sycl::nd_range<1>{x_groups * batch_size_ * y_groups * wg_size,
                                 wg_size};
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
    index_t k = a_.get_size_col();

    const index_t lda = a_.getSizeL();
    const index_t ldb = b_.getSizeL();
    const index_t ldc = c_.getSizeL();

    const index_t wg_id = static_cast<index_t>(id.get_group(0));
    // The batch index that each workgroup should start working with
    const index_t x_groups = (get_wg_x_cluster() - 1) / jm_row_frags + 1;
    const index_t y_groups = (get_wg_y_cluster() - 1) / jm_col_frags + 1;
    const index_t wg_batch_id = wg_id / (x_groups * y_groups);
    // This will disable all workgroups that dont have any batch to work on
    if (wg_batch_id >= batch_size_) {
      return;
    }
    const index_t batch_stride = id.get_group_range(0) / (x_groups * y_groups);

    auto scratch = scratch_acc.localAcc.get_pointer();

    // The number of work-group required to executed each batch efficiently
    const index_t wg_id_x = wg_id % x_groups;
    const index_t wg_id_y = (wg_id / x_groups) % y_groups;

    const index_t a_size = trans_a ? m * lda : k * lda;
    const index_t b_size = trans_b ? ldb * k : n * ldb;
    const index_t c_size = ldc * n;

    using address_t = cl::sycl::access::address_space;
    auto ptr_A = cl::sycl::multi_ptr<const element_t, address_t::global_space>(
                     a_.get_pointer()) +
                 (wg_batch_id * stridea_);
    auto ptr_B = cl::sycl::multi_ptr<const element_t, address_t::global_space>(
                     b_.get_pointer()) +
                 (wg_batch_id * strideb_);
    auto ptr_C = cl::sycl::multi_ptr<element_t, address_t::global_space>(
                     c_.get_pointer()) +
                 (wg_batch_id * stridec_);

    auto sg = id.get_sub_group();
    const index_t sg_id = sg.get_group_linear_id();
    const index_t item_id = id.get_local_linear_id();

    const index_t wg_row = wg_id_x * block_rows;
    const index_t wg_col = wg_id_y * block_cols;
    const bool out_of_range = (wg_row >= m || wg_col >= n);
    const bool internal = m - wg_row >= block_rows && n - wg_col >= block_cols;

    ptr_C += (wg_row + wg_col * ldc);

    const index_t mc = m - wg_row;
    const index_t nc = n - wg_col;

    const index_t it_mod_brows = item_id % block_rows;
    const index_t it_div_brows = item_id / block_rows;

    ptr_C += (it_mod_brows + it_div_brows * ldc);

    const index_t it_mod_bcols = item_id % block_cols;
    const index_t it_div_bcols = item_id / block_cols;

    const index_t it_mod_cl = item_id % cl_elems;
    const index_t it_div_cl = item_id / cl_elems;

    ptr_B += (trans_b ? it_div_bcols * ldb + (wg_col + it_mod_bcols)
                      : it_mod_cl + (wg_col + it_div_cl) * ldb);

    n = n - wg_col - (trans_b ? it_mod_bcols : it_div_cl);
    ptr_A += (trans_a ? (wg_row + it_div_cl) * lda + it_mod_cl
                      : (wg_row + it_mod_brows) + it_div_brows * lda);

    m = m - wg_row - (trans_a ? it_div_cl : it_mod_brows);

    const index_t s1_offset = (trans_b ? it_div_bcols + it_mod_bcols * ldsb
                                       : it_mod_cl + it_div_cl * ldsb);

    const index_t s2_offset =
        (sg_id / jm_row_frags) * tile_type::joint_matrix_N * ldsb;

    const index_t ofs = (double_buffer + 1) * ldsb * block_cols;

    const index_t s3_offset =
        ofs + (trans_a ? it_div_cl + (it_mod_cl)*ldsa
                       : it_mod_brows + it_div_brows * ldsa);

    const index_t s4_offset =
        ofs + (sg_id % jm_row_frags) * tile_type::joint_matrix_M;

    if constexpr (std::is_same<typename tile_type::jmInpType,
                               cl::sycl::ext::oneapi::experimental::matrix::
                                   precision::tf32>::value) {
      auto s1 = scratch + s1_offset;
      auto s2 = scratch + s2_offset;
      auto s3 = scratch + s3_offset;
      auto s4 = scratch + s4_offset;

      if (internal) {
        compute_panel_gemm<double_buffer, false, false>(
            id, item_id, m, n, k, mc, nc, a_size, b_size, c_size, ptr_A, lda,
            ptr_B, ldb, ptr_C, ldc, scratch, s1, s2, s3, s4, out_of_range,
            batch_stride, wg_batch_id, batch_size_);
      } else {
        compute_panel_gemm<double_buffer, true, true>(
            id, item_id, m, n, k, mc, nc, a_size, b_size, c_size, ptr_A, lda,
            ptr_B, ldb, ptr_C, ldc, scratch, s1, s2, s3, s4, out_of_range,
            batch_stride, wg_batch_id, batch_size_);
      }
    } else {
      auto input_scratch = *reinterpret_cast<cl::sycl::multi_ptr<
          typename tile_type::jmInpType, address_t::local_space> *>(&scratch);

      auto s1 = input_scratch + s1_offset;
      auto s2 = input_scratch + s2_offset;
      auto s3 = input_scratch + s3_offset;
      auto s4 = input_scratch + s4_offset;
      if (internal) {
        compute_panel_gemm<double_buffer, false, false>(
            id, item_id, m, n, k, mc, nc, a_size, b_size, c_size, ptr_A, lda,
            ptr_B, ldb, ptr_C, ldc, scratch, s1, s2, s3, s4, out_of_range,
            batch_stride, wg_batch_id, batch_size_);
      } else {
        compute_panel_gemm<double_buffer, true, true>(
            id, item_id, m, n, k, mc, nc, a_size, b_size, c_size, ptr_A, lda,
            ptr_B, ldb, ptr_C, ldc, scratch, s1, s2, s3, s4, out_of_range,
            batch_stride, wg_batch_id, batch_size_);
      }
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
  /*!
   * @brief Compute a GEMM of a block-row of A (transpose) and a
   * block-column of B (transpose).
   *
   * This is a collective operation between all items in as work-group.
   * This method should actually be a generic lambda (as in C++20), it
   * only forwards parameters from GemmFactory::run().
   *
   * @tparam check_m_limit  iff true, check if row indices of C are
   *                        out-of-bound
   * @tparam check_n_limit  iff true, check if column indices of C are
   *                        out-of-bound
   */
  template <bool double_buffer, bool check_m_limit, bool check_n_limit,
            typename InputPointerType, typename OutputPointerType,
            typename OutputScratchPointerType, typename InputScratchPointerType>
  PORTBLAS_INLINE void compute_panel_gemm(
      const cl::sycl::nd_item<1> &id, const index_t &item_id, const index_t &m,
      const index_t &n, const index_t &orig_k, const index_t &mc,
      const index_t &nc, const index_t &a_size, const index_t &b_size,
      const index_t &c_size, InputPointerType orig_A, const index_t &lda,
      InputPointerType orig_B, const index_t &ldb, OutputPointerType orig_C,
      const index_t &ldc, OutputScratchPointerType s0,
      InputScratchPointerType s1, InputScratchPointerType s2,
      InputScratchPointerType s3, InputScratchPointerType s4,
      const bool out_of_range, index_t batch_stride, index_t wg_batch_id,
      index_t batch_size) noexcept {
    index_t ofs = 1;
    using namespace cl::sycl::ext::oneapi::experimental::matrix;
    using CType =
        joint_matrix<cl::sycl::sub_group, typename tile_type::jmOutType,
                     use::accumulator, tile_type::joint_matrix_M,
                     tile_type::joint_matrix_N>;
    do {
      auto A = orig_A;
      auto B = orig_B;
      auto C = orig_C;
      auto k = orig_k;
      CType reg_res[frags_per_sg] = {};
      while (k >= cl_elems) {
        extract_input_blocks<check_m_limit, check_n_limit, false>(
            item_id, m, n, k, A, lda, B, ldb, s1, s3, out_of_range);
        id.barrier(cl::sycl::access::fence_space::local_space);
        compute_block_gemm<check_m_limit, check_n_limit>(id, s2, s4, reg_res);
        A += cl_elems * (trans_a ? 1 : lda);
        B += cl_elems * (trans_b ? ldb : 1);

        sync_smem<double_buffer, ldsb * block_cols, ldsb * block_cols,
                  ldsa * cl_elems, ldsa * cl_elems>(id, ofs, s1, s2, s3, s4);
        k -= cl_elems;
      }

      if (k > 0) {
        extract_input_blocks<check_m_limit, check_n_limit, true>(
            item_id, m, n, k, A, lda, B, ldb, s1, s3, out_of_range);
        id.barrier(cl::sycl::access::fence_space::local_space);
        compute_block_gemm<check_m_limit, check_n_limit>(id, s2, s4, reg_res);

        sync_smem<double_buffer, ldsb * block_cols, ldsb * block_cols,
                  ldsa * cl_elems, ldsa * cl_elems>(id, ofs, s1, s2, s3, s4);
      }

      // store the output
      store_output_block<check_m_limit, check_n_limit>(id, mc, nc, C, s0, ldc,
                                                       reg_res, out_of_range);
      orig_A += (stridea_ * batch_stride);
      orig_B += (strideb_ * batch_stride);
      orig_C += (stridec_ * batch_stride);
      // batch_size_ must be signed as the negative value has meaning here.
      batch_size -= batch_stride;
    } while (batch_size > wg_batch_id);
  }

  /*!
   * @brief Store the computed gemm result to the C matrix after performing
   * input and output scaling. Output scaling only takes place if beta is
   * non-zero.
   *
   * @tparam check_m_limit  iff true, check if row indices of C are
   *                        out-of-bound
   * @tparam check_n_limit  iff true, check if no indices of C are
   *                        out-of-bound
   * @tparam ScratchPointerType the type of shared local memory pointer
   * @tparam OutputPointerType the type of C
   * @tparam CType the type for joint_matrix fragment to store output
   * @param mc the computed boundary limit of m in matrix C
   * @param nc the computed boundary limit of n in matrix C
   * @param alpha  scaling factor of AB
   * @param beta  scaling factor of C
   * @param C  pointer to the first element of C
   * @param ldc  leading dimension of C
   * @param reg_res  joint_matrix fragment array containing the partial result
   * of C per sub-group
   */

  template <bool check_m_limit, bool check_n_limit, typename OutputPointerType,
            typename ScratchPointerType, typename CType>
  PORTBLAS_INLINE void store_output_block(cl::sycl::nd_item<1> id, index_t mc,
                                          index_t nc, OutputPointerType C,
                                          ScratchPointerType scratch,
                                          index_t ldc,
                                          CType (&reg_res)[frags_per_sg],
                                          const bool out_of_range) noexcept {
    if (out_of_range) {
      return;
    }
    using namespace cl::sycl::ext::oneapi::experimental::matrix;
    using Cfloat_Type =
        joint_matrix<cl::sycl::sub_group, element_t, use::accumulator,
                     tile_type::joint_matrix_M, tile_type::joint_matrix_N>;

    Cfloat_Type float_out;
    auto sg = id.get_sub_group();

    const index_t item_id = static_cast<index_t>(id.get_local_linear_id());
    const index_t local_range = static_cast<index_t>(id.get_local_range(0));
    const index_t sg_id = static_cast<index_t>(sg.get_group_linear_id());

    const index_t output_local_store_offset =
        (sg_id % jm_row_frags) * tile_type::joint_matrix_M +
        (sg_id / jm_row_frags) * ldsc * tile_type::joint_matrix_N;
    const index_t output_local_load_offset =
        item_id % block_rows + (item_id / block_rows) * ldsc;
    const index_t rows_per_iter = local_range / block_rows;
    const index_t loop_limit =
        (frags_per_sg == 1 ? block_cols : tile_type::joint_matrix_N) /
        rows_per_iter;

    const index_t output_global_outer_offset = ldc * tile_type::joint_matrix_N;
    const index_t output_global_inner_offset = ldc * rows_per_iter;
    const index_t output_local_inner_offset = ldsc * rows_per_iter;

    for (index_t frag = 0; frag < frags_per_sg;
         frag++, C += output_global_outer_offset) {
      auto new_C = C;
      auto new_scratch = scratch + output_local_load_offset;

      joint_matrix_copy(sg, reg_res[frag], float_out);
      joint_matrix_apply(sg, float_out, [=](element_t &x) { x *= alpha_; });

      id.barrier(cl::sycl::access::fence_space::local_space);

      joint_matrix_store(sg, float_out, scratch + output_local_store_offset,
                         ldsc, layout::col_major);

      id.barrier(cl::sycl::access::fence_space::local_space);

      for (int i = 0; i < loop_limit; i++, new_C += output_global_inner_offset,
               new_scratch += output_local_inner_offset) {
        *new_C = *new_scratch;
      }
    }
  }

  /*!
   * @brief Extract a block of A, and a conformant block of B.
   *
   * @see GemmFactory::extract_block()
   */
  template <bool check_m_limit, bool check_n_limit, bool check_k_limit,
            typename InputPointerType, typename ScratchPointerType>
  PORTBLAS_INLINE void extract_input_blocks(
      index_t item_id, index_t m, index_t n, index_t k, InputPointerType A,
      index_t lda, InputPointerType B, index_t ldb, ScratchPointerType sB,
      ScratchPointerType sA, const bool out_of_range) noexcept {
    if (out_of_range) {
      return;
    }

    extract_block<!check_m_limit && !check_n_limit, check_m_limit,
                  check_k_limit, trans_a, block_rows, cl_elems, ldsa>(
        item_id, A, lda, sA,
        [&](index_t, index_t cr) PORTBLAS_ALWAYS_INLINE { return cr < m; },
        [&](index_t ic, index_t cc)
            PORTBLAS_ALWAYS_INLINE { return cc < k - ic; });
    extract_block<!check_m_limit && !check_n_limit, check_k_limit,
                  check_n_limit, trans_b, cl_elems, block_cols, ldsb>(
        item_id, B, ldb, sB,
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
            bool trans, index_t rows, index_t cols, index_t lds,
            typename InputPointerType, typename ScratchPointerType,
            typename RowPredicate, typename ColPredicate>
  PORTBLAS_INLINE typename std::enable_if<!trans>::type extract_block(
      index_t item_id, InputPointerType ptr, index_t ld,
      ScratchPointerType scratch, RowPredicate in_row, ColPredicate in_col) {
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

      packetize_t::template load<trans, internal, lds>(
          in_range, ptr + col_ofs * ld, scratch + col_ofs * lds,
          [&](const index_t &ofs) {
            return in_row((item_id * multiplier) % rows, ofs) &&
                   in_col((item_id * multiplier) / rows, col_ofs);
          });
    }
  }
  template <bool internal, bool check_row_limit, bool check_col_limit,
            bool trans, index_t rows, index_t cols, index_t lds,
            typename InputPointerType, typename ScratchPointerType,
            typename RowPredicate, typename ColPredicate>
  PORTBLAS_INLINE typename std::enable_if<trans>::type extract_block(
      index_t item_id, InputPointerType ptr, index_t ld,
      ScratchPointerType scratch, RowPredicate in_row, ColPredicate in_col) {
    constexpr index_t bs = rows * cols;
    constexpr index_t multiplier = internal ? packetize_t::packet_size : 1;
    constexpr index_t loop_iterations = (bs - 1) / (wg_size * multiplier) + 1;
    constexpr index_t divisor =
        rows > cols ? block_rows == 128 ? 2 : 4 : loop_iterations / 2;
#pragma unroll
    for (index_t i = 0; i < loop_iterations; ++i) {
      if (!do_check<((bs % (wg_size * multiplier)) != 0)>(
              item_id + i * (wg_size * multiplier) < bs))
        continue;
      const index_t local_row_ofs =
          (i % divisor) * ((wg_size * multiplier) / cols) +
          i / divisor * lds * cols;
      const index_t row_ofs = i * ((wg_size * multiplier) / cols);
      const bool in_range = do_check<check_row_limit>(in_row(
                                (item_id * multiplier) / cols, row_ofs)) &&
                            do_check<check_col_limit>(in_col(
                                (item_id * multiplier) % cols, multiplier - 1));

      packetize_t::template load<trans, internal, lds>(
          in_range, ptr + row_ofs * ld, scratch + local_row_ofs,
          [&](const index_t &ofs) PORTBLAS_ALWAYS_INLINE {
            return in_col((item_id * multiplier) % cols, ofs) &&
                   in_row((item_id * multiplier) / cols, row_ofs);
          });
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
   */
  template <bool check_m_limit, bool check_n_limit, typename InputPointerType,
            typename CType>
  PORTBLAS_INLINE void compute_block_gemm(
      const cl::sycl::nd_item<1> &id, InputPointerType s2, InputPointerType s4,
      CType (&reg_res)[frags_per_sg]) noexcept {
    using namespace cl::sycl::ext::oneapi::experimental::matrix;
    using AType =
        joint_matrix<cl::sycl::sub_group, typename tile_type::jmInpType, use::a,
                     tile_type::joint_matrix_M, tile_type::joint_matrix_K,
                     layout::col_major>;
    using BType =
        joint_matrix<cl::sycl::sub_group, typename tile_type::jmInpType, use::b,
                     tile_type::joint_matrix_K, tile_type::joint_matrix_N,
                     layout::col_major>;

    AType inA;
    BType inB;

    const index_t strideA = ldsa;
    const index_t strideB = ldsb;

    auto sg = id.get_sub_group();

#pragma unroll
    for (index_t frag = 0; frag < frags_per_sg; frag++) {
      auto new_B = s2 + frag * tile_type::joint_matrix_N * ldsb;
      auto new_A = s4;

      for (index_t i = 0; i < cl_elems / tile_type::joint_matrix_K; i++) {
        joint_matrix_load(sg, inA, new_A, strideA);  // M
        joint_matrix_load(sg, inB, new_B, strideB);  // N

        joint_matrix_mad(sg, reg_res[frag], inA, inB, reg_res[frag]);

        new_A += tile_type::joint_matrix_K * strideA;
        new_B += tile_type::joint_matrix_K;
      }
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
      Ps &...ss) noexcept {
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

};  // Gemm

}  // namespace blas

#endif  // SB_ENABLE_JOINT_MATRIX
#endif  // PORTBLAS_BLAS3_LOCAL_GEMM_JOINT_MATRIX_HPP
