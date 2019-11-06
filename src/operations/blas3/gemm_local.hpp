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
#define SHOULD_PRINT
#define ID_TO_PRINT 0
#define WG_TO_PRINT 0
// Vectorization stuff

template <bool trans, size_t packet_size, typename value_t>
struct Packetize {
  template <bool internal, int ld, bool check_row = false,
            typename packet_t = cl::sycl::vec<value_t, packet_size>,
            typename SrcPointerType, typename DestPointerType, typename index_t,
            typename EdgePredicate>
  static SYCL_BLAS_INLINE typename std::enable_if<(internal == false)>::type
  load(const bool in_range, SrcPointerType src, const index_t &src_offset,
       DestPointerType dest, const index_t &dest_offset, EdgePredicate) {
    *(dest + dest_offset) = in_range ? *(src + src_offset) : value_t{0};
  }
  // template <bool internal, bool check_row = false,
  //           typename packet_t = cl::sycl::vec<value_t, packet_size>,
  //           typename SrcPointerType, typename DestPointerType, typename
  //           index_t, typename RowPredicate>
  // static SYCL_BLAS_INLINE
  //     typename std::enable_if<(internal == true && check_row)>::type
  //     load(bool in_range, SrcPointerType src, index_t src_offset,
  //          DestPointerType dest, index_t dest_offset, index_t item_id,
  //          RowPredicate in_row) {}
  template <bool internal, int ld,
            typename packet_t = cl::sycl::vec<value_t, packet_size>,
            typename SrcPointerType, typename DestPointerType, typename index_t,
            typename EdgePredicate>
  static SYCL_BLAS_INLINE typename std::enable_if<(internal == true)>::type
  load(const bool in_range, SrcPointerType src, const index_t &src_offset,
       DestPointerType dest, const index_t &dest_offset,
       EdgePredicate edge_in_range) {
    using address_t = cl::sycl::access::address_space;
    packet_t packet{0};
    if (in_range) {
      packet.template load<address_t::global_space>(0, src + src_offset);
    } else {
#pragma unroll
      for (index_t i = 0; i < packet_size; i++) {
        reinterpret_cast<value_t *>(&packet)[i] =
            edge_in_range(i) ? *(src + src_offset + i) : 0;
      }
    }

    store<trans, ld>(packet, dest, dest_offset);
  }
  template <bool tran, int ld,
            typename packet_t = cl::sycl::vec<value_t, packet_size>,
            typename DestPointerType, typename index_t>
  static SYCL_BLAS_INLINE typename std::enable_if<tran>::type store(
      packet_t &packet, DestPointerType dest, const index_t &dest_offset) {
#pragma unroll
    for (index_t i = 0; i < packet_size; i++) {
      *(dest + dest_offset + ld * i) = reinterpret_cast<value_t *>(&packet)[i];
    }
  }
  template <bool tran, int ld,
            typename packet_t = cl::sycl::vec<value_t, packet_size>,
            typename DestPointerType, typename index_t>
  static SYCL_BLAS_INLINE typename std::enable_if<!tran>::type store(
      packet_t &packet, DestPointerType dest, const index_t &dest_offset) {
    using address_t = cl::sycl::access::address_space;
    packet.template store<address_t::local_space>(0, dest + dest_offset);
  }
};
template <typename T, size_t Size>
struct VectorizationParams {
#ifdef GEMM_VECTORISATION_SUPPORT
  using vectorised_t = cl::sycl::vec<T, Size>;
  static constexpr size_t packet_size = Size;
#else
  // In the case where vectorization is not enabled, always set to 1
  using vectorised_t = cl::sycl::vec<T, 1>;
  static constexpr size_t packet_size = 1;
#endif
};

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
          typename element_t, bool is_beta_zero, int VectorSize>
class Gemm<input_t, output_t, DoubleBuffer, NbcA, NbcB, ClSize, TileType,
           TransA, TransB, element_t, is_beta_zero,
           static_cast<int>(gemm_memory_t::local),
           static_cast<int>(gemm_algorithm_t::standard), VectorSize> {
 public:
  using tile_type = TileType;
  using value_t = element_t;
  using vector_params = VectorizationParams<value_t, VectorSize>;
  using vector_t = typename vector_params::vectorised_t;
  using index_t = typename std::make_signed<typename input_t::index_t>::type;
  using address_t = cl::sycl::access::address_space;

  static constexpr index_t packet_size = vector_params::packet_size;
  static constexpr bool aligned = false;

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

  static_assert(item_rows % packet_size == 0,
                "Item rows must be a multiple of the vector packet size");

  static_assert(cl_elems % packet_size == 0,
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

  SYCL_BLAS_INLINE Gemm(input_t A, input_t B, output_t C, element_t alpha,
                        element_t beta, index_t batch_size)
      : a_(A),
        b_(B),
        c_(C),
        alpha_(alpha),
        beta_(beta / alpha),
        batch_size_(batch_size) {}

  /*!
   * @brief Get the type of this GemmFactory as a human readable string.
   */
  static SYCL_BLAS_INLINE std::string get_type_string() noexcept {
    std::ostringstream str{};
    str << "GemmFactory<" << double_buffer << ", " << nbc_a << ", " << nbc_b
        << ", " << cl_elems * sizeof(element_t) << ", "
        << tile_type::get_type_string() << ", "
        << type_string<value_t>::get_value() << ">";
    return str.str();
  }

  /*!
   *@brief gt_workgroup_cluster. This function is used to find the optimum
   *number of work_group required to execute each GEMM.
   *
   */
  SYCL_BLAS_INLINE index_t get_workgroup_cluster() const noexcept {
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
  SYCL_BLAS_INLINE index_t
  get_num_workgroup_cluster(index_t compute_units) const noexcept {
    return ((4 * compute_units - 1) / get_workgroup_cluster() + 1);
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
  SYCL_BLAS_INLINE cl::sycl::nd_range<1> get_nd_range(
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

  SYCL_BLAS_INLINE index_t get_size() const {
    return a_.get_size_row() * b_.get_size_col();
  }

  /*!
   * @brief Run the generated GEMM device function.
   * @tparam local_memory_t LocalMemory type
   * @param id  nd_item used for calls to local barriers
   * @param scratch local memory
   */
  template <typename local_memory_t>
  SYCL_BLAS_INLINE void eval(local_memory_t scratch_acc,
                             const cl::sycl::nd_item<1> &id) noexcept {
    index_t m = a_.get_size_row();
    index_t n = b_.get_size_col();
    index_t k = a_.get_size_col();

    const index_t lda = a_.getSizeL();
    const index_t ldb = b_.getSizeL();
    // printf("n: %d ldb: %d\n", n, ldb);
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
    using ScratchPointerType = decltype(scratch);
    // PointerWrapper<ScratchPointerType> s1, s2, s3, s4;
    // The number of work-group required to executed each batch efficiently
    const index_t wg_id = id.get_group(0) % get_workgroup_cluster();

    const index_t a_size = trans_a ? m * lda : k * lda;
    const index_t b_size = trans_b ? ldb * k : n * ldb;
    const index_t c_size = ldc * n;
    // PointerWrapper orig_A, orig_B, orig_C;

    auto ptr_A = a_.get_data().get_pointer() + a_.get_access_displacement() +
                 (wg_batch_id * a_size);
    auto ptr_B = b_.get_data().get_pointer() + b_.get_access_displacement() +
                 (wg_batch_id * b_size);
    auto ptr_C = c_.get_data().get_pointer() + c_.get_access_displacement() +
                 (wg_batch_id * c_size);
    // using PointerType = decltype(orig_A);
    // PointerWrapper<PointerType> ptr_A{orig_A, orig_A};
    // PointerWrapper<PointerType> ptr_B{orig_B, orig_B};
    // PointerWrapper<PointerType> ptr_C{orig_C, orig_C};
    // auto ptr_A = orig_A;
    // auto ptr_B = orig_B;
    // auto ptr_C = orig_C;
    const index_t item_id = id.get_local_id(0);
    const index_t tile_id = wg_id / tile_size;
    const index_t tile_local_id = wg_id % tile_size;
    const index_t tiles_per_col = (m - 1) / big_tile_rows + 1;
    const index_t tile_row = (tile_id % tiles_per_col) * tl_rows;
    const index_t tile_col = (tile_id / tiles_per_col) * tl_cols;
    const index_t wg_row = (tile_row + tile_local_id % tl_rows) * block_rows;
    const index_t wg_col = (tile_col + tile_local_id / tl_rows) * block_rows;
    const bool out_of_range = (wg_row >= m || wg_col >= n);
    const bool internal = m - wg_row >= block_rows && n - wg_col >= block_cols;
    const index_t vector_offset = internal ? packet_size : 1;
    const index_t item_id_ofs = item_id * vector_offset;
    // const index_t item_row = item_id % wg_rows * vector_offset;
    // const index_t item_col = (item_id / wg_rows) * item_cols;
    const index_t row = wg_row + item_id % wg_rows * vector_offset;
    const index_t col = wg_col + (item_id / wg_rows) * item_cols;

    element_t reg_a[item_rows];
    element_t reg_b;
    // ptr_C.ptr += wg_row + (item_id % wg_rows) + col * ldc;
    ptr_C += row + col * ldc;

    // printf("[%d/%d] row:%d col:%d c_ofs: %d\n", wg_id, item_id, row, col,
    //        static_cast<int>(orig_C - test_c));
    const index_t mc = m - row;
    const index_t nc = n - col;

    ptr_B += (trans_b ? (item_id_ofs / block_cols) * ldb +
                            (wg_col + item_id_ofs % block_cols)
                      : item_id_ofs % cl_elems +
                            (wg_col + item_id_ofs / cl_elems) * ldb);

    // ptr_B.ptr +=
    //     (trans_b
    //          ? (item_id / block_cols) * ldb + (wg_col + item_id %
    //          block_cols) : item_id % cl_elems + (wg_col + item_id / cl_elems)
    //          * ldb);
    n = n - wg_col -
        (trans_b ? item_id_ofs % block_cols : item_id_ofs / cl_elems);
    ptr_A += (trans_a ? (wg_row + item_id_ofs / cl_elems) * lda +
                            (item_id_ofs % cl_elems)
                      : (wg_row + item_id_ofs % block_rows) +
                            (item_id_ofs / block_rows) * lda);
    // printf("[%d/%d] %d + %d * %d = %d\n", item_id, wg_id,
    //        (wg_row + item_id_ofs % block_rows), (item_id_ofs / block_rows),
    //        lda,
    //        (wg_row + item_id_ofs % block_rows) +
    //            (item_id_ofs / block_rows) * lda);
    // ptr_A.ptr +=
    //     (trans_a
    //          ? (wg_row + item_id / cl_elems) * lda + (item_id % cl_elems)
    //          : (wg_row + item_id % block_rows) + (item_id / block_rows) *
    //          lda);

    // printf("[%d] %d - %d - (%d mod %d) = %d\n", item_id, m, wg_row,
    //        item_id_ofs, block_rows, m - wg_row - (item_id_ofs %
    //        block_rows));
    m = m - wg_row -
        (trans_a ? item_id_ofs / cl_elems : item_id_ofs % block_rows);
    // s1.ptr = scratch +
    //          (trans_b ? item_id / block_cols + (item_id % block_cols) * ldsb
    //                   : item_id % cl_elems + (item_id / cl_elems) * ldsb);
    auto s1 =
        scratch +
        (trans_b ? item_id_ofs / block_cols + (item_id_ofs % block_cols) * ldsb
                 : item_id_ofs % cl_elems + (item_id_ofs / cl_elems) * ldsb);
    auto s2 = scratch + (item_id / wg_rows) * item_cols * ldsb;
    index_t ofs = (double_buffer + 1) * block_cols * ldsb;
    // s3.ptr = scratch + ofs +
    //          (trans_a ? item_id / cl_elems + (item_id % cl_elems) * ldsa
    //                   : item_id % block_rows + (item_id / block_rows) *
    //                   ldsa);
    auto s3 =
        scratch + ofs +
        (trans_a
             ? item_id_ofs / cl_elems + (item_id_ofs % cl_elems) * ldsa
             : item_id_ofs % block_rows + (item_id_ofs / block_rows) * ldsa);
    // printf("[%d] s3.ofs: %d / %d + (%d mod %d) * %d\n", item_id, item_id_ofs,
    //        cl_elems, item_id_ofs, cl_elems, ldsa);
    auto s4 = scratch + ofs + (item_id % wg_rows * vector_offset);
    // printf("[%d/%d] a: %d b: %d sa: %d sb: %d m: %d n: %d\n", wg_id, item_id,
    //        static_cast<int>(orig_A - test_ptr),
    //        static_cast<int>(orig_B - test_ptrb), static_cast<int>(s2 -
    //        scratch), static_cast<int>(s4 - (scratch + ofs)), m, n);
    if (internal) {
      compute_panel_gemm<double_buffer, false, false>(
          id, item_id, m, n, k, mc, nc, a_size, b_size, c_size, ptr_A, lda,
          ptr_B, ldb, ptr_C, ldc, s1, s2, s3, s4, reg_a, reg_b, out_of_range,
          batch_stride, wg_batch_id);
    } else {
      compute_panel_gemm<double_buffer, true, true>(
          id, item_id, m, n, k, mc, nc, a_size, b_size, c_size, ptr_A, lda,
          ptr_B, ldb, ptr_C, ldc, s1, s2, s3, s4, reg_a, reg_b, out_of_range,
          batch_stride, wg_batch_id);
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
  SYCL_BLAS_INLINE bool valid_thread(const cl::sycl::nd_item<1> &ndItem) const {
    return true;
  }

 private:
  /** @brief If beta is not zero then this function will load in values from C,
  multiply them by the beta value and store them in the results register. If
  beta is zero then this function does nothing. */
  template <bool check_m_limit, bool check_n_limit, typename InputPointerType,
            bool beta_zero = is_beta_zero>
  SYCL_BLAS_INLINE typename std::enable_if<!beta_zero>::type scaling_c(
      element_t *reg_res, InputPointerType C, const index_t &mc,
      const index_t &nc, const index_t &ldc, const bool out_of_range) {
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
          reg_res[i * item_rows + j] = beta_ * *(C + j * wg_rows);
        }
      }
      C = C + ldc;
    }
  }

  template <bool check_m_limit, bool check_n_limit, typename InputPointerType,
            bool beta_zero = is_beta_zero>
  SYCL_BLAS_INLINE typename std::enable_if<beta_zero>::type scaling_c(
      element_t *reg_res, InputPointerType, const index_t &, const index_t &,
      const index_t &, const bool out_of_range) {
    if (out_of_range) {
      return;
    }
#pragma unroll
    for (index_t i = 0; i < item_cols * item_rows; ++i) {
      reg_res[i] = 0;
    }
  }
  // template <address_t src_address_space, address_t dest_address_space,
  //           index_t p_size = packet_size, typename SourcePointerType,
  //           typename DestPointerType>
  // static SYCL_BLAS_INLINE typename std::enable_if<(p_size > 1)>::type
  // load_val(
  //     bool aligned, SourcePointerType src, index_t src_index,
  //     DestPointerType dest, index_t dest_index) {
  //   if (aligned) {
  //     reinterpret_cast<vector_t *>(dest + dest_index)[0] =
  //         reinterpret_cast<vector_t *>(src + src_index)[0];
  //   } else {
  //     vector_t value{0};
  //     value.template load<src_address_space>(src_index / packet_size, src);
  //     value.template store<dest_address_space>(dest_index / packet_size,
  //     dest);
  //   }
  // }
  // template <address_t src_address_space, address_t dest_address_space,
  //           index_t p_size = packet_size, typename SourcePointerType,
  //           typename DestPointerType>
  // static SYCL_BLAS_INLINE typename std::enable_if<(p_size == 1)>::type
  // load_val(
  //     bool aligned, SourcePointerType src, index_t src_index,
  //     DestPointerType dest, index_t dest_index) {
  //   dest[dest_index] = src[src_index];
  // }

  // template <address_t address_space, index_t p_size = packet_size,
  //           typename DestPointerType>
  // static SYCL_BLAS_INLINE typename std::enable_if<(p_size > 1)>::type
  // store_value(bool aligned, DestPointerType dest, index_t dest_index,
  //             vector_t value) {
  //   if (aligned) {
  //     reinterpret_cast<vector_t *>(dest + dest_index)[0] = value;
  //   } else {
  //     value.template store<address_space>(0, dest + dest_index);
  //   }
  // }
  // template <address_t address_space, index_t p_size = packet_size,
  //           typename DestPointerType>
  // static SYCL_BLAS_INLINE typename std::enable_if<(p_size == 1)>::type
  // store_value(bool aligned, DestPointerType dest, index_t dest_index,
  //             vector_t value) {
  //   dest[dest_index] = value;
  // }

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
  SYCL_BLAS_INLINE void compute_panel_gemm(
      const cl::sycl::nd_item<1> &id, const index_t &item_id, const index_t &m,
      const index_t &n, const index_t &orig_k, const index_t &mc,
      const index_t &nc, const index_t &a_size, const index_t &b_size,
      const index_t &c_size, InputPointerType orig_A, const index_t &lda,
      InputPointerType orig_B, const index_t &ldb, OutputPointerType orig_C,
      const index_t &ldc, ScratchPointerType s1, ScratchPointerType s2,
      ScratchPointerType s3, ScratchPointerType s4, element_t *reg_a,
      element_t &reg_b, const bool out_of_range, const index_t &batch_stride,
      const index_t &wg_batch_id) noexcept {
    index_t ofs = 1;
    do {
      auto A = orig_A;
      auto B = orig_B;
      auto C = orig_C;
      auto k = orig_k;
      element_t reg_res[item_rows * item_cols];
      scaling_c<check_m_limit, check_n_limit>(reg_res, C, mc, nc, ldc,
                                              out_of_range);
      while (k >= cl_elems) {
        extract_input_blocks<check_m_limit, check_n_limit, false>(
            item_id, m, n, k, A, lda, B, ldb, s1, s3, out_of_range);
        id.barrier(cl::sycl::access::fence_space::local_space);
        compute_block_gemm<check_m_limit, check_n_limit>(item_id, s2, s4, reg_a,
                                                         reg_b, reg_res);
        A += cl_elems * (trans_a ? 1 : lda);
        // A.ptr_vec += cl_elems * (trans_a ? 1 : lda);
        B += cl_elems * (trans_b ? ldb : 1);
        // B.ptr_vec += cl_elems * (trans_b ? ldb : 1);
        // if (id.get_group(0) == WG_TO_PRINT)
        //   print_mat<WG_TO_PRINT, ldsa, cl_elems>(item_id, s3, k);
        // id.barrier();
        sync_smem<double_buffer, block_cols * ldsb, block_cols * ldsb,
                  ldsa * cl_elems, ldsa * cl_elems>(id, ofs, s1, s2, s3, s4);
        k -= cl_elems;
      }

      if (k > 0) {
        extract_input_blocks<check_m_limit, check_n_limit, true>(
            item_id, m, n, k, A, lda, B, ldb, s1, s3, out_of_range);
        id.barrier(cl::sycl::access::fence_space::local_space);
        compute_block_gemm<check_m_limit, check_n_limit>(item_id, s2, s4, reg_a,
                                                         reg_b, reg_res);

        // if (id.get_group(0) == WG_TO_PRINT)
        //   print_mat<WG_TO_PRINT, ldsb, block_cols>(item_id, s1, k);
        // id.barrier();
        sync_smem<double_buffer, block_cols * ldsb, block_cols * ldsb,
                  ldsa * cl_elems, ldsa * cl_elems>(id, ofs, s1, s2, s3, s4);
        // if (id.get_group(0) == 0) print_mat<0, ldsa, cl_elems>(item_id,
        // s3); if (id.get_group(0) == WG_TO_PRINT)
        //   print_array<item_rows * item_cols>(item_id, reg_res);
        // id.barrier();
      }

      // store the output
      store_output_block<check_m_limit, check_n_limit>(item_id, mc, nc, C, ldc,
                                                       reg_res, out_of_range);
      orig_A += (a_size * batch_stride);
      // orig_A.ptr_vec += (a_size * batch_stride);
      orig_B += (b_size * batch_stride);
      // orig_B.ptr_vec += (b_size * batch_stride);
      orig_C += (c_size * batch_stride);
      // batch_size_ must be signed as the negative value has meaning here.
      batch_size_ -= batch_stride;
    } while (batch_size_ > wg_batch_id);
  }

  //   template <index_t p_size = packet_size>
  //   static SYCL_BLAS_INLINE typename std::enable_if<(p_size == 1),
  //   vector_t>::type collect_value(element_t
  //   (&reg_res)[item_rows][item_cols], index_t row,
  //                 index_t col) {
  //     return reg_res[row][col];
  //   }
  //   template <index_t p_size = packet_size>
  //   static SYCL_BLAS_INLINE typename std::enable_if<(p_size > 1),
  //   vector_t>::type collect_value(element_t
  //   (&reg_res)[item_rows][item_cols], index_t row,
  //                 index_t col) {
  //     vector_t value{0};
  // #pragma unroll
  //     for (index_t i = 0; i < packet_size; ++i) {
  //       reinterpret_cast<element_t *>(&value)[i] = reg_res[row +
  //       i][col];
  //     }
  //     return value;
  //   }
  /*template<typename T = vector_t>
    static SYCL_BLAS_INLINE typename std::enable_if<typename
    std::is_same<vector_t, cl::sycl::float4>::type, float >::type
    scalarize(Index i, x){ switch(i ) { case 1 : return x.x() case 2:
    return x.y(); case 3: return x.z(); case 4 return x.w();
      }
      __builtin_unreachable();
    }*/
  // template <index_t p_size = packet_size>
  // static SYCL_BLAS_INLINE typename std::enable_if<(p_size == 1)>::type
  // set_scalar(index_t i, vector_t &x, element_t &val) {
  //   x = val;
  // }
  // template <index_t p_size = packet_size>
  // static SYCL_BLAS_INLINE typename std::enable_if<(p_size > 1)>::type
  // set_scalar(index_t i, vector_t &x, element_t &val) {
  //   switch (i) {
  //     case 1:
  //       x.x() = val;
  //       return;
  //     case 2:
  //       x.y() = val;
  //       return;
  //     case 3:
  //       x.z() = val;
  //       return;
  //     case 4:
  //       x.w() = val;
  //       return;
  //   }
  //   __builtin_unreachable();
  // }
  template <bool internal, index_t p_size = packet_size,
            typename OutputPointerType>
  SYCL_BLAS_INLINE typename std::enable_if<!internal>::type store_packet(
      element_t *reg, OutputPointerType out_ptr) {
    *out_ptr = alpha_ * *reg;
    // *out_ptr = item_id;
  }

  template <bool internal, index_t p_size = packet_size,
            typename OutputPointerType>
  SYCL_BLAS_INLINE typename std::enable_if<internal>::type store_packet(
      element_t *reg, OutputPointerType out_ptr) {
    vector_t out_vec{0};
    // value_t* out[stride];
    // #pragma unroll
    //     for (int i = 0; i < packet_size; i++) {
    //       reinterpret_cast<element_t *>(&out_vec)[i] = *(reg);
    //       // set_scalar(i, out_vec, *(reg + i * item_cols));
    //     }
    out_vec.template load<address_t::private_space>(0, reg);
    out_vec *= alpha_;

    // tmp += (alpha * out_vec);
    // tmp = vector_t{static_cast<float>(item_id)};
    // printf("value: %f\n", reinterpret_cast<value_t *>(&out_vec)[0]);
    out_vec.template store<address_t::global_space>(0, out_ptr);
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
  SYCL_BLAS_INLINE void store_output_block(index_t item_id, index_t mc,
                                           index_t nc, OutputPointerType C,
                                           index_t ldc, element_t *reg_res,
                                           const bool out_of_range) noexcept {
    if (out_of_range) {
      return;
    }
    constexpr index_t offset =
        (!check_m_limit && !check_n_limit) ? packet_size : 1;
#pragma unroll
    for (index_t i = 0; i < item_cols; ++i) {
#pragma unroll
      for (index_t j = 0; j < item_rows / offset; j++) {
        const bool in_range =
            do_check<check_m_limit>(j * wg_rows * offset < mc) &&
            do_check<check_n_limit>(i < nc);

        if (in_range) {
          store_packet<!check_m_limit && !check_n_limit>(
              reg_res, C + j * (wg_rows * offset));
          // cl::sycl::vec<element_t, offset> out_vec{0};
          // out_vec.template load<address_t::private_space>(0, reg_res);
          // out_vec *= alpha_;

          // out_vec.template store<address_t::global_space>(
          //     0, C + j * (wg_rows * offset));
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
            typename InputPointerType, typename ScratchPointerType>
  SYCL_BLAS_INLINE void extract_input_blocks(
      index_t item_id, index_t m, index_t n, index_t k, InputPointerType A,
      index_t lda, InputPointerType B, index_t ldb, ScratchPointerType sB,
      ScratchPointerType sA, const bool out_of_range) noexcept {
    if (out_of_range) {
      return;
    }

    extract_block<!check_m_limit && !check_n_limit, check_m_limit,
                  check_k_limit, trans_a, block_rows, cl_elems, ldsa>(
        item_id, A, lda, sA,
        [&](index_t ir, index_t cr) SYCL_BLAS_ALWAYS_INLINE { return cr < m; },
        [&](index_t ic, index_t cc)
            SYCL_BLAS_ALWAYS_INLINE { return cc < k - ic; });
    extract_block<!check_m_limit && !check_n_limit, check_k_limit,
                  check_n_limit, trans_b, cl_elems, block_cols, ldsb>(
        item_id, B, ldb, sB,
        [&](index_t ir, index_t cr)
            SYCL_BLAS_ALWAYS_INLINE { return cr < k - ir; },
        [&](index_t ic, index_t cc) SYCL_BLAS_ALWAYS_INLINE { return cc < n; });
  }
  // template <bool trans, index_t lds, bool check_row_limit,
  //           typename SrcPointerType, typename DestPointerType,
  //           typename RowPredicate>
  // static SYCL_BLAS_INLINE typename std::enable_if<!trans>::type
  // load_vector(
  //     SrcPointerType src, index_t src_index, DestPointerType dest,
  //     index_t dest_index, RowPredicate in_row, bool in_range, index_t
  //     item_id) {
  //   *reinterpret_cast<vector_t *>(static_cast<element_t *>(dest) +
  //   dest_index) =
  //       *reinterpret_cast<vector_t *>(static_cast<element_t *>(src) +
  //                                     src_index);
  // }
  //   template <bool trans, index_t lds, bool check_row_limit,
  //             typename SrcPointerType, typename DestPointerType,
  //             typename RowPredicate>
  //   static SYCL_BLAS_INLINE
  //       typename std::enable_if<!trans && check_row_limit>::type
  //       load_vector(SrcPointerType src, index_t src_index, DestPointerType
  //       dest,
  //                   index_t dest_index, RowPredicate in_row, bool in_range,
  //                   index_t item_id) {
  //     vector_t vec{0};
  //     vec.template load<address_t::global_space>(0, src + src_index);
  // // vec = vector_t{static_cast<float>(item_id)};
  // #pragma unroll
  //     for (index_t i = 0; i < packet_size; ++i) {
  //       const bool in_range = do_check<check_row_limit>(in_row(i));
  //       *(dest + dest_index + i) =
  //           in_range ? reinterpret_cast<value_t *>(&vec)[i] : value_t(i);
  //     }
  //   }
  //   template <bool trans, index_t lds, bool check_row_limit,
  //             typename SrcPointerType, typename DestPointerType,
  //             typename RowPredicate>
  //   static SYCL_BLAS_INLINE typename std::enable_if<!trans>::type
  //   load_vector(
  //       SrcPointerType src, index_t src_index, DestPointerType dest,
  //       index_t dest_index, RowPredicate in_row, bool in_range, index_t
  //       item_id) {
  //     vector_t vec{0};
  //     if (in_range) {
  //       vec.template load<address_t::global_space>(0, src + src_index);
  //     }
  //     vec.template store<address_t::local_space>(0, dest + dest_index);
  //   }
  //   template <bool trans, index_t lds, bool check_row_limit,
  //             typename SrcPointerType, typename DestPointerType,
  //             typename RowPredicate>
  //   static SYCL_BLAS_INLINE typename std::enable_if<trans>::type load_vector(
  //       SrcPointerType src, index_t src_index, DestPointerType dest,
  //       index_t dest_index, RowPredicate in_row, bool in_range, index_t
  //       item_id) {
  //     vector_t vec{0};
  //     vec.template load<address_t::global_space>(0, src + src_index);
  // // vec = vector_t{static_cast<float>(item_id)};
  // #pragma unroll
  //     for (index_t i = 0; i < packet_size; ++i) {
  //       const bool in_range = do_check<check_row_limit>(in_row(i));
  //       *(dest + dest_index + i * lds) =
  //           in_range ? reinterpret_cast<value_t *>(&vec)[i] : value_t(0);
  //     }
  //   }

  //   template <bool internal, bool check_row_limit = false, bool trans =
  //   false,
  //             index_t lds = 0, index_t p_size = packet_size,
  //             typename SrcPointerType, typename DestPointerType,
  //             typename RowPredicate>
  //   static SYCL_BLAS_INLINE typename std::enable_if<internal>::type
  //   load_value(
  //       SrcPointerType src, index_t src_index, DestPointerType dest,
  //       index_t dest_index, index_t item_id, RowPredicate in_row, bool
  //       in_range) {
  //     load_vector<trans, lds, check_row_limit>(src, src_index, dest,
  //     dest_index,
  //                                              in_row, in_range, item_id);
  //   }
  //   template <bool internal, bool check_row_limit = false, bool trans =
  //   false,
  //             index_t lds = 0, index_t p_size = packet_size,
  //             typename SrcPointerType, typename DestPointerType,
  //             typename RowPredicate>
  //   static SYCL_BLAS_INLINE typename std::enable_if<!internal>::type
  //   load_value(
  //       SrcPointerType src, index_t src_index, DestPointerType dest,
  //       index_t dest_index, index_t item_id, RowPredicate in_row, bool
  //       in_range) {
  // #pragma unroll
  //     for (index_t i = 0; i < packet_size; i++) {
  //       *(dest + dest_index + i) = in_range ? *(src + src_index + i) :
  //       value_t{0};
  //       // *(dest + dest_index + i) = value_t(item_id);
  //     }
  //   }

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
  SYCL_BLAS_INLINE typename std::enable_if<!trans>::type extract_block(
      index_t item_id, InputPointerType ptr, index_t ld,
      ScratchPointerType scratch, RowPredicate in_row, ColPredicate in_col) {
    constexpr index_t bs = rows * cols;
    constexpr index_t multiplier = internal ? packet_size : 1;
    // const bool aligned = ld % packet_size == 0;
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
      // load_value<internal, check_row_limit, false, lds>(
      //     ptr, col_ofs * ld, scratch, col_ofs * lds, item_id,
      //     [&](index_t ofs) SYCL_BLAS_ALWAYS_INLINE {
      //       // if (in_row(item_id % rows * multiplier, ofs) && item_id ==
      //       // 0)
      //       //   printf("[%d] %d < %d - %d mod %d (%d) * %d (%d)\n",
      //       //   item_id, ofs,
      //       //          2, item_id, rows, item_id % rows, multiplier,
      //       //          item_id % rows * multiplier);
      //       return in_row(((item_id * multiplier) % rows) + ofs, 0);
      //     },
      //     in_range);

      Packetize<trans, packet_size, element_t>::template load<internal, lds>(
          in_range, ptr, col_ofs * ld, scratch, col_ofs * lds,
          [&](const index_t &ofs) {
            return in_row(item_id * multiplier % rows, ofs);
          });

      // *(scratch + col_ofs * lds) =
      //     in_range ? *(ptr + col_ofs * ld) : element_t(0);
    }
  }
  template <bool internal, bool check_row_limit, bool check_col_limit,
            bool trans, index_t rows, index_t cols, index_t lds,
            typename InputPointerType, typename ScratchPointerType,
            typename RowPredicate, typename ColPredicate>
  SYCL_BLAS_INLINE typename std::enable_if<trans>::type extract_block(
      index_t item_id, InputPointerType ptr, index_t ld,
      ScratchPointerType scratch, RowPredicate in_row, ColPredicate in_col) {
    const index_t bs = rows * cols;
    constexpr index_t multiplier = internal ? packet_size : 1;
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

      Packetize<trans, packet_size, element_t>::template load<internal, lds>(
          in_range, ptr, row_ofs * ld, scratch, row_ofs,
          [&](const index_t &ofs) SYCL_BLAS_ALWAYS_INLINE {
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
   * @param reg_a  temporary register array used to prefetch columns of A
   * @param reg_b  temporary register used to prefetch elements of B
   * @param reg_res  2D register array used to store the result C
   */
  template <index_t stride, typename ScratchPointerType>
  static SYCL_BLAS_INLINE void load_packet(element_t *reg,
                                           ScratchPointerType ptr, index_t j) {
// constexpr index_t stride = internal ? packet_size : 1;
#pragma unroll
    for (int i = 0; i < stride; i++) {
      reg[i + j * stride] = *(ptr + (i + j * wg_rows * stride));
    }
  }
  template <bool check_m_limit, bool check_n_limit, typename InputPointerType>
  SYCL_BLAS_INLINE void compute_block_gemm(index_t item_id, InputPointerType B,
                                           InputPointerType A, element_t *reg_a,
                                           element_t &reg_b,
                                           element_t *reg_res) noexcept {
    // NOTE: Adding "#pragma unroll" here reduces performance on AMD R9
    // Nano.
    //       Seems that the small reduction of arithmetic operations does
    //       not amortize the cost of loading the larger kernel binary
    //       resulting from loop unrollment.
    constexpr index_t work_per_load =
        !check_m_limit && !check_n_limit ? packet_size : 1;
    for (index_t i = 0; i < cl_elems; ++i) {
#pragma unroll
      for (index_t j = 0; j < item_rows / work_per_load; ++j) {
        // reg_a[j] = A[j * wg_rows];
        load_packet<work_per_load>(reg_a, A, j);
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
    // #pragma unroll
    //     for (index_t j = 0; j < item_cols; ++j) {
    // #pragma unroll
    //       for (index_t l = 0; l < item_rows; ++l) {
    //         if (item_id == 0) printf("[%d][%d] %f\n", l, j,
    //         reg_res[l][j]);
    //       }
    //     }
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
      const cl::sycl::nd_item<1> &id, index_t &ofs_sign, P &s,
      Ps &... ss) noexcept {
    s += ofs_sign * o;
    // s.ptr_vec += ofs_sign * o;
    sync_smem<db, os...>(id, ofs_sign, ss...);
  }

  template <bool db>
  static SYCL_BLAS_INLINE typename std::enable_if<db>::type sync_smem(
      const cl::sycl::nd_item<1> &, index_t &ofs_sign) noexcept {
    ofs_sign = -ofs_sign;
  }

  template <bool db, index_t..., typename... Ps>
  static SYCL_BLAS_INLINE typename std::enable_if<!db>::type sync_smem(
      const cl::sycl::nd_item<1> &id, index_t &, Ps &...) noexcept {
    id.barrier(cl::sycl::access::fence_space::local_space);
  }
  //   template <index_t num, index_t leading_dim, index_t other_dim,
  //             typename PointerType>
  //   static SYCL_BLAS_INLINE void print_mat(index_t id, PointerType ptr,
  //                                          index_t k) {
  // #ifdef SHOULD_PRINT
  //     if (id == 0) printf("======Matrix[%d]======\n", k);
  //     for (index_t i = 0; i < leading_dim; i++) {
  //       for (index_t j = 0; j < other_dim; j++) {
  //         if (id == 0) {
  //           printf("%.2f ", *(ptr + j * leading_dim + i));
  //         }
  //       }
  //       if (id == 0) printf("\n");
  //     }
  // #endif
  //   }
  //   template <index_t size, typename PointerType>
  //   static SYCL_BLAS_INLINE void print_array(index_t id, PointerType ptr) {
  //     for (index_t i = 0; i < size; i++) {
  //       if (id == 0) printf("[%d] %.2f\n", i, ptr[i]);
  //     }
  //   }
};  // Gemm

}  // namespace blas

#endif  // SYCL_BLAS_BLAS3_LOCAL_GEMM_HPP
