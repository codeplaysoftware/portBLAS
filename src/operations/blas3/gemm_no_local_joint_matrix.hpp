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
 *  @filename gemm_no_local_joint_matrix.hpp
 *
 **************************************************************************/

#ifndef PORTBLAS_BLAS3_NO_LOCAL_GEMM_JOINT_MATRIX_HPP
#define PORTBLAS_BLAS3_NO_LOCAL_GEMM_JOINT_MATRIX_HPP

#if defined SB_ENABLE_JOINT_MATRIX_PVC || defined SB_ENABLE_JOINT_MATRIX_ARC

#include "gemm_common.hpp"
#include "gemm_load_store_joint_matrix.hpp"

namespace blas {

using namespace cl::sycl;
using namespace cl::sycl::ext::oneapi::experimental;
using namespace cl::sycl::ext::oneapi::experimental::matrix;
using namespace cl::sycl::ext::oneapi;

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
           static_cast<int>(gemm_memory_t::no_local),
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

  using AFrag = joint_matrix<cl::sycl::sub_group, typename tile_type::jmInpType,
                             use::a, tile_type::joint_matrix_M,
                             tile_type::joint_matrix_K, layout::row_major>;
  using BFrag =
      joint_matrix<cl::sycl::sub_group, typename tile_type::jmInpType, use::b,
                   tile_type::joint_matrix_K, tile_type::joint_matrix_N,
                   layout::ext_intel_packed>;
  using CFrag = joint_matrix<cl::sycl::sub_group, typename tile_type::jmOutType,
                             use::accumulator, tile_type::joint_matrix_M,
                             tile_type::joint_matrix_N>;

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

  static constexpr int local_memory_size = 0;

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
  static constexpr index_t num_sub_groups = wg_size / sg_size;
  static constexpr index_t KK = cl_elems / tile_type::joint_matrix_K;
  static constexpr index_t a_frags = jm_row_frags * KK;
  static constexpr index_t b_frags = jm_col_frags * KK;
  static constexpr index_t c_frags = jm_row_frags * jm_col_frags;

  static constexpr index_t vnni_factor =
      sizeof(float) / sizeof(typename tile_type::jmInpType);

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

  static_assert(sg_size % 8 == 0, "Sub_group size must be a multiple of 8");

  // TODO: this assert should change to compare value_t with bfloat16
  static_assert(std::is_same<value_t, float>::value,
                "This code is only supported for float data type.");

  static_assert(vnni_factor == 2,
                "This implementation only works for vnni factor 2.");

  // these two asserts are required to make sure that we have a square (or
  // multiple square) fragment(s) which we can transpose using sg.shuffle
  // instruction
  static_assert(tile_type::joint_matrix_M == 8 && jm_row_frags % 2 == 0,
                "matrix A fragment should be square or multiple squares "
                "(16x16 for Arc and PVC)");

  static_assert((tile_type::joint_matrix_N * jm_col_frags) % 16 == 0,
                "matrix B fragment should also be square or multiple squares "
                "(16x16 for Arc and PVC)");
  input_t a_;
  input_t b_;
  output_t c_;
  const element_t alpha_;
  const element_t beta_;
  const index_t batch_size_;
  const index_t stridea_;
  const index_t strideb_;
  const index_t stridec_;
  const index_t frags_sg_m_;
  const index_t frags_sg_n_;
  const index_t sg_m_;
  const index_t sg_n_;
  const index_t total_sg_reqd_;
  const index_t total_wg_reqd_;
  const bool jm_feasible_;

  PORTBLAS_ALWAYS_INLINE Gemm(input_t A, input_t B, output_t C, element_t alpha,
                              element_t beta, index_t batch_size,
                              index_t stride_a, index_t stride_b,
                              index_t stride_c)
      : a_(A),
        b_(B),
        c_(C),
        alpha_(alpha),
        beta_(beta / alpha),
        batch_size_(batch_size),
        stridea_{stride_a},
        strideb_{stride_b},
        stridec_{stride_c},
        frags_sg_m_{(a_.get_size_row() - 1) / tile_type::joint_matrix_M + 1},
        frags_sg_n_{(b_.get_size_col() - 1) / tile_type::joint_matrix_N + 1},
        sg_m_{(frags_sg_m_ - 1) / jm_row_frags + 1},
        sg_n_{(frags_sg_n_ - 1) / jm_col_frags + 1},
        total_sg_reqd_{sg_m_ * sg_n_},
        total_wg_reqd_{(total_sg_reqd_ - 1) / num_sub_groups + 1},
        jm_feasible_{
            (a_.get_size_row() % (jm_row_frags * tile_type::joint_matrix_M) ==
             0) &&
            (b_.get_size_col() % (jm_col_frags * tile_type::joint_matrix_N) ==
             0) &&
            (b_.get_size_row() % tile_type::joint_matrix_K == 0) && !trans_a &&
            !trans_b} {}

  /*!
   * @brief Get the type of this GemmFactory as a human readable string.
   */
  static PORTBLAS_ALWAYS_INLINE std::string get_type_string() noexcept {
    std::ostringstream str{};
    str << "Gemm <" << double_buffer << ", " << nbc_a << ", " << nbc_b << ", "
        << cl_elems * sizeof(element_t) << ", " << tile_type::get_type_string()
        << ", " << type_string<value_t>::get_value() << "gemm_memory:no_local, "
        << "gemm_algorithm:standard, "
        << "gemm_vectorization:none, "
        << "vector size" << VectorSize << ", batch_type:strided> "
        << "with joint_matrix extension";
    return str.str();
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
  PORTBLAS_ALWAYS_INLINE cl::sycl::nd_range<1> get_nd_range(
      index_t) const noexcept {
#ifdef VERBOSE
    std::cout << " M: " << a_.get_size_row() << " , N " << b_.get_size_col()
              << " , big_tile_rows: " << big_tile_rows
              << " , big_tile_cols: " << big_tile_cols
              << " , wg_size: " << wg_size << " , nwg : " << total_wg_reqd_
              << std::endl;
#endif
    return cl::sycl::nd_range<1>{total_wg_reqd_ * batch_size_ * wg_size,
                                 wg_size};
  }

  PORTBLAS_ALWAYS_INLINE index_t get_size() const {
    return a_.get_size_row() * b_.get_size_col();
  }

  /*!
   * @brief Run the generated GEMM device function.
   * @param id  nd_item
   */
  PORTBLAS_ALWAYS_INLINE void eval(const cl::sycl::nd_item<1> &id) noexcept {
    index_t m = a_.get_size_row();
    index_t n = b_.get_size_col();
    index_t k = a_.get_size_col();

    const index_t lda = a_.getSizeL();
    const index_t ldb = b_.getSizeL();
    const index_t ldc = c_.getSizeL();
    // The batch index that each workgroup should start working with
    const index_t wg_id = id.get_group(0);
    const index_t wg_batch_id = wg_id / total_wg_reqd_;
    // This will disable all workgroups that dont have any batch to work on
    if (wg_batch_id >= batch_size_) {
      return;
    }
    const index_t batch_stride = id.get_group_range(0) / total_wg_reqd_;

    using address_t = cl::sycl::access::address_space;
    auto ptr_A = cl::sycl::multi_ptr<const element_t, address_t::global_space>(
                     a_.get_pointer()) +
                 wg_batch_id * stridea_;
    auto ptr_B = cl::sycl::multi_ptr<const element_t, address_t::global_space>(
                     b_.get_pointer()) +
                 wg_batch_id * strideb_;
    auto ptr_C = cl::sycl::multi_ptr<element_t, address_t::global_space>(
                     c_.get_pointer()) +
                 wg_batch_id * stridec_;

    auto sg = id.get_sub_group();
    const index_t item_global_id = id.get_global_linear_id();
    const index_t item_local_id = sg.get_local_linear_id();
    const index_t sg_global_id = item_global_id / sg_size;
    const index_t sg_start_y = sg_global_id % sg_n_;
    const index_t sg_start_x = sg_global_id / sg_n_;

    const index_t start_m =
        sg_start_x * tile_type::joint_matrix_M * jm_row_frags;
    const index_t start_n =
        sg_start_y * tile_type::joint_matrix_N * jm_col_frags;

    const bool out_of_range = start_m >= m || start_n >= n;
    const bool internal =
        m - start_m >= jm_row_frags * tile_type::joint_matrix_M &&
        n - start_n >= jm_col_frags * tile_type::joint_matrix_N;

    ptr_C += (start_n + start_m * ldc);

    // TODO: remove this reinterpret_cast once input pointers are of type
    // bfloat16
    auto computeA = *reinterpret_cast<cl::sycl::multi_ptr<
        const bfloat16, address_t::global_space, access::decorated::yes> *>(
        &ptr_A);
    auto computeB = *reinterpret_cast<cl::sycl::multi_ptr<
        const bfloat16, address_t::global_space, access::decorated::yes> *>(
        &ptr_B);
    // auto computeA = ptr_A;
    // auto computeB = ptr_B;

    if (out_of_range) {
      return;
    }

    computeB += start_n * (trans_b ? ldb : 1);
    computeA += start_m * (trans_a ? 1 : lda);

    if (jm_feasible_) {
      compute_panel_gemm<true, false, false>(
          id, item_local_id, m, n, k, start_n, computeA, lda, computeB, ldb,
          ptr_C, ldc, batch_stride, wg_batch_id, batch_size_);
    } else {
      m = m - start_m;
      n = n - start_n;

      if (internal) {
        compute_panel_gemm<false, false, false>(
            id, item_local_id, m, n, k, start_n, computeA, lda, computeB, ldb,
            ptr_C, ldc, batch_stride, wg_batch_id, batch_size_);
      } else {
        compute_panel_gemm<false, true, true>(
            id, item_local_id, m, n, k, start_n, computeA, lda, computeB, ldb,
            ptr_C, ldc, batch_stride, wg_batch_id, batch_size_);
      }
    }
  }

  template <bool beta_zero, bool internal, bool check_m_limit,
            bool check_n_limit, typename PointerType>
  PORTBLAS_ALWAYS_INLINE typename std::enable_if<!beta_zero>::type scale_c(
      const cl::sycl::nd_item<1> id, const index_t item_id, const index_t m,
      const index_t n, PointerType C, const index_t ldc, CFrag *sub_c) {
    auto sg = id.get_sub_group();
    if constexpr (internal) {
      auto row_ptr = C;
#pragma unroll
      for (index_t i = 0, idx = 0; i < jm_row_frags;
           i++, row_ptr += tile_type::joint_matrix_M * ldc) {
        auto col_ptr = row_ptr;
#pragma unroll
        for (index_t j = 0; j < jm_col_frags;
             j++, col_ptr += tile_type::joint_matrix_N) {
          joint_matrix_load(sg, sub_c[idx], col_ptr, ldc, layout::row_major);
          joint_matrix_apply(
              sg, sub_c[idx++],
              [=](typename tile_type::jmOutType &x) { x *= beta_; });
        }
      }
    } else {
      const auto row_lambda =
          [&](index_t ic, index_t cc)
              PORTBLAS_ALWAYS_INLINE { return cc < m - ic; };
      const auto col_lambda = [&](index_t, index_t cc)
                                  PORTBLAS_ALWAYS_INLINE { return cc < n; };
      constexpr index_t rows = jm_row_frags * tile_type::joint_matrix_M;
      constexpr index_t cols = jm_col_frags * tile_type::joint_matrix_N;
      typename tile_type::jmOutType reg[(rows * cols) / sg_size];
      extract_block<internal, check_m_limit, check_n_limit, false, rows, cols,
                    1, 1, 1, 1>(id, item_id, C, ldc, sub_c, reg, row_lambda,
                                col_lambda);
#pragma unroll
      for (index_t i = 0, c_idx = 0; i < jm_row_frags; i++) {
        index_t start_idx = i * tile_type::joint_matrix_M;
#pragma unroll
        for (index_t j = 0; j < jm_col_frags; j++, start_idx += rows) {
          auto wi_data_c =
              sycl::ext::oneapi::detail::get_wi_data(sg, sub_c[c_idx++]);
#pragma unroll
          for (index_t si = 0, reg_idx = start_idx; si < wi_data_c.length();
               si++, reg_idx++) {
            wi_data_c[si] = beta_ * reg[reg_idx];
          }
        }
      }
    }
  }

  template <bool beta_zero, bool internal, bool check_m_limit,
            bool check_n_limit, typename PointerType>
  PORTBLAS_ALWAYS_INLINE typename std::enable_if<beta_zero>::type scale_c(
      const cl::sycl::nd_item<1> id, const index_t, const index_t,
      const index_t, PointerType, const index_t, CFrag *sub_c) {
    auto sg = id.get_sub_group();
    for (index_t i = 0; i < c_frags; i++) {
      joint_matrix_fill(sg, sub_c[i], typename tile_type::jmOutType{0});
    }
  }

  template <bool jm_feasible, bool check_m_limit, bool check_n_limit,
            typename InputPointerType, typename OutputPointerType>
  PORTBLAS_ALWAYS_INLINE typename std::enable_if<jm_feasible>::type
  compute_panel_gemm(const cl::sycl::nd_item<1> &id, const index_t &item_id,
                     const index_t &m, const index_t &n, const index_t &orig_k,
                     const index_t &start_n, InputPointerType orig_A,
                     const index_t &lda, InputPointerType orig_B,
                     const index_t &ldb, OutputPointerType orig_C,
                     const index_t &ldc, index_t batch_stride,
                     index_t wg_batch_id, index_t batch_size) noexcept {
    auto sg = id.get_sub_group();
    do {
      auto A = orig_A;
      auto B = orig_B;
      auto C = orig_C;
      auto k = orig_k;

      AFrag sub_a[jm_row_frags];
      BFrag sub_b[jm_col_frags];
      CFrag sub_c[c_frags];

      scale_c<is_beta_zero, true, false, false>(id, item_id, m, n, C, ldc,
                                                sub_c);
#pragma unroll
      for (index_t k_idx = 0, new_k = k; k_idx < k;
           k_idx += tile_type::joint_matrix_K, A += tile_type::joint_matrix_K,
                   B += (tile_type::joint_matrix_K * ldb),
                   new_k -= tile_type::joint_matrix_K) {
        // load A matrix fragments
        {
          auto new_A = A;
#pragma unroll
          for (index_t i = 0; i < jm_row_frags;
               i++, new_A += tile_type::joint_matrix_M * lda) {
            joint_matrix_load(sg, sub_a[i], new_A, lda);
          }
        }
        // load the B matrix fragments
        // TODO: use joint_matrix instead of normal loads once the input
        // is in packed format.
        {
          constexpr index_t load_b_cols =
              tile_type::joint_matrix_N * jm_col_frags;
          typename tile_type::jmInpType
              reg_b[(load_b_cols * tile_type::joint_matrix_K) / sg_size];
          constexpr index_t rows = tile_type::joint_matrix_K;
          constexpr index_t cols = load_b_cols;
          const auto k_lambda =
              [&](index_t ir, index_t cr)
                  PORTBLAS_ALWAYS_INLINE { return cr < new_k - ir; };
          const auto n_lambda =
              [&](index_t, index_t cc)
                  PORTBLAS_ALWAYS_INLINE { return cc < (n - start_n); };
          extract_block<false, false, false, false, rows, cols, 1, 1, 1, 1>(
              id, item_id, B, ldb, sub_b, reg_b, k_lambda, n_lambda);

#pragma unroll
          for (index_t i = 0, b_idx = 0; i < jm_col_frags; i++) {
            using storage_element_type = typename sycl::ext::oneapi::detail::
                jm_type_interpretation_helper_trait<
                    typename tile_type::jmInpType>::storage_element_type;
            auto wi_data_c =
                sycl::ext::oneapi::detail::get_wi_data(sg, sub_b[i]);
#pragma unroll
            for (index_t si = 0; si < wi_data_c.length(); si++) {
              wi_data_c[si] = static_cast<storage_element_type>(reg_b[b_idx++]);
            }
          }
        }
#pragma unroll
        for (index_t i = 0, out_idx = 0; i < jm_row_frags; i++) {
#pragma unroll
          for (index_t j = 0; j < jm_col_frags; j++) {
            joint_matrix_mad(sg, sub_c[out_idx], sub_a[i], sub_b[j],
                             sub_c[out_idx]);
            out_idx++;
          }
        }
      }

#pragma unroll
      for (index_t i = 0, out_idx = 0; i < jm_row_frags;
           i++, C += tile_type::joint_matrix_M * ldc) {
        auto new_C = C;
#pragma unroll
        for (index_t j = 0; j < jm_col_frags;
             j++, new_C += tile_type::joint_matrix_N) {
          joint_matrix_apply(
              sg, sub_c[out_idx],
              [=](typename tile_type::jmOutType &x) { x *= alpha_; });
          joint_matrix_store(sg, sub_c[out_idx], new_C, ldc, layout::row_major);
          out_idx++;
        }
      }

      orig_A += (stridea_ * batch_stride);
      orig_B += (strideb_ * batch_stride);
      orig_C += (stridec_ * batch_stride);
      // batch_size_ must be signed as the negative value has meaning here.
      batch_size -= batch_stride;
    } while (batch_size > wg_batch_id);
  }

  template <bool jm_feasible, bool check_m_limit, bool check_n_limit,
            typename InputPointerType, typename OutputPointerType>
  PORTBLAS_ALWAYS_INLINE typename std::enable_if<!jm_feasible>::type
  compute_panel_gemm(const cl::sycl::nd_item<1> &id, const index_t &item_id,
                     const index_t &m, const index_t &n, const index_t &orig_k,
                     const index_t &start_n, InputPointerType orig_A,
                     const index_t &lda, InputPointerType orig_B,
                     const index_t &ldb, OutputPointerType orig_C,
                     const index_t &ldc, index_t batch_stride,
                     index_t wg_batch_id, index_t batch_size) noexcept {
    do {
      auto A = orig_A;
      auto B = orig_B;
      auto C = orig_C;
      auto k = orig_k;
      CFrag sub_c[c_frags];

      // initialize the output fragments and scale them with beta if required
      scale_c<is_beta_zero, !check_m_limit && !check_n_limit, check_m_limit,
              check_n_limit>(id, item_id, m, n, C, ldc, sub_c);
      while (k >= cl_elems) {
        AFrag sub_a[a_frags];
        BFrag sub_b[b_frags];
        extract_input_blocks<check_m_limit, check_n_limit, false>(
            id, item_id, m, n, k, A, lda, B, ldb, sub_a, sub_b);
        compute_block_gemm(id, sub_a, sub_b, sub_c);
        A += cl_elems * (trans_a ? lda : 1);
        B += cl_elems * (trans_b ? 1 : ldb);
        k -= cl_elems;
      }

      if (k > 0) {
        AFrag sub_a[a_frags];
        BFrag sub_b[b_frags];
        extract_input_blocks<check_m_limit, check_n_limit, true>(
            id, item_id, m, n, k, A, lda, B, ldb, sub_a, sub_b);
        compute_block_gemm(id, sub_a, sub_b, sub_c);
      }

      // store the output
      store_output_block<check_m_limit, check_n_limit>(id, item_id, m, n, C,
                                                       ldc, sub_c);
      orig_A += (stridea_ * batch_stride);
      orig_B += (strideb_ * batch_stride);
      orig_C += (stridec_ * batch_stride);
      // batch_size_ must be signed as the negative value has meaning here.
      batch_size -= batch_stride;
    } while (batch_size > wg_batch_id);
  }

  template <bool check_m_limit, bool check_n_limit, typename OutputPointerType>
  PORTBLAS_ALWAYS_INLINE
      typename std::enable_if<!(check_m_limit && check_n_limit)>::type
      store_output_block(const cl::sycl::nd_item<1> id, const index_t,
                         index_t mc, index_t nc, OutputPointerType C,
                         const index_t ldc, CFrag *sub_c) noexcept {
    auto sg = id.get_sub_group();
    const index_t out_offset = tile_type::joint_matrix_M * ldc;
#pragma unroll
    for (index_t i = 0, c_idx = 0; i < jm_row_frags; i++, C += out_offset) {
      auto new_C = C;
#pragma unroll
      for (index_t j = 0; j < jm_col_frags;
           j++, new_C += tile_type::joint_matrix_N) {
        joint_matrix_apply(
            sg, sub_c[c_idx],
            [=](typename tile_type::jmOutType &x) { x *= alpha_; });
        joint_matrix_store(sg, sub_c[c_idx++], new_C, ldc, layout::row_major);
      }
    }
  }

  template <bool check_m_limit, bool check_n_limit, typename OutputPointerType>
  PORTBLAS_ALWAYS_INLINE
      typename std::enable_if<check_m_limit || check_n_limit>::type
      store_output_block(const cl::sycl::nd_item<1> id, const index_t item_id,
                         index_t mc, index_t nc, OutputPointerType C,
                         const index_t ldc, CFrag *sub_c) noexcept {
    auto sg = id.get_sub_group();
    const index_t out_offset = tile_type::joint_matrix_M * ldc;

    constexpr index_t row_bound = tile_type::joint_matrix_M * jm_row_frags;
    constexpr index_t col_bound = tile_type::joint_matrix_N * jm_col_frags;
#pragma unroll
    for (index_t i = 0, c_idx = 0; i < jm_row_frags; i++, C += out_offset) {
      auto new_C = C;
      const index_t stg_loop_limit =
          mc >= tile_type::joint_matrix_M ? tile_type::joint_matrix_M : mc;
      index_t new_nc = nc;
#pragma unroll
      for (index_t j = 0; j < jm_col_frags;
           j++, new_C += tile_type::joint_matrix_N) {
        const index_t item_limit = new_nc >= tile_type::joint_matrix_N
                                       ? tile_type::joint_matrix_N
                                       : new_nc;
        if (item_limit == tile_type::joint_matrix_N &&
            stg_loop_limit == tile_type::joint_matrix_M) {
          joint_matrix_apply(
              sg, sub_c[c_idx],
              [=](typename tile_type::jmOutType &x) { x *= alpha_; });
          joint_matrix_store(sg, sub_c[c_idx++], new_C, ldc, layout::row_major);
        } else {
          auto wi_data_c =
              sycl::ext::oneapi::detail::get_wi_data(sg, sub_c[c_idx++]);
          new_C += item_id;
#pragma unroll
          for (index_t si = 0; si < stg_loop_limit; si++, new_C += ldc) {
            if (item_id < item_limit) {
              *new_C = alpha_ * wi_data_c[si];
            }
          }
        }
        new_nc -= tile_type::joint_matrix_N;
      }
      mc -= tile_type::joint_matrix_M;
    }
  }

  /*!
   * @brief Extract a block of A, and a conformant block of B.
   *
   * @see GemmFactory::extract_block()
   */
  template <bool check_m_limit, bool check_n_limit, bool check_k_limit,
            typename InputPointerType>
  PORTBLAS_ALWAYS_INLINE
      typename std::enable_if<!check_m_limit && !check_n_limit &&
                              !check_k_limit>::type
      extract_input_blocks(const sycl::nd_item<1> id, const index_t item_id,
                           const index_t m, const index_t n, const index_t k,
                           InputPointerType A, const index_t lda,
                           InputPointerType B, const index_t ldb, AFrag *frag_a,
                           BFrag *frag_b) noexcept {
    auto sg = id.get_sub_group();

    typename tile_type::jmInpType *reg;
    constexpr index_t a_frags_row = trans_a ? KK : jm_row_frags;
    constexpr index_t a_frags_col = trans_a ? jm_row_frags : KK;
    const auto lambda_expr = [&](index_t, index_t) PORTBLAS_ALWAYS_INLINE {};
    extract_block<true, check_m_limit, check_k_limit, trans_a, 1, 1,
                  tile_type::joint_matrix_M, tile_type::joint_matrix_K,
                  a_frags_row, a_frags_col>(id, item_id, A, lda, frag_a, reg,
                                            lambda_expr, lambda_expr);
    {
      constexpr index_t load_b_cols = tile_type::joint_matrix_N * jm_col_frags;
      typename tile_type::jmInpType reg_b[(load_b_cols * cl_elems) / sg_size];
      constexpr index_t rows = trans_b ? load_b_cols : cl_elems;
      constexpr index_t cols = trans_b ? cl_elems : load_b_cols;
      const auto k_lambda = [&](index_t ir, index_t cr)
                                PORTBLAS_ALWAYS_INLINE { return cr < k - ir; };
      const auto n_lambda = [&](index_t, index_t cc)
                                PORTBLAS_ALWAYS_INLINE { return cc < n; };
      if constexpr (trans_b) {
        extract_block<false, check_k_limit, check_n_limit, trans_b, rows, cols,
                      1, 1, 1, 1>(id, item_id, B, ldb, frag_b, reg_b, n_lambda,
                                  k_lambda);
      } else {
        extract_block<false, check_k_limit, check_n_limit, trans_b, rows, cols,
                      1, 1, 1, 1>(id, item_id, B, ldb, frag_b, reg_b, k_lambda,
                                  n_lambda);
      }

#pragma unroll
      for (index_t i = 0, b_idx = 0; i < jm_col_frags; i++) {
#pragma unroll
        for (index_t kk = 0; kk < KK; kk++) {
          using storage_element_type = typename sycl::ext::oneapi::detail::
              jm_type_interpretation_helper_trait<
                  typename tile_type::jmInpType>::storage_element_type;
          auto wi_data_c = sycl::ext::oneapi::detail::get_wi_data(
              sg, frag_b[jm_col_frags * kk + i]);
#pragma unroll
          for (index_t si = 0; si < wi_data_c.length(); si++) {
            wi_data_c[si] = static_cast<storage_element_type>(reg_b[b_idx++]);
          }
        }
      }
    }
  }

  /*!
   * @brief Extract a block of A, and a conformant block of B.
   *
   * @see GemmFactory::extract_block()
   */
  template <bool check_m_limit, bool check_n_limit, bool check_k_limit,
            typename InputPointerType>
  PORTBLAS_ALWAYS_INLINE
      typename std::enable_if<check_m_limit || check_n_limit ||
                              check_k_limit>::type
      extract_input_blocks(const sycl::nd_item<1> id, const index_t item_id,
                           const index_t m, const index_t n, const index_t k,
                           InputPointerType A, const index_t lda,
                           InputPointerType B, const index_t ldb, AFrag *frag_a,
                           BFrag *frag_b) noexcept {
    auto sg = id.get_sub_group();
    {
      constexpr index_t load_a_rows = tile_type::joint_matrix_M * jm_row_frags;
      typename tile_type::jmInpType reg_a[(cl_elems * load_a_rows) / sg_size];
      constexpr index_t rows = trans_a ? cl_elems : load_a_rows;
      constexpr index_t cols = trans_a ? load_a_rows : cl_elems;
      const auto k_lambda = [&](index_t, index_t cc)
                                PORTBLAS_ALWAYS_INLINE { return cc < k; };
      const auto m_lambda = [&](index_t ic, index_t cc)
                                PORTBLAS_ALWAYS_INLINE { return cc < m - ic; };
      if constexpr (trans_a) {
        extract_block<false, check_k_limit, check_m_limit, trans_a, rows, cols,
                      1, 1, 1, 1>(id, item_id, A, lda, frag_a, reg_a, k_lambda,
                                  m_lambda);
      } else {
        extract_block<false, check_m_limit, check_k_limit, trans_a, rows, cols,
                      1, 1, 1, 1>(id, item_id, A, lda, frag_a, reg_a, m_lambda,
                                  k_lambda);
      }

#pragma unroll
      for (index_t kk = 0, a_idx = 0; kk < KK; kk++) {
#pragma unroll
        for (index_t i = 0; i < jm_row_frags; i++) {
          using storage_element_type = typename sycl::ext::oneapi::detail::
              jm_type_interpretation_helper_trait<
                  typename tile_type::jmInpType>::storage_element_type;
          auto wi_data_c = sycl::ext::oneapi::detail::get_wi_data(
              sg, frag_a[i + kk * jm_row_frags]);
#pragma unroll
          for (index_t si = 0; si < wi_data_c.length(); si++) {
            wi_data_c[si] = static_cast<storage_element_type>(reg_a[a_idx++]);
          }
        }
      }
    }

    {
      constexpr index_t load_b_cols = tile_type::joint_matrix_N * jm_col_frags;
      typename tile_type::jmInpType reg_b[(load_b_cols * cl_elems) / sg_size];
      constexpr index_t rows = trans_b ? load_b_cols : cl_elems;
      constexpr index_t cols = trans_b ? cl_elems : load_b_cols;
      const auto k_lambda = [&](index_t ir, index_t cr)
                                PORTBLAS_ALWAYS_INLINE { return cr < k - ir; };
      const auto n_lambda = [&](index_t, index_t cc)
                                PORTBLAS_ALWAYS_INLINE { return cc < n; };
      if constexpr (trans_b) {
        extract_block<false, check_n_limit, check_k_limit, trans_b, rows, cols,
                      1, 1, 1, 1>(id, item_id, B, ldb, frag_b, reg_b, n_lambda,
                                  k_lambda);
      } else {
        extract_block<false, check_k_limit, check_n_limit, trans_b, rows, cols,
                      1, 1, 1, 1>(id, item_id, B, ldb, frag_b, reg_b, k_lambda,
                                  n_lambda);
      }

#pragma unroll
      for (index_t i = 0, b_idx = 0; i < jm_col_frags; i++) {
#pragma unroll
        for (index_t kk = 0; kk < KK; kk++) {
          using storage_element_type = typename sycl::ext::oneapi::detail::
              jm_type_interpretation_helper_trait<
                  typename tile_type::jmInpType>::storage_element_type;
          auto wi_data_c = sycl::ext::oneapi::detail::get_wi_data(
              sg, frag_b[jm_col_frags * kk + i]);
#pragma unroll
          for (index_t si = 0; si < wi_data_c.length(); si++) {
            wi_data_c[si] = static_cast<storage_element_type>(reg_b[b_idx++]);
          }
        }
      }
    }
  }

  template <bool internal, bool check_row_limit, bool check_col_limit,
            bool trans, index_t rows, index_t cols, index_t frag_rows,
            index_t frag_cols, index_t num_frags_row, index_t num_frags_col,
            typename InputPointerType, typename FragT, typename RegT,
            typename RowPredicate, typename ColPredicate>
  PORTBLAS_ALWAYS_INLINE typename std::enable_if<internal>::type extract_block(
      const sycl::nd_item<1> id, const index_t, InputPointerType ptr,
      index_t ld, FragT *frag, RegT *, RowPredicate, ColPredicate) {
    auto sg = id.get_sub_group();

    auto row_ptr = ptr;
#pragma unroll
    for (index_t i = 0, idx = 0; i < num_frags_row;
         i++, row_ptr += frag_rows * ld) {
      auto col_ptr = row_ptr;
#pragma unroll
      for (index_t j = 0; j < num_frags_col; j++, col_ptr += frag_cols) {
        joint_matrix_load(sg, frag[idx + j * num_frags_row], col_ptr, ld);
      }
      idx++;
    }

    // TODO: perform transpose on the loaded fragment if trans == true
  }

  template <bool internal, bool check_row_limit, bool check_col_limit,
            bool trans, index_t rows, index_t cols, index_t frag_rows,
            index_t frag_cols, index_t num_frags_row, index_t num_frags_col,
            typename InputPointerType, typename FragT, typename RegT,
            typename RowPredicate, typename ColPredicate>
  PORTBLAS_ALWAYS_INLINE typename std::enable_if<!internal>::type extract_block(
      const sycl::nd_item<1> id, const index_t item_id, InputPointerType ptr,
      index_t ld, FragT *, RegT *reg, RowPredicate in_row,
      ColPredicate in_col) {
    constexpr index_t bs = rows * cols;
    constexpr index_t arr_size = (bs - 1) / sg_size + 1;
    RegT temp[arr_size];

#pragma unroll
    for (index_t i = 0; i < rows; i++, ptr += ld) {
      const bool row_check = do_check<check_row_limit>(in_row(i, 0));
      if (row_check) {
#pragma unroll
        for (index_t j = item_id, ofs = i; j < cols;
             j += sg_size, ofs += rows) {
          const bool col_check = do_check<check_col_limit>(in_col(0, j));
          if (col_check) {
            *(temp + ofs) = static_cast<RegT>(*(ptr + j));
          } else {
            *(temp + ofs) = RegT{0};
          }
        }
      } else {
#pragma unroll
        for (index_t j = item_id, ofs = i; j < cols;
             j += sg_size, ofs += rows) {
          *(temp + ofs) = RegT{0};
        }
      }
    }

    // transpose the input to get correct layout for joint_matrix
#pragma unroll
    for (index_t ofs = 0; ofs < arr_size; ofs += sg_size) {
      transpose<trans>(id, item_id, temp + ofs, reg + ofs);
    }
  }

  template <bool trans, typename RegT>
  PORTBLAS_ALWAYS_INLINE typename std::enable_if<trans>::type transpose(
      const sycl::nd_item<1> id, const index_t item_id, RegT *reg_in,
      RegT *reg_out) {
    auto sg = id.get_sub_group();
#pragma unroll
    for (index_t j = 0; j < sg_size; j++) {
      if (j == item_id) {
#pragma unroll
        for (index_t idx = 0; idx < sg_size; idx++) {
          reg_out[idx] = sg.shuffle(reg_in[item_id], idx);
        }
      }
    }
  }

  template <bool trans, typename RegT>
  PORTBLAS_ALWAYS_INLINE typename std::enable_if<!trans>::type transpose(
      const sycl::nd_item<1>, const index_t, RegT *reg_in, RegT *reg_out) {
#pragma unroll
    for (index_t j = 0; j < sg_size; j++) {
      reg_out[j] = reg_in[j];
    }
  }

  PORTBLAS_ALWAYS_INLINE void compute_block_gemm(const cl::sycl::nd_item<1> &id,
                                                 AFrag *sub_a, BFrag *sub_b,
                                                 CFrag *sub_c) noexcept {
    auto sg = id.get_sub_group();
#pragma unroll
    for (index_t kk = 0; kk < KK; kk++) {
      index_t c_idx = 0;
#pragma unroll
      for (index_t i = 0, a_idx = kk * jm_row_frags; i < jm_row_frags;
           i++, a_idx++) {
#pragma unroll
        for (index_t j = 0, b_idx = kk * jm_col_frags; j < jm_col_frags;
             j++, b_idx++) {
          joint_matrix_mad(sg, sub_c[c_idx], sub_a[a_idx], sub_b[b_idx],
                           sub_c[c_idx]);
          c_idx++;
        }
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
  PORTBLAS_ALWAYS_INLINE bool valid_thread(
      const cl::sycl::nd_item<1> &ndItem) const {
    return true;
  }
};
}  // namespace blas

#endif  // SB_ENABLE_JOINT_MATRIX
#endif  // PORTBLAS_BLAS3_NO_LOCAL_GEMM_JOINT_MATRIX_HPP
