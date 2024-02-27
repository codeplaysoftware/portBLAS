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
 *  @filename gemm_partial_local.hpp
 *
 **************************************************************************/

#ifndef PORTBLAS_BLAS3_PARTIAL_GEMM_HPP
#define PORTBLAS_BLAS3_PARTIAL_GEMM_HPP

#include "gemm_common.hpp"

namespace blas {

template <typename input_t, typename output_t, bool DoubleBuffer, bool NbcA,
          bool NbcB, int ClSize, typename tile_type, bool TransA, bool TransB,
          bool IsFinal, bool IsBetaZero, typename element_t>
class GemmPartial<input_t, output_t, DoubleBuffer, NbcA, NbcB, ClSize,
                  tile_type, TransA, TransB, IsFinal, IsBetaZero, element_t,
                  static_cast<int>(gemm_memory_t::local)> {
 public:
  using index_t = typename std::make_signed<typename input_t::index_t>::type;
  using value_t = element_t;

 private:
  /* This structure holds information about the block loading pattern */
  template <index_t loads, index_t tile_ld, index_t tile_size_row,
            index_t tile_size_col, bool transpose, bool rhs>
  struct BlockProperties {
    static constexpr bool transpose_at_load = transpose;
    static constexpr index_t loads_per_thread = loads;
    static constexpr index_t tile_leading_dim = tile_ld;
    static constexpr index_t tile_row = tile_size_row;
    static constexpr index_t tile_col = tile_size_col;
    static constexpr index_t col_stride = tile_col / loads_per_thread;
    static constexpr index_t local_mem_increment =
        col_stride * (transpose_at_load ? 1 : tile_leading_dim);
  };

 public:
  input_t a_;
  input_t b_;
  output_t cube_;

  element_t alpha_;
  element_t beta_;

  /* Matrix dimensions */
  const index_t m_;
  const index_t n_;
  const index_t k_;
  const index_t lda_;
  const index_t ldb_;
  const index_t ldc_;

  /* Calculating the number of threads */
  static constexpr index_t local_thread_size =
      tile_type::wg_rows * tile_type::wg_cols;

  /* The number of elements per cache line size depends on the element type */
  static constexpr index_t cl_elems = ClSize / sizeof(element_t);

  /* Checking if the tile is valid */
  static_assert(cl_elems % tile_type::wg_rows == 0,
                "The number of item-level tiles within each work group column "
                "must divide the number of elements per cache line.");
  static_assert(cl_elems % tile_type::wg_cols == 0,
                "The number of item-level tiles within each work group row "
                "must divide the number of elements per cache line.");

  /* The dimensions of a single tile */
  static constexpr index_t tile_size_dim_m =
      tile_type::wg_rows * tile_type::item_rows;
  static constexpr index_t tile_size_dim_n =
      tile_type::wg_cols * tile_type::item_cols;
  /* Using an alias here as it might be changed later */
  static constexpr index_t tile_size_dim_k = cl_elems;

  /* The dimensions of the LHS and RHS tiles in global memory */
  static constexpr index_t lhs_tile_rows =
      TransA ? tile_size_dim_k : tile_size_dim_m;
  static constexpr index_t lhs_tile_cols =
      TransA ? tile_size_dim_m : tile_size_dim_k;
  static constexpr index_t rhs_tile_rows =
      TransB ? tile_size_dim_n : tile_size_dim_k;
  static constexpr index_t rhs_tile_cols =
      TransB ? tile_size_dim_k : tile_size_dim_n;

  /* Check that the number of threads is divisible by the numbers of rows in
   * LHS and RHS tiles (to simplify loads) */
  static_assert(
      (tile_type::wg_rows * tile_type::wg_cols) % lhs_tile_rows == 0,
      "The number of threads must be divisible by the tile dimension m");
  static_assert(
      (tile_type::wg_rows * tile_type::wg_cols) % rhs_tile_rows == 0,
      "The number of threads must be divisible by the tile dimension k");

  /* Number of loads per thread for LHS and RHS tiles */
  static constexpr index_t loads_per_thread_lhs =
      (tile_size_dim_k * tile_type::item_rows) / tile_type::wg_cols;
  static constexpr index_t loads_per_thread_rhs =
      (tile_size_dim_k * tile_type::item_cols) / tile_type::wg_rows;

  /* The leading dimension of the LHS and RHS tiles */
  static constexpr index_t lhs_tile_ld = tile_size_dim_m + (NbcA && TransA);
  static constexpr index_t rhs_tile_ld = tile_size_dim_n + (NbcB && !TransB);

  /* Local memory size of a LHS and RHS tile */
  static constexpr index_t lhs_tile_mem_size = lhs_tile_ld * tile_size_dim_k;
  static constexpr index_t rhs_tile_mem_size = rhs_tile_ld * tile_size_dim_k;

  /* Local memory size */
  static constexpr index_t local_memory_size =
      (DoubleBuffer + 1) * (lhs_tile_mem_size + rhs_tile_mem_size);

  /* Where the RHS tiles are located in the scratch buffer */
  static constexpr index_t rhs_scratch_offset =
      (DoubleBuffer + 1) * lhs_tile_mem_size;

  /* Number of private summation registers */
  static constexpr index_t private_res_size =
      tile_type::item_rows * tile_type::item_cols;

  /* Blocks properties */
  using LHSBlockProperties =
      BlockProperties<loads_per_thread_lhs, lhs_tile_ld, lhs_tile_rows,
                      lhs_tile_cols, TransA, false>;
  using RHSBlockProperties =
      BlockProperties<loads_per_thread_rhs, rhs_tile_ld, rhs_tile_rows,
                      rhs_tile_cols, !TransB, true>;

  /* Work groups per dimension m, n, k */
  const index_t group_count_m;
  const index_t group_count_n;
  const index_t group_count_k;

  /* The number of tiles to be processed */
  const index_t num_tiles;

  PORTBLAS_INLINE
  GemmPartial(input_t A, input_t B, output_t cube_buffer, element_t alpha,
              element_t beta, index_t wg_count_k)
      : a_(A),
        b_(B),
        cube_(cube_buffer),
        alpha_(alpha),
        beta_(beta),
        m_(a_.get_size_row()),
        n_(b_.get_size_col()),
        k_(a_.get_size_col()),
        lda_(a_.getSizeL()),
        ldb_(b_.getSizeL()),
        ldc_(cube_.getSizeL()),
        group_count_m((m_ - 1) / tile_size_dim_m + 1),
        group_count_n((n_ - 1) / tile_size_dim_n + 1),
        group_count_k(wg_count_k),
        num_tiles((k_ - 1) / (tile_size_dim_k * group_count_k) + 1) {}

  void bind(cl::sycl::handler& h) {
    a_.bind(h);
    b_.bind(h);
    cube_.bind(h);
  }
  void adjust_access_displacement() {
    a_.adjust_access_displacement();
    b_.adjust_access_displacement();
    cube_.adjust_access_displacement();
  }

  /*!
   * @brief This function returns the depth of the cube buffer that should give
   * the best performance.
   */
  static PORTBLAS_INLINE index_t get_ideal_cube_depth(index_t compute_units,
                                                      index_t m, index_t n,
                                                      index_t k) noexcept {
    const index_t group_count_mn =
        ((m - 1) / tile_size_dim_m + 1) * ((n - 1) / tile_size_dim_n + 1);
    /* The depth of the cube buffer is calculated so that each compute unit
     * will compute 4 work groups. This value is empirical */
    return (4 * compute_units - 1) / group_count_mn + 1;
  }

  /*!
   * @brief This function is used to find the optimum number of work groups
   * required to execute each partial GEMM.
   */
  PORTBLAS_INLINE index_t
  get_workgroup_cluster(index_t compute_units) noexcept {
    return ((m_ - 1) / tile_size_dim_m + 1) * ((n_ - 1) / tile_size_dim_n + 1) *
           group_count_k;
  }

  /*!
   * @brief Get the nd_range value which has to be used for kernels that
   * intend to call GemmPartial::run().
   */
  PORTBLAS_INLINE cl::sycl::nd_range<1> get_nd_range(
      index_t compute_units) noexcept {
    const cl::sycl::range<1> nwg(get_workgroup_cluster(compute_units));
    const cl::sycl::range<1> wgs(local_thread_size);
    return cl::sycl::nd_range<1>(nwg * wgs, wgs);
  }

  template <typename local_memory_t>
  PORTBLAS_INLINE void eval(local_memory_t scratch,
                            cl::sycl::nd_item<1> id) noexcept {
    /* Pointers to the scratch memory (lhs and rhs) */
    value_t* scratch_ptr = scratch.localAcc.get_pointer();
    value_t* rhs_scratch_ptr = scratch_ptr + rhs_scratch_offset;

    /* Create and initialise the private res summation registers */
    element_t private_res[private_res_size] = {element_t(0)};

    /* workgroup id */
    const index_t group_id = id.get_group(0);
    /* Local thread id */
    const index_t local_id = id.get_local_id(0);

    /* Local ID column and row */
    const index_t n_local_id = local_id / tile_type::wg_rows;
    const index_t m_local_id = local_id - (n_local_id * tile_type::wg_rows);

    /* Workgroup id m, k and n */
    const index_t group_count_mn = group_count_m * group_count_n;
    const index_t kgroup_id = IsFinal ? 0 : (group_id / group_count_mn);
    const index_t mn_group_id =
        IsFinal ? group_id : (group_id - kgroup_id * group_count_mn);
    const index_t ngroup_id = mn_group_id / group_count_m;
    const index_t mgroup_id = mn_group_id - ngroup_id * group_count_m;

    /* register offsets */
    const index_t global_m_offset = mgroup_id * tile_size_dim_m;
    const index_t global_n_offset = ngroup_id * tile_size_dim_n;
    const index_t global_k_offset =
        IsFinal ? 0 : (kgroup_id * tile_size_dim_k * num_tiles);

    /* Find out whether we need to check the limits when loading the tiles */
    const bool check_m_limit = global_m_offset + tile_size_dim_m > m_;
    const bool check_n_limit = global_n_offset + tile_size_dim_n > n_;

    /* Calculate the starting rows and columns for LHS and RHS tile loads */
    const index_t lhs_row = local_id % LHSBlockProperties::tile_row;
    const index_t lhs_col = local_id / LHSBlockProperties::tile_row;
    const index_t rhs_row = local_id % RHSBlockProperties::tile_row;
    const index_t rhs_col = local_id / RHSBlockProperties::tile_row;

    /* The first tile is pre-loaded before the loop if double buffering is
     * enabled */
    if (DoubleBuffer) {
      extract_input_blocks(lhs_row, lhs_col, rhs_row, rhs_col, 0, scratch_ptr,
                           rhs_scratch_ptr, global_m_offset, global_n_offset,
                           global_k_offset, check_m_limit, check_n_limit);
    }

    /* Loop over all the tiles in this work group */
    for (index_t tile_id = 0; tile_id < num_tiles; tile_id++) {
      id.barrier(cl::sycl::access::fence_space::local_space);

      // Start loading the next tile
      index_t next_tile = DoubleBuffer ? (tile_id + 1) : tile_id;
      const bool tile_nb_check = do_check<DoubleBuffer>(next_tile < num_tiles);
      if (tile_nb_check) {
        extract_input_blocks(lhs_row, lhs_col, rhs_row, rhs_col, next_tile,
                             scratch_ptr, rhs_scratch_ptr, global_m_offset,
                             global_n_offset, global_k_offset, check_m_limit,
                             check_n_limit);
      }

      if (!DoubleBuffer) {
        id.barrier(cl::sycl::access::fence_space::local_space);
      }

      // Calculate offsets in the temporary memory.
      index_t lhs_offset =
          (DoubleBuffer * (tile_id & 1) * lhs_tile_mem_size) + m_local_id;
      index_t rhs_offset =
          (DoubleBuffer * (tile_id & 1) * rhs_tile_mem_size) + n_local_id;

      /* Loop over the values of a single tile */
      for (index_t k = 0; k < tile_size_dim_k; k++) {
        index_t idx = 0;
        index_t rhs_index = 0;
#pragma unroll
        for (index_t wLPTN = 0; wLPTN < tile_type::item_cols; wLPTN++) {
          // load a RHS element from the scratch buffer
          const value_t privateRhs = rhs_scratch_ptr[rhs_index + rhs_offset];

          index_t lhs_index = 0;
#pragma unroll
          for (index_t wLPTM = 0; wLPTM < tile_type::item_rows; wLPTM++) {
            // load a LHS element from the scratch buffer
            const value_t privateLhs = scratch_ptr[lhs_index + lhs_offset];

            private_res[wLPTM + idx] =
                mul_add(privateLhs, privateRhs, private_res[wLPTM + idx]);

            lhs_index += tile_type::wg_rows;
          }
          idx += tile_type::item_rows;
          rhs_index += tile_type::wg_cols;
        }
        lhs_offset += lhs_tile_ld;
        rhs_offset += rhs_tile_ld;
      }
    }

    // Store the final results in the cube buffer
    index_t slice_col = (ngroup_id * tile_size_dim_n) + (n_local_id);
    const index_t slice_row_offset =
        (mgroup_id * tile_size_dim_m) + (m_local_id);
    const index_t cube_depth_offset = IsFinal ? 0 : (kgroup_id * ldc_ * n_);
    index_t cube_index = slice_col * ldc_;
    index_t private_index_offset = 0;
    const index_t cube_index_inc = tile_type::wg_cols * ldc_;

#pragma unroll
    for (index_t wLPTN = 0; wLPTN < tile_type::item_cols; wLPTN++) {
      index_t private_index = private_index_offset;

      index_t slice_row = slice_row_offset;
#pragma unroll
      for (index_t wLPTM = 0; wLPTM < tile_type::item_rows; wLPTM++) {
        if (slice_row < m_ && slice_col < n_) {
          const index_t write_idx = cube_index + slice_row + cube_depth_offset;
          cube_.template eval<true>(write_idx) =
              IsBetaZero ? (alpha_ * private_res[wLPTM + private_index])
                         : (alpha_ * private_res[wLPTM + private_index] +
                            beta_ * cube_.template eval<true>(write_idx));
        }
        slice_row += tile_type::wg_rows;
      }
      private_index += tile_type::item_rows;

      slice_col += tile_type::wg_cols;
      cube_index += cube_index_inc;
      private_index_offset += tile_type::item_rows;
    }
  }

  template <typename local_ptr_t>
  PORTBLAS_INLINE void extract_input_blocks(
      const index_t& lhs_row, const index_t& lhs_col, const index_t& rhs_row,
      const index_t& rhs_col, const index_t& tile_idx, local_ptr_t scratch_ptr,
      local_ptr_t rhs_scratch_ptr, const index_t& global_m_offset,
      const index_t& global_n_offset, const index_t& global_k_offset,
      bool check_m_limit, bool check_n_limit) {
    const bool check_k_limit =
        IsFinal ? ((tile_idx + 1) * tile_size_dim_k > k_)
                : (global_k_offset + (tile_idx + 1) * tile_size_dim_k > k_);
    const bool check_limits = check_m_limit || check_n_limit || check_k_limit;
    if (check_limits)
      load_blocks<true, true, true>(
          lhs_row, lhs_col, rhs_row, rhs_col, tile_idx, scratch_ptr,
          rhs_scratch_ptr, global_m_offset, global_n_offset, global_k_offset);
    else
      load_blocks<false, false, false>(
          lhs_row, lhs_col, rhs_row, rhs_col, tile_idx, scratch_ptr,
          rhs_scratch_ptr, global_m_offset, global_n_offset, global_k_offset);
  }

  template <bool check_m_limit, bool check_n_limit, bool check_k_limit,
            typename local_ptr_t>
  PORTBLAS_INLINE void load_blocks(
      const index_t& lhs_row, const index_t& lhs_col, const index_t& rhs_row,
      const index_t& rhs_col, const index_t& tile_idx, local_ptr_t scratch_ptr,
      local_ptr_t rhs_scratch_ptr, const index_t& global_m_offset,
      const index_t& global_n_offset, const index_t& global_k_offset) {
    // LHS tile
    constexpr bool a_check_row_limit = TransA ? check_k_limit : check_m_limit;
    constexpr bool a_check_col_limit = TransA ? check_m_limit : check_k_limit;
    load_block<LHSBlockProperties, a_check_row_limit, a_check_col_limit>(
        lhs_row, lhs_col, tile_idx, a_, lda_, scratch_ptr,
        TransA ? global_k_offset : global_m_offset,
        TransA ? global_m_offset : global_k_offset, TransA ? k_ : m_,
        TransA ? m_ : k_);

    // RHS tile
    constexpr bool b_check_row_limit = TransB ? check_n_limit : check_k_limit;
    constexpr bool b_check_col_limit = TransB ? check_k_limit : check_n_limit;
    load_block<RHSBlockProperties, b_check_row_limit, b_check_col_limit>(
        rhs_row, rhs_col, tile_idx, b_, ldb_, rhs_scratch_ptr,
        TransB ? global_n_offset : global_k_offset,
        TransB ? global_k_offset : global_n_offset, TransB ? n_ : k_,
        TransB ? k_ : n_);
  }

  template <typename BlockPropertiesType, bool check_row_limit,
            bool check_col_limit, typename local_ptr_t>
  static PORTBLAS_INLINE void load_block(
      const index_t& local_row, const index_t& local_col,
      const index_t& tile_idx, const input_t& in_view,
      const index_t& leading_dim, local_ptr_t local_ptr,
      const index_t& global_row_offset, const index_t& global_col_offset,
      const index_t& global_rows, const index_t& global_cols) {
    const index_t global_tile_row_offset =
        global_row_offset + BlockPropertiesType::transpose_at_load *
                                (BlockPropertiesType::tile_row * tile_idx);
    const index_t global_tile_col_offset =
        global_col_offset + (1 - BlockPropertiesType::transpose_at_load) *
                                (BlockPropertiesType::tile_col * tile_idx);

    // Double buffering
    constexpr index_t block_size =
        BlockPropertiesType::tile_leading_dim * tile_size_dim_k;
    const index_t local_mem_offset = DoubleBuffer * (tile_idx & 1) * block_size;

    const index_t local_mem_index =
        local_mem_offset +
        (BlockPropertiesType::transpose_at_load
             ? (local_col + local_row * BlockPropertiesType::tile_leading_dim)
             : (local_row + local_col * BlockPropertiesType::tile_leading_dim));

    const index_t global_col_index = global_tile_col_offset + local_col;
    const index_t global_row_index = global_tile_row_offset + local_row;
    index_t global_mem_index =
        global_col_index * leading_dim + global_row_index;

    const bool valid_row =
        do_check<check_row_limit>(global_row_index < global_rows);

    const index_t global_mem_increment =
        BlockPropertiesType::col_stride * leading_dim;

#pragma unroll
    for (index_t lpt = 0; lpt < BlockPropertiesType::loads_per_thread; lpt++) {
      const bool in_range =
          valid_row &&
          do_check<check_col_limit>(global_col_index +
                                        lpt * BlockPropertiesType::col_stride <
                                    global_cols);
      element_t val = in_range ? in_view.template eval<true>(global_mem_index)
                               : element_t(0);

      local_ptr[local_mem_index +
                lpt * BlockPropertiesType::local_mem_increment] = val;
      global_mem_index += global_mem_increment;
    }
  }
};

}  // namespace blas

#endif  // PORTBLAS_BLAS3_PARTIAL_GEMM_HPP
