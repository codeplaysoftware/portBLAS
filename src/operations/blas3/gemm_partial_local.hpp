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
 *  @filename gemm_partial_local.hpp
 *
 **************************************************************************/

#ifndef SYCL_BLAS_BLAS3_PARTIAL_GEMM_HPP
#define SYCL_BLAS_BLAS3_PARTIAL_GEMM_HPP

#include "gemm_common.hpp"

namespace blas {

template <typename input_t, typename output_t, bool DoubleBuffer, bool NbcA,
          bool NbcB, int ClSize, typename tile_type, bool TransA, bool TransB,
          bool IsFinal, bool IsBetaZero, typename element_t>
class GemmPartial<input_t, output_t, DoubleBuffer, NbcA, NbcB, ClSize,
                  tile_type, TransA, TransB, IsFinal, IsBetaZero, element_t,
                  static_cast<int>(Gemm_memory_t::local_memory)> {
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

  /* If used for tsgemm: false ; for gemm: true */
  static constexpr bool is_final = IsFinal;

  /* Should we read C and multiply it by beta. */
  static constexpr bool is_beta_zero = IsBetaZero;

  /* Workload per work item on each dimension m and n */
  static constexpr index_t work_per_thread_m = tile_type::item_rows;
  static constexpr index_t work_per_thread_n = tile_type::item_cols;

  /* Calculating the number of threads */
  static constexpr index_t local_thread_size_m = tile_type::wg_rows;
  static constexpr index_t local_thread_size_n = tile_type::wg_cols;
  static constexpr index_t local_thread_size =
      local_thread_size_m * local_thread_size_n;

  /* The number of elements per cache line size depends on the element type */
  static constexpr index_t cl_elems = ClSize / sizeof(element_t);

  /* Checking if the tile is valid */
  static_assert(cl_elems % local_thread_size_m == 0,
                "The number of item-level tiles within each work group column "
                "must divide the number of elements per cache line.");
  static_assert(cl_elems % local_thread_size_n == 0,
                "The number of item-level tiles within each work group row "
                "must divide the number of elements per cache line.");

  /* The dimensions of a single tile */
  static constexpr index_t tile_size_dim_m =
      local_thread_size_m * work_per_thread_m;
  static constexpr index_t tile_size_dim_n =
      local_thread_size_n * work_per_thread_n;
  static constexpr index_t tile_size_dim_k = cl_elems;

  /* Number of threads in the work group */
  static constexpr index_t nb_threads =
      local_thread_size_m * local_thread_size_n;

  /* Transpose mode for matrices A and B */
  static constexpr bool trans_a = TransA;
  static constexpr bool trans_b = TransB;

  /* The dimensions of the LHS and RHS tiles in global memory */
  static constexpr index_t lhs_tile_rows =
      trans_a ? tile_size_dim_k : tile_size_dim_m;
  static constexpr index_t lhs_tile_cols =
      trans_a ? tile_size_dim_m : tile_size_dim_k;
  static constexpr index_t rhs_tile_rows =
      trans_b ? tile_size_dim_n : tile_size_dim_k;
  static constexpr index_t rhs_tile_cols =
      trans_b ? tile_size_dim_k : tile_size_dim_n;

  /* Check that the number of threads is divisible by the numbers of rows in
   * LHS and RHS tiles (to simplify loads) */
  static_assert(
      nb_threads % lhs_tile_rows == 0,
      "The number of threads must be divisible by the tile dimension m");
  static_assert(
      nb_threads % rhs_tile_rows == 0,
      "The number of threads must be divisible by the tile dimension k");

  /* Number of loads per thread for LHS and RHS tiles */
  static constexpr index_t loads_per_thread_lhs =
      (tile_size_dim_k * work_per_thread_m) / local_thread_size_n;
  static constexpr index_t loads_per_thread_rhs =
      (tile_size_dim_k * work_per_thread_n) / local_thread_size_m;

  /* The leading dimension of the LHS and RHS tiles in local memory */
  static constexpr index_t lhs_tile_ld = tile_size_dim_m + (NbcA && TransA);
  static constexpr index_t rhs_tile_ld = tile_size_dim_n + (NbcB && !TransB);

  /* Local memory size of a LHS and RHS tile */
  static constexpr index_t lhs_tile_mem_size = lhs_tile_ld * tile_size_dim_k;
  static constexpr index_t rhs_tile_mem_size = rhs_tile_ld * tile_size_dim_k;

  /* If double buffering should be used */
  static constexpr bool double_buffer = DoubleBuffer;

  static constexpr index_t scratch_padding =
      (NbcA && NbcB && TransB && !TransA) ? 1 : 0;

  /* Local memory size */
  static constexpr index_t local_memory_size =
      (double_buffer + 1) * (lhs_tile_mem_size + rhs_tile_mem_size) +
      scratch_padding;

  /* Where the RHS tiles are located in the scratch buffer */
  static constexpr index_t rhs_scratch_offset =
      (double_buffer + 1) * lhs_tile_mem_size + scratch_padding;

  // Number of private summation registers
  static constexpr index_t private_res_size =
      work_per_thread_m * work_per_thread_n;

  /* Blocks properties */
  using LHSBlockProperties =
      BlockProperties<loads_per_thread_lhs, lhs_tile_ld, lhs_tile_rows,
                      lhs_tile_cols, trans_a, false>;
  using RHSBlockProperties =
      BlockProperties<loads_per_thread_rhs, rhs_tile_ld, rhs_tile_rows,
                      rhs_tile_cols, !trans_b, true>;

  /* Work groups per dimension m, n, k */
  const index_t group_count_m;
  const index_t group_count_n;
  const index_t group_count_k;

  /* The number of tiles to be processed */
  const index_t num_tiles;

  SYCL_BLAS_INLINE
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
  static SYCL_BLAS_INLINE index_t get_ideal_cube_depth(index_t compute_units,
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
  SYCL_BLAS_INLINE index_t
  get_workgroup_cluster(index_t compute_units) noexcept {
    return ((m_ - 1) / tile_size_dim_m + 1) * ((n_ - 1) / tile_size_dim_n + 1) *
           group_count_k;
  }

  /*!
   * @brief Get the nd_range value which has to be used for kernels that
   * intend to call GemmPartial::run().
   */
  SYCL_BLAS_INLINE cl::sycl::nd_range<1> get_nd_range(
      index_t compute_units) noexcept {
    const cl::sycl::range<1> nwg(get_workgroup_cluster(compute_units));
    const cl::sycl::range<1> wgs(local_thread_size);
    return cl::sycl::nd_range<1>(nwg * wgs, wgs);
  }

  template <typename local_memory_t>
  SYCL_BLAS_INLINE void eval(local_memory_t scratch,
                             cl::sycl::nd_item<1> id) noexcept {
    /* Pointers to the scratch memory (lhs and rhs) */
    value_t* scratch_ptr = scratch.localAcc.get_pointer().get();
    value_t* rhs_scratch_ptr = scratch_ptr + rhs_scratch_offset;

    /* Create and initialise the private res summation registers */
    element_t private_res[private_res_size];
    for (auto i = 0; i < private_res_size; i++) {
      private_res[i] = static_cast<element_t>(0);
    }

    /* workgroup id */
    const index_t group_id = id.get_group(0);
    /* Local thread id */
    const index_t local_id = id.get_local_id(0);

    /* Local ID column and row */
    const index_t n_local_id = local_id / local_thread_size_m;
    const index_t m_local_id = local_id - (n_local_id * local_thread_size_m);

    /* Workgroup id m, k and n */
    const index_t group_count_mn = group_count_m * group_count_n;
    const index_t kgroup_id = is_final ? 0 : (group_id / group_count_mn);
    const index_t mn_group_id =
        is_final ? group_id : (group_id - kgroup_id * group_count_mn);
    const index_t ngroup_id = mn_group_id / group_count_m;
    const index_t mgroup_id = mn_group_id - ngroup_id * group_count_m;

    /* register offsets */
    const index_t global_m_offset = mgroup_id * tile_size_dim_m;
    const index_t global_n_offset = ngroup_id * tile_size_dim_n;
    const index_t global_k_offset =
        is_final ? 0 : (kgroup_id * tile_size_dim_k * num_tiles);

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
    if (double_buffer) {
      extract_input_blocks(lhs_row, lhs_col, rhs_row, rhs_col, 0, scratch_ptr,
                           rhs_scratch_ptr, global_m_offset, global_n_offset,
                           global_k_offset, check_m_limit, check_n_limit);
    }

    /* Loop over all the tiles in this work group */
    for (index_t tile_id = 0; tile_id < num_tiles; tile_id++) {
      id.barrier(cl::sycl::access::fence_space::local_space);

      // Start loading the next tile
      index_t next_tile = double_buffer ? (tile_id + 1) : tile_id;
      const bool tile_nb_check = do_check<double_buffer>(next_tile < num_tiles);
      if (tile_nb_check) {
        extract_input_blocks(lhs_row, lhs_col, rhs_row, rhs_col, next_tile,
                             scratch_ptr, rhs_scratch_ptr, global_m_offset,
                             global_n_offset, global_k_offset, check_m_limit,
                             check_n_limit);
      }

      if (!double_buffer) {
        id.barrier(cl::sycl::access::fence_space::local_space);
      }

      // Calculate offsets in the temporary memory.
      index_t lhs_offset =
          (double_buffer * (tile_id & 1) * lhs_tile_mem_size) + m_local_id;
      index_t rhs_offset =
          (double_buffer * (tile_id & 1) * rhs_tile_mem_size) + n_local_id;

      /* Loop over the values of a single tile */
      for (index_t k = 0; k < tile_size_dim_k; k++) {
        index_t idx = 0;
        index_t rhs_index = 0;
#pragma unroll
        for (index_t wLPTN = 0; wLPTN < work_per_thread_n; wLPTN++) {
          // load a RHS element from the scratch buffer
          const value_t privateRhs = rhs_scratch_ptr[rhs_index + rhs_offset];

          index_t lhs_index = 0;
#pragma unroll
          for (index_t wLPTM = 0; wLPTM < work_per_thread_m; wLPTM++) {
            // load a LHS element from the scratch buffer
            const value_t privateLhs = scratch_ptr[lhs_index + lhs_offset];

            // Perform a manual MAD.
            private_res[wLPTM + idx] += (privateLhs * privateRhs);

            lhs_index += local_thread_size_m;
          }
          idx += work_per_thread_m;
          rhs_index += local_thread_size_n;
        }
        lhs_offset += lhs_tile_ld;
        rhs_offset += rhs_tile_ld;
      }
    }

    // Store the final results in the cube buffer
    index_t slice_col = (ngroup_id * tile_size_dim_n) + (n_local_id);
    const index_t slice_row_offset =
        (mgroup_id * tile_size_dim_m) + (m_local_id);
    const index_t cube_depth_offset = is_final ? 0 : (kgroup_id * ldc_ * n_);
    index_t cube_index = slice_col * ldc_;
    index_t private_index_offset = 0;
    const index_t cube_index_inc = local_thread_size_n * ldc_;

#pragma unroll
    for (index_t wLPTN = 0; wLPTN < work_per_thread_n; wLPTN++) {
      index_t private_index = private_index_offset;

      index_t slice_row = slice_row_offset;
#pragma unroll
      for (index_t wLPTM = 0; wLPTM < work_per_thread_m; wLPTM++) {
        if (slice_row < m_ && slice_col < n_) {
          const index_t write_idx = cube_index + slice_row + cube_depth_offset;
          cube_.template eval<true>(write_idx) =
              is_beta_zero ? (alpha_ * private_res[wLPTM + private_index])
                           : (alpha_ * private_res[wLPTM + private_index] +
                              beta_ * cube_.template eval<true>(write_idx));
        }
        slice_row += local_thread_size_m;
      }
      private_index += work_per_thread_m;

      slice_col += local_thread_size_n;
      cube_index += cube_index_inc;
      private_index_offset += work_per_thread_m;
    }
  }

  template <typename local_ptr_t>
  SYCL_BLAS_INLINE void extract_input_blocks(
      const index_t& lhs_row, const index_t& lhs_col, const index_t& rhs_row,
      const index_t& rhs_col, const index_t& tile_idx, local_ptr_t scratch_ptr,
      local_ptr_t rhs_scratch_ptr, const index_t& global_m_offset,
      const index_t& global_n_offset, const index_t& global_k_offset,
      bool check_m_limit, bool check_n_limit) {
    const bool check_k_limit =
        is_final ? ((tile_idx + 1) * tile_size_dim_k > k_)
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
  SYCL_BLAS_INLINE void load_blocks(
      const index_t& lhs_row, const index_t& lhs_col, const index_t& rhs_row,
      const index_t& rhs_col, const index_t& tile_idx, local_ptr_t scratch_ptr,
      local_ptr_t rhs_scratch_ptr, const index_t& global_m_offset,
      const index_t& global_n_offset, const index_t& global_k_offset) {
    // LHS tile
    if (trans_a) {
      load_block<LHSBlockProperties, check_k_limit, check_m_limit>(
          lhs_row, lhs_col, tile_idx, a_, lda_, scratch_ptr, global_k_offset,
          global_m_offset, k_, m_);
    } else {
      load_block<LHSBlockProperties, check_m_limit, check_k_limit>(
          lhs_row, lhs_col, tile_idx, a_, lda_, scratch_ptr, global_m_offset,
          global_k_offset, m_, k_);
    }
    // RHS tile
    if (trans_b) {
      load_block<RHSBlockProperties, check_n_limit, check_k_limit>(
          rhs_row, rhs_col, tile_idx, b_, ldb_, rhs_scratch_ptr,
          global_n_offset, global_k_offset, n_, k_);
    } else {
      load_block<RHSBlockProperties, check_k_limit, check_n_limit>(
          rhs_row, rhs_col, tile_idx, b_, ldb_, rhs_scratch_ptr,
          global_k_offset, global_n_offset, k_, n_);
    }
  }

  template <typename BlockPropertiesType, bool check_row_limit,
            bool check_col_limit, typename local_ptr_t>
  static SYCL_BLAS_INLINE void load_block(
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
    const index_t local_mem_offset =
        double_buffer * (tile_idx & 1) * block_size;

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

#endif  // SYCL_BLAS_BLAS3_PARTIAL_GEMM_HPP
