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
          typename element_t>
class GemmPartial<input_t, output_t, DoubleBuffer, NbcA, NbcB, ClSize,
                  tile_type, TransA, TransB, element_t,
                  static_cast<int>(Gemm_memory_t::local_memory)> {
 public:
  using index_t = typename std::make_signed<typename input_t::index_t>::type;
  using value_t = element_t;

  input_t a_;
  input_t b_;
  output_t cube_;

  element_t alpha;

  /* Matrix dimensions */
  const index_t m_;
  const index_t n_;
  const index_t k_;
  const index_t lda_;
  const index_t ldb_;
  const index_t ldc_;

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

  /* The number of tiles to be processed */
  static constexpr index_t num_tiles = 12;
  // TODO: don't hardcode that!!! See Eigen

  /* Number of loads per thread for LHS and RHS tiles */
  static constexpr index_t loads_per_thread_lhs =
      (tile_size_dim_k * work_per_thread_m) / local_thread_size_n;
  static constexpr index_t loads_per_thread_rhs =
      (tile_size_dim_k * work_per_thread_n) / local_thread_size_m;

  /* Local memory size */
  static constexpr index_t local_memory_size =
      2 * tile_size_dim_m * tile_size_dim_k +
      2 * tile_size_dim_k * tile_size_dim_n;

  // Number of private summation registers
  static constexpr index_t private_res_size =
      work_per_thread_m * work_per_thread_n;

  /* If double buffering should be used */
  static constexpr bool double_buffer = DoubleBuffer;
  // TODO: this is not used yet

  /* Transpose mode for matrices A and B */
  static constexpr bool trans_a = TransA;
  static constexpr bool trans_b = TransB;

  /* Work groups per dimension m, n, k */
  const index_t group_count_m;
  const index_t group_count_n;
  const index_t group_count_k;

  SYCL_BLAS_INLINE GemmPartial(input_t A, input_t B, output_t cube_buffer,
                               element_t alpha)
      : a_(A),
        b_(B),
        cube_(cube_buffer),
        alpha(alpha),
        m_(a_.get_size_row()),
        n_(b_.get_size_col()),
        k_(a_.get_size_col()),
        lda_(a_.getSizeL()),
        ldb_(b_.getSizeL()),
        ldc_(cube_.getSizeL()),
        group_count_m((m_ - 1) / tile_size_dim_m + 1),
        group_count_n((n_ - 1) / tile_size_dim_n + 1),
        group_count_k((k_ - 1) / (tile_size_dim_k * num_tiles) + 1) {}

  /*!
   * @brief Get the type of this GemmPartial as a human readable string.
   */
  static SYCL_BLAS_INLINE std::string get_type_string() noexcept {
    std::ostringstream str{};
    str << "GemmPartial<" << double_buffer << ", "
        << tile_type::get_type_string() << ", "
        << type_string<element_t>::get_value() << ">";
    return str.str();
  }

  void bind(cl::sycl::handler &h) {
    a_.bind(h);
    b_.bind(h);
    cube_.bind(h);
  }
  void adjust_access_displacement() {
    a_.adjust_access_displacement();
    b_.adjust_access_displacement();
    cube_.adjust_access_displacement();
  }
  SYCL_BLAS_INLINE bool valid_thread(cl::sycl::nd_item<1> ndItem) const {
    return true;
  }

  /*!
   * @brief get_workgroup_cluster. This function is used to find the optimum
   * number of work_group required to execute each partial GEMM.
   */
  static SYCL_BLAS_INLINE index_t get_workgroup_cluster(
      index_t m, index_t n, index_t k, index_t compute_units) noexcept {
    return ((m - 1) / tile_size_dim_m + 1) * ((n - 1) / tile_size_dim_n + 1) *
           ((k - 1) / (tile_size_dim_k * num_tiles) + 1);
  }
  /*!
   * @brief get_num_workgroup_cluster. This function is used to extend the
   * number of work_group cluster, in order to make sure that atleast 4 gemm
   * operations are available per work group. The number 4 is used based on
   * empirical research.
   */
  static SYCL_BLAS_INLINE index_t get_num_workgroup_cluster(
      index_t m, index_t n, index_t k, index_t compute_units) noexcept {
    // return ((4 * compute_units - 1) / get_workgroup_cluster(m, n) + 1);
    return 1;  // TODO: optimize that later
  }

  /*!
   * @brief Get the nd_range value which has to be used for kernels that
   *        intend to call GemmPartial::run().
   */
  static SYCL_BLAS_INLINE cl::sycl::nd_range<1> get_nd_range(
      index_t m, index_t n, index_t k, index_t compute_units) noexcept {
    const cl::sycl::range<1> nwg(
        get_workgroup_cluster(m, n, k, compute_units) *
        get_num_workgroup_cluster(m, n, k, compute_units));
    const cl::sycl::range<1> wgs(local_thread_size);
    return cl::sycl::nd_range<1>(nwg * wgs, wgs);
    // TODO: add verbose
  }

  SYCL_BLAS_INLINE index_t get_size() const { return m_ * n_; }

  template <typename local_memory_t>
  SYCL_BLAS_INLINE void eval(local_memory_t scratch,
                             cl::sycl::nd_item<1> id) noexcept {
    /* references to the matrices */
    auto A = a_.get_pointer();
    auto B = b_.get_pointer();
    auto cube_buffer = cube_.get_pointer();

    /* references to the scratch memory (lhs and rhs) */
    auto scratch_ptr = scratch.localAcc.get_pointer().get();
    auto rhs_scratch_ptr =
        scratch_ptr + (2 * tile_size_dim_m * tile_size_dim_k);

    /* create and initialise the private res summation registers */
    element_t private_res[private_res_size];
    for (auto i = 0; i < private_res_size; i++) {
      private_res[i] = static_cast<element_t>(0);
    }

    /* workgroup id */
    const index_t wg_id = id.get_group(0);
    /* Local thread id */
    const index_t local_id = id.get_local_id(0);

    /* Local ID column and row */
    const index_t n_local_id = local_id / local_thread_size_m;
    const index_t m_local_id = local_id - (n_local_id * local_thread_size_m);

    /* Workgroup id m, k and n */
    const index_t group_count_mn = group_count_m * group_count_n;
    const index_t kgroup_id = wg_id / group_count_mn;
    const index_t mn_group_id = wg_id - kgroup_id * group_count_mn;
    const index_t ngroup_id = mn_group_id / group_count_m;
    const index_t mgroup_id = mn_group_id - ngroup_id * group_count_m;

    /* register offsets */
    const index_t global_m_offset = mgroup_id * tile_size_dim_m;
    const index_t global_n_offset = ngroup_id * tile_size_dim_n;
    const index_t global_k_offset = kgroup_id * tile_size_dim_k;

    extract_input_blocks(local_id, 0, m_, n_, k_, A, lda_, B, ldb_, scratch_ptr,
                         rhs_scratch_ptr, global_m_offset, global_n_offset,
                         global_k_offset);

    const index_t start_lhs_index = m_local_id;
    const index_t start_rhs_index = n_local_id;
    index_t tile_id = 0;
    /* Loop over all tiles allocated to this particular workgroup size */
    do {
      // Make sure the current tile is fully loaded
      id.barrier(cl::sycl::access::fence_space::local_space);

      // Start loading the next tile
      index_t next_tile = tile_id + 1;
      if (next_tile < num_tiles) {
        extract_input_blocks(local_id, next_tile, m_, n_, k_, A, lda_, B, ldb_,
                             scratch_ptr, rhs_scratch_ptr, global_m_offset,
                             global_n_offset, global_k_offset);
      }

      // Calculate offsets into the temporary memory.
      index_t lhs_offset =
          ((tile_id & 1) * (tile_size_dim_m * tile_size_dim_k)) +
          start_lhs_index;
      index_t rhs_offset =
          ((tile_id & 1) * (tile_size_dim_k * tile_size_dim_n)) +
          start_rhs_index;

      // TODO: remove me
      if (local_id == 0 && wg_id == 0) {
        printf("tile %d\n", tile_id);
        printf(" - LHS -\n");
        for (int si = 0; si < tile_size_dim_m; si++) {
          for (int sj = 0; sj < tile_size_dim_k; sj++) {
            printf("%f ", scratch_ptr[lhs_offset + si + tile_size_dim_m * sj]);
          }
          printf("\n");
        }
        printf(" - RHS -\n");
        for (int si = 0; si < tile_size_dim_k; si++) {
          for (int sj = 0; sj < tile_size_dim_n; sj++) {
            printf("%f ",
                   rhs_scratch_ptr[rhs_offset + tile_size_dim_n * si + sj]);
          }
          printf("\n");
        }
        printf("\n");
      }

      /* Loop over the values of a single tile */
      for (index_t k = 0; k < tile_size_dim_k; k++) {
        auto idx = 0;
        auto rhs_index = 0;
        for (index_t wLPTN = 0; wLPTN < work_per_thread_n; wLPTN++) {
          // load a RHS element from the scratch buffer
          element_t privateRhs = rhs_scratch_ptr[rhs_index + rhs_offset];

          index_t lhs_index = 0;
          for (index_t wLPTM = 0; wLPTM < work_per_thread_m; wLPTM++) {
            // load a LHS element from the scratch buffer
            element_t privateLhs = scratch_ptr[lhs_index + lhs_offset];

            // Perform a manual MAD.
            private_res[wLPTM + idx] =
                private_res[wLPTM + idx] + (privateLhs * privateRhs);

            lhs_index += local_thread_size_m;
          }
          idx += work_per_thread_m;
          rhs_index += local_thread_size_n;
        }
        lhs_offset += tile_size_dim_m;
        rhs_offset += tile_size_dim_n;
      }
      tile_id++;
    } while (tile_id < num_tiles);

    id.barrier(cl::sycl::access::fence_space::local_space);

    // Store the final results in the cube buffer
    index_t cube_col_offset = (ngroup_id * tile_size_dim_n) + (n_local_id);
    index_t cube_row_offset = (mgroup_id * tile_size_dim_m) + (m_local_id);
    index_t cube_depth_offset = kgroup_id * m_ * n_;
    index_t cube_index = cube_col_offset * m_;
    index_t private_index_offset = 0;

    for (index_t wLPTN = 0; wLPTN < work_per_thread_n; wLPTN++) {
      index_t private_index = private_index_offset;

      index_t cube_col = cube_col_offset;
      index_t cube_row = cube_row_offset;
      for (index_t wLPTM = 0; wLPTM < work_per_thread_m; wLPTM++) {
        if (/*(NoEdge) ||*/ (cube_row < m_ && cube_col < n_)) {
          cube_buffer[cube_index + cube_row + cube_depth_offset] =
              private_res[wLPTM + private_index];
        }
        cube_row += local_thread_size_m;
      }
      cube_index += m_;
      private_index += work_per_thread_m;

      cube_col_offset += local_thread_size_n;
      cube_index = cube_col_offset * m_;
      private_index_offset += work_per_thread_m;
    }
  }

  template <typename global_ptr_t, typename local_ptr_t>
  static SYCL_BLAS_INLINE void extract_input_blocks(
      index_t local_id, index_t tile_idx, index_t m_, index_t n_, index_t k_,
      global_ptr_t A, index_t lda, global_ptr_t B, index_t ldb,
      local_ptr_t scratch_ptr, local_ptr_t rhs_scratch_ptr,
      index_t global_m_offset, index_t global_n_offset,
      index_t global_k_offset) {
    // LHS tile
    if (trans_a) {
      load_and_transpose_block<loads_per_thread_lhs>(
          local_id, tile_idx, A, lda, scratch_ptr, global_k_offset,
          global_m_offset, k_, m_, tile_size_dim_k, tile_size_dim_m);
    } else {
      load_block<loads_per_thread_lhs>(local_id, tile_idx, A, lda, scratch_ptr,
                                       global_m_offset, global_k_offset, m_, k_,
                                       tile_size_dim_m, tile_size_dim_k);
    }
    // RHS tile
    if (trans_b) {
      load_block<loads_per_thread_rhs>(
          local_id, tile_idx, B, ldb, rhs_scratch_ptr, global_n_offset,
          global_k_offset, n_, k_, tile_size_dim_n, tile_size_dim_k);
    } else {
      load_and_transpose_block<loads_per_thread_rhs>(
          local_id, tile_idx, B, ldb, rhs_scratch_ptr, global_k_offset,
          global_n_offset, k_, n_, tile_size_dim_k, tile_size_dim_n);
    }
  }

  // TODO: less duplication with templates?

  template <index_t loads_per_thread, typename global_ptr_t,
            typename local_ptr_t>
  static SYCL_BLAS_INLINE void load_block(
      index_t local_id, index_t tile_idx, global_ptr_t global_ptr,
      index_t leading_dim, local_ptr_t local_ptr, index_t global_row_offset,
      index_t global_col_offset, index_t global_rows, index_t global_cols,
      index_t block_rows, index_t block_cols) {
    index_t local_mem_id = local_id;
    const index_t global_tile_col_offset =
        global_col_offset + block_cols * tile_idx;
    const index_t local_thread_size = local_thread_size_n * local_thread_size_m;

    // Double buffering
    index_t local_mem_offset = (tile_idx & 1) * (block_rows * block_cols);

    for (index_t lPT = 0; lPT < loads_per_thread; lPT++) {
      index_t local_thread_col = local_mem_id / block_rows;
      index_t local_thread_row = local_mem_id - (local_thread_col * block_rows);

      index_t global_col_index = global_tile_col_offset + local_thread_col;
      index_t global_row_index = global_row_offset + local_thread_row;

      element_t val = 0;
      if (/*(NoEdge) ||*/ ((global_row_index < global_rows) &&
                           (global_col_index < global_cols))) {
        val = global_ptr[global_col_index * leading_dim + global_row_index];
      }

      local_ptr[local_mem_offset + local_mem_id] = val;

      local_mem_id += local_thread_size;
    }
  }

  template <index_t loads_per_thread, typename global_ptr_t,
            typename local_ptr_t>
  static SYCL_BLAS_INLINE void load_and_transpose_block(
      index_t local_id, index_t tile_idx, global_ptr_t global_ptr,
      index_t leading_dim, local_ptr_t local_ptr, index_t global_row_offset,
      index_t global_col_offset, index_t global_rows, index_t global_cols,
      index_t block_rows, index_t block_cols) {
    index_t local_linear_id = local_id;
    const index_t global_tile_row_offset =
        global_row_offset + block_rows * tile_idx;
    const index_t local_thread_size = local_thread_size_n * local_thread_size_m;

    // Double buffering
    index_t local_mem_offset = (tile_idx & 1) * (block_rows * block_cols);

    for (index_t lPT = 0; lPT < loads_per_thread; lPT++) {
      index_t local_thread_col = local_linear_id / block_rows;
      index_t local_thread_row =
          local_linear_id - (local_thread_col * block_rows);

      index_t global_col_index = global_col_offset + local_thread_col;
      index_t global_row_index = global_tile_row_offset + local_thread_row;

      // Transpose on the fly
      index_t local_mem_id = local_thread_col + (local_thread_row * block_cols);

      element_t val = 0;
      if (/*(NoEdge) ||*/ ((global_row_index < global_rows) &&
                           (global_col_index < global_cols))) {
        val = global_ptr[global_col_index * leading_dim + global_row_index];
        // todo: change global_rows to leading dimension
      }

      local_ptr[local_mem_offset + local_mem_id] = val;

      local_linear_id += local_thread_size;
    }
  }
};

}  // namespace blas

#endif  // SYCL_BLAS_BLAS3_PARTIAL_GEMM_HPP
