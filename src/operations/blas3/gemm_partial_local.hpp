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
 *  @filename gemm_tall_skinny.hpp
 *
 **************************************************************************/

#ifndef SYCL_BLAS_BLAS3_PARTIAL_GEMM_HPP
#define SYCL_BLAS_BLAS3_PARTIAL_GEMM_HPP

#include "gemm_common.hpp"

#include <sstream>

#define dbg(expr)                                    \
  [&]() -> decltype(expr) {                          \
    auto __macro_val = expr;                         \
    std::cerr << #expr " = " << __macro_val << ", "; \
    return __macro_val;                              \
  }()

#define dbg_str(expr)                   \
  [&]() -> std::string {                \
    std::stringstream sstr;             \
    auto __macro_val = expr;            \
    sstr << #expr " = " << __macro_val; \
    return sstr.str();                  \
  }()

namespace blas {

// So far, more or less just copied from eigen.
// template <typename OutScalar, typename LhsScalar, typename RhsScalar,
// typename OutAccessor, typename TempAcc, typename LhsMapper,
// typename RhsMapper, typename Scratch, typename Index,
// typename PanelParameters, bool Vectorizable, bool NoEdge, boolIsFinal>
template <typename input_t, typename output_t, bool DoubleBuffer, bool NbcA,
          bool NbcB, int ClSize, typename tile_type, bool TransA, bool TransB,
          typename element_t, bool is_beta_zero>
class GemmPartial<input_t, output_t, DoubleBuffer, NbcA, NbcB,
                  ClSize, tile_type, TransA, TransB, element_t, is_beta_zero,
                  static_cast<int>(Gemm_t::tall_skinny_local_memory)> {
 public:
  // using index_t = typename std::make_signed<typename input_t::index_t>::type;
  using index_t = int;
  using value_t = element_t;

  // Temporary
  // using scratch_t =
  //     cl::sycl::accessor<element_t, 1, cl::sycl::access::mode::read_write,
  //                        cl::sycl::access::target::local>;

  input_t a_;
  input_t b_;
  output_t c_;

  element_t alpha;
  element_t beta;

  // OutAccessor out_res;

  // const LhsMapper lhs;
  // const RhsMapper rhs;

  /* Matrix dimensions */
  const index_t m_;
  const index_t n_;
  const index_t k_;
  index_t lda_;
  index_t ldb_;
  index_t ldc_;

  static constexpr index_t local_thread_size_n = tile_type::local_thread_size_n;
  static constexpr index_t local_thread_size_m = tile_type::local_thread_size_m;

  static constexpr int local_thread_size = local_thread_size_m * local_thread_size_n;

  /* The number of tiles to be processed */
  static constexpr index_t num_tiles = tile_type::num_tiles;

  /* The dimensions of a single tile */
  static constexpr index_t tile_size_dim_m = tile_type::tile_size_dim_m;
  static constexpr index_t tile_size_dim_n = tile_type::tile_size_dim_n;
  static constexpr index_t tile_size_dim_k = tile_type::tile_size_dim_k;

  /* Local memory */
  static constexpr int local_memory_size = 2 * tile_size_dim_m * tile_size_dim_k + 2 * tile_size_dim_k * tile_size_dim_n;

  /* Workload per work item on each dimension m and n */
  static constexpr index_t work_per_thread_m = tile_type::work_per_thread_m;
  static constexpr index_t work_per_thread_n = tile_type::work_per_thread_n;

  /* If double buffering should be used */
  static constexpr bool double_buffer = DoubleBuffer;

  /* Transpose mode for matrices A and B */
  static constexpr bool trans_a = TransA;
  static constexpr bool trans_b = TransB;

  // TODO: what is this? Is this relevant here?
  static constexpr bool nbc_a = NbcA;
  static constexpr bool nbc_b = NbcB;

  /* The number of groups on each dimension m, n, k */
  const index_t group_count_m;
  const index_t group_count_n;
  const index_t group_count_k;

  // input_t A, input_t B, output_t C, element_t alpha,
  //                       element_t beta, index_t batch_size
  SYCL_BLAS_INLINE GemmPartial(input_t A, input_t B, output_t C,
                               element_t alpha, element_t beta,
                               index_t group_count_m, index_t group_count_n, index_t group_count_k)
      : a_(A),
        b_(B),
        c_(C),
        alpha(alpha),
        beta(beta),
        m_(a_.get_size_row()),
        n_(b_.get_size_col()),
        k_(a_.get_size_col()),
        lda_(a_.getSizeL()),
        ldb_(b_.getSizeL()),
        ldc_(c_.getSizeL()),
        group_count_m(group_count_m),
        group_count_n(group_count_n),
        group_count_k(group_count_k) {}

  /*!
   * @brief Get the type of this GemmPartial as a human readable string.
   */
  static SYCL_BLAS_INLINE std::string get_type_string() noexcept {
    std::ostringstream str{};
    str << "GemmPartial<" << double_buffer << ", " << nbc_a << ", " << nbc_b
        << ", " << tile_type::get_type_string() << ", "
        << type_string<element_t>::get_value() << ">";
    return str.str();
  }

  void bind(cl::sycl::handler &h) {
    a_.bind(h);
    b_.bind(h);
    c_.bind(h);
  }
  SYCL_BLAS_INLINE bool valid_thread(cl::sycl::nd_item<1> ndItem) const {
    return true;
  }

  /*!
   * @brief get_workgroup_cluster. This function is used to find the optimum
   * number of work_group required to execute each partial GEMM.
   */
  static SYCL_BLAS_INLINE index_t get_workgroup_cluster(index_t m,
                                                        index_t n) noexcept {
    return 1; // TODO: use real value here
    // return group_count_m * group_count_n * group_count_k;
  }
  /*!
   * @brief get_num_workgroup_cluster. This function is used to extend the number
   * of work_group cluster, in order to make sure that atleast 4 gemm operations
   * are available per work group. The number 4 is used based on empirical
   * research.
   */
  static SYCL_BLAS_INLINE index_t get_num_workgroup_cluster(
      index_t m, index_t n, index_t compute_units) noexcept {
    // return ((4 * compute_units - 1) / get_workgroup_cluster(m, n) + 1);
    return 1; // TODO: optimize that later
  }

  /*!
   * @brief Get the nd_range value which has to be used for kernels that
   *        intend to call GemmPartial::run().
   */
  static SYCL_BLAS_INLINE cl::sycl::nd_range<1> get_nd_range(
      index_t m, index_t n, index_t compute_units) noexcept {
    const cl::sycl::range<1> nwg(
        get_workgroup_cluster(m, n) *
        get_num_workgroup_cluster(m, n, compute_units));
    const cl::sycl::range<1> wgs(local_thread_size);
    return cl::sycl::nd_range<1>(nwg * wgs, wgs);
    // TODO: add verbose
  }

  SYCL_BLAS_INLINE index_t get_size() const { return m_ * n_; }

  template <typename local_memory_t>
  SYCL_BLAS_INLINE void eval(local_memory_t scratch,
                   cl::sycl::nd_item<1> id) noexcept {
    /* references to the matrices */
    auto A = a_.get_data().get_pointer().get() + a_.get_access_displacement();
    auto B = b_.get_data().get_pointer().get() + b_.get_access_displacement();
    auto C = c_.get_data().get_pointer().get() + c_.get_access_displacement();

    /* references to the temporary memory, scratch memory, and rhs scratch
     * memory*/
    auto scratch_ptr = scratch.localAcc.get_pointer().get();

    auto rhs_scratch_ptr =
        scratch_ptr + (2 * tile_size_dim_m * tile_size_dim_k);

    element_t private_res[work_per_thread_m * work_per_thread_n];

    /* workgroup id */
    const index_t wg_id = id.get_group(0);
    /* Local thread id */
    const index_t local_id = id.get_local_id(0);

    /* Local ID column and row */
    const index_t n_local_id = local_id / local_thread_size_m;
    const index_t m_local_id = local_id - (n_local_id * local_thread_size_m);

    /* workgroup id, local column */
    const index_t tmp = wg_id / group_count_m;

    /* Workgroup id row */
    const index_t mgroup_id = wg_id % group_count_m;
    // wg_id - (tmp * group_count_m);  // isn't this always 0???

    const index_t kgroup_id = (wg_id / group_count_m) / group_count_n;
    const index_t ngroup_id = wg_id % group_count_n;

    // tmp - (kgroup_id * group_count_n);

    /* register offsets */
    const index_t global_mix_offset = mgroup_id * tile_size_dim_m;
    const index_t global_nix_offset = ngroup_id * tile_size_dim_n;
    const index_t global_kix_offset = kgroup_id * tile_size_dim_k;

    /* initialise the private res summation registers */
    for (auto i = 0; i < work_per_thread_m * work_per_thread_n; i++) {
      private_res[i] = static_cast<element_t>(0);
    }

    /* Load tiles, LHS and RHS */

    // Is there any reason why we "preload" these here? We have a do while
    // loop... -> we preload next iteration each time (double buffering)?

    // Tile LHS for now, assume that the LHS isn't transposed.
    load_tile(A, scratch_ptr, local_id, global_mix_offset, global_kix_offset, 0,
              work_per_thread_m, m_, k_);
    // Tile RHS
    load_and_transpose_tile(B, rhs_scratch_ptr, local_id, global_nix_offset,
                            global_kix_offset, 0, work_per_thread_n, k_, n_);

    id.barrier(cl::sycl::access::fence_space::local_space);

    const index_t start_lhs_index = m_local_id;
    const index_t start_rhs_index = n_local_id;
    index_t tile_id = 0;
    /* Loop over all tiles allocated to this particular workgroup size */
    do {
      // Synchronise

      id.barrier(cl::sycl::access::fence_space::local_space);

      index_t next_tile = tile_id + 1;
      // If we need to swap, or not? -> no need to preload next tile if there is
      // no next tile
      if (next_tile < num_tiles) {
        /* Load tiles, LHS and RHS into local memory */
        // Tile LHS
        load_tile(A, scratch_ptr, local_id, global_mix_offset,
                  global_kix_offset, next_tile, work_per_thread_m, m_, k_);
        // Tile RHS
        load_and_transpose_tile(B, rhs_scratch_ptr, local_id, global_nix_offset,
                                global_kix_offset, next_tile, work_per_thread_n,
                                k_, n_);
      }
      // Calculate offsets into the temporary memory.

      index_t lhs_offset =
          ((tile_id & 1) * (tile_size_dim_m * tile_size_dim_k)) +
          start_lhs_index;
      index_t rhs_offset =
          ((tile_id & 1) * (tile_size_dim_k * tile_size_dim_n)) +
          start_rhs_index;

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
            private_res[wLPTM + idx] =  // local_id;
                private_res[wLPTM + idx] + (privateLhs * privateRhs);

            lhs_index += local_thread_size_m;
          }
          idx += work_per_thread_m;
          rhs_index += local_thread_size_n;
        }
        lhs_offset += tile_size_dim_m;
        rhs_offset += tile_size_dim_n;
      }
      // Next tile
      tile_id++;
    } while (tile_id < num_tiles);

    id.barrier(cl::sycl::access::fence_space::local_space);

    // Store the final results in C
    index_t global_col_offset = (ngroup_id * tile_size_dim_n) + (n_local_id);
    index_t global_row_offset = (mgroup_id * tile_size_dim_m) + (m_local_id);
    index_t global_k_offset = kgroup_id * m_ * n_;
    index_t c_index = global_col_offset * m_;
    index_t private_index_offset = 0;

    for (index_t wLPTN = 0; wLPTN < work_per_thread_n; wLPTN++) {
      // Disregard anything involving `i` - it simply specifies a stride
      // for (index_t i = 0; i < PacketSize; i++) {
      index_t global_col = global_col_offset;  // + i;
      index_t global_row = global_row_offset;
      for (index_t wLPTM = 0; wLPTM < work_per_thread_m; wLPTM++) {
        if (/*(NoEdge) ||*/ (global_row < m_ && global_col < n_)) {
          // Store the final results in C

          C[c_index + global_row + global_k_offset] =
              private_res[wLPTM + private_index_offset];
        }
      }
      c_index += m_;
      //}
      global_col_offset += local_thread_size_n;
      c_index = global_col_offset * m_;
      private_index_offset += work_per_thread_m;
    }
  }
  // We need two load functions: one that loads normally, one that
  // loads + transposes on load.

  // Load a "left hand" tile, or "right hand transposed" tile
  // What is NoEdge for?? -> bypass test if we know we're in an inside block
  template <typename GlobalPointerType, typename LocalPointerType>
  static inline void load_tile(GlobalPointerType glb_ptr,
                               LocalPointerType lcl_ptr,
                               index_t linear_local_thread_id,
                               index_t global_m_offset, index_t global_k_offset,
                               index_t next_tile, index_t load_per_thread_lhs,
                               index_t M, index_t K) {
    // Our rhs linear id is the same as our thread id to start with
    index_t local_lhs_linear_id = linear_local_thread_id;

    // Local id offset depends on whether we're on the first or second "half" of
    // the scratch memory. If we're on the first half (i.e. the lowest bit is
    // set to 0), then the offset is 0. If we're on the second half (i.e. the
    // lowest bit is set to 1), then the offset is the linear size of a RHS
    // tile: tile_size_dim_m * tile_size_dim_k
    index_t linear_local_id_offset =
        (next_tile & 1) * (tile_size_dim_m * tile_size_dim_k);

    for (index_t lPTL = 0; lPTL < load_per_thread_lhs; lPTL++) {
      index_t local_thread_k = local_lhs_linear_id / tile_size_dim_m;
      index_t local_thread_m =
          local_lhs_linear_id - (local_thread_k * tile_size_dim_m);

      index_t global_k_index = global_k_offset + local_thread_k;
      index_t global_m_index = global_m_offset + local_thread_m;

      index_t linear_local_id_index =
          local_thread_m + (local_thread_k * tile_size_dim_m);

      // We can ignore this check, as we're not using packet types right now
      // if (/*(NoEdge) ||*/ ((global_m_index < M) && (global_k_index < K))) {
      // load from matrix according to global_m_index and
      // global_k_index

      element_t val = glb_ptr[global_m_index + (global_k_index * M)];

      lcl_ptr[linear_local_id_index + linear_local_id_offset] = val;
      // }

      local_lhs_linear_id += (local_thread_size_n * local_thread_size_m);
    }
  }

  template <typename GlobalPointerType, typename LocalPointerType>
  static inline void load_and_transpose_tile(
      GlobalPointerType glb_ptr, LocalPointerType lcl_ptr,
      index_t linear_local_thread_id, index_t global_n_offset,
      index_t global_k_offset, index_t next_tile, index_t load_per_thread_rhs,
      index_t K, index_t N) {
    // Our rhs linear id is the same as our thread id to start with
    index_t local_rhs_linear_id = linear_local_thread_id;
    // Local id offset depends on whether we're on the first or second "half" of
    // the scratch memory. If we're on the first half (i.e. the lowest bit is
    // set to 0), then the offset is 0. If we're on the second half (i.e. the
    // lowest bit is set to 1), then the offset is the linear size of a RHS
    // tile: tile_size_dim_k * tile_size_dim_n
    index_t linear_local_id_offset =
        (next_tile & 1) * (tile_size_dim_k * tile_size_dim_n);
    for (index_t lPTR = 0; lPTR < load_per_thread_rhs; lPTR++) {
      // Calculate the index in the 'n' dimension that this thread should access
      index_t local_thread_n = local_rhs_linear_id / tile_size_dim_k;
      // Calculate the index in the 'k' dimension that this thread should access
      index_t local_thread_k =
          local_rhs_linear_id - (tile_size_dim_k * local_thread_n);

      index_t global_k_index = global_k_offset + local_thread_k;
      index_t global_n_index = global_n_offset + local_thread_n;

      // Transpose RHS on the fly
      // index_t linear_local_id_index =
      //     local_thread_n + (local_thread_k * tile_size_dim_n);
      index_t linear_local_id_index =
          local_thread_k + (local_thread_n * tile_size_dim_k);

      element_t val = 0;
      if (/*(NoEdge) ||*/ ((global_k_index < K) && (global_n_index < N))) {
        val = glb_ptr[global_n_index + (global_k_index * N)];
      }

      lcl_ptr[linear_local_id_index + linear_local_id_offset] = val;
      local_rhs_linear_id += local_thread_size_n * local_thread_size_m;
    }
  }
};

}  // namespace blas

#endif  // SYCL_BLAS_BLAS3_PARTIAL_GEMM_HPP
