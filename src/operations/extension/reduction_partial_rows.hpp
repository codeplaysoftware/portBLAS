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
 *  @filename reduction_partial_rows.hpp
 *
 **************************************************************************/

#ifndef SYCL_BLAS_EXTENSION_REDUCTION_PARTIAL_ROWS_HPP
#define SYCL_BLAS_EXTENSION_REDUCTION_PARTIAL_ROWS_HPP

#include "operations/extension_trees.h"
#include "views/view.h"
#include <CL/sycl.hpp>
#include <string>

namespace blas {

// Definition of the reduction step class
template <typename input_t, typename output_t, int ClSize, int WgSize,
          int WorkPerItem, typename element_t, bool IsFinal>
class ReductionPartialRows {
 public:
  using index_t = typename std::make_signed<typename input_t::index_t>::type;
  using value_t = element_t;

  // static constexpr bool is_final = IsFinal;
  // TODO: either use IsFinal or remove it

  /* The number of elements per cache line size depends on the element type */
  static constexpr index_t cl_elems = ClSize / sizeof(element_t);

  input_t in_;
  output_t out_;

  /* Matrix dimensions */
  const index_t rows_;
  const index_t cols_;
  const index_t leading_dim_;

  /* Workload per work item on each dimension m and n */
  static constexpr index_t rows_per_item = WorkPerItem;

  /* Checking if the parameters are valid */
  static_assert(cl_elems % rows_per_item == 0,
                "The number of rows processed per item must divide the number "
                "of elements per cache line.");

  /* Work group dimensions */
  static constexpr index_t work_group_size = WgSize;
  static constexpr index_t work_group_rows = cl_elems / rows_per_item;
  static constexpr index_t work_group_cols = work_group_size / work_group_rows;
  // TODO: we may want more columns there

  /* Local memory dimensions */
  static constexpr index_t local_memory_rows =
      work_group_rows * rows_per_item;
  static constexpr index_t local_memory_cols = work_group_cols;
  static constexpr index_t local_memory_size =
      local_memory_rows * local_memory_cols;

  /* Work groups per dimension */
  const index_t group_count_rows_;
  const index_t group_count_cols_;

  SYCL_BLAS_INLINE ReductionPartialRows(input_t in, output_t out,
                                        index_t num_rows, index_t num_cols,
                                        index_t group_count_cols)
      : in_(in),
        out_(out),
        rows_(in_.get_size_row()),
        cols_(in_.get_size_col()),
        leading_dim_(in_.getSizeL()),
        group_count_rows_((num_rows - 1) / local_memory_rows + 1),
        group_count_cols_(group_count_cols) {}

  /*!
   * @brief Get the type of this reduction as a human readable string.
   */
  static SYCL_BLAS_INLINE std::string get_type_string() noexcept {
    std::ostringstream str{};
    str << "ReductionPartialRows<>";
    // TODO: add type string
    return str.str();
  }

  void bind(cl::sycl::handler &h) {
    in_.bind(h);
    out_.bind(h);
  }
  void adjust_access_displacement() {
    in_.adjust_access_displacement();
    out_.adjust_access_displacement();
  }
  SYCL_BLAS_INLINE bool valid_thread(cl::sycl::nd_item<1> ndItem) const {
    return true;
  }

  /*!
   * @brief get_workgroup_cluster. This function is used to find the optimum
   * number of work_group required to execute each partial reduction step.
   */
  SYCL_BLAS_INLINE index_t get_workgroup_cluster() noexcept {
    return ((rows_ - 1) / local_memory_rows + 1) * group_count_cols_;
  }
  /*!
   * @brief get_num_workgroup_cluster. This function is used to extend the
   * number of work_group cluster, in order to make sure that at least 4
   * operations are available per work group. The number 4 is used based on
   * empirical research.
   */
  SYCL_BLAS_INLINE index_t get_num_workgroup_cluster(index_t compute_units) noexcept {
    return 1;  // TODO: optimize that later
  }

  /*!
   * @brief Get the nd_range value which has to be used for kernels that
   *        intend to call GemmPartial::run().
   */
  SYCL_BLAS_INLINE cl::sycl::nd_range<1> get_nd_range(index_t compute_units) noexcept {
    const cl::sycl::range<1> nwg(
        get_workgroup_cluster() *
        get_num_workgroup_cluster(compute_units));
    const cl::sycl::range<1> wgs(work_group_size);
    return cl::sycl::nd_range<1>(nwg * wgs, wgs);
    // TODO: add verbose
  }

  // TODO: I'm not sure if this method should stay here or not
  SYCL_BLAS_INLINE index_t get_size() const { return rows_; }

  template <typename local_memory_t>
  SYCL_BLAS_INLINE void eval(local_memory_t scratch,
                             cl::sycl::nd_item<1> id) noexcept {
    /* references to the input and output data */
    auto in_ptr = in_.get_pointer();
    auto out_ptr = out_.get_pointer();

    /* reference to the scratch memory */
    auto scratch_ptr = scratch.localAcc.get_pointer().get();

    /* workgroup id */
    const index_t group_id = id.get_group(0);
    /* Local thread id */
    const index_t local_id = id.get_local_id(0);

    /* Block row and column */
    const index_t group_row = group_id % group_count_rows_;  // TODO: no modulo
    const index_t group_col = group_id / group_count_rows_;

    /* Item row and column within a block */
    const index_t local_row = local_id % work_group_rows;  // TODO: no modulo
    const index_t local_col = local_id / work_group_rows;

    /* Global position of the first element processed by the thread */
    const index_t global_row = group_row * local_memory_rows + local_row;
    const index_t global_col = group_col * local_memory_cols + local_col;

    /* Total number of item cols in all work groups */
    const index_t total_item_cols = local_memory_cols * group_count_cols_;

    /* Initialize private reduction registers */
    element_t accumulators[rows_per_item] {element_t(0)};

    /* Sequential reduction level:
     * Load multiple elements from the global memory, reduce them together and
     * store them in the local memory */
    {
      index_t elem_col = global_col;
      while(elem_col < cols_) {
        index_t elem_col_idx = leading_dim_ * elem_col;
        /* Each thread is responsible for multiple independent rows */
        index_t elem_row = global_row;
        #pragma unroll
        for (index_t wpr = 0; wpr < rows_per_item; wpr++) {
          if (elem_row < rows_) {
            accumulators[wpr] += in_ptr[elem_col_idx + elem_row];
            // TODO: use reduction actual operation
          }
          elem_row += work_group_rows;
        }
        elem_col += total_item_cols;
      };
    }

    /* Copy the accumulation registers into the local memory */
    {
      index_t local_memory_idx = local_memory_rows * local_col + local_row;
      #pragma unroll
      for (index_t wpr = 0; wpr < rows_per_item; wpr++) {
        scratch_ptr[local_memory_idx] = accumulators[wpr];
        local_memory_idx += work_group_rows;
      }
    }
    // TODO: use finalize operation if supporting mean reductions

    /* Parallel-reduction level:
     * Tree-based reduction in local memory */
    {
      const index_t local_memory_lhs = local_col * local_memory_rows;
      #pragma unroll
      for (index_t stride = work_group_cols / 2; stride > 0; stride /= 2) {
        const index_t local_memory_rhs =
            (local_col + stride) * local_memory_rows;
        /* Synchronize group */
        id.barrier(cl::sycl::access::fence_space::local_space);

        /* Only the lhs performs the reduction */
        if (local_col < stride) {
          /* Each thread is responsible for multiple independent rows */
          index_t local_memory_row = local_row;
          #pragma unroll
          for (index_t wpr = 0; wpr < rows_per_item; wpr++) {
            /* Reduce left-hand and right-hand elements together */
            scratch_ptr[local_memory_lhs + local_memory_row] +=
                scratch_ptr[local_memory_rhs + local_memory_row];
            local_memory_row += work_group_rows;
          }
        }
      }
    }

    /* Threads of the first column write their results in the output buffer */
    if (local_col == 0) {
      const index_t global_row_offset = group_row * local_memory_rows;
      index_t local_memory_row = local_row;
      index_t out_memory_idx =
          group_col * rows_ + global_row_offset;
      #pragma unroll
      for (index_t wpr = 0; wpr < rows_per_item; wpr++) {
        if(global_row_offset + local_memory_row < rows_) {
          out_ptr[out_memory_idx + local_memory_row] =
              scratch_ptr[local_memory_row];
        }
        local_memory_row += work_group_rows;
      }
    }
  }
};

}  // namespace blas

#endif  // SYCL_BLAS_EXTENSION_REDUCTION_PARTIAL_ROWS_HPP
