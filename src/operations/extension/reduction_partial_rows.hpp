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
// #include <type_traits>

namespace blas {

template <typename input_t, typename output_t, typename temp_t, int ClSize,
          typename tile_type, typename element_t, bool IsFinal>
class ReductionPartialRows {
 public:
  using index_t = typename std::make_signed<typename input_t::index_t>::type;
  using value_t = element_t;

  static constexpr index_t cl_size = ClSize;
  static constexpr bool is_final = IsFinal;

  input_t in_;
  output_t out_;
  temp_t temp_;

  /* Matrix dimensions */
  const index_t rows_;
  const index_t cols_;
  const index_t leading_dim_;

  /* Checking if the tile is valid */
  // static_assert(, "");
  // TODO: some checks here?

  /* Workload per work item on each dimension m and n */
  static constexpr index_t work_per_item_rows = tile_type::item_rows;
  static constexpr index_t work_per_item_cols = tile_type::item_cols;

  /* Work group dimensions */
  static constexpr index_t work_group_rows = tile_type::wg_rows;
  static constexpr index_t work_group_cols = tile_type::wg_cols;
  static constexpr index_t work_group_size = work_group_rows * work_group_cols;
  // TODO: we may want more columns there

  /* Local memory dimensions */
  static constexpr index_t local_memory_rows = work_group_rows * work_per_item_rows;
  static constexpr index_t local_memory_cols = work_group_cols;
  static constexpr index_t local_memory_size = local_memory_rows * local_memory_cols;

  /* Work groups per dimension */
  const index_t group_count_rows;
  const index_t group_count_cols;

  SYCL_BLAS_INLINE ReductionPartialRows(input_t in, output_t out, temp_t temp,
                                        index_t num_rows, index_t num_cols)
      : in_(in),
        out_(out),
        temp_(temp),
        rows_(in_.get_size_row()),
        cols_(in_.get_size_col()),
        leading_dim_(in_.getSizeL()),
        group_count_rows((num_rows - 1) / work_group_rows + 1),
        group_count_cols((num_cols - 1) / (work_group_cols * work_per_item_cols) + 1) {}

  /*!
   * @brief Get the type of this Reduction as a human readable string.
   */
  static SYCL_BLAS_INLINE std::string get_type_string() noexcept {
    std::ostringstream str{};
    str << "ReductionPartialRows<>";
    // TODO: add type string
    return str.str();
  }

  void bind(cl::sycl::handler &h) {
    in_.bind(h);
    if(is_final) {
      out_.bind(h);
    } else {
      temp_.bind(h);
    }
  }
  void adjust_access_displacement() {
    in_.adjust_access_displacement();
    if(is_final) {
      out_.adjust_access_displacement();
    } else {
      temp_.adjust_access_displacement();
    }
  }
  SYCL_BLAS_INLINE bool valid_thread(cl::sycl::nd_item<1> ndItem) const {
    return true;
  }

  /*!
   * @brief get_workgroup_cluster. This function is used to find the optimum
   * number of work_group required to execute each partial reduction step.
   */
  static SYCL_BLAS_INLINE index_t get_workgroup_cluster(
      index_t num_rows, index_t num_cols, index_t compute_units) noexcept {
    return 1;
    // TODO: real value
  }
  /*!
   * @brief get_num_workgroup_cluster. This function is used to extend the
   * number of work_group cluster, in order to make sure that at least 4
   * operations are available per work group. The number 4 is used based on
   * empirical research.
   */
  static SYCL_BLAS_INLINE index_t get_num_workgroup_cluster(
      index_t num_rows, index_t num_cols, index_t compute_units) noexcept {
    return 1;  // TODO: optimize that later
  }

  /*!
   * @brief Get the nd_range value which has to be used for kernels that
   *        intend to call GemmPartial::run().
   */
  static SYCL_BLAS_INLINE cl::sycl::nd_range<1> get_nd_range(
      index_t num_rows, index_t num_cols, index_t compute_units) noexcept {
    const cl::sycl::range<1> nwg(
        get_workgroup_cluster(num_rows, num_cols, compute_units) *
        get_num_workgroup_cluster(num_rows, num_cols, compute_units));
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
    auto out_ptr = is_final ? out_.get_pointer() : temp_.get_pointer();

    /* reference to the scratch memory */
    auto scratch_ptr = scratch.localAcc.get_pointer().get();

    /* workgroup id */
    const index_t wg_id = id.get_group(0);
    /* Local thread id */
    const index_t local_id = id.get_local_id(0);

    //
    // TODO: implement that lol
    //

    printf("Hello from thread %d\n", local_id);
  }

};

}  // namespace blas

#endif  // SYCL_BLAS_EXTENSION_REDUCTION_PARTIAL_ROWS_HPP
