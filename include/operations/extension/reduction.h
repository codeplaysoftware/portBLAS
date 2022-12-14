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
 *  @filename reduction.h
 *
 **************************************************************************/

#ifndef SYCL_BLAS_EXTENSION_REDUCTION_H
#define SYCL_BLAS_EXTENSION_REDUCTION_H

#include <CL/sycl.hpp>

#include "container/sycl_iterator.h"

namespace blas {

/*!
 * @brief Determines which type of reduction to perform
 *
 * - inner: partial reduction along the columns
 * - outer: partial reduction along the rows
 */
enum class reduction_dim_t : int { inner = 0, outer = 1 };

/*!
 * @brief Calculates the parameters of the reduction step
 *
 * @tparam index_type Datatype of indices
 * @tparam element_type Datatype of matrix elements
 * @tparam ClSize Cache line size
 * @tparam WgSize Workgroup size
 * @tparam ReductionsPerThread Reductions to perform per thread
 * @tparam reduction_dim Dimension along which to perform the reduction (see
 * `reduction_dim_t` enum)
 * @tparam bank_offset Whether to use an offset to prevent bank conflicts
 */
template <typename index_type, typename element_type, int ClSize, int WgSize,
          int ReductionsPerThread, int reduction_dim, bool bank_offset = true>
struct ReductionParams {
  using index_t = index_type;
  using element_t = element_type;

  // The number of elements per cache line size depends on the element type
  static constexpr index_t get_num_cache_line_elems() {
    return ClSize / sizeof(element_t);
  }

  static constexpr index_t get_workgroup_col() {
    return get_num_cache_line_elems();
  }

  static constexpr index_t get_workgroup_row() {
    return WgSize / get_num_cache_line_elems();
  }

  static constexpr index_t get_local_thread_size_preserve() {
    return is_outer_dim() ? get_num_cache_line_elems()
                          : WgSize / get_num_cache_line_elems();
  }

  static constexpr index_t get_local_thread_size_reduce() {
    return is_outer_dim() ? WgSize / get_num_cache_line_elems()
                          : get_num_cache_line_elems();
  }

  static constexpr int get_reduction_dim() { return reduction_dim; }

  // Offset to use to prevent bank conflicts
  static constexpr bool use_bank_offset() { return bank_offset; }

  // Local memory dimensions
  static constexpr index_t get_local_memory_size() {
    return get_local_thread_size_reduce() *
           (get_local_thread_size_preserve() + use_bank_offset());
  }

  static constexpr bool is_outer_dim() {
    return reduction_dim == static_cast<int>(reduction_dim_t::outer);
  }

  static constexpr index_t get_reductions_per_thread() {
    return ReductionsPerThread;
  }

  static index_t calculate_reduced_group_count(index_t rows, index_t cols) {
    constexpr index_t reductions_per_thread = get_reductions_per_thread();
    constexpr index_t local_range =
        get_local_thread_size_preserve() * get_local_thread_size_reduce();
    const auto num_elems_to_reduce = is_outer_dim() ? cols : rows;
    const index_t max_group_count_reduce =
        (num_elems_to_reduce - 1) / get_local_thread_size_reduce() + index_t(1);
    index_t reduced_group_count =
        std::min(get_local_thread_size_reduce(), max_group_count_reduce);
    reduced_group_count =
        num_elems_to_reduce > reductions_per_thread * local_range
            ? std::min(reduced_group_count, local_range)
            : index_t(1);

    return reduced_group_count;
  }
};

/*!
 * @brief This class holds the kernel for the partial reduction of the rows.
 *
 * The output buffer will contain the same number of rows as the input buffer
 * and a smaller number of columns if reducing along the outer dimension and
 * the same number of columns and a smaller number of rows if reducing along the
 * inner dimension. Eventually this will result in a single column or row. The
 * number of work groups can be chosen to control the number of steps before the
 * reduction is complete.
 *
 * The reduction kernel uses the following algorithm:
 *
 * 1. Load multiple values from global memory.
 * 2. Reduce them together using the reduction operator (`operator_t`)
 * 3. Store the result in local memory.
 * 4. Perform the reduction operation on the element in local memory with
 * current local id and the corresponding element in the second half of local
 * memory.
 * 5. Store the result in the appropriate part of the output vector.
 *
 * @tparam operator_t Reduction operation to perform (one of AddOperator,
 * AbsoluteAddOperator, ProductOperator, DivisionOperator, MaxOperator,
 * MinOperator, MeanOperator)
 * @tparam params_t ReductionParams to use
 * @tparam input_t The input matrix type
 * @tparam output_t The output matrix type
 */
template <typename operator_t, typename params_t, typename input_t,
          typename output_t>
class Reduction {
 public:
  using index_t = typename params_t::index_t;
  using element_t = typename params_t::element_t;
  using value_t = element_t;
  /// Neutral value for this reduction operator
  /// TODO(Peter): This should be constexpr once half supports it
  static const element_t init_val;
  /* Input and output buffers */
  input_t in_;
  output_t out_;
  /* Matrix dimensions */
  const index_t rows_;
  const index_t cols_;
  const index_t leading_dim_;
  const index_t ld_mul_;
  /* Work groups per dimension */
  const index_t reduced_group_count_;
  const index_t group_count_rows_;
  const index_t group_count_cols_;
  const index_t preserve_elements_num_groups_;
  const index_t reduce_elements_num_groups_;
  const index_t num_elems_to_preserve_;
  const index_t num_elems_to_reduce_;
  Reduction(input_t in, output_t out);
  bool valid_thread(cl::sycl::nd_item<1> id) const;
  void bind(cl::sycl::handler& h);
  void adjust_access_displacement();
  cl::sycl::nd_range<1> get_nd_range(index_t compute_units) noexcept;
  void reduce(index_t global_reduce_id, index_t global_preserve_id,
              element_t& accumulator) noexcept;
  template <typename local_memory_t>
  void eval(local_memory_t scratch, cl::sycl::nd_item<1> id) noexcept;
};

/*!
 * @brief Helper function used for constructing the Reduction kernel.
 *
 * @tparam operator_t Reduction operator to use (one of AddOperator,
 * AbsoluteAddOperator, ProductOperator, DivisionOperator, MaxOperator,
 * MinOperator, MeanOperator)
 * @tparam params_t Reduction parameters to use (see `ReductionParams` struct)
 * @tparam input_t Type of the input matrix
 * @tparam output_t Type of the output matrix
 * @param in Input matrix
 * @param out Output matrix
 */
template <typename operator_t, typename params_t, typename input_t,
          typename output_t>
inline Reduction<operator_t, params_t, input_t, output_t> make_reduction(
    input_t in, output_t out) {
  return Reduction<operator_t, params_t, input_t, output_t>(in, out);
}

}  // namespace blas

#endif  // SYCL_BLAS_EXTENSION_REDUCTION_H
