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
 *  @filename reduction.hpp
 *
 **************************************************************************/

#ifndef SYCL_BLAS_EXTENSION_REDUCTION_HPP
#define SYCL_BLAS_EXTENSION_REDUCTION_HPP

#include "operations/extension/reduction.h"
#include "views/view.h"
#include <CL/sycl.hpp>
#include <string>
namespace blas {

template <typename operator_t, typename params_t, typename input_t,
          typename output_t>
SYCL_BLAS_INLINE Reduction<operator_t, params_t, input_t, output_t>::Reduction(
    input_t in, output_t out, index_t reduced_group_count)
    : in_(in),
      out_(out),
      rows_(in_.get_size_row()),
      cols_(in_.get_size_col()),
      leading_dim_(in_.getSizeL()),
      ld_mul_(in.getSizeL() / in_.get_size_row()),
      group_count_rows_(params_t::is_outer_dim()
                            ? ((rows_ - 1) / params_t::get_workgroup_row() + 1)
                            : reduced_group_count),
      group_count_cols_(
          params_t::is_outer_dim()
              ? reduced_group_count
              : ((rows_ - 1) / params_t::get_workgroup_row() + 1)),
      preserve_elements_num_groups_(
          params_t::is_outer_dim() ? group_count_rows_ : group_count_cols_),
      reduce_elements_num_groups_(params_t::is_outer_dim() ? group_count_cols_
                                                           : group_count_rows_),
      num_elems_to_preserve_(params_t::is_outer_dim() ? rows_ : cols_),
      num_elems_to_reduce_(params_t::is_outer_dim() ? cols_ : rows_) {}

/*!
 * @brief Tells the runtime whether a work item "ndItem" should execute. We
 * handle this in the kernel itself so always return true
 */
template <typename operator_t, typename params_t, typename input_t,
          typename output_t>
SYCL_BLAS_INLINE bool
Reduction<operator_t, params_t, input_t, output_t>::valid_thread(
    cl::sycl::nd_item<1> id) const {
  return true;
}

template <typename operator_t, typename params_t, typename input_t,
          typename output_t>
SYCL_BLAS_INLINE void Reduction<operator_t, params_t, input_t, output_t>::bind(
    cl::sycl::handler& h) {
  in_.bind(h);
  out_.bind(h);
}

template <typename operator_t, typename params_t, typename input_t,
          typename output_t>
SYCL_BLAS_INLINE void Reduction<operator_t, params_t, input_t,
                                output_t>::adjust_access_displacement() {
  in_.adjust_access_displacement();
  out_.adjust_access_displacement();
}

/*!
 * @brief Get the nd_range value which has to be used for kernels that
 *        intend to call Reduction::eval().
 */
template <typename operator_t, typename params_t, typename input_t,
          typename output_t>
SYCL_BLAS_INLINE cl::sycl::nd_range<1>
Reduction<operator_t, params_t, input_t, output_t>::get_nd_range(
    index_t compute_units) noexcept {
  auto get_nearest_power_of_2 = [](size_t val) -> size_t {
    val--;
    val |= val >> 1;
    val |= val >> 2;
    val |= val >> 4;
    val |= val >> 8;
    val |= val >> 16;
    // If 64-bit
    if (sizeof(void*) == 8) {
      val |= val >> 32;
    }

    return val++;
  };

  auto round_up = [](index_t x, index_t y) -> index_t {
    return ((x + y - 1) / y) * y;
  };

  constexpr index_t local_range = params_t::get_local_thread_size_preserve() *
                                  params_t::get_local_thread_size_reduce();
  const index_t round_up_p = round_up(
      num_elems_to_preserve_, params_t::get_local_thread_size_preserve());
  constexpr index_t reductions_per_thread = 64;
  const index_t preserve_num_groups =
      round_up_p / params_t::get_local_thread_size_preserve();
  index_t reduce_groups =
      (get_nearest_power_of_2(compute_units) + preserve_num_groups - 1) /
      preserve_num_groups;
  index_t reduce_num_groups =
      num_elems_to_reduce_ > reductions_per_thread * local_range
          ? std::min(reduce_groups, local_range)
          : 1;
  const index_t global_range =
      preserve_num_groups * reduce_num_groups * local_range;

  return cl::sycl::nd_range<1>(cl::sycl::range<1>(global_range),
                               cl::sycl::range<1>(local_range));
}

/*!
 * @brief Loads multiple elements from the global memory, reduces them together
 * and stores the result in accumulator
 */
template <typename operator_t, typename params_t, typename input_t,
          typename output_t>
SYCL_BLAS_INLINE void
Reduction<operator_t, params_t, input_t, output_t>::reduce(
    index_t global_reduce_id, index_t global_preserve_id,
    element_t& accumulator) noexcept {
  if (global_preserve_id >= num_elems_to_preserve_) {
    return;
  }

  index_t global_offset =
      params_t::is_outer_dim()
          ? global_preserve_id +
                (ld_mul_ * global_reduce_id * num_elems_to_preserve_)
          : global_reduce_id +
                (ld_mul_ * global_preserve_id * num_elems_to_reduce_);

  const index_t per_thread_local_stride =
      params_t::get_local_thread_size_reduce() * reduce_elements_num_groups_;
  const index_t per_thread_global_stride =
      params_t::is_outer_dim()
          ? ld_mul_ * num_elems_to_preserve_ * per_thread_local_stride
          : per_thread_local_stride;

  for (index_t i = global_reduce_id; i < num_elems_to_reduce_;
       i += per_thread_local_stride) {
    accumulator =
        operator_t::eval(accumulator, in_.template eval<true>(global_offset));
    global_offset += per_thread_global_stride;
  }
}

/*!
 * @brief The main implementation of the Reduction kernel
 */
template <typename operator_t, typename params_t, typename input_t,
          typename output_t>
template <typename local_memory_t>
SYCL_BLAS_INLINE void Reduction<operator_t, params_t, input_t, output_t>::eval(
    local_memory_t scratch, cl::sycl::nd_item<1> id) noexcept {
  const index_t local_id = id.get_local_id(0);
  const index_t group_id = id.get_group(0);
  index_t preserve_local_id =
      params_t::is_outer_dim()
          ? local_id % params_t::get_local_thread_size_preserve()
          : local_id / params_t::get_local_thread_size_preserve();
  index_t reduce_local_id =
      params_t::is_outer_dim()
          ? local_id / params_t::get_local_thread_size_preserve()
          : local_id % params_t::get_local_thread_size_preserve();
  const index_t preserve_group_id =
      params_t::is_outer_dim() ? group_id % preserve_elements_num_groups_
                               : group_id / reduce_elements_num_groups_;
  const index_t reduce_group_id = params_t::is_outer_dim()
                                      ? group_id / preserve_elements_num_groups_
                                      : group_id % reduce_elements_num_groups_;
  index_t global_preserve_id =
      preserve_group_id * params_t::get_local_thread_size_preserve() +
      preserve_local_id;
  const index_t global_reduce_id =
      reduce_group_id * params_t::get_local_thread_size_reduce() +
      reduce_local_id;
  element_t* scratch_ptr = scratch.localAcc.get_pointer();
  element_t accumulator = init_val;
  const index_t out_offset = reduce_elements_num_groups_ > 1
                                 ? reduce_group_id * num_elems_to_preserve_
                                 : 0;

  // Reduce elements from global memory
  reduce(global_reduce_id, global_preserve_id, accumulator);

  accumulator = operator_t::get_final_value(accumulator, num_elems_to_reduce_);
  const index_t scratch_idx =
      preserve_local_id +
      reduce_local_id * (params_t::get_local_thread_size_preserve() +
                         params_t::use_bank_offset());
  // Write the accumulator in local memory
  scratch_ptr[scratch_idx] = accumulator;
  if (!params_t::is_outer_dim()) {
    preserve_local_id = local_id / params_t::get_local_thread_size_preserve();
    reduce_local_id = local_id % params_t::get_local_thread_size_preserve();
    global_preserve_id =
        preserve_group_id * params_t::get_local_thread_size_preserve() +
        preserve_local_id;
  }

  element_t* out_scratch_ptr = scratch_ptr + scratch_idx;
  id.barrier(cl::sycl::access::fence_space::local_space);
  if (!params_t::is_outer_dim()) {
    accumulator = *out_scratch_ptr;
  }

// Perform reduction on the element with current local id and the corresponding
// element in the second half of tne local memory
#pragma unroll
  for (index_t offset = params_t::get_local_thread_size_reduce() >> 1;
       offset > 0; offset >>= 1) {
    if (reduce_local_id < offset) {
      accumulator = operator_t::eval(
          accumulator,
          out_scratch_ptr[(params_t::get_local_thread_size_preserve() +
                           params_t::use_bank_offset()) *
                          offset]);
      *out_scratch_ptr = accumulator;
    }

    id.barrier(cl::sycl::access::fence_space::local_space);
  }

  // Write result to the output vector
  if (reduce_local_id == 0 && (global_preserve_id < num_elems_to_preserve_)) {
    out_.template eval<true>(out_offset + global_preserve_id) = accumulator;
  }
}

template <typename operator_t, typename params_t, typename input_t,
          typename output_t>
const typename params_t::element_t
    Reduction<operator_t, params_t, input_t, output_t>::init_val =
        operator_t::template init<output_t>();

}  // namespace blas

#endif  // SYCL_BLAS_EXTENSION_REDUCTION_HPP
