/***************************************************************************
 *
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
 *  @filename IndexMaxMin.hpp
 *
 **************************************************************************/

#ifndef INDEX_MAX_MIN_HPP
#define INDEX_MAX_MIN_HPP
#include "operations/blas1_trees.h"
#include "operations/blas_operators.hpp"

namespace blas {

/**
 * Temporary class to select the type of blas::Operator to use.
 * @tparam max Indicate whether the desired operation is iamax or not.
 */
template <bool max>
struct SelectOperator;

template <>
struct SelectOperator<true> {
  using op = IMaxOperator;
};

template <>
struct SelectOperator<false> {
  using op = IMinOperator;
};

/*! IndexMaxMin.
 * @brief Generic implementation for operators that require a
 * reduction inside kernel code for computing index of max/min
 * value within the input (i.e. iamax and iamin).
 * */
template <bool is_max, bool is_step0, typename lhs_t, typename rhs_t>
IndexMaxMin<is_max, is_step0, lhs_t, rhs_t>::IndexMaxMin(lhs_t& _l, rhs_t& _r)
    : lhs_(_l), rhs_(_r){};

template <bool is_max, bool is_step0, typename lhs_t, typename rhs_t>
PORTBLAS_INLINE typename IndexMaxMin<is_max, is_step0, lhs_t, rhs_t>::index_t
IndexMaxMin<is_max, is_step0, lhs_t, rhs_t>::get_size() const {
  return rhs_.get_size();
}

template <bool is_max, bool is_step0, typename lhs_t, typename rhs_t>
PORTBLAS_INLINE bool IndexMaxMin<is_max, is_step0, lhs_t, rhs_t>::valid_thread(
    cl::sycl::nd_item<1> ndItem) const {
  return true;
}

/**
 * eval() function without local memory.
 */
template <bool is_max, bool is_step0, typename lhs_t, typename rhs_t>
PORTBLAS_INLINE void IndexMaxMin<is_max, is_step0, lhs_t, rhs_t>::eval(
    cl::sycl::nd_item<1> ndItem) {
  using op = typename SelectOperator<is_max>::op;
  const auto size = rhs_.get_size();
  auto sg = ndItem.get_sub_group();
  const auto local_range = ndItem.get_local_range(0);
  int lid = ndItem.get_global_linear_id();
  const index_t local_id = ndItem.get_local_id();
  value_t val = op::template init<rhs_t>();
  const auto loop_stride = local_range * ndItem.get_group_range(0);

  // First loop for big arrays
  for (int id = lid; id < size; id += loop_stride) {
    val = op::eval(val, rhs_.eval(id));
  }

  const index_t sg_local_id = sg.get_local_id();
  const index_t sg_local_range = sg.get_local_range()[0];

  using element_t =
      typename ResolveReturnType<op, rhs_t>::type::value_t::value_t;

  // reduction within the sub_group
  for (index_t i = sg_local_range >> 1; i > 0; i >>= 1) {
    element_t shfl_val = cl::sycl::shift_group_left(sg, val.get_value(), i);
    index_t shfl_idx = cl::sycl::shift_group_left(sg, val.get_index(), i);
    value_t shfl{shfl_idx, shfl_val};
    val = op::eval(val, shfl);
  }

  const index_t lhs_idx =
      ndItem.get_group_linear_id() * (local_range / sg_local_range) +
      sg.get_group_linear_id();

  // write IndexValueTuple to Global Memory iff reduction step0
  // or write Index to Global Memory iff reduction step1.
  // only 1 work item per sub_group performs this operation.
  if (sg_local_id == 0) {
    if constexpr (is_step0) {
      lhs_.eval(lhs_idx) = val;
    } else {
      lhs_.eval(lhs_idx) = val.get_index();
    }
  }

  return;
}

/**
 * eval() function with local memory.
 */
template <bool is_max, bool is_step0, typename lhs_t, typename rhs_t>
template <typename sharedT>
PORTBLAS_INLINE void IndexMaxMin<is_max, is_step0, lhs_t, rhs_t>::eval(
    sharedT scratch, cl::sycl::nd_item<1> ndItem) {
  using op = typename SelectOperator<is_max>::op;
  const auto size = rhs_.get_size();
  const auto local_range = ndItem.get_local_range(0);
  int lid = ndItem.get_global_linear_id();
  const auto group_id = ndItem.get_group(0);
  const index_t local_id = ndItem.get_local_id();
  value_t val = op::template init<rhs_t>();
  const auto loop_stride = local_range * ndItem.get_group_range(0);

  // First loop for big arrays
  for (int id = lid; id < size; id += loop_stride) {
    val = op::eval(val, rhs_.eval(id));
  }

  scratch[local_id] = val;
  ndItem.barrier(cl::sycl::access::fence_space::local_space);

  value_t local_val = op::template init<rhs_t>();
  // reduction within the work group
  for (index_t i = local_range >> 1; i > 0; i >>= 1) {
    if (local_id < i) {
      val = scratch[local_id];
      local_val = scratch[local_id + i];
      scratch[local_id] = op::eval(val, local_val);
    }
    ndItem.barrier(cl::sycl::access::fence_space::local_space);
  }

  // write IndexValueTuple to Global Memory iff reduction step0
  // or write Index to Global Memory iff reduction step1.
  // only 1 work item per work group performs this operation.
  if (local_id == 0) {
    val = scratch[local_id];
    if constexpr (is_step0) {
      lhs_.eval(group_id) = val;
    } else {
      lhs_.eval(group_id) = val.get_index();
    }
  }

  return;
}

template <bool is_max, bool is_step0, typename lhs_t, typename rhs_t>
PORTBLAS_INLINE void IndexMaxMin<is_max, is_step0, lhs_t, rhs_t>::bind(
    cl::sycl::handler& h) {
  lhs_.bind(h);
  rhs_.bind(h);
}

template <bool is_max, bool is_step0, typename lhs_t, typename rhs_t>
PORTBLAS_INLINE void
IndexMaxMin<is_max, is_step0, lhs_t, rhs_t>::adjust_access_displacement() {
  lhs_.adjust_access_displacement();
  rhs_.adjust_access_displacement();
}
}  // namespace blas

#endif
