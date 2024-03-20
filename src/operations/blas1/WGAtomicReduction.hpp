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
 *  @filename WGAtomicReduction.hpp
 *
 **************************************************************************/

#ifndef WG_ATOMIC_REDUCTION_HPP
#define WG_ATOMIC_REDUCTION_HPP
#include "operations/blas1_trees.h"
#include "operations/blas_operators.hpp"

namespace blas {

/*! WGAtomicReduction.
 * @brief This class implement a device size reduction using all WG to compute
 * and atomics operation to combine the results.
 *
 * */
template <typename operator_t, bool managed_mem, typename lhs_t, typename rhs_t>
WGAtomicReduction<operator_t, managed_mem, lhs_t, rhs_t>::WGAtomicReduction(
    lhs_t& _l, rhs_t& _r)
    : lhs_(_l), rhs_(_r){};

template <typename operator_t, bool managed_mem, typename lhs_t, typename rhs_t>
PORTBLAS_INLINE
    typename WGAtomicReduction<operator_t, managed_mem, lhs_t, rhs_t>::index_t
    WGAtomicReduction<operator_t, managed_mem, lhs_t, rhs_t>::get_size() const {
  return rhs_.get_size();
}

template <typename operator_t, bool managed_mem, typename lhs_t, typename rhs_t>
PORTBLAS_INLINE bool
WGAtomicReduction<operator_t, managed_mem, lhs_t, rhs_t>::valid_thread(
    cl::sycl::nd_item<1> ndItem) const {
  return true;
}

template <typename operator_t, bool managed_mem, typename lhs_t, typename rhs_t>
PORTBLAS_INLINE
    typename WGAtomicReduction<operator_t, managed_mem, lhs_t, rhs_t>::value_t
    WGAtomicReduction<operator_t, managed_mem, lhs_t, rhs_t>::eval(
        cl::sycl::nd_item<1> ndItem) {
  auto atomic_res =
      cl::sycl::atomic_ref<value_t, cl::sycl::memory_order::relaxed,
                           cl::sycl::memory_scope::device,
                           cl::sycl::access::address_space::global_space>(
          lhs_.get_data()[0]);
  const auto size = get_size();
  int lid = ndItem.get_global_linear_id();
  value_t val = operator_t::template init<rhs_t>();
  const auto loop_stride =
      ndItem.get_local_range(0) * ndItem.get_group_range(0);

  // First loop for big arrays
  for (int id = lid; id < size; id += loop_stride) {
    val = operator_t::eval(val, rhs_.eval(id));
  }

  val = cl::sycl::reduce_over_group(ndItem.get_sub_group(), val,
                                    cl::sycl::plus<value_t>());

  if ((ndItem.get_local_id()[0] &
       (ndItem.get_sub_group().get_local_range()[0] - 1)) == 0) {
    atomic_res += val;
  }
  return {};
}

template <typename operator_t, bool managed_mem, typename lhs_t, typename rhs_t>
template <typename sharedT>
PORTBLAS_INLINE
    typename WGAtomicReduction<operator_t, managed_mem, lhs_t, rhs_t>::value_t
    WGAtomicReduction<operator_t, managed_mem, lhs_t, rhs_t>::eval(
        sharedT scratch, cl::sycl::nd_item<1> ndItem) {
  const auto size = get_size();
  const int lid = static_cast<int>(ndItem.get_global_linear_id());
  const auto loop_stride =
      ndItem.get_local_range(0) * ndItem.get_group_range(0);
  value_t val = operator_t::template init<rhs_t>();

  // First loop for big arrays
  for (int id = lid; id < size; id += loop_stride) {
    val = operator_t::eval(val, rhs_.eval(id));
  }

  val = cl::sycl::reduce_over_group(ndItem.get_sub_group(), val,
                                    cl::sycl::plus<value_t>());

  if (ndItem.get_sub_group().get_local_id()[0] == 0) {
    scratch[ndItem.get_sub_group().get_group_linear_id()] = val;
  }
  ndItem.barrier();

  val =
      (ndItem.get_local_id()[0] < (ndItem.get_local_range(0) /
                                   ndItem.get_sub_group().get_local_range()[0]))
          ? scratch[ndItem.get_sub_group().get_local_id()]
          : 0;
  if (ndItem.get_sub_group().get_group_id()[0] == 0) {
    val = cl::sycl::reduce_over_group(ndItem.get_sub_group(), val,
                                      cl::sycl::plus<value_t>());
  }
  if (ndItem.get_local_id()[0] == 0) {
    if constexpr (!managed_mem) {
      auto atomic_res =
          cl::sycl::atomic_ref<value_t, cl::sycl::memory_order::relaxed,
                               cl::sycl::memory_scope::device,
                               cl::sycl::access::address_space::global_space>(
              lhs_.get_data()[0]);
      atomic_res += val;
    } else {
      auto atomic_res =
          cl::sycl::atomic_ref<value_t, cl::sycl::memory_order::relaxed,
                               cl::sycl::memory_scope::device,
                               cl::sycl::access::address_space::generic_space>(
              lhs_.get_data()[0]);
      atomic_res += val;
    }
  }

  return {};
}

template <typename operator_t, bool managed_mem, typename lhs_t, typename rhs_t>
PORTBLAS_INLINE void WGAtomicReduction<operator_t, managed_mem, lhs_t,
                                       rhs_t>::bind(cl::sycl::handler& h) {
  lhs_.bind(h);
  rhs_.bind(h);
}

template <typename operator_t, bool managed_mem, typename lhs_t, typename rhs_t>
PORTBLAS_INLINE void WGAtomicReduction<operator_t, managed_mem, lhs_t,
                                       rhs_t>::adjust_access_displacement() {
  lhs_.adjust_access_displacement();
  rhs_.adjust_access_displacement();
}
}  // namespace blas

#endif
