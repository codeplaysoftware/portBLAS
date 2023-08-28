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
 *  @filename asum.hpp
 *
 **************************************************************************/

#ifndef ASUM_HPP
#define ASUM_HPP
#include "operations/blas1_trees.h"
#include "operations/blas_operators.hpp"
#include "views/view_sycl.hpp"

namespace blas {

/*! Asum.
 * @brief Implements the reduction operation for assignments
 * (in the form y = x) with y a scalar and x a subexpression tree.
 */
template <typename lhs_t, typename rhs_t>
Asum<lhs_t, rhs_t>::Asum(lhs_t &_l, rhs_t &_r) : lhs_(_l), rhs_(_r){};

template <typename lhs_t, typename rhs_t>
PORTBLAS_INLINE typename Asum<lhs_t, rhs_t>::index_t
Asum<lhs_t, rhs_t>::get_size() const {
  return rhs_.get_size();
}

template <typename lhs_t, typename rhs_t>
PORTBLAS_INLINE bool Asum<lhs_t, rhs_t>::valid_thread(
    cl::sycl::nd_item<1> ndItem) const {
  return true;
}

template <typename lhs_t, typename rhs_t>
PORTBLAS_INLINE typename Asum<lhs_t, rhs_t>::value_t Asum<lhs_t, rhs_t>::eval(
    cl::sycl::nd_item<1> ndItem) {
  auto atomic_res = sycl::atomic_ref<value_t, sycl::memory_order::relaxed,
                                     sycl::memory_scope::device,
                                     sycl::access::address_space::global_space>(
      lhs_.get_data()[0]);
  const auto size = rhs_.get_size();
  int lid = ndItem.get_global_linear_id();
  value_t in_val{0};

  // First loop for big arrays
  for (int id = lid; id < size;
       id += ndItem.get_local_range()[0] * ndItem.get_group_range()[0]) {
    in_val += sycl::abs(rhs_.eval(id));
  }

  in_val =
      sycl::reduce_over_group(ndItem.get_sub_group(), in_val, sycl::plus<>());

  if ((ndItem.get_local_id() &
       (ndItem.get_sub_group().get_local_range() - 1)) == 0) {
    atomic_res += in_val;
  }
  return {};
}
template <typename lhs_t, typename rhs_t>
template <typename sharedT>
PORTBLAS_INLINE typename Asum<lhs_t, rhs_t>::value_t Asum<lhs_t, rhs_t>::eval(
    sharedT scratch, cl::sycl::nd_item<1> ndItem) {
  auto atomic_res = sycl::atomic_ref<value_t, sycl::memory_order::relaxed,
                                     sycl::memory_scope::device,
                                     sycl::access::address_space::global_space>(
      lhs_.get_data()[0]);
  const auto size = rhs_.get_size();
  const int lid = static_cast<int>(ndItem.get_global_linear_id());
  const auto loop_stride =
      ndItem.get_local_range()[0] * ndItem.get_group_range()[0];
  value_t in_val{0};

  // First loop for big arrays
  for (int id = lid; id < size; id += loop_stride) {
    in_val += sycl::abs(rhs_.eval(id));
  }

  in_val =
      sycl::reduce_over_group(ndItem.get_sub_group(), in_val, sycl::plus<>());

  if (ndItem.get_sub_group().get_local_id() == 0) {
    scratch[ndItem.get_sub_group().get_group_linear_id()] = in_val;
  }
  ndItem.barrier();

  in_val =
      (ndItem.get_local_id() < (ndItem.get_local_range()[0] /
                                ndItem.get_sub_group().get_local_range()[0]))
          ? scratch[ndItem.get_sub_group().get_local_id()]
          : 0;
  if (ndItem.get_sub_group().get_group_id() == 0) {
    in_val =
        sycl::reduce_over_group(ndItem.get_sub_group(), in_val, sycl::plus<>());
  }
  if (ndItem.get_local_id() == 0) {
    atomic_res += in_val;
  }

  return {};
}

template <typename lhs_t, typename rhs_t>
PORTBLAS_INLINE void Asum<lhs_t, rhs_t>::bind(cl::sycl::handler &h) {
  lhs_.bind(h);
  rhs_.bind(h);
}

template <typename lhs_t, typename rhs_t>
PORTBLAS_INLINE void Asum<lhs_t, rhs_t>::adjust_access_displacement() {
  lhs_.adjust_access_displacement();
  rhs_.adjust_access_displacement();
}
}  // namespace blas

#endif
