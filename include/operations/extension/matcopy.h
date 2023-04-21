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
 *  @filename matcopy.h
 *
 **************************************************************************/

#ifndef SYCL_BLAS_EXTENSION_MATCOPY_H
#define SYCL_BLAS_EXTENSION_MATCOPY_H

#include <CL/sycl.hpp>

#include "container/sycl_iterator.h"

namespace blas {

enum class matcopy_op : int { inplace = 0, outplace = 1, outplaceadd = 2 };

template <matcopy_op op, int ClSize, bool trans_rhs_1, bool trans_rhs_2,
          typename Tile, typename lhs_t, typename rhs_1_t, typename rhs_2_t>
struct MatCopy {
  using value_t = typename rhs_1_t::value_t;
  using index_t = typename rhs_2_t::index_t;

  lhs_t lhs_;
  rhs_1_t rhs_1_;
  rhs_2_t rhs_2_;
  value_t alpha_, beta_;
  index_t m_, n_, lhs_ld_, rhs_1_ld_, rhs_2_ld_, lhs_stride_, rhs_1_stride_,
      rhs_2_stride_;

  MatCopy(lhs_t lhs, rhs_1_t rhs_1, rhs_2_t rhs_2, value_t alpha, value_t beta,
          index_t m, index_t n, index_t lhs_ld, index_t rhs_1_ld,
          index_t rhs_2_ld, index_t lhs_stride, index_t rhs_1_stride,
          index_t rhs_2_stride);
  index_t get_size() const;
  bool valid_thread(cl::sycl::nd_item<1> ndItem) const;
  value_t eval(index_t i);
  value_t eval(cl::sycl::nd_item<1> ndItem);
  template <typename sharedT>
  value_t eval(sharedT shMem, cl::sycl::nd_item<1> ndItem);
  void bind(cl::sycl::handler &h);
  void adjust_access_displacement();
};

}  // namespace blas

#endif  // SYCL_BLAS_EXTENSION_MATCOPY_H
