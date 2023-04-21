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
 *  @filename matcopy.hpp
 *
 **************************************************************************/

#ifndef SYCL_BLAS_EXTENSION_MATCOPY_HPP
#define SYCL_BLAS_EXTENSION_MATCOPY_HPP

#include "operations/extension/matcopy.h"

namespace blas {

template <matcopy_op op, int ClSize, bool trans_rhs_1, bool trans_rhs_2,
          typename Tile, typename lhs_t, typename rhs_1_t, typename rhs_2_t>
MatCopy<op, ClSize, trans_rhs_1, trans_rhs_2, Tile, lhs_t, rhs_1_t,
        rhs_2_t>::MatCopy(lhs_t lhs, rhs_1_t rhs_1, rhs_2_t rhs_2,
                          value_t alpha, value_t beta,
                          typename rhs_2_t::index_t m,
                          typename rhs_2_t::index_t n,
                          typename rhs_2_t::index_t lhs_ld,
                          typename rhs_2_t::index_t rhs_1_ld,
                          typename rhs_2_t::index_t rhs_2_ld,
                          typename rhs_2_t::index_t lhs_stride,
                          typename rhs_2_t::index_t rhs_1_stride,
                          typename rhs_2_t::index_t rhs_2_stride)
    : lhs_(lhs),
      rhs_1_(rhs_1),
      rhs_2_(rhs_2),
      alpha_(alpha),
      beta_(beta),
      m_(m),
      n_(n),
      lhs_ld_(lhs_ld),
      rhs_1_ld_(rhs_1_ld),
      rhs_2_ld_(rhs_2_ld),
      lhs_stride_(lhs_stride),
      rhs_1_stride_(rhs_1_stride),
      rhs_2_stride_(rhs_2_stride) {}

template <matcopy_op op, int ClSize, bool trans_rhs_1, bool trans_rhs_2,
          typename Tile, typename lhs_t, typename rhs_1_t, typename rhs_2_t>
typename rhs_1_t::value_t
MatCopy<op, ClSize, trans_rhs_1, trans_rhs_2, Tile, lhs_t, rhs_1_t,
        rhs_2_t>::eval(cl::sycl::nd_item<1> ndItem) {}

template <matcopy_op op, int ClSize, bool trans_rhs_1, bool trans_rhs_2,
          typename Tile, typename lhs_t, typename rhs_1_t, typename rhs_2_t>
SYCL_BLAS_INLINE void MatCopy<op, ClSize, trans_rhs_1, trans_rhs_2, Tile, lhs_t,
                              rhs_1_t, rhs_2_t>::bind(cl::sycl::handler &h) {
  lhs_.bind(h);
  rhs_1_.bind(h);
  rhs_2_.bind(h);
}

template <matcopy_op op, int ClSize, bool trans_rhs_1, bool trans_rhs_2,
          typename Tile, typename lhs_t, typename rhs_1_t, typename rhs_2_t>
SYCL_BLAS_INLINE void MatCopy<op, ClSize, trans_rhs_1, trans_rhs_2, Tile, lhs_t,
                              rhs_1_t, rhs_2_t>::adjust_access_displacement() {
  lhs_.adjust_access_displacement();
  rhs_1_.adjust_access_displacement();
  rhs_2_.adjust_access_displacement();
}

template <matcopy_op op, int ClSize, bool trans_rhs_1, bool trans_rhs_2,
          typename Tile, typename lhs_t, typename rhs_1_t, typename rhs_2_t>
SYCL_BLAS_INLINE typename rhs_2_t::index_t
MatCopy<op, ClSize, trans_rhs_1, trans_rhs_2, Tile, lhs_t, rhs_1_t,
        rhs_2_t>::get_size() const {
  return m_ * n_;
}

template <matcopy_op op, int ClSize, bool trans_rhs_1, bool trans_rhs_2,
          typename Tile, typename lhs_t, typename rhs_1_t, typename rhs_2_t>
SYCL_BLAS_INLINE bool
MatCopy<op, ClSize, trans_rhs_1, trans_rhs_2, Tile, lhs_t, rhs_1_t,
        rhs_2_t>::valid_thread(cl::sycl::nd_item<1> ndItem) const {
  return true;
}

}  // namespace blas

#endif  // SYCL_BLAS_EXTENSION_MATCOPY_HPP