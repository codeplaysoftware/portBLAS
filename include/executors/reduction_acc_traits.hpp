/***************************************************************************
 *
 *  @license
 *  Copyright (C) 2016 Codeplay Software Limited
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
 *  @filename reduction_acc_traits.hpp
 *
 **************************************************************************/

#ifndef BLAS_REDUCTION_ACC_TRAITS_HPP
#define BLAS_REDUCTION_ACC_TRAITS_HPP

#include <evaluators/blas_functor_traits.hpp>

namespace blas {

/*! accum_functor_traits<functor_traits>
 * @brief An optimization for the first for-loop of a reduction, which helps to
 * avoid extra branching.
 */
template <typename DevFunctor> struct accum_functor_traits {
  template <typename L, typename R>
  static void acc(L &l, R r) {
    l = DevFunctor::eval(l, r);
  }
};
template <typename ScalarT, typename Device>
struct accum_functor_traits<functor_traits<addOp2_struct, ScalarT, Device>> {
  template <typename R> static inline R abs(R r) { return functor_traits<absOp1_struct, ScalarT, Device>::eval(r); }
  template <typename L, typename R>
  static void acc(L &l, R r) {
    l += r;
  }
};
template <typename ScalarT, typename Device>
struct accum_functor_traits<functor_traits<prdOp2_struct, ScalarT, Device>> {
  template <typename R> static inline R abs(R r) { return functor_traits<absOp1_struct, ScalarT, Device>::eval(r); }
  template <typename L, typename R>
  static void acc(L &l, R r) {
    l *= r;
  }
};
template <typename ScalarT, typename Device>
struct accum_functor_traits<functor_traits<addAbsOp2_struct, ScalarT, Device>> {
  template <typename R> static inline R abs(R r) { return functor_traits<absOp1_struct, ScalarT, Device>::eval(r); }
  template <typename L, typename R>
  static void acc(L &l, R r) {
    l += abs(r);
  }
};
template <typename ScalarT, typename Device>
struct accum_functor_traits<functor_traits<maxIndOp2_struct, ScalarT, Device>> {
  template <typename R> static inline R abs(R r) { return functor_traits<absOp1_struct, ScalarT, Device>::eval(r); }
  template <typename L, typename R>
  static void acc(L &l, R r) {
    if(abs(l.getVal()) < abs(r.getVal())) {
      l = r;
    }
  }
};
template <typename ScalarT, typename Device>
struct accum_functor_traits<functor_traits<minIndOp2_struct, ScalarT, Device>> {
  template <typename R> static inline R abs(R r) { return functor_traits<absOp1_struct, ScalarT, Device>::eval(r); }
  template <typename L, typename R>
  static void acc(L &l, R r) {
    if(abs(l.getVal()) > abs(r.getVal())) {
      l = r;
    }
  }
};

} // namespace blas

#endif /* end of include guard: BLAS_REDUCTION_ACC_TRAITS_HPP */
