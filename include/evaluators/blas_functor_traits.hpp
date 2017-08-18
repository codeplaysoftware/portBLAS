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
 *  @filename blas_functor_traits.hpp
 *
 **************************************************************************/

#ifndef BLAS_FUNCTOR_TRAITS_HPP
#define BLAS_FUNCTOR_TRAITS_HPP

#include <iostream>
#include <stdexcept>
#include <vector>

#include <CL/sycl.hpp>

#include <operations/blas_operators.hpp>
#include <executors/blas_device_sycl.hpp>

namespace blas {

/*!
 * functor_traits.
 * @brief Changes the functor if it has a different implementation on the
 * device.
 */
template <class Functor, typename ScalarT, typename Device>
struct functor_traits : Functor {
};

template <typename ScalarT, typename Device>
struct functor_traits<iniAddOp1_struct, ScalarT, Device> : iniAddOp1_struct {
};
template <typename ScalarT, typename Device>
struct functor_traits<iniPrdOp1_struct, ScalarT, Device> : iniPrdOp1_struct {
};
template <typename ScalarT, typename Device>
struct functor_traits<posOp1_struct, ScalarT, Device> : posOp1_struct {
};
template <typename ScalarT, typename Device>
struct functor_traits<negOp1_struct, ScalarT, Device> : negOp1_struct {
};
template <typename ScalarT, typename Device>
struct functor_traits<sqtOp1_struct, ScalarT, Device> : sqtOp1_struct {
  template <typename R>
  static R eval(R r) {
    return cl::sycl::sqrt(r);
  }
};
template <typename ScalarT, typename Device>
struct functor_traits<tupOp1_struct, ScalarT, Device> : tupOp1_struct {
};
template <typename ScalarT, typename Device>
struct functor_traits<addOp1_struct, ScalarT, Device> : addOp1_struct {
};
template <typename ScalarT, typename Device>
struct functor_traits<prdOp1_struct, ScalarT, Device> : prdOp1_struct {
};
template <typename ScalarT, typename Device>
struct functor_traits<absOp1_struct, ScalarT, Device> : absOp1_struct {
  template <typename R>
  static R eval(R r) {
    return cl::sycl::fabs(r);
  }
};
template <typename ScalarT, typename Device>
struct functor_traits<addOp2_struct, ScalarT, Device> : addOp2_struct {
};
template <typename ScalarT, typename Device>
struct functor_traits<prdOp2_struct, ScalarT, Device> : prdOp2_struct {
};
template <typename ScalarT, typename Device>
struct functor_traits<divOp2_struct, ScalarT, Device> : divOp2_struct {
};
template <typename ScalarT, typename Device>
struct functor_traits<maxOp2_struct, ScalarT, Device> : maxOp2_struct {
  template <typename L, typename R>
  static R eval(L l, R r) {
    return ((functor_traits<absOp1_struct, L, Device>::eval(l.getVal()) <
             functor_traits<absOp1_struct, R, Device>::eval(r.getVal())) ||
            (functor_traits<absOp1_struct, L, Device>::eval(l.getVal()) ==
                 functor_traits<absOp1_struct, R, Device>::eval(r.getVal()) &&
             l.getInd() > r.getInd()))
               ? r
               : l;
  }
};
template <typename ScalarT, typename Device>
struct functor_traits<minOp2_struct, ScalarT, Device> : minOp2_struct {
  template <typename L, typename R>
  static R eval(L l, R r) {
    return ((functor_traits<absOp1_struct, L, Device>::eval(l.getVal()) >
             functor_traits<absOp1_struct, R, Device>::eval(r.getVal())) ||
            (functor_traits<absOp1_struct, L, Device>::eval(l.getVal()) ==
                 functor_traits<absOp1_struct, R, Device>::eval(r.getVal()) &&
             l.getInd() > r.getInd()))
               ? r
               : l;
  }
};
template <typename ScalarT, typename Device>
struct functor_traits<addAbsOp2_struct, ScalarT, Device> : addAbsOp2_struct {
  template <typename L, typename R>
  static R eval(L l, R r) {
    return functor_traits<addOp2_struct, ScalarT, Device>::eval(
        functor_traits<absOp1_struct, L, Device>::eval(l),
        functor_traits<absOp1_struct, R, Device>::eval(r));
  }
};

}  // namespace blas

#endif
