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

#ifndef BLAS_FUNCTOR_TRAITS_HPP_7FBGTUQ8
#define BLAS_FUNCTOR_TRAITS_HPP_7FBGTUQ8

#include <iostream>
#include <stdexcept>
#include <vector>

#include <CL/sycl.hpp>

#include <executors/blas_packet_traits_sycl.hpp>
#include <operations/blas_operators.hpp>

namespace blas {

template <class Functor, typename ScalarT, typename Device>
struct functor_traits : Functor {
  static constexpr bool supported = false;
};

template <typename ScalarT, typename Device>
struct functor_traits<iniAddOp1_struct, ScalarT, Device> : iniAddOp1_struct {
  static constexpr bool supported = true;
};
template <typename ScalarT, typename Device>
struct functor_traits<iniPrdOp1_struct, ScalarT, Device> : iniPrdOp1_struct {
  static constexpr bool supported = true;
};
template <typename ScalarT, typename Device>
struct functor_traits<posOp1_struct, ScalarT, Device> : posOp1_struct {
  static constexpr bool supported = true;
};
template <typename ScalarT, typename Device>
struct functor_traits<negOp1_struct, ScalarT, Device> : negOp1_struct {
  static constexpr bool supported = Packet_traits<ScalarT, Device>::has_neg;
};
template <typename ScalarT, typename Device>
struct functor_traits<sqtOp1_struct, ScalarT, Device> : sqtOp1_struct {
  static constexpr bool supported = Packet_traits<ScalarT, Device>::has_sqrt;
  template <typename R = ScalarT>
  static R eval(R r) {
    return cl::sycl::sqrt(r);
  }
};
template <typename ScalarT, typename Device>
struct functor_traits<tupOp1_struct, ScalarT, Device> : tupOp1_struct {
  static constexpr bool supported = true;
};
template <typename ScalarT, typename Device>
struct functor_traits<addOp1_struct, ScalarT, Device> : addOp1_struct {
  static constexpr bool supported = Packet_traits<ScalarT, Device>::has_add ||
                                    Packet_traits<ScalarT, Device>::has_mul;
};
template <typename ScalarT, typename Device>
struct functor_traits<prdOp1_struct, ScalarT, Device> : prdOp1_struct {
  static constexpr bool supported = Packet_traits<ScalarT, Device>::has_mul;
};
template <typename ScalarT, typename Device>
struct functor_traits<absOp1_struct, ScalarT, Device> : absOp1_struct {
  static constexpr bool supported = Packet_traits<ScalarT, Device>::has_abs;
  template <typename R = ScalarT>
  static R eval(R r) {
    return cl::sycl::abs(r);
  }
};
template <typename ScalarT, typename Device>
struct functor_traits<addOp2_struct, ScalarT, Device> : addOp2_struct {
  static constexpr bool supported = Packet_traits<ScalarT, Device>::has_add;
};
template <typename ScalarT, typename Device>
struct functor_traits<prdOp2_struct, ScalarT, Device> : prdOp2_struct {
  static constexpr bool supported = Packet_traits<ScalarT, Device>::has_mul;
};
template <typename ScalarT, typename Device>
struct functor_traits<divOp2_struct, ScalarT, Device> : divOp2_struct {
  static constexpr bool supported = Packet_traits<ScalarT, Device>::has_div;
};
template <typename ScalarT, typename Device>
struct functor_traits<maxOp2_struct, ScalarT, Device> : maxOp2_struct {
  static constexpr bool supported = Packet_traits<ScalarT, Device>::has_max;
  template <typename L = ScalarT, typename R = ScalarT>
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
  static constexpr bool supported = Packet_traits<ScalarT, Device>::has_min;
  template <typename L = ScalarT, typename R = ScalarT>
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
  static constexpr bool supported = Packet_traits<ScalarT, Device>::has_abs &&
                                    Packet_traits<ScalarT, Device>::has_add;
  template <typename L = ScalarT, typename R = ScalarT>
  static R eval(L l, R r) {
    return functor_traits<addOp2_struct, ScalarT, Device>::eval(
        functor_traits<absOp1_struct, L, Device>::eval(l),
        functor_traits<absOp1_struct, R, Device>::eval(r));
  }
};

}  // namespace blas

#endif
