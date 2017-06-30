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
 *  @filename blas1_interface_sycl_unverified.hpp
 *
 **************************************************************************/

#ifndef BLAS1_INTERFACE_SYCL_UNVERIFIED_HPP
#define BLAS1_INTERFACE_SYCL_UNVERIFIED_HPP

#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>

#include <executors/executor_sycl.hpp>
#include <operations/blas1_trees.hpp>

namespace blas {

// UNVERIFIED ROUTINE
#ifdef BLAS_EXPERIMENTAL
template <typename T>
void _rotmg(T &_d1, T &_d2, T &_x1, T &_y1, VectorSYCL<T> _param) {
  T flag, h11, h12, h21, h22;
  T p1, p2, q1, q2, temp, su;
  T gam = 4096, gamsq = 16777216, rgamsq = 5.9604645e-8;

  if (_d1 < constant<T, const_val::one>::value) {
    // GO ZERO-H-D-AND-_X1..
    flag = constant<T, const_val::m_one>::value;
    h11 = constant<T, const_val::zero>::value;
    h12 = constant<T, const_val::zero>::value;
    h21 = constant<T, const_val::zero>::value;
    h22 = constant<T, const_val::zero>::value;
    _d1 = constant<T, const_val::zero>::value;
    _d2 = constant<T, const_val::zero>::value;
    _x1 = constant<T, const_val::zero>::value;
  } else {
    // CASE-SD1-NONNEGATIVE
    p2 = _d2 * _y1;
    if (p2 == constant<T, const_val::zero>::value) {
      flag = constant<T, const_val::m_two>::value;
      _param.eval(0) = flag;
      return;
    }
    // REGULAR-CASE..
    p1 = _d1 * _x1;
    q2 = p2 * _y1;
    q1 = p1 * _x1;
    if (std::abs(q1) > std::abs(q2)) {
      h21 = -_y1 / _x1;
      h12 = p2 / p1;
      su = constant<T, const_val::one>::value - (h12 * h21);
      if (su > constant<T, const_val::zero>::value) {
        flag = constant<T, const_val::zero>::value;
        _d1 = _d1 / su;
        _d2 = _d2 / su;
        _x1 = _x1 / su;
      }
    } else {
      if (q2 < constant<T, const_val::zero>::value) {
        // GO zero-H-D-AND-_X1..
        flag = constant<T, const_val::m_one>::value;
        h11 = constant<T, const_val::zero>::value;
        h12 = constant<T, const_val::zero>::value;
        h21 = constant<T, const_val::zero>::value;
        h22 = constant<T, const_val::zero>::value;
        _d1 = constant<T, const_val::zero>::value;
        _d2 = constant<T, const_val::zero>::value;
        _x1 = constant<T, const_val::zero>::value;

      } else {
        flag = constant<T, const_val::one>::value;
        h11 = p1 / p2;
        h22 = _x1 / _y1;
        su = constant<T, const_val::one>::value + (h11 * h22);
        temp = _d2 / su;
        _d2 = _d1 / su;
        _d1 = temp;
        _x1 = _y1 * su;
        h12 = constant<T, const_val::zero>::value;
        h21 = constant<T, const_val::zero>::value;
        _d1 = constant<T, const_val::zero>::value;
        _d2 = constant<T, const_val::zero>::value;
        _x1 = constant<T, const_val::zero>::value;
      }
    }
    // PROCEDURE..SCALE-CHECK
    if (_d1 != constant<T, const_val::zero>::value) {
      while ((_d1 < rgamsq) || (_d1 >= gamsq)) {
        if (flag == constant<T, const_val::zero>::value) {
          h11 = constant<T, const_val::one>::value;
          h22 = constant<T, const_val::one>::value;
          flag = constant<T, const_val::m_one>::value;
        } else {
          h21 = constant<T, const_val::m_one>::value;
          h12 = constant<T, const_val::one>::value;
          flag = constant<T, const_val::m_one>::value;
        }
        if (_d1 <= rgamsq) {
          _d1 *= gam * gam;
          _x1 /= gam;
          h11 /= gam;
          h12 /= gam;
        } else {
          _d1 /= gam * gam;
          _x1 *= gam;
          h11 *= gam;
          h12 *= gam;
        }
      }
    }
    if (_d2 != constant<T, const_val::zero>::value) {
      while ((_d2 < rgamsq) || (std::abs(_d2) >= gamsq)) {
        if (flag == constant<T, const_val::zero>::value) {
          h11 = constant<T, const_val::one>::value;
          h22 = constant<T, const_val::one>::value;
          flag = constant<T, const_val::m_one>::value;
        } else {
          h21 = constant<T, const_val::m_one>::value;
          h12 = constant<T, const_val::one>::value;
          flag = constant<T, const_val::m_one>::value;
        }
        if (std::abs(_d2) <= rgamsq) {
          _d2 *= gam * gam;
          h21 /= gam;
          h22 /= gam;
        } else {
          _d2 /= gam * gam;
          h21 *= gam;
          h22 *= gam;
        }
      }
    }
  }
  if (flag < constant<T, const_val::zero>::value) {
    _param.eval(1) = h11;
    _param.eval(2) = h21;
    _param.eval(3) = h12;
    _param.eval(4) = h22;
  } else if (flag == constant<T, const_val::zero>::value) {
    _param.eval(2) = h21;
    _param.eval(3) = h12;
  } else {
    _param.eval(1) = h11;
    _param.eval(4) = h22;
  }
  _param.eval(0) = flag;
}
#endif  // BLAS_EXPERIMENTAL
}  // namespace blas

#endif  // BLAS1_INTERFACE_SYCL_UNVERIFIED_HPP
