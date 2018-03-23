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
 *  @filename blas1_interface_sycl.hpp
 *
 **************************************************************************/

#ifndef BLAS1_INTERFACE_SYCL_HPP
#define BLAS1_INTERFACE_SYCL_HPP

#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>

#include <executors/executor_sycl.hpp>
#include <operations/blas1_trees.hpp>

namespace blas {
/**
 * \brief AXPY constant times a vector plus a vector.
 *
 * Implements AXPY \f$y = ax + y\f$
 *
 * @param Executor<ExecutorType> ex
 * @param _vx  VectorView
 * @param _incx Increment in X axis
 * @param _vy  VectorView
 * @param _incy Increment in Y axis
 */
template <typename ExecutorType, typename T>
void _axpy(Executor<ExecutorType> &ex, int _N, T _alpha, T *_vx, int _incx,
           T *_vy, int _incy) {
  using RHS =
      vector_view<T, typename Executor<ExecutorType>::template ContainerT<T>>;
  auto vx_container = ex.get_buffer(_vx);
  RHS my_vx(vx_container, ex.get_offset(_vx), _incx, _N);
  auto vy_container = ex.get_buffer(_vy);
  RHS my_vy(vy_container, ex.get_offset(_vy), _incy, _N);
#ifdef VERBOSE
  std::cout << "alpha = " << _alpha << std::endl;
  my_vx.printH("VX");
  my_vy.printH("VY");
#endif  //  VERBOSE
  auto scalOp = make_op<ScalarOp, prdOp2_struct>(_alpha, my_vx);
  auto addOp = make_op<BinaryOp, addOp2_struct>(my_vy, scalOp);
  auto assignOp = make_op<Assign>(my_vy, addOp);
  ex.execute(assignOp);
#ifdef VERBOSE
  my_vy.printH("VY");
#endif  //  VERBOSE
}

/**
 * \brief COPY copies a vector, x, to a vector, y.
 *
 * @param Executor<ExecutorType> ex
 * @param _vx  VectorView
 * @param _incx Increment in X axis
 * @param _vy  VectorView
 * @param _incy Increment in Y axis
 */
template <typename ExecutorType, typename T>
void _copy(Executor<ExecutorType> &ex, int _N, T *_vx, int _incx, T *_vy,
           int _incy) {
  using RHS =
      vector_view<T, typename Executor<ExecutorType>::template ContainerT<T>>;
  auto vx_container = ex.get_buffer(_vx);
  RHS my_vx(vx_container, ex.get_offset(_vx), _incx, _N);
  auto vy_container = ex.get_buffer(_vy);
  RHS my_vy(vy_container, ex.get_offset(_vy), _incy, _N);

#ifdef VERBOSE
  my_vx.printH("VX");
  my_vy.printH("VY");
#endif  //  VERBOSE
  auto assignOp2 = make_op<Assign>(my_vy, my_vx);
  ex.execute(assignOp2);
#ifdef VERBOSE
  my_vx.printH("VX");
  my_vy.printH("VY");
#endif  //  VERBOSE
}

/**
 * \brief Compute the inner product of two vectors with extended precision
    accumulation.
 * @param Executor<ExecutorType> ex
 * @param _vx  VectorView
 * @param _incx Increment in X axis
 * @param _vx  VectorView
 * @param _incy Increment in Y axis
 */
template <typename ExecutorType, typename T>
void _dot(Executor<ExecutorType> &ex, int _N, T *_vx, int _incx, T *_vy,
          int _incy, T *_rs) {
  using RHS =
      vector_view<T, typename Executor<ExecutorType>::template ContainerT<T>>;
  auto vx_container = ex.get_buffer(_vx);
  RHS my_vx(vx_container, ex.get_offset(_vx), _incx, _N);
  auto vy_container = ex.get_buffer(_vy);
  RHS my_vy(vy_container, ex.get_offset(_vy), _incy, _N);
  auto rs_container = ex.get_buffer(_rs);
  RHS my_rs(rs_container, ex.get_offset(_rs), 1, 1);
#ifdef VERBOSE
  my_vx.printH("VX");
  my_vy.printH("VY");
  my_rs.printH("VR");
#endif  //  VERBOSE
  auto prdOp = make_op<BinaryOp, prdOp2_struct>(my_vx, my_vy);
  auto localSize = 256;
  auto nWG = 512;
  auto assignOp1 =
      make_addAssignReduction(my_rs, prdOp, localSize, localSize * nWG);
  ex.reduce(assignOp1);
#ifdef VERBOSE
  my_rs.printH("VR");
#endif  //  VERBOSE
}

/**
 * \briefCompute the inner product of two vectors with extended
    precision accumulation and result.
 *
 * @param Executor<ExecutorType> ex
 * @param _vx  VectorView
 * @param _incx Increment in X axis
 * @param _vx  VectorView
 * @param _incy Increment in Y axis
 */
template <typename ExecutorType, typename T>
T _dot(Executor<ExecutorType> &ex, int _N, T *_vx, int _incx, T *_vy,
       int _incy) {
#ifdef VERBOSE
  my_vx.printH("VX");
  my_vy.printH("VY");
#endif  //  VERBOSE
  auto val_ptr = ex.template allocate<T>(1);
  auto res = std::vector<T>(1);
  _dot(ex, _N, _vx, _incx, _vy, _incy, val_ptr);
  ex.copy_to_host(val_ptr, res.data(), 1);
  ex.template deallocate<T>(val_ptr);

#ifdef VERBOSE
  std::cout << "val = " << res << std::endl;
#endif  //  VERBOSE
  return res[0];
}

/**
 * \brief IAMAX finds the index of the first element having maximum
 * @param _vx  VectorView
 * @param _incx Increment in X axis
 */
template <typename ExecutorType, typename T, typename I>
void _iamax(Executor<ExecutorType> &ex, int _N, T *_vx, int _incx, I *_rs) {
  using RHS =
      vector_view<T, typename Executor<ExecutorType>::template ContainerT<T>>;
  using RHS1 =
      vector_view<I, typename Executor<ExecutorType>::template ContainerT<I>>;
  auto vx_container = ex.get_buffer(_vx);
  RHS my_vx(vx_container, ex.get_offset(_vx), _incx, _N);
  auto rs_container = ex.get_buffer(_rs);
  RHS1 my_rs(rs_container, ex.get_offset(_rs), 1, 1);
#ifdef VERBOSE
  my_vx.printH("VX");
#endif  //  VERBOSE
  size_t localSize = 256, nWG = 512;
  auto tupOp = TupleOp<RHS>(my_vx);
  auto assignOp2 =
      make_maxIndAssignReduction(my_rs, tupOp, localSize, localSize * nWG);
  ex.reduce(assignOp2);
}

/**
 * \brief ICAMAX finds the index of the first element having maximum
 * @param _vx  VectorView
 * @param _incx Increment in X axis
 */
template <typename ExecutorType, typename T>
size_t _iamax(Executor<ExecutorType> &ex, int _N, T *_vx, int _incx) {
  std::vector<IndVal<T>> rsT(1);
  auto val_ptr1 = ex.template allocate<IndVal<T>>(1);
  _iamax(ex, _N, _vx, _incx, val_ptr1);
  ex.copy_to_host(val_ptr1, rsT.data(), 1);
  ex.template deallocate<IndVal<T>>(val_ptr1);
  return rsT[0].getInd();
}

/**
 * \brief IAMIN finds the index of the first element having minimum
 * @param _vx  VectorView
 * @param _incx Increment in X axis
 */
template <typename ExecutorType, typename T, typename I>
void _iamin(Executor<ExecutorType> &ex, int _N, T *_vx, int _incx, I *_rs) {
  using RHS =
      vector_view<T, typename Executor<ExecutorType>::template ContainerT<T>>;
  using RHS1 =
      vector_view<I, typename Executor<ExecutorType>::template ContainerT<I>>;
  auto vx_container = ex.get_buffer(_vx);
  RHS my_vx(vx_container, ex.get_offset(_vx), _incx, _N);
  auto rs_container = ex.get_buffer(_rs);
  RHS1 my_rs(rs_container, ex.get_offset(_rs), 1, 1);

#ifdef VERBOSE
  my_vx.printH("VX");
#endif  //  VERBOSE
  size_t localSize = 256, nWG = 512;
  auto tupOp = TupleOp<RHS>(my_vx);
  auto assignOp2 =
      make_minIndAssignReduction(my_rs, tupOp, localSize, localSize * nWG);
  ex.reduce(assignOp2);
}

/**
 * \brief ICAMIN finds the index of the first element having minimum
 * @param _vx  VectorView
 * @param _incx Increment in X axis
 */
template <typename ExecutorType, typename T>
size_t _iamin(Executor<ExecutorType> &ex, int _N, T *_vx, int _incx) {
  std::vector<IndVal<T>> rsT(1);
  auto val_ptr1 = ex.template allocate<IndVal<T>>(1);
  _iamin(ex, _N, _vx, _incx, val_ptr1);
  ex.copy_to_host(val_ptr1, rsT.data(), 1);
  ex.template deallocate<IndVal<T>>(val_ptr1);
  return rsT[0].getInd();
}

/**
 * \brief SWAP interchanges two vectors
 *
 * @param Executor<ExecutorType> ex
 * @param _vx  VectorView
 * @param _incx Increment in X axis
 * @param _vy  VectorView
 * @param _incy Increment in Y axis
 */
template <typename ExecutorType, typename T>
void _swap(Executor<ExecutorType> &ex, int _N, T *_vx, int _incx, T *_vy,
           int _incy) {
  using RHS =
      vector_view<T, typename Executor<ExecutorType>::template ContainerT<T>>;
  auto vx_container = ex.get_buffer(_vx);
  RHS my_vx(vx_container, ex.get_offset(_vx), _incx, _N);
  auto vy_container = ex.get_buffer(_vy);
  RHS my_vy(vy_container, ex.get_offset(_vy), _incy, _N);
#ifdef VERBOSE
  my_vx.printH("VX");
  my_vy.printH("VY");
#endif  //  VERBOSE
  auto swapOp = make_op<DobleAssign>(my_vy, my_vx, my_vx, my_vy);
  ex.execute(swapOp);
#ifdef VERBOSE
  my_vx.printH("VX");
  my_vy.printH("VY");
#endif  //  VERBOSE
}

template <typename ExecutorType, typename T>
void _scal(Executor<ExecutorType> &ex, int _N, T _alpha, T *_vx, int _incx) {
  using RHS =
      vector_view<T, typename Executor<ExecutorType>::template ContainerT<T>>;
  auto vx_container = ex.get_buffer(_vx);
  RHS my_vx(vx_container, ex.get_offset(_vx), _incx, _N);
#ifdef VERBOSE
  std::cout << "alpha = " << _alpha << std::endl;
  my_vx.printH("VX");
#endif  //  VERBOSE
  auto scalOp = make_op<ScalarOp, prdOp2_struct>(_alpha, my_vx);
  auto assignOp = make_op<Assign>(my_vx, scalOp);
  ex.execute(assignOp);
#ifdef VERBOSE
  my_vx.printH("VX");
#endif  //  VERBOSE
}

/**
 * \brief NRM2 Returns the euclidian norm of a vector
 *
 * @param Executor<ExecutorType> ex
 * @param _vx  VectorView
 * @param _incx Increment in X axis
 */
template <typename ExecutorType, typename T>
T _nrm2(Executor<ExecutorType> &ex, int _N, T *_vx, int _incx) {
  std::vector<T> rs(1, T(0));
  auto _rs = ex.template allocate<T>(1);
  auto rs_container = ex.get_buffer(_rs);
  _nrm2(ex, _N, _vx, _incx, _rs);
  ex.copy_to_host(_rs, rs.data(), 1);
  ex.template deallocate<T>(_rs);
#ifdef VERBOSE
  my_vx.printH("VX");
#endif  //  VERBOSE

#ifdef VERBOSE
//  std::cout << "val = " << std::sqrt(val2.eval(0)) << std::endl;
#endif  //  VERBOSE
  return rs[0];
}

/**
 * \brief NRM2 Returns the euclidian norm of a vector
 * @param Executor<ExecutorType> ex
 * @param _vx  VectorView
 * @param _incx Increment in X axis
 */
template <typename ExecutorType, typename T>
void _nrm2(Executor<ExecutorType> &ex, int _N, T *_vx, int _incx, T *_rs) {
  using RHS =
      vector_view<T, typename Executor<ExecutorType>::template ContainerT<T>>;
  auto vx_container = ex.get_buffer(_vx);
  RHS my_vx(vx_container, ex.get_offset(_vx), _incx, _N);
  auto vr_container = ex.get_buffer(_rs);
  RHS my_rs(vr_container, 0, 1, 1);
#ifdef VERBOSE
  my_vx.printH("VX");
#endif  //  VERBOSE
  auto prdOp = make_op<UnaryOp, prdOp1_struct>(my_vx);
  auto localSize = 256;
  auto nWG = 512;
  auto assignOp2 =
      make_addAssignReduction(my_rs, prdOp, localSize, localSize * nWG);
  ex.reduce(assignOp2);
  auto sqrtOp = make_op<UnaryOp, sqtOp1_struct>(my_rs);
  auto assignOp3 = make_op<Assign>(my_rs, sqrtOp);
  ex.execute(assignOp3);
}

/**
 * \brief ASUM Takes the sum of the absolute values
 * @param Executor<ExecutorType> ex
 * @param _vx  VectorView
 * @param _incx Increment in X axis
 */
template <typename ExecutorType, typename T>
void _asum(Executor<ExecutorType> &ex, int _N, T *_vx, int _incx, T *_rs) {
  using RHS =
      vector_view<T, typename Executor<ExecutorType>::template ContainerT<T>>;
  auto vx_container = ex.get_buffer(_vx);
  RHS my_vx(vx_container, ex.get_offset(_vx), _incx, _N);
  auto rs_container = ex.get_buffer(_rs);
  RHS my_rs(rs_container, ex.get_offset(_rs), 1, 1);

#ifdef VERBOSE
  my_vx.printH("VX");
  my_rs.printH("VR");
#endif  //  VERBOSE
  auto localSize = 256;
  auto nWG = 512;
  auto assignOp =
      make_addAbsAssignReduction(my_rs, my_vx, localSize, localSize * nWG);
  ex.reduce(assignOp);
#ifdef VERBOSE
  my_rs.printH("VR");
#endif  //  VERBOSE
}

/**
 * \brief ASUM Takes the sum of the absolute values
 *
 * @param Executor<ExecutorType> ex
 * @param _vx  VectorView
 * @param _incx Increment in X axis
 */
template <typename ExecutorType, typename T>
T _asum(Executor<ExecutorType> &ex, int _N, T *_vx, int _incx) {
#ifdef VERBOSE
  my_vx.printH("VX");
#endif  //  VERBOSE
  std::vector<T> vR(1, T(0));
  auto gpu_vR = ex.template allocate<T>(1);
  ex.copy_to_device(vR.data(), gpu_vR, 1);
  _asum(ex, _N, _vx, _incx, gpu_vR);
  ex.copy_to_host(gpu_vR, vR.data(), 1);
  ex.template deallocate<T>(gpu_vR);
#ifdef VERBOSE
  std::cout << "val = " << vR[0] << std::endl;
#endif  //  VERBOSE
  return vR[0];
}

/**
 * ROTG.
 * @brief Consturcts given plane rotation
 * Not implemented.
 */
template <typename T>
void _rotg(T &_alpha, T &_beta, T &_cos, T &_sin) {
  T abs_alpha = std::abs(_alpha);
  T abs_beta = std::abs(_beta);
  T roe = (abs_alpha > abs_beta) ? _alpha : _beta;
  T scale = abs_alpha + abs_beta;
  T norm;
  T aux;

  if (scale == constant<T, const_val::zero>::value) {
    _cos = constant<T, const_val::one>::value;
    _sin = constant<T, const_val::zero>::value;
    norm = constant<T, const_val::zero>::value;
    aux = constant<T, const_val::zero>::value;
  } else {
    norm = scale * std::sqrt((_alpha / scale) * (_alpha / scale) +
                             (_beta / scale) * (_beta / scale));
    if (roe < constant<T, const_val::zero>::value) norm = -norm;
    _cos = _alpha / norm;
    _sin = _beta / norm;
    if (abs_alpha > abs_beta) {
      aux = _sin;
    } else if (_cos != constant<T, const_val::zero>::value) {
      aux = constant<T, const_val::one>::value / _cos;
    } else {
      aux = constant<T, const_val::one>::value;
    }
  }
  _alpha = norm;
  _beta = aux;
}

/**
 * ROTG.
 * @brief Consturcts given plane rotation
 * Not implemented.
 */
template <typename ExecutorType, typename T>
void _rot(Executor<ExecutorType> &ex, int _N, T *_vx, int _incx, T *_vy,
          int _incy, T _cos, T _sin) {
  using RHS =
      vector_view<T, typename Executor<ExecutorType>::template ContainerT<T>>;
  auto vx_container = ex.get_buffer(_vx);
  RHS my_vx(vx_container, ex.get_offset(_vx), _incx, _N);
  auto vy_container = ex.get_buffer(_vy);
  RHS my_vy(vy_container, ex.get_offset(_vy), _incy, _N);
#ifdef VERBOSE
  std::cout << "cos = " << _cos << " , sin = " << _sin << std::endl;
  my_vx.printH("VX");
  my_vy.printH("VY");
#endif  //  VERBOSE
  auto scalOp1 = make_op<ScalarOp, prdOp2_struct>(_cos, my_vx);
  auto scalOp2 = make_op<ScalarOp, prdOp2_struct>(_sin, my_vy);
  auto scalOp3 = make_op<ScalarOp, prdOp2_struct>(-_sin, my_vx);
  auto scalOp4 = make_op<ScalarOp, prdOp2_struct>(_cos, my_vy);
  auto addOp12 = make_op<BinaryOp, addOp2_struct>(scalOp1, scalOp2);
  auto addOp34 = make_op<BinaryOp, addOp2_struct>(scalOp3, scalOp4);
  auto dobleAssignView = make_op<DobleAssign>(my_vx, my_vy, addOp12, addOp34);
  ex.execute(dobleAssignView);
#ifdef VERBOSE
  my_vx.printH("VX");
  my_vy.printH("VY");
#endif  //  VERBOSE
}

// THIS ROUTINE IS UNVERIFIED AND HAS NOT BEEN TESTED
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

#endif  // BLAS1_INTERFACE_SYCL_HPP
