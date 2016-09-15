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
template <typename ExecutorType, typename T, typename ContainerT>
void _axpy(Executor<ExecutorType> ex, int _N, T _alpha,
           vector_view<T, ContainerT> _vx, int _incx,
           vector_view<T, ContainerT> _vy, int _incy) {
  auto my_vx = vector_view<T, ContainerT>(_vx, _vx.getDisp(), _incx, _N);
  auto my_vy = vector_view<T, ContainerT>(_vy, _vy.getDisp(), _incy, _N);
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
template <typename ExecutorType, typename T, typename ContainerT>
void _copy(Executor<ExecutorType> ex, int _N, vector_view<T, ContainerT> _vx,
           int _incx, vector_view<T, ContainerT> _vy, int _incy) {
  auto my_vx = vector_view<T, ContainerT>(_vx, _vx.getDisp(), _incx, _N);
  auto my_vy = vector_view<T, ContainerT>(_vy, _vy.getDisp(), _incy, _N);
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
 * \brief SWAP interchanges two vectors
 *
 * @param Executor<ExecutorType> ex
 * @param _vx  VectorView
 * @param _incx Increment in X axis
 * @param _vy  VectorView
 * @param _incy Increment in Y axis
 */
template <typename ExecutorType, typename T, typename ContainerT>
void _swap(Executor<ExecutorType> ex, int _N, vector_view<T, ContainerT> _vx,
           int _incx, vector_view<T, ContainerT> _vy, int _incy) {
  auto my_vx = vector_view<T, ContainerT>(_vx, _vx.getDisp(), _incx, _N);
  auto my_vy = vector_view<T, ContainerT>(_vy, _vy.getDisp(), _incy, _N);
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

/**
 * \brief SCAL scales a vector by a constant
 *
 * @param Executor<ExecutorType> ex
 * @param _vx  VectorView
 * @param _incx Increment in X axis
 */
template <typename ExecutorType, typename T, typename ContainerT>
void _scal(Executor<ExecutorType> ex, int _N, T _alpha,
           vector_view<T, ContainerT> _vx, int _incx) {
  auto my_vx = vector_view<T, ContainerT>(_vx, _vx.getDisp(), _incx, _N);
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
 * \briefCompute the inner product of two vectors with extended
    precision accumulation and result.
 *
 * @param Executor<ExecutorType> ex
 * @param _vx  VectorView
 * @param _incx Increment in X axis
 * @param _vx  VectorView
 * @param _incy Increment in Y axis
 */
template <typename ExecutorType, typename T, typename ContainerT>
T _dot(Executor<ExecutorType> ex, int _N, vector_view<T, ContainerT> _vx,
       int _incx, vector_view<T, ContainerT> _vy, int _incy) {
  auto my_vx = vector_view<T, ContainerT>(_vx, _vx.getDisp(), _incx, _N);
  auto my_vy = vector_view<T, ContainerT>(_vy, _vy.getDisp(), _incy, _N);
#ifdef VERBOSE
  my_vx.printH("VX");
  my_vy.printH("VY");
#endif  //  VERBOSE
  auto prdOp = make_op<BinaryOp, prdOp2_struct>(my_vx, my_vy);

  auto localSize = 256;
  auto nWG = 512;
  ContainerT valT1(nWG);
  auto val1 = vector_view<T, ContainerT>(valT1, 0, 1, nWG);
  auto assignOp1 =
      make_addReducAssignNewOp2(val1, prdOp, localSize, nWG * localSize);
  ex.execute(assignOp1);

  ContainerT valT2(1);
  auto val2 = vector_view<T, ContainerT>(valT2, 0, 1, 1);
  auto assignOp2 = make_addReducAssignNewOp2(val2, val1, localSize, nWG);
  ex.execute(assignOp2);

#ifdef VERBOSE
  std::cout << "val = " << val2.eval(0) << std::endl;
#endif  //  VERBOSE
  return val2.eval(0);
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
template <typename ExecutorType, typename T, typename ContainerT>
void _dot(Executor<ExecutorType> ex, int _N, vector_view<T, ContainerT> _vx,
          int _incx, vector_view<T, ContainerT> _vy, int _incy,
          vector_view<T, ContainerT> _rs) {
  auto my_vx = vector_view<T, ContainerT>(_vx, _vx.getDisp(), _incx, _N);
  auto my_vy = vector_view<T, ContainerT>(_vy, _vy.getDisp(), _incy, _N);
  auto my_rs = vector_view<T, ContainerT>(_rs, _rs.getDisp(), 1, 1);
#ifdef VERBOSE
  my_vx.printH("VX");
  my_vy.printH("VY");
  my_rs.printH("VR");
#endif  //  VERBOSE
  auto prdOp = make_op<BinaryOp, prdOp2_struct>(my_vx, my_vy);
  auto localSize = 256;
  auto nWG = 512;
  auto assignOp1 =
      make_addReducAssignNewOp2(my_rs, prdOp, localSize, localSize * nWG);
  ex.reduce(assignOp1);
#ifdef VERBOSE
  my_rs.printH("VR");
#endif  //  VERBOSE
}

/**
 * \brief NRM2 Returns the euclidian norm of a vector
 *
 * @param Executor<ExecutorType> ex
 * @param _vx  VectorView
 * @param _incx Increment in X axis
 */
template <typename ExecutorType, typename T, typename ContainerT>
T _nrm2(Executor<ExecutorType> ex, int _N, vector_view<T, ContainerT> _vx,
        int _incx) {
  auto my_vx = vector_view<T, ContainerT>(_vx, _vx.getDisp(), _incx, _N);
#ifdef VERBOSE
  my_vx.printH("VX");
#endif  //  VERBOSE
  auto prdOp = make_op<UnaryOp, prdOp1_struct>(my_vx);

  auto localSize = 256;
  auto nWG = 512;
  ContainerT valT1(nWG);
  auto val1 = vector_view<T, ContainerT>(valT1, 0, 1, nWG);
  auto assignOp1 =
      make_addReducAssignNewOp2(val1, prdOp, localSize, localSize * nWG);
  ex.execute(assignOp1);

  ContainerT valT2(1);
  auto val2 = vector_view<T, ContainerT>(valT2, 0, 1, 1);
  auto assignOp2 = make_addReducAssignNewOp2(val2, val1, localSize, nWG);
  ex.execute(assignOp2);
#ifdef VERBOSE
  std::cout << "val = " << std::sqrt(val2.eval(0)) << std::endl;
#endif  //  VERBOSE
  return std::sqrt(val2.eval(0));
}

/**
 * \brief NRM2 Returns the euclidian norm of a vector
 * @param Executor<ExecutorType> ex
 * @param _vx  VectorView
 * @param _incx Increment in X axis
 */
template <typename ExecutorType, typename T, typename ContainerT>
void _nrm2(Executor<ExecutorType> ex, int _N, vector_view<T, ContainerT> _vx,
           int _incx, vector_view<T, ContainerT> _rs) {
  auto my_vx = vector_view<T, ContainerT>(_vx, _vx.getDisp(), _incx, _N);
  auto my_rs = vector_view<T, ContainerT>(_rs, _rs.getDisp(), 1, 1);
#ifdef VERBOSE
  my_vx.printH("VX");
  my_rs.printH("VR");
#endif  //  VERBOSE

  auto prdOp = make_op<UnaryOp, prdOp1_struct>(my_vx);
  auto localSize = 256;
  auto nWG = 512;
  auto assignOp1 =
      make_addReducAssignNewOp2(my_rs, prdOp, localSize, localSize * nWG);
  ex.reduce(assignOp1);
  auto sqrtOp = make_op<UnaryOp, sqtOp1_struct>(my_rs);
  auto assignOp2 = make_op<Assign>(my_rs, sqrtOp);
  ex.execute(assignOp2);
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
template <typename ExecutorType, typename T, typename ContainerT>
T _asum(Executor<ExecutorType> ex, int _N, vector_view<T, ContainerT> _vx,
        int _incx) {
  auto my_vx = vector_view<T, ContainerT>(_vx, _vx.getDisp(), _incx, _N);
#ifdef VERBOSE
  my_vx.printH("VX");
#endif  //  VERBOSE

  auto localSize = 256;
  auto nWG = 128;
  ContainerT valT1(nWG);
  auto val1 = vector_view<T, ContainerT>(valT1, 0, 1, nWG);
  auto assignOp1 =
      make_addAbsReducAssignNewOp2(val1, my_vx, localSize, nWG * localSize);
  ex.execute(assignOp1);

  ContainerT valT2(1);
  auto val2 = vector_view<T, ContainerT>(valT2, 0, 1, 1);
  auto assignOp2 = make_addAbsReducAssignNewOp2(val2, val1, localSize, nWG);
  ex.execute(assignOp2);

#ifdef VERBOSE
  std::cout << "val = " << val2.eval(0) << std::endl;
#endif  //  VERBOSE
  return val2.eval(0);
}

/**
 * \brief ASUM Takes the sum of the absolute values
 * @param Executor<ExecutorType> ex
 * @param _vx  VectorView
 * @param _incx Increment in X axis
 */
template <typename ExecutorType, typename T, typename ContainerT>
void _asum(Executor<ExecutorType> ex, int _N, vector_view<T, ContainerT> _vx,
           int _incx, vector_view<T, ContainerT> _rs) {
  auto my_vx = vector_view<T, ContainerT>(_vx, _vx.getDisp(), _incx, _N);
  auto my_rs = vector_view<T, ContainerT>(_rs, _rs.getDisp(), 1, 1);
#ifdef VERBOSE
  my_vx.printH("VX");
  my_rs.printH("VR");
#endif  //  VERBOSE
  auto localSize = 256;
  auto nWG = 512;
  auto assignOp =
      make_addAbsReducAssignNewOp2(my_rs, my_vx, localSize, localSize * nWG);
  ex.reduce(assignOp);
#ifdef VERBOSE
  my_rs.printH("VR");
#endif  //  VERBOSE
}

/**
 * \brief ICAMAX finds the index of the first element having maximum
 * @param _vx  VectorView
 * @param _incx Increment in X axis
 */
template <typename ExecutorType, typename T, typename ContainerT>
size_t _iamax(Executor<ExecutorType> ex, int _N, vector_view<T, ContainerT> _vx,
              int _incx) {
  auto my_vx = vector_view<T, ContainerT>(_vx, _vx.getDisp(), _incx, _N);
#ifdef VERBOSE
  my_vx.printH("VX");
#endif  //  VERBOSE

  auto localSize = 256;
  auto nWG = 512;
  std::vector<IndVal<T>> valT1(nWG);
  auto val1 = vector_view<IndVal<T>, std::vector<IndVal<T>>>(valT1, 0, 1, nWG);
  auto assignOp1 =
      make_maxIndReducAssignNewOp3(val1, my_vx, localSize, nWG * localSize);
  ex.execute(assignOp1);

  std::vector<IndVal<T>> valT2(1);
  auto val2 = vector_view<IndVal<T>, std::vector<IndVal<T>>>(valT1, 0, 1, 1);
  auto assignOp2 = make_maxIndReducAssignNewOp2(val2, val1, localSize, nWG);
  ex.execute(assignOp2);
#ifdef VERBOSE
  std::cout << "ind = " << val1.eval(0).getInd() << std::endl;
#endif  //  VERBOSE
  return val1.eval(0).getInd();
}

/**
 * \brief IAMAX finds the index of the first element having maximum
 * @param _vx  VectorView
 * @param _incx Increment in X axis
 */
template <typename ExecutorType, typename T, typename ContainerT, typename I,
          typename ContainerI>
void _iamax(Executor<ExecutorType> ex, int _N, vector_view<T, ContainerT> _vx,
            int _incx, vector_view<I, ContainerI> _rs) {
  auto my_vx = vector_view<T, ContainerT>(_vx, _vx.getDisp(), _incx, _N);
  auto my_rs = vector_view<I, ContainerI>(_rs, _rs.getDisp(), 1, 1);
#ifdef VERBOSE
  my_vx.printH("VX");
#endif  //  VERBOSE
  auto localSize = 256;
  auto nWG = 512;
  auto assignOp =
      make_maxIndReducAssignNewOp3(my_rs, my_vx, localSize, localSize * nWG);
  ex.reduce(assignOp);
#ifdef VERBOSE
  std::cout << "ind = " << val1.eval(0).getInd() << std::endl;
#endif  //  VERBOSE
}

/**
 * ROTG.
 * @brief Consturcts given plane rotation
 * Not implemented.
 */
template <typename T>
void _rotg(T& _alpha, T& _beta, T& _cos, T& _sin) {
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
template <typename ExecutorType, typename T, typename ContainerT>
void _rot(Executor<ExecutorType> ex, int _N, vector_view<T, ContainerT> _vx,
          int _incx, vector_view<T, ContainerT> _vy, int _incy, T _cos,
          T _sin) {
  auto my_vx = vector_view<T, ContainerT>(_vx, _vx.getDisp(), _incx, _N);
  auto my_vy = vector_view<T, ContainerT>(_vy, _vy.getDisp(), _incy, _N);
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

}  // namespace blas

#endif  // BLAS1_INTERFACE_SYCL_HPP
