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
  auto scalExpr = make_expr<ScalarExpr, prdOp2_struct>(_alpha, my_vx);
  auto addExpr = make_expr<BinaryExpr, addOp2_struct>(my_vy, scalExpr);
  auto assignExpr = make_expr<AssignExpr>(my_vy, addExpr);
  ex.execute(assignExpr);
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
  auto assignExpr = make_expr<AssignExpr>(my_vy, my_vx);
  ex.execute(assignExpr);
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
  auto swapExpr = make_expr<DoubleAssignExpr>(my_vy, my_vx, my_vx, my_vy);
  ex.execute(swapExpr);
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
  auto scalExpr = make_expr<ScalarExpr, prdOp2_struct>(_alpha, my_vx);
  auto assignExpr = make_expr<AssignExpr>(my_vx, scalExpr);
  ex.execute(assignExpr);
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
/* template <typename ExecutorType, typename T, typename ContainerT> */
/* T _dot(Executor<ExecutorType> ex, int _N, vector_view<T, ContainerT> _vx, */
/*        int _incx, vector_view<T, ContainerT> _vy, int _incy) { */
/*   auto my_vx = vector_view<T, ContainerT>(_vx, _vx.getDisp(), _incx, _N); */
/*   auto my_vy = vector_view<T, ContainerT>(_vy, _vy.getDisp(), _incy, _N); */
/* #ifdef VERBOSE */
/*   my_vx.printH("VX"); */
/*   my_vy.printH("VY"); */
/* #endif  //  VERBOSE */
/*   auto prdOp = make_expr<BinaryOp, prdOp2_struct>(my_vx, my_vy); */

/*   auto localSize = 256; */
/*   auto nWG = 512; */
/*   ContainerT valT1(nWG); */
/*   auto val1 = vector_view<T, ContainerT>(valT1, 0, 1, nWG); */
/*   auto assignExpr1 = */
/*       make_addAssignReductionExpr(val1, prdOp, localSize, nWG * localSize);
 */
/*   ex.execute(assignExpr1); */

/*   ContainerT valT2(1); */
/*   auto val2 = vector_view<T, ContainerT>(valT2, 0, 1, 1); */
/*   auto assignExpr2 = make_addAssignReductionExpr(val2, val1, localSize, nWG);
 */
/*   ex.execute(assignExpr2); */

/* #ifdef VERBOSE */
/*   std::cout << "val = " << val2.eval(0) << std::endl; */
/* #endif  //  VERBOSE */
/*   return val2.eval(0); */
/* } */

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
void _dot(Executor<ExecutorType> ex, int _N, vector_view<T, ContainerT> _vx, int _incx, vector_view<T, ContainerT> _vy, int _incy, vector_view<T, ContainerT> _rs) {
  auto my_vx = vector_view<T, ContainerT>(_vx, _vx.getDisp(), _incx, _N);
  auto my_vy = vector_view<T, ContainerT>(_vy, _vy.getDisp(), _incy, _N);
  auto my_rs = vector_view<T, ContainerT>(_rs, _rs.getDisp(), 1, 1);
#ifdef VERBOSE
  my_vx.printH("VX");
  my_vy.printH("VY");
  my_rs.printH("VR");
#endif  //  VERBOSE
  auto prdExpr = make_expr<BinaryExpr, prdOp2_struct>(my_vx, my_vy);
  auto localSize = 256;
  auto nWG = 512;
  auto assignExpr = make_addAssignReduction(my_rs, prdExpr);
  ex.execute(assignExpr);
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
T _nrm2(Executor<ExecutorType> ex, int _N, vector_view<T, ContainerT> _vx, int _incx) {
  auto my_vx = vector_view<T, ContainerT>(_vx, _vx.getDisp(), _incx, _N);
#ifdef VERBOSE
  my_vx.printH("VX");
#endif  //  VERBOSE
  auto prdExpr = make_expr<UnaryExpr, prdOp1_struct>(my_vx);

  auto localSize = 256;
  auto nWG = 512;
  ContainerT valT1(nWG);
  auto val1 = vector_view<T, ContainerT>(valT1, 0, 1, nWG);
  auto assignExpr = make_addAssignReductionExpr(val1, prdExpr);
  ex.execute(assignExpr);
#ifdef VERBOSE
  std::cout << "val = " << std::sqrt(val1.eval(0)) << std::endl;
#endif  //  VERBOSE
  return std::sqrt(val1.eval(0));
}

/**
 * \brief NRM2 Returns the euclidian norm of a vector
 * @param Executor<ExecutorType> ex
 * @param _vx  VectorView
 * @param _incx Increment in X axis
 */
template <typename ExecutorType, typename T, typename ContainerT>
void _nrm2(Executor<ExecutorType> ex, int _N, vector_view<T, ContainerT> _vx, int _incx, vector_view<T, ContainerT> _rs) {
  auto my_vx = vector_view<T, ContainerT>(_vx, _vx.getDisp(), _incx, _N);
  auto my_rs = vector_view<T, ContainerT>(_rs, _rs.getDisp(), 1, 1);
#ifdef VERBOSE
  my_vx.printH("VX");
  my_rs.printH("VR");
#endif  //  VERBOSE

  auto prdExpr = make_expr<UnaryExpr, prdOp1_struct>(my_vx);
  auto localSize = 256;
  auto nWG = 512;
  auto assignExpr = make_addAssignReductionExpr(my_rs, prdExpr, localSize, localSize * nWG);
  ex.execute(assignExpr);
  auto sqrtExpr = make_expr<UnaryExpr, sqtOp1_struct>(my_rs);
  auto assignExpr2 = make_expr<AssignExpr>(my_rs, sqrtExpr);
  ex.execute(assignExpr2);
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
T _asum(Executor<ExecutorType> ex, int _N, vector_view<T, ContainerT> _vx, int _incx) {
  auto my_vx = vector_view<T, ContainerT>(_vx, _vx.getDisp(), _incx, _N);
#ifdef VERBOSE
  my_vx.printH("VX");
#endif  //  VERBOSE

  auto localSize = 256;
  auto nWG = 128;
  ContainerT valT1(nWG);
  auto val1 = vector_view<T, ContainerT>(valT1, 0, 1, nWG);
  auto assignExpr = make_addAbsAssignReductionExpr(val1, my_vx, localSize, nWG * localSize);
  ex.execute(assignExpr);
#ifdef VERBOSE
  std::cout << "val = " << val2.eval(0) << std::endl;
#endif  //  VERBOSE
  return val1.eval(0);
}

/**
 * \brief ASUM Takes the sum of the absolute values
 * @param Executor<ExecutorType> ex
 * @param _vx  VectorView
 * @param _incx Increment in X axis
 */
template <typename ExecutorType, typename T, typename ContainerT>
void _asum(Executor<ExecutorType> ex, int _N, vector_view<T, ContainerT> _vx, int _incx, vector_view<T, ContainerT> _rs) {
  auto my_vx = vector_view<T, ContainerT>(_vx, _vx.getDisp(), _incx, _N);
  auto my_rs = vector_view<T, ContainerT>(_rs, _rs.getDisp(), 1, 1);
#ifdef VERBOSE
  my_vx.printH("VX");
  my_rs.printH("VR");
#endif  //  VERBOSE
  auto localSize = 256;
  auto nWG = 512;
  auto assignExpr = make_addAbsAssignReductionExpr(my_rs, my_vx);
  ex.execute(assignExpr);
#ifdef VERBOSE
  my_rs.printH("VR");
#endif  //  VERBOSE
}

/**
 * \brief IAMAX finds the index of the first element having maximum
 * @param _vx  VectorView
 * @param _incx Increment in X axis
 */
template <typename ExecutorType, typename T, typename ContainerT, typename I, typename ContainerI>
void _iamax(Executor<ExecutorType> ex, int _N, vector_view<T, ContainerT> _vx, int _incx, vector_view<I, ContainerI> _rs) {
  auto my_vx = vector_view<T, ContainerT>(_vx, _vx.getDisp(), _incx, _N);
  auto my_rs = vector_view<I, ContainerI>(_rs, _rs.getDisp(), 1, 1);
#ifdef VERBOSE
  my_vx.printH("VX");
#endif  //  VERBOSE
  size_t localSize = 256, nWG = 512;
  auto tupExpr = TupleExpr<vector_view<T, ContainerT>>(my_vx);
  std::vector<IndVal<T>> valT1(nWG, constant<IndVal<T>, const_val::imax>::value);
  auto bvalT1 = cl::sycl::buffer<IndVal<T>, 1>(valT1.data(), cl::sycl::range<1>{nWG});
  auto val1 = BufferVectorView<IndVal<T>>(bvalT1, 0, 1, nWG);
  auto assignExpr = make_maxIndAssignReductionExpr(val1, tupExpr);
  ex.execute(assignExpr);
#ifdef VERBOSE
  std::cout << "ind = " << val1.eval(0).getInd() << std::endl;
#endif  //  VERBOSE
}

/**
 * \brief ICAMAX finds the index of the first element having maximum
 * @param _vx  VectorView
 * @param _incx Increment in X axis
 */
template <typename ExecutorType, typename T, typename ContainerT>
size_t _iamax(Executor<ExecutorType> ex, int _N, vector_view<T, ContainerT> _vx, int _incx) {
  IndVal<T> rsT;
  auto brsT = cl::sycl::buffer<IndVal<T>, 1>(&rsT, cl::sycl::range<1>{1});
  auto rs = BufferVectorView<IndVal<T>>(brsT, 0, 1, 1);
  _iamax(ex, _N, _vx, _incx, rs);
  return rs.eval(0).getInd();
}

/**
 * \brief IAMIN finds the index of the first element having minimum
 * @param _vx  VectorView
 * @param _incx Increment in X axis
 */
template <typename ExecutorType, typename T, typename ContainerT, typename I, typename ContainerI>
void _iamin(Executor<ExecutorType> ex, int _N, vector_view<T, ContainerT> _vx, int _incx, vector_view<I, ContainerI> _rs) {
  auto my_vx = vector_view<T, ContainerT>(_vx, _vx.getDisp(), _incx, _N);
  auto my_rs = vector_view<I, ContainerI>(_rs, _rs.getDisp(), 1, 1);
#ifdef VERBOSE
  my_vx.printH("VX");
#endif  //  VERBOSE
  size_t localSize = 256, nWG = 512;
  auto tupExpr = TupleExpr<vector_view<T, ContainerT>>(my_vx);
  std::vector<IndVal<T>> valT1(nWG, constant<IndVal<T>, const_val::imin>::value);
  auto bvalT1 = cl::sycl::buffer<IndVal<T>, 1>(valT1.data(), cl::sycl::range<1>{nWG});
  auto val1 = BufferVectorView<IndVal<T>>(bvalT1, 0, 1, nWG);
  auto assignExpr = make_minIndAssignReduction(val1, tupExpr, localSize, localSize * nWG);
  ex.execute(assignExpr);
#ifdef VERBOSE
  std::cout << "ind = " << val1.eval(0).getInd() << std::endl;
#endif  //  VERBOSE
}

/**
 * \brief ICAMIN finds the index of the first element having minimum
 * @param _vx  VectorView
 * @param _incx Increment in X axis
 */
template <typename ExecutorType, typename T, typename ContainerT>
size_t _iamin(Executor<ExecutorType> ex, int _N, vector_view<T, ContainerT> _vx, int _incx) {
  IndVal<T> rsT;
  auto brsT = cl::sycl::buffer<IndVal<T>, 1>(&rsT, cl::sycl::range<1>{1});
  auto rs = BufferVectorView<IndVal<T>>(brsT, 0, 1, 1);
  _iamin(ex, _N, _vx, _incx, rs);
  return rs.eval(0).getInd();
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
template <typename ExecutorType, typename T, typename ContainerT>
void _rot(Executor<ExecutorType> ex, int _N, vector_view<T, ContainerT> _vx, int _incx, vector_view<T, ContainerT> _vy, int _incy, T _cos, T _sin) {
  auto my_vx = vector_view<T, ContainerT>(_vx, _vx.getDisp(), _incx, _N);
  auto my_vy = vector_view<T, ContainerT>(_vy, _vy.getDisp(), _incy, _N);
#ifdef VERBOSE
  std::cout << "cos = " << _cos << " , sin = " << _sin << std::endl;
  my_vx.printH("VX");
  my_vy.printH("VY");
#endif  //  VERBOSE
  auto scalExpr1 = make_expr<ScalarExpr, prdOp2_struct>(_cos, my_vx);
  auto scalExpr2 = make_expr<ScalarExpr, prdOp2_struct>(_sin, my_vy);
  auto scalExpr3 = make_expr<ScalarExpr, prdOp2_struct>(-_sin, my_vx);
  auto scalExpr4 = make_expr<ScalarExpr, prdOp2_struct>(_cos, my_vy);
  auto addExpr12 = make_expr<BinaryExpr, addOp2_struct>(scalExpr1, scalExpr2);
  auto addExpr34 = make_expr<BinaryExpr, addOp2_struct>(scalExpr3, scalExpr4);
  auto doubleAssignExpr = make_expr<DoubleAssignExpr>(my_vx, my_vy, addExpr12, addExpr34);
  ex.execute(doubleAssignExpr);
#ifdef VERBOSE
  my_vx.printH("VX");
  my_vy.printH("VY");
#endif  //  VERBOSE
}

}  // namespace blas

#endif  // BLAS1_INTERFACE_SYCL_HPP
