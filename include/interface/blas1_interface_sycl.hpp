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

#include <CL/sycl.hpp>

#include <executors/executor_sycl.hpp>
#include <executors/reduction_sycl.hpp>
#include <operations/blas1_trees.hpp>

namespace blas {

/**
 * \brief AXPY constant times a vector plus a vector.
 *
 * Implements AXPY \f$y = ax + y\f$
 *
 * @param Device &dev
 * @param _vx  VectorView
 * @param _incx Increment in X axis
 * @param _vy  VectorView
 * @param _incy Increment in Y axis
 */
template <typename Device, typename ScalarT, typename ContainerT>
void _axpy(Device &dev, int _N, ScalarT _alpha,
           vector_view<ScalarT, ContainerT> _vx, int _incx,
           vector_view<ScalarT, ContainerT> _vy, int _incy) {
  auto my_vx = vector_view<ScalarT, ContainerT>(_vx, _vx.getDisp(), _incx, _N);
  auto my_vy = vector_view<ScalarT, ContainerT>(_vy, _vy.getDisp(), _incy, _N);
  auto scalExpr = make_expr<ScalarExpr, prdOp2_struct>(_alpha, my_vx);
  auto addExpr = make_expr<BinaryExpr, addOp2_struct>(my_vy, scalExpr);
  auto assignExpr = make_expr<AssignExpr>(my_vy, addExpr);
  blas::execute(dev, assignExpr);
}

/**
 * \brief COPY copies a vector, x, to a vector, y.
 *
 * @param Device &dev
 * @param _vx  VectorView
 * @param _incx Increment in X axis
 * @param _vy  VectorView
 * @param _incy Increment in Y axis
 */
template <typename Device, typename ScalarT, typename ContainerT>
void _copy(Device &dev, int _N, vector_view<ScalarT, ContainerT> _vx, int _incx,
           vector_view<ScalarT, ContainerT> _vy, int _incy) {
  auto my_vx = vector_view<ScalarT, ContainerT>(_vx, _vx.getDisp(), _incx, _N);
  auto my_vy = vector_view<ScalarT, ContainerT>(_vy, _vy.getDisp(), _incy, _N);
  auto assignExpr = make_expr<AssignExpr>(my_vy, my_vx);
  blas::execute(dev, assignExpr);
}

/**
 * \brief SWAP interchanges two vectors
 *
 * @param Device &dev
 * @param _vx  VectorView
 * @param _incx Increment in X axis
 * @param _vy  VectorView
 * @param _incy Increment in Y axis
 */
template <typename Device, typename ScalarT, typename ContainerT>
void _swap(Device &dev, int _N, vector_view<ScalarT, ContainerT> _vx, int _incx,
           vector_view<ScalarT, ContainerT> _vy, int _incy) {
  auto my_vx = vector_view<ScalarT, ContainerT>(_vx, _vx.getDisp(), _incx, _N);
  auto my_vy = vector_view<ScalarT, ContainerT>(_vy, _vy.getDisp(), _incy, _N);
  auto swapExpr = make_expr<DoubleAssignExpr>(my_vy, my_vx, my_vx, my_vy);
  blas::execute(dev, swapExpr);
}

/**
 * \brief SCAL scales a vector by a constant
 *
 * @param Device &dev
 * @param _vx  VectorView
 * @param _incx Increment in X axis
 */
template <typename Device, typename ScalarT, typename ContainerT>
void _scal(Device &dev, int _N, ScalarT _alpha,
           vector_view<ScalarT, ContainerT> _vx, int _incx) {
  auto my_vx = vector_view<ScalarT, ContainerT>(_vx, _vx.getDisp(), _incx, _N);
  auto scalExpr = make_expr<ScalarExpr, prdOp2_struct>(_alpha, my_vx);
  auto assignExpr = make_expr<AssignExpr>(my_vx, scalExpr);
  blas::execute(dev, assignExpr);
}

/**
 * \briefCompute the inner product of two vectors with extended
    precision accumulation and result.
 *
 * @param Device &dev
 * @param _vx  VectorView
 * @param _incx Increment in X axis
 * @param _vx  VectorView
 * @param _incy Increment in Y axis
 */
template <typename Device, typename ScalarT, typename ContainerT>
ScalarT _dot(Device &dev, int _N, vector_view<ScalarT, ContainerT> _vx,
             int _incx, vector_view<ScalarT, ContainerT> _vy, int _incy) {
  auto my_vx = vector_view<ScalarT, ContainerT>(_vx, _vx.getDisp(), _incx, _N);
  auto my_vy = vector_view<ScalarT, ContainerT>(_vy, _vy.getDisp(), _incy, _N);

  auto prdExpr = make_expr<BinaryExpr, prdOp2_struct>(my_vx, my_vy);
  ScalarT result;
  cl::sycl::buffer<ScalarT, 1> buf_result(&result, cl::sycl::range<1>{1});
  auto rs = vector_view<ScalarT, ContainerT>(buf_result, 0, 1, 1);
  auto dotExpr = make_addReductionExpr(prdExpr);
  auto assignExpr = make_expr<AssignExpr>(rs, dotExpr);
  blas::execute(dev, assignExpr);
  return rs.eval(0);
}

/**
 * \brief Compute the inner product of two vectors with extended precision
    accumulation.
 * @param Device &dev
 * @param _vx  VectorView
 * @param _incx Increment in X axis
 * @param _vx  VectorView
 * @param _incy Increment in Y axis
 */
template <typename Device, typename ScalarT, typename ContainerT>
void _dot(Device &dev, int _N, vector_view<ScalarT, ContainerT> _vx, int _incx,
          vector_view<ScalarT, ContainerT> _vy, int _incy,
          vector_view<ScalarT, ContainerT> _rs) {
  auto my_vx = vector_view<ScalarT, ContainerT>(_vx, _vx.getDisp(), _incx, _N);
  auto my_vy = vector_view<ScalarT, ContainerT>(_vy, _vy.getDisp(), _incy, _N);
  auto my_rs = vector_view<ScalarT, ContainerT>(_rs, _rs.getDisp(), 1, 1);
  auto prdExpr = make_expr<BinaryExpr, prdOp2_struct>(my_vx, my_vy);
  auto dotExpr = make_addReductionExpr(prdExpr);
  auto assignExpr = make_expr<AssignExpr>(my_rs, dotExpr);
  blas::execute(dev, assignExpr);
}

template <typename Device, typename ScalarT, typename ContainerT>
void _dot_tree(Device &dev, int _N, vector_view<ScalarT, ContainerT> _vx,
               int _incx, vector_view<ScalarT, ContainerT> _vy, int _incy,
               vector_view<ScalarT, ContainerT> _rs) {
  auto my_vx = vector_view<ScalarT, ContainerT>(_vx, _vx.getDisp(), _incx, _N);
  auto my_vy = vector_view<ScalarT, ContainerT>(_vy, _vy.getDisp(), _incy, _N);
  auto my_rs = vector_view<ScalarT, ContainerT>(_rs, _rs.getDisp(), 1, 1);
  auto prdExpr = make_expr<BinaryExpr, prdOp2_struct>(my_vx, my_vy);
  auto dotExpr = make_addReductionExpr(prdExpr);
  auto assignExpr = make_expr<AssignExpr>(my_rs, dotExpr);
  blas::execute(dev, assignExpr);
}

/**
 * \brief NRM2 Returns the euclidian norm of a vector
 *
 * @param Device &dev
 * @param _vx  VectorView
 * @param _incx Increment in X axis
 */
template <typename Device, typename ScalarT, typename ContainerT>
ScalarT _nrm2(Device &dev, int _N, vector_view<ScalarT, ContainerT> _vx,
              int _incx) {
  auto my_vx = vector_view<ScalarT, ContainerT>(_vx, _vx.getDisp(), _incx, _N);
  auto prdExpr = make_expr<UnaryExpr, prdOp1_struct>(my_vx);

  ContainerT valT1(1);
  auto rs = vector_view<ScalarT, ContainerT>(valT1, 0, 1, 1);
  auto nrm2Expr = make_addReductionExpr(prdExpr);
  auto assignExpr = make_expr<AssignExpr>(rs, nrm2Expr);
  blas::execute(dev, assignExpr);
  return std::sqrt(rs.eval(0));
}

/**
 * \brief NRM2 Returns the euclidian norm of a vector
 * @param Device &dev
 * @param _vx  VectorView
 * @param _incx Increment in X axis
 */
template <typename Device, typename ScalarT, typename ContainerT>
void _nrm2(Device &dev, int _N, vector_view<ScalarT, ContainerT> _vx, int _incx,
           vector_view<ScalarT, ContainerT> _rs) {
  auto my_vx = vector_view<ScalarT, ContainerT>(_vx, _vx.getDisp(), _incx, _N);
  auto my_rs = vector_view<ScalarT, ContainerT>(_rs, _rs.getDisp(), 1, 1);

  auto prdExpr = make_expr<UnaryExpr, prdOp1_struct>(my_vx);
  auto nrm2Expr = make_addReductionExpr(prdExpr);
  auto sqrtExpr = make_expr<UnaryExpr, sqtOp1_struct>(nrm2Expr);
  auto assignExpr = make_expr<AssignExpr>(my_rs, sqrtExpr);
  blas::execute(dev, assignExpr);
}

/**
 * \brief ASUM Takes the sum of the absolute values
 * @param Device &dev
 * @param _vx  VectorView
 * @param _incx Increment in X axis
 */
template <typename Device, typename ScalarT, typename ContainerT>
void _asum(Device &dev, int _N, vector_view<ScalarT, ContainerT> _vx, int _incx,
           vector_view<ScalarT, ContainerT> _rs) {
  auto my_vx = vector_view<ScalarT, ContainerT>(_vx, _vx.getDisp(), _incx, _N);
  auto my_rs = vector_view<ScalarT, ContainerT>(_rs, _rs.getDisp(), 1, 1);
  auto asumExpr = make_addAbsReductionExpr(my_vx);
  auto assignExpr = make_expr<AssignExpr>(my_rs, asumExpr);
  blas::execute(dev, assignExpr);
}

/**
 * \brief IAMAX finds the index of the first element having maximum
 * @param _vx  VectorView
 * @param _incx Increment in X axis
 */
template <typename Device, typename ScalarT, typename ContainerT, typename I,
          typename ContainerI>
void _iamax(Device &dev, int _N, vector_view<ScalarT, ContainerT> _vx,
            int _incx, vector_view<I, ContainerI> _rs) {
  auto my_vx = vector_view<ScalarT, ContainerT>(_vx, _vx.getDisp(), _incx, _N);
  auto my_rs = vector_view<I, ContainerI>(_rs, _rs.getDisp(), 1, 1);
  auto tupExpr = TupleExpr<vector_view<ScalarT, ContainerT>>(my_vx);
  auto iamaxExpr = make_maxIndReductionExpr(tupExpr);
  auto assignExpr = make_expr<AssignExpr>(my_rs, iamaxExpr);
  blas::execute(dev, assignExpr);
}

/**
 * \brief ICAMAX finds the index of the first element having maximum
 * @param _vx  VectorView
 * @param _incx Increment in X axis
 */
template <typename Device, typename T, typename ContainerT>
size_t _iamax(Device &dev, int _N, vector_view<T, ContainerT> _vx, int _incx) {
  IndVal<T> result;
  cl::sycl::buffer<IndVal<T>, 1> buf_result(&result, cl::sycl::range<1>{1});
  auto rs = BufferVectorView<IndVal<T>>(buf_result, 0, 1, 1);
  _iamax(dev, _N, _vx, _incx, rs);
  return rs.eval(0).getInd();
}

/**
 * \brief IAMIN finds the index of the first element having minimum
 * @param _vx  VectorView
 * @param _incx Increment in X axis
 */
template <typename Device, typename T, typename ContainerT, typename I,
          typename ContainerI>
void _iamin(Device &dev, int _N, vector_view<T, ContainerT> _vx, int _incx,
            vector_view<I, ContainerI> _rs) {
  auto my_vx = vector_view<T, ContainerT>(_vx, _vx.getDisp(), _incx, _N);
  auto my_rs = vector_view<I, ContainerI>(_rs, _rs.getDisp(), 1, 1);
  auto tupExpr = TupleExpr<vector_view<T, ContainerT>>(my_vx);
  auto iaminExpr = make_minIndReductionExpr(tupExpr);
  auto assignExpr = make_expr<AssignExpr>(my_rs, iaminExpr);
  blas::execute(dev, assignExpr);
}

/**
 * \brief ICAMIN finds the index of the first element having minimum
 * @param _vx  VectorView
 * @param _incx Increment in X axis
 */
template <typename Device, typename T, typename ContainerT>
size_t _iamin(Device &dev, int _N, vector_view<T, ContainerT> _vx, int _incx) {
  IndVal<T> result;
  auto buf_result =
      cl::sycl::buffer<IndVal<T>, 1>(&result, cl::sycl::range<1>{1});
  auto rs = BufferVectorView<IndVal<T>>(buf_result, 0, 1, 1);
  _iamin(dev, _N, _vx, _incx, rs);
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
template <typename Device, typename T, typename ContainerT>
void _rot(Device &dev, int _N, vector_view<T, ContainerT> _vx, int _incx,
          vector_view<T, ContainerT> _vy, int _incy, T _cos, T _sin) {
  auto my_vx = vector_view<T, ContainerT>(_vx, _vx.getDisp(), _incx, _N);
  auto my_vy = vector_view<T, ContainerT>(_vy, _vy.getDisp(), _incy, _N);
  auto scalExpr1 = make_expr<ScalarExpr, prdOp2_struct>(_cos, my_vx);
  auto scalExpr2 = make_expr<ScalarExpr, prdOp2_struct>(_sin, my_vy);
  auto scalExpr3 = make_expr<ScalarExpr, prdOp2_struct>(-_sin, my_vx);
  auto scalExpr4 = make_expr<ScalarExpr, prdOp2_struct>(_cos, my_vy);
  auto addExpr12 = make_expr<BinaryExpr, addOp2_struct>(scalExpr1, scalExpr2);
  auto addExpr34 = make_expr<BinaryExpr, addOp2_struct>(scalExpr3, scalExpr4);
  auto doubleAssignExpr =
      make_expr<DoubleAssignExpr>(my_vx, my_vy, addExpr12, addExpr34);
  blas::execute(dev, doubleAssignExpr);
}

}  // namespace blas

#endif  // BLAS1_INTERFACE_SYCL_HPP
