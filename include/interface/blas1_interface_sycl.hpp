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

#define RETURNCXX11(expr) \
  ->decltype(expr) { return expr; }

/**
 * \brief COPY copies a vector, x, to a vector, y.
 *
 * @param Device &dev
 * @param _N Number of elements
 * @param _x  X
 * @param _offx Offset in X axis
 * @param _incx Increment in X axis
 * @param _y  Y
 * @param _offy Offset in Y axis
 * @param _incy Increment in Y axis
 */
template <typename X, typename Y>
auto _copy(int _N, X _x, int _offx, int _incx, Y _y, int _offy, int _incy) RETURNCXX11(
  make_expr<AssignExpr>(
    make_strdExpr(_x, _offx, _incx, _N),
    make_strdExpr(_y, _offy, _incy, _N))
)

/**
 * \brief SWAP interchanges two vectors
 *
 * @param Device &dev
 * @param _N Number of elements
 * @param _x X
 * @param _offx Offset in X axis
 * @param _incx Increment in X axis
 * @param _y Y
 * @param _offy Offset in Y axis
 * @param _incy Increment in Y axis
 */
template <typename X, typename Y>
auto _swap(int _N, X _x, int _offx, int _incx, Y _y, int _offy, int _incy) RETURNCXX11(
  make_expr<DoubleAssignExpr>(
    make_strdExpr(_y, _offy, _incy, _N),
    make_strdExpr(_x, _offx, _incx, _N),
    make_strdExpr(_x, _offx, _incx, _N),
    make_strdExpr(_y, _offy, _incy, _N))
)

/**
 * \brief SCAL scales a vector by a constant
 *
 * @param Device &dev
 * @param _N Number of elements
 * @param _x X
 * @param _offx Offset in X axis
 * @param _incx Increment in X axis
 */
template <typename ScalarT, typename X>
auto _scal(int _N, ScalarT _alpha, X _x, int _offx, int _incx) RETURNCXX11(
  make_expr<AssignExpr>(
    make_strdExpr(_x, _offx, _incx, _N),
    make_expr<ScalarExpr, prdOp2_struct>(
      _alpha,
      make_strdExpr(_x, _offx, _incx, _N)))
)

/**
 * \brief AXPY constant times a vector plus a vector.
 *
 * Implements AXPY \f$y = ax + y\f$
 *
 * @param Device &dev
 * @param _N Number of elements
 * @param _x X
 * @param _offx Offset in X axis
 * @param _incx Increment in X axis
 * @param _y Y
 * @param _offy Offset in Y axis
 * @param _incy Increment in Y axis
 */
template <typename ScalarT, typename X, typename Y>
auto _axpy(int _N, ScalarT _alpha, X _x, int _offx, int _incx, Y _y, int _offy, int _incy) RETURNCXX11(
  make_expr<AssignExpr>(
    make_strdExpr(_y, _offy, _incy, _N),
    make_expr<BinaryExpr, addOp2_struct>(
      make_strdExpr(_y, _offx, _incx, _N),
      make_expr<ScalarExpr, prdOp2_struct>(
        _alpha,
        make_strdExpr(_x, _offy, _incy, _N))))
)

/**
 * \brief Compute the inner product of two vectors with extended precision
 * accumulation.
 * @param Device &dev
 * @param _N Number of elements
 * @param _x X
 * @param _offx Offset in X axis
 * @param _incx Increment in X axis
 * @param _Y Y
 * @param _offy Offset in Y axis
 * @param _incy Increment in Y axis
 * @param _rs SYCL buffer for the result.
 */
template <typename X, typename Y, typename ScalarT>
auto _dot(int _N, X _x, int _offx, int _incx, Y _y, int _offy, int _incy, cl::sycl::buffer<ScalarT, 1> _rs) RETURNCXX11(
  make_expr<AssignExpr>(
    vector_view<ScalarT, cl::sycl::buffer<ScalarT, 1>>(_rs, 0, 1, 1),
    make_addReductionExpr(
      make_expr<BinaryExpr, prdOp2_struct>(
        make_strdExpr(_x, _offx, _incx, _N),
        make_strdExpr(_y, _offy, _incy, _N))))
)

/**
 * \brief NRM2 Returns the euclidian norm of a vector
 * @param Device &dev
 * @param _N Number of elements
 * @param _x X
 * @param _offx Offset in X axis
 * @param _incx Increment in X axis
 * @param _rs SYCL buffer for the result
 */
template <typename X, typename ScalarT>
auto _nrm2(int _N, X _x, int _offx, int _incx, cl::sycl::buffer<ScalarT, 1> _rs) RETURNCXX11(
  make_expr<AssignExpr>(
    vector_view<ScalarT, cl::sycl::buffer<ScalarT, 1>>(_rs, 0, 1, 1),
    make_expr<UnaryExpr, sqtOp1_struct>(
      make_addReductionExpr(
        make_expr<UnaryExpr, prdOp1_struct>(
          make_strdExpr(_x, _offx, _incx, _N)))))
)

/**
 * \brief ASUM Takes the sum of the absolute values
 * @param Device &dev
 * @param _N Number of elements
 * @param _x X
 * @param _offx Offset in X axis
 * @param _incx Increment in X axis
 * @param _rs SYCL buffer for the result
 */
template <typename X, typename ScalarT>
auto _asum(int _N, X _x, int _offx, int _incx, cl::sycl::buffer<ScalarT, 1> _rs) RETURNCXX11(
  make_expr<AssignExpr>(
    vector_view<ScalarT, cl::sycl::buffer<ScalarT, 1>>(_rs, 0, 1, 1),
    make_addAbsReductionExpr(
      make_strdExpr(_x, _offx, _incx, _N)))
)

/**
 * \brief IAMAX finds the index of the first element having maximum
 * @param _N Number of elements
 * @param _x X
 * @param _offx Offset in X axis
 * @param _incx Increment in X axis
 * @param _rs SYCL buffer for the result
 */
template <typename X, typename ScalarI>
auto _iamax(int _N, X _x, int _offx, int _incx, cl::sycl::buffer<ScalarI, 1> _rs) RETURNCXX11(
  make_expr<AssignExpr>(
    vector_view<ScalarI, cl::sycl::buffer<ScalarI, 1>>(_rs, 0, 1, 1),
    make_maxIndReductionExpr(
      make_tplExpr(
        make_strdExpr(_x, _offx, _incx, _N))))
)

/**
 * \brief IAMIN finds the index of the first element having minimum
 * @param _N Number of elements
 * @param _x X
 * @param _offx Offset in X axis
 * @param _incx Increment in X axis
 * @param _rs SYCL buffer for the result
 */
template <typename X, typename ScalarI>
auto _iamin(int _N, X _x, int _offx, int _incx, cl::sycl::buffer<ScalarI, 1> _rs) RETURNCXX11(
  make_expr<AssignExpr>(
    vector_view<ScalarI, cl::sycl::buffer<ScalarI, 1>>(_rs, 0, 1, 1),
    make_minIndReductionExpr(
      make_tplExpr(
        make_strdExpr(_x, _offx, _incx, _N))))
)

/**
 * ROTG.
 * @brief Consturcts given plane rotation
 * Not implemented.
 */
template <typename ScalarT>
void _rotg(ScalarT &_alpha, ScalarT &_beta, ScalarT &_cos, ScalarT &_sin) {
  ScalarT abs_alpha = std::abs(_alpha);
  ScalarT abs_beta = std::abs(_beta);
  ScalarT roe = (abs_alpha > abs_beta) ? _alpha : _beta;
  ScalarT scale = abs_alpha + abs_beta;
  ScalarT norm;
  ScalarT aux;

  if (scale == constant<ScalarT, const_val::zero>::value) {
    _cos = constant<ScalarT, const_val::one>::value;
    _sin = constant<ScalarT, const_val::zero>::value;
    norm = constant<ScalarT, const_val::zero>::value;
    aux = constant<ScalarT, const_val::zero>::value;
  } else {
    norm = scale * std::sqrt((_alpha / scale) * (_alpha / scale) +
                             (_beta / scale) * (_beta / scale));
    if (roe < constant<ScalarT, const_val::zero>::value) norm = -norm;
    _cos = _alpha / norm;
    _sin = _beta / norm;
    if (abs_alpha > abs_beta) {
      aux = _sin;
    } else if (_cos != constant<ScalarT, const_val::zero>::value) {
      aux = constant<ScalarT, const_val::one>::value / _cos;
    } else {
      aux = constant<ScalarT, const_val::one>::value;
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
template <typename X, typename Y, typename ScalarT>
auto _rot(int _N, X _x, int _offx, int _incx, Y _y, int _offy, int _incy, ScalarT _cos, ScalarT _sin) RETURNCXX11(
  make_expr<DoubleAssignExpr>(
    make_strdExpr(_x, _offx, _incx, _N),
    make_strdExpr(_y, _offy, _incy, _N),
    make_expr<BinaryExpr, addOp2_struct>(
      make_expr<ScalarExpr, prdOp2_struct>(
        _cos, make_strdExpr(_x, _offx, _incx, _N)),
      make_expr<ScalarExpr, prdOp2_struct>(
        _sin, make_strdExpr(_y, _offy, _incy, _N))),
    make_expr<BinaryExpr, addOp2_struct>(
      make_expr<ScalarExpr, prdOp2_struct>(
        -_sin, make_strdExpr(_x, _offx, _incx, _N)),
      make_expr<ScalarExpr, prdOp2_struct>(
        _cos, make_strdExpr(_y, _offy, _incy, _N))))
)

}  // namespace blas

#endif  // BLAS1_INTERFACE_SYCL_HPP
