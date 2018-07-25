/***************************************************************************
 *
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
 *  @filename blas1_interface.hpp
 *
 **************************************************************************/

#ifndef BLAS1_INTERFACE_HPP
#define BLAS1_INTERFACE_HPP

#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>

#include <sycl-blas/executors/executor_sycl.hpp>
#include <sycl-blas/interface/blas_interface_sycl.hpp>
#include <sycl-blas/operations/blas1_trees.hpp>
#include <sycl-blas/types/sycl_types.hpp>

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
template <typename Executor, typename ContainerT0, typename ContainerT1,
          typename T, typename IndexType, typename IncrementType>
typename Executor::Return_Type _axpy(Executor &ex, IndexType _N, T _alpha,
                                     ContainerT0 &_vx, IncrementType _incx,
                                     ContainerT1 &_vy, IncrementType _incy) {
  auto vx = make_vector_view(ex, _vx, _incx, _N);
  auto vy = make_vector_view(ex, _vy, _incy, _N);

  auto scalOp = make_op<ScalarOp, prdOp2_struct>(_alpha, vx);
  auto addOp = make_op<BinaryOp, addOp2_struct>(vy, scalOp);
  auto assignOp = make_op<Assign>(vy, addOp);
  auto ret = ex.execute(assignOp);
  return ret;
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
template <typename Executor, typename IndexType, typename ContainerT0,
          typename ContainerT1, typename IncrementType>
typename Executor::Return_Type _copy(Executor &ex, IndexType _N,
                                     ContainerT0 _vx, IncrementType _incx,
                                     ContainerT1 _vy, IncrementType _incy) {
  auto vx = make_vector_view(ex, _vx, _incx, _N);
  auto vy = make_vector_view(ex, _vy, _incy, _N);
  auto assignOp2 = make_op<Assign>(vy, vx);
  auto ret = ex.execute(assignOp2);
  return ret;
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
template <typename Executor, typename ContainerT0, typename ContainerT1,
          typename ContainerT2, typename IndexType, typename IncrementType>
typename Executor::Return_Type _dot(Executor &ex, IndexType _N, ContainerT0 _vx,
                                    IncrementType _incx, ContainerT1 _vy,
                                    IncrementType _incy, ContainerT2 _rs) {
  auto vx = make_vector_view(ex, _vx, _incx, _N);
  auto vy = make_vector_view(ex, _vy, _incy, _N);
  auto rs = make_vector_view(ex, _rs, static_cast<IncrementType>(1),
                             static_cast<IndexType>(1));
  auto prdOp = make_op<BinaryOp, prdOp2_struct>(vx, vy);
  // TODO: (Mehdi) read them from the device
  auto localSize = 256;
  auto nWG = 512;
  auto assignOp =
      make_addAssignReduction(rs, prdOp, localSize, localSize * nWG);
  auto ret = ex.reduce(assignOp);
  return ret;
}

/**
 * \brief ASUM Takes the sum of the absolute values
 * @param Executor<ExecutorType> ex
 * @param _vx  VectorView
 * @param _incx Increment in X axis
 */
template <typename Executor, typename ContainerT0, typename ContainerT1,
          typename IndexType, typename IncrementType>
typename Executor::Return_Type _asum(Executor &ex, IndexType _N,
                                     ContainerT0 _vx, IncrementType _incx,
                                     ContainerT1 _rs) {
  auto vx = make_vector_view(ex, _vx, _incx, _N);
  auto rs = make_vector_view(ex, _rs, static_cast<IncrementType>(1),
                             static_cast<IndexType>(1));
  // TODO: (Mehdi) read them from the device
  auto localSize = 256;
  auto nWG = 512;
  auto assignOp =
      make_addAbsAssignReduction(rs, vx, localSize, localSize * nWG);
  auto ret = ex.reduce(assignOp);
  return ret;
}

/**
 * \brief IAMAX finds the index of the first element having maximum
 * @param _vx  VectorView
 * @param _incx Increment in X axis
 */
template <typename Executor, typename ContainerT, typename ContainerI,
          typename IndexType, typename IncrementType>
typename Executor::Return_Type _iamax(Executor &ex, IndexType _N,
                                      ContainerT _vx, IncrementType _incx,
                                      ContainerI _rs) {
  auto vx = make_vector_view(ex, _vx, _incx, _N);
  auto rs = make_vector_view(ex, _rs, static_cast<IncrementType>(1),
                             static_cast<IndexType>(1));
  // TODO: (Mehdi) take this value from device
  size_t localSize = 256, nWG = 512;
  auto tupOp = make_tuple_op(vx);
  auto assignOp =
      make_maxIndAssignReduction(rs, tupOp, localSize, localSize * nWG);
  auto ret = ex.reduce(assignOp);
  return ret;
}

/**
 * \brief IAMIN finds the index of the first element having minimum
 * @param _vx  VectorView
 * @param _incx Increment in X axis
 */
template <typename Executor, typename ContainerT, typename ContainerI,
          typename IndexType, typename IncrementType>
typename Executor::Return_Type _iamin(Executor &ex, IndexType _N,
                                      ContainerT _vx, IncrementType _incx,
                                      ContainerI _rs) {
  auto vx = make_vector_view(ex, _vx, _incx, _N);
  auto rs = make_vector_view(ex, _rs, static_cast<IncrementType>(1),
                             static_cast<IndexType>(1));

  // TODO: (Mehdi) read them from the device
  size_t localSize = 256, nWG = 512;
  auto tupOp = make_tuple_op(vx);
  auto assignOp =
      make_minIndAssignReduction(rs, tupOp, localSize, localSize * nWG);
  auto ret = ex.reduce(assignOp);
  return ret;
}

/**
 * \brief SWAP interchanges two vectors
 *
 * @param Executor ex
 * @param _vx  VectorView
 * @param _incx Increment in X axis
 * @param _vy  VectorView
 * @param _incy Increment in Y axis
 */
template <typename Executor, typename ContainerT0, typename ContainerT1,
          typename IndexType, typename IncrementType>
typename Executor::Return_Type _swap(Executor &ex, IndexType _N,
                                     ContainerT0 _vx, IncrementType _incx,
                                     ContainerT1 _vy, IncrementType _incy) {
  auto vx = make_vector_view(ex, _vx, _incx, _N);
  auto vy = make_vector_view(ex, _vy, _incy, _N);
  auto swapOp = make_op<DobleAssign>(vy, vx, vx, vy);
  auto ret = ex.execute(swapOp);

  return ret;
}

/**
 * \brief SCALAR  operation on a vector
 * @param Executor ex
 * @param _vx  VectorView
 * @param _incx Increment in X axis
 */
template <typename Executor, typename T, typename ContainerT0,
          typename IndexType, typename IncrementType>
typename Executor::Return_Type _scal(Executor &ex, IndexType _N, T _alpha,
                                     ContainerT0 _vx, IncrementType _incx) {
  auto vx = make_vector_view(ex, _vx, _incx, _N);
  auto scalOp = make_op<ScalarOp, prdOp2_struct>(_alpha, vx);
  auto assignOp = make_op<Assign>(vx, scalOp);
  auto ret = ex.execute(assignOp);
  return ret;
}

/**
 * \brief NRM2 Returns the euclidian norm of a vector
 * @param Executor<ExecutorType> ex
 * @param _vx  VectorView
 * @param _incx Increment in X axis
 */
template <typename Executor, typename ContainerT0, typename ContainerT1,
          typename IndexType, typename IncrementType>
typename Executor::Return_Type _nrm2(Executor &ex, IndexType _N,
                                     ContainerT0 _vx, IncrementType _incx,
                                     ContainerT1 _rs) {
  auto vx = make_vector_view(ex, _vx, _incx, _N);
  auto rs = make_vector_view(ex, _rs, static_cast<IncrementType>(1),
                             static_cast<IndexType>(1));
  auto prdOp = make_op<UnaryOp, prdOp1_struct>(vx);
  // TODO: (Mehdi) read them from the deivce
  auto localSize = 256;
  auto nWG = 512;
  auto assignOp =
      make_addAssignReduction(rs, prdOp, localSize, localSize * nWG);
  ex.reduce(assignOp);
  auto sqrtOp = make_op<UnaryOp, sqtOp1_struct>(rs);
  auto assignOpFinal = make_op<Assign>(rs, sqrtOp);
  auto ret = ex.execute(assignOpFinal);
  return ret;
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
template <typename Executor, typename ContainerT0, typename ContainerT1,
          typename T, typename IndexType, typename IncrementType>
typename Executor::Return_Type _rot(Executor &ex, IndexType _N, ContainerT0 _vx,
                                    IncrementType _incx, ContainerT1 _vy,
                                    IncrementType _incy, T _cos, T _sin) {
  auto vx = make_vector_view(ex, _vx, _incx, _N);
  auto vy = make_vector_view(ex, _vy, _incy, _N);
  auto scalOp1 = make_op<ScalarOp, prdOp2_struct>(_cos, vx);
  auto scalOp2 = make_op<ScalarOp, prdOp2_struct>(_sin, vy);
  auto scalOp3 = make_op<ScalarOp, prdOp2_struct>(-_sin, vx);
  auto scalOp4 = make_op<ScalarOp, prdOp2_struct>(_cos, vy);
  auto addOp12 = make_op<BinaryOp, addOp2_struct>(scalOp1, scalOp2);
  auto addOp34 = make_op<BinaryOp, addOp2_struct>(scalOp3, scalOp4);
  auto dobleAssignView = make_op<DobleAssign>(vx, vy, addOp12, addOp34);
  auto ret = ex.execute(dobleAssignView);
  return ret;
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

/**
 * \brief Compute the inner product of two vectors with extended
    precision accumulation and result.
 *
 * @param Executor<ExecutorType> ex
 * @param _vx  VectorView
 * @param _incx Increment in X axis
 * @param _vx  VectorView
 * @param _incy Increment in Y axis
 */
template <typename Executor, typename ContainerT0, typename ContainerT1,
          typename IndexType, typename IncrementType>
typename scalar_type<ContainerT0>::ScalarT _dot(Executor &ex, IndexType _N,
                                                ContainerT0 _vx,
                                                IncrementType _incx,
                                                ContainerT1 _vy,
                                                IncrementType _incy) {
  using T = typename scalar_type<ContainerT0>::ScalarT;
  auto res = std::vector<T>(1);
  auto gpu_res = helper::make_sycl_iteator_buffer<T>(static_cast<IndexType>(1));
  _dot(ex, _N, _vx, _incx, _vy, _incy, gpu_res);
  gpu_res.copy_to_host(ex, res.data());
  return res[0];
}

/**
 * \brief ICAMAX finds the index of the first element having maximum
 * @param _vx  VectorView
 * @param _incx Increment in X axis
 */
template <typename Executor, typename ContainerT, typename IndexType,
          typename IncrementType>
IndexType _iamax(Executor &ex, IndexType _N, ContainerT _vx,
                 IncrementType _incx) {
  using T = typename scalar_type<ContainerT>::ScalarT;
  using IndValTuple = IndexValueTuple<T>;
  std::vector<IndValTuple> rsT(1);
  auto gpu_res =
      helper::make_sycl_iteator_buffer<IndValTuple>(static_cast<IndexType>(1));
  _iamax(ex, _N, _vx, _incx, gpu_res);
  gpu_res.copy_to_host(ex, rsT.data());
  return rsT[0].get_index();
}

/**
 * \brief ICAMIN finds the index of the first element having minimum
 * @param _vx  VectorView
 * @param _incx Increment in X axis
 */
template <typename Executor, typename ContainerT, typename IndexType,
          typename IncrementType>
IndexType _iamin(Executor &ex, IndexType _N, ContainerT _vx,
                 IncrementType _incx) {
  using T = typename scalar_type<ContainerT>::ScalarT;
  using IndValTuple = IndexValueTuple<T>;
  std::vector<IndValTuple> rsT(1);
  auto gpu_res =
      helper::make_sycl_iteator_buffer<IndValTuple>(static_cast<IndexType>(1));
  _iamin(ex, _N, _vx, _incx, gpu_res);
  gpu_res.copy_to_host(ex, rsT.data());
  return rsT[0].get_index();
}

/**
 * \brief ASUM Takes the sum of the absolute values
 *
 * @param Executor<ExecutorType> ex
 * @param _vx  VectorView
 * @param _incx Increment in X axis
 */
template <typename Executor, typename ContainerT, typename IndexType,
          typename IncrementType>
typename scalar_type<ContainerT>::ScalarT _asum(Executor &ex, IndexType _N,
                                                ContainerT _vx,
                                                IncrementType _incx) {
  using T = typename scalar_type<ContainerT>::ScalarT;
  auto res = std::vector<T>(1, T(0));
  auto gpu_res = helper::make_sycl_iteator_buffer<T>(static_cast<IndexType>(1));
  _asum(ex, _N, _vx, _incx, gpu_res);
  gpu_res.copy_to_host(ex, res.data());
  return res[0];
}

/**
 * \brief NRM2 Returns the euclidian norm of a vector
 *
 * @param Executor<ExecutorType> ex
 * @param _vx  VectorView
 * @param _incx Increment in X axis
 */
template <typename Executor, typename ContainerT, typename IndexType,
          typename IncrementType>
typename scalar_type<ContainerT>::ScalarT _nrm2(Executor &ex, IndexType _N,
                                                ContainerT _vx,
                                                IncrementType _incx) {
  using T = typename scalar_type<ContainerT>::ScalarT;
  auto res = std::vector<T>(1, T(0));
  auto gpu_res = helper::make_sycl_iteator_buffer<T>(static_cast<IndexType>(1));
  _nrm2(ex, _N, _vx, _incx, gpu_res);
  gpu_res.copy_to_host(ex, res.data());
  return res[0];
}

}  // namespace blas

#endif  // BLAS1_INTERFACE_HPP
