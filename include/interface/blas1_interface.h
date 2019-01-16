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
 *  @filename blas1_interface.h
 *
 **************************************************************************/
#ifndef SYCL_BLAS_BLAS1_INTERFACE_H
#define SYCL_BLAS_BLAS1_INTERFACE_H
#include "blas_meta.h"

namespace blas {
namespace internal {
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
typename Executor::Policy::event_type _axpy(Executor &ex, IndexType _N,
                                            T _alpha, ContainerT0 _vx,
                                            IncrementType _incx,
                                            ContainerT1 _vy,
                                            IncrementType _incy);

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
typename Executor::Policy::event_type _copy(Executor &ex, IndexType _N,
                                            ContainerT0 _vx,
                                            IncrementType _incx,
                                            ContainerT1 _vy,
                                            IncrementType _incy);

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
typename Executor::Policy::event_type _dot(Executor &ex, IndexType _N,
                                           ContainerT0 _vx, IncrementType _incx,
                                           ContainerT1 _vy, IncrementType _incy,
                                           ContainerT2 _rs);
/**
 * \brief ASUM Takes the sum of the absolute values
 * @param Executor<ExecutorType> ex
 * @param _vx  VectorView
 * @param _incx Increment in X axis
 */
template <typename Executor, typename ContainerT0, typename ContainerT1,
          typename IndexType, typename IncrementType>
typename Executor::Policy::event_type _asum(Executor &ex, IndexType _N,
                                            ContainerT0 _vx,
                                            IncrementType _incx,
                                            ContainerT1 _rs);
/**
 * \brief IAMAX finds the index of the first element having maximum
 * @param _vx  VectorView
 * @param _incx Increment in X axis
 */
template <typename Executor, typename ContainerT, typename ContainerI,
          typename IndexType, typename IncrementType>
typename Executor::Policy::event_type _iamax(Executor &ex, IndexType _N,
                                             ContainerT _vx,
                                             IncrementType _incx,
                                             ContainerI _rs);
/**
 * \brief IAMIN finds the index of the first element having minimum
 * @param _vx  VectorView
 * @param _incx Increment in X axis
 */
template <typename Executor, typename ContainerT, typename ContainerI,
          typename IndexType, typename IncrementType>
typename Executor::Policy::event_type _iamin(Executor &ex, IndexType _N,
                                             ContainerT _vx,
                                             IncrementType _incx,
                                             ContainerI _rs);

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
typename Executor::Policy::event_type _swap(Executor &ex, IndexType _N,
                                            ContainerT0 _vx,
                                            IncrementType _incx,
                                            ContainerT1 _vy,
                                            IncrementType _incy);

/**
 * \brief SCALAR  operation on a vector
 * @param Executor ex
 * @param _vx  VectorView
 * @param _incx Increment in X axis
 */
template <typename Executor, typename T, typename ContainerT0,
          typename IndexType, typename IncrementType>
typename Executor::Policy::event_type _scal(Executor &ex, IndexType _N,
                                            T _alpha, ContainerT0 _vx,
                                            IncrementType _incx);

/**
 * \brief NRM2 Returns the euclidian norm of a vector
 * @param Executor<ExecutorType> ex
 * @param _vx  VectorView
 * @param _incx Increment in X axis
 */
template <typename Executor, typename ContainerT0, typename ContainerT1,
          typename IndexType, typename IncrementType>
typename Executor::Policy::event_type _nrm2(Executor &ex, IndexType _N,
                                            ContainerT0 _vx,
                                            IncrementType _incx,
                                            ContainerT1 _rs);

/**
 * @brief _rot constructor given plane rotation
 * @param Executor<ExecutorType> ex
 * @param _vx  VectorView
 * @param _incx Increment in X axis
 * @param _vx  VectorView
 * @param _incy Increment in Y axis
 * @param _sin  sine
 * @param _cos cosine
 * @param _N data size
 */
template <typename Executor, typename ContainerT0, typename ContainerT1,
          typename T, typename IndexType, typename IncrementType>
typename Executor::Policy::event_type _rot(Executor &ex, IndexType _N,
                                           ContainerT0 _vx, IncrementType _incx,
                                           ContainerT1 _vy, IncrementType _incy,
                                           T _cos, T _sin);

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
typename scalar_type<ContainerT0>::type _dot(Executor &ex, IndexType _N,
                                             ContainerT0 _vx,
                                             IncrementType _incx,
                                             ContainerT1 _vy,
                                             IncrementType _incy);
/**
 * \brief ICAMAX finds the index of the first element having maximum
 * @param _vx  VectorView
 * @param _incx Increment in X axis
 */
template <typename Executor, typename ContainerT, typename IndexType,
          typename IncrementType>
IndexType _iamax(Executor &ex, IndexType _N, ContainerT _vx,
                 IncrementType _incx);

/**
 * \brief ICAMIN finds the index of the first element having minimum
 * @param _vx  VectorView
 * @param _incx Increment in X axis
 */
template <typename Executor, typename ContainerT, typename IndexType,
          typename IncrementType>
IndexType _iamin(Executor &ex, IndexType _N, ContainerT _vx,
                 IncrementType _incx);

/**
 * \brief ASUM Takes the sum of the absolute values
 *
 * @param Executor<ExecutorType> ex
 * @param _vx  VectorView
 * @param _incx Increment in X axis
 */
template <typename Executor, typename ContainerT, typename IndexType,
          typename IncrementType>
typename scalar_type<ContainerT>::type _asum(Executor &ex, IndexType _N,
                                             ContainerT _vx,
                                             IncrementType _incx);

/**
 * \brief NRM2 Returns the euclidian norm of a vector
 *
 * @param Executor<ExecutorType> ex
 * @param _vx  VectorView
 * @param _incx Increment in X axis
 */
template <typename Executor, typename ContainerT, typename IndexType,
          typename IncrementType>
typename scalar_type<ContainerT>::type _nrm2(Executor &ex, IndexType _N,
                                             ContainerT _vx,
                                             IncrementType _incx);
}  // namespace internal

template <typename Executor, typename ContainerT0, typename ContainerT1,
          typename T, typename IndexType, typename IncrementType>
typename Executor::Policy::event_type _axpy(Executor &ex, IndexType _N,
                                            T _alpha, ContainerT0 _vx,
                                            IncrementType _incx,
                                            ContainerT1 _vy,
                                            IncrementType _incy) {
  return internal::_axpy(ex, _N, _alpha,
                         ex.get_policy_handler().get_buffer(_vx), _incx,
                         ex.get_policy_handler().get_buffer(_vy), _incy);
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
typename Executor::Policy::event_type _copy(Executor &ex, IndexType _N,
                                            ContainerT0 _vx,
                                            IncrementType _incx,
                                            ContainerT1 _vy,
                                            IncrementType _incy) {
  return internal::_copy(ex, _N, ex.get_policy_handler().get_buffer(_vx), _incx,
                         ex.get_policy_handler().get_buffer(_vy), _incy);
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
typename Executor::Policy::event_type _dot(Executor &ex, IndexType _N,
                                           ContainerT0 _vx, IncrementType _incx,
                                           ContainerT1 _vy, IncrementType _incy,
                                           ContainerT2 _rs) {
  return internal::_dot(ex, _N, ex.get_policy_handler().get_buffer(_vx), _incx,
                        ex.get_policy_handler().get_buffer(_vy), _incy,
                        ex.get_policy_handler().get_buffer(_rs));
}

/**
 * \brief ASUM Takes the sum of the absolute values
 * @param Executor<ExecutorType> ex
 * @param _vx  VectorView
 * @param _incx Increment in X axis
 */
template <typename Executor, typename ContainerT0, typename ContainerT1,
          typename IndexType, typename IncrementType>
typename Executor::Policy::event_type _asum(Executor &ex, IndexType _N,
                                            ContainerT0 _vx,
                                            IncrementType _incx,
                                            ContainerT1 _rs) {
  return internal::_asum(ex, _N, ex.get_policy_handler().get_buffer(_vx), _incx,
                         ex.get_policy_handler().get_buffer(_rs));
}

/**
 * \brief IAMAX finds the index of the first element having maximum
 * @param _vx  VectorView
 * @param _incx Increment in X axis
 */
template <typename Executor, typename ContainerT, typename ContainerI,
          typename IndexType, typename IncrementType>
typename Executor::Policy::event_type _iamax(Executor &ex, IndexType _N,
                                             ContainerT _vx,
                                             IncrementType _incx,
                                             ContainerI _rs) {
  return internal::_iamax(ex, _N, ex.get_policy_handler().get_buffer(_vx),
                          _incx, ex.get_policy_handler().get_buffer(_rs));
}

/**
 * \brief IAMIN finds the index of the first element having minimum
 * @param _vx  VectorView
 * @param _incx Increment in X axis
 */
template <typename Executor, typename ContainerT, typename ContainerI,
          typename IndexType, typename IncrementType>
typename Executor::Policy::event_type _iamin(Executor &ex, IndexType _N,
                                             ContainerT _vx,
                                             IncrementType _incx,
                                             ContainerI _rs) {
  return internal::_iamin(ex, _N, ex.get_policy_handler().get_buffer(_vx),
                          _incx, ex.get_policy_handler().get_buffer(_rs));
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
typename Executor::Policy::event_type _swap(Executor &ex, IndexType _N,
                                            ContainerT0 _vx,
                                            IncrementType _incx,
                                            ContainerT1 _vy,
                                            IncrementType _incy) {
  return internal::_swap(ex, _N, ex.get_policy_handler().get_buffer(_vx), _incx,
                         ex.get_policy_handler().get_buffer(_vy), _incy);
}

/**
 * \brief SCALAR  operation on a vector
 * @param Executor ex
 * @param _vx  VectorView
 * @param _incx Increment in X axis
 */
template <typename Executor, typename T, typename ContainerT0,
          typename IndexType, typename IncrementType>
typename Executor::Policy::event_type _scal(Executor &ex, IndexType _N,
                                            T _alpha, ContainerT0 _vx,
                                            IncrementType _incx) {
  return internal::_scal(ex, _N, _alpha,
                         ex.get_policy_handler().get_buffer(_vx), _incx);
}

/**
 * \brief NRM2 Returns the euclidian norm of a vector
 * @param Executor<ExecutorType> ex
 * @param _vx  VectorView
 * @param _incx Increment in X axis
 */
template <typename Executor, typename ContainerT0, typename ContainerT1,
          typename IndexType, typename IncrementType>
typename Executor::Policy::event_type _nrm2(Executor &ex, IndexType _N,
                                            ContainerT0 _vx,
                                            IncrementType _incx,
                                            ContainerT1 _rs) {
  return internal::_nrm2(ex, _N, ex.get_policy_handler().get_buffer(_vx), _incx,
                         ex.get_policy_handler().get_buffer(_rs));
}

/**
 * .
 * @brief _rot constructor given plane rotation
 *  *
 * @param Executor<ExecutorType> ex
 * @param _vx  VectorView
 * @param _incx Increment in X axis
 * @param _vx  VectorView
 * @param _incy Increment in Y axis
 * @param _sin  sine
 * @param _cos cosine
 * @param _N data size
 *
 */
template <typename Executor, typename ContainerT0, typename ContainerT1,
          typename T, typename IndexType, typename IncrementType>
typename Executor::Policy::event_type _rot(Executor &ex, IndexType _N,
                                           ContainerT0 _vx, IncrementType _incx,
                                           ContainerT1 _vy, IncrementType _incy,
                                           T _cos, T _sin) {
  return internal::_rot(ex, _N, ex.get_policy_handler().get_buffer(_vx), _incx,
                        ex.get_policy_handler().get_buffer(_vy), _incy, _cos,
                        _sin);
}

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
typename scalar_type<ContainerT0>::type _dot(Executor &ex, IndexType _N,
                                             ContainerT0 _vx,
                                             IncrementType _incx,
                                             ContainerT1 _vy,
                                             IncrementType _incy) {
  return internal::_dot(ex, _N, ex.get_policy_handler().get_buffer(_vx), _incx,
                        ex.get_policy_handler().get_buffer(_vy), _incy);
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
  return internal::_iamax(ex, _N, ex.get_policy_handler().get_buffer(_vx),
                          _incx);
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
  return internal::_iamin(ex, _N, ex.get_policy_handler().get_buffer(_vx),
                          _incx);
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
typename scalar_type<ContainerT>::type _asum(Executor &ex, IndexType _N,
                                             ContainerT _vx,
                                             IncrementType _incx) {
  return internal::_asum(ex, _N, ex.get_policy_handler().get_buffer(_vx),
                         _incx);
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
typename scalar_type<ContainerT>::type _nrm2(Executor &ex, IndexType _N,
                                             ContainerT _vx,
                                             IncrementType _incx) {
  return internal::_nrm2(ex, _N, ex.get_policy_handler().get_buffer(_vx),
                         _incx);
}

}  // end namespace blas
#endif  // SYCL_BLAS_BLAS1_INTERFACE
