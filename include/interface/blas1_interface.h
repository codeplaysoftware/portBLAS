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
 * @param ex Executor
 * @param _vx BufferIterator
 * @param _incx Increment for the vector X
 * @param _vy BufferIterator
 * @param _incy Increment for the vector Y
 */
template <typename executor_t, typename container_0_t, typename container_1_t,
          typename element_t, typename index_t, typename increment_t>
typename executor_t::policy_t::event_t _axpy(
    executor_t &ex, index_t _N, element_t _alpha, container_0_t _vx,
    increment_t _incx, container_1_t _vy, increment_t _incy);

/**
 * \brief COPY copies a vector, x, to a vector, y.
 *
 * @param ex Executor
 * @param _vx BufferIterator
 * @param _incx Increment for the vector X
 * @param _vy BufferIterator
 * @param _incy Increment for the vector Y
 */
template <typename executor_t, typename index_t, typename container_0_t,
          typename container_1_t, typename increment_t>
typename executor_t::policy_t::event_t _copy(executor_t &ex, index_t _N,
                                             container_0_t _vx,
                                             increment_t _incx,
                                             container_1_t _vy,
                                             increment_t _incy);

/**
 * \brief Compute the inner product of two vectors with extended precision
    accumulation.
 * @param ex Executor
 * @param _vx BufferIterator
 * @param _incx Increment for the vector X
 * @param _vx BufferIterator
 * @param _incy Increment for the vector Y
 */
template <typename executor_t, typename container_0_t, typename container_1_t,
          typename container_2_t, typename index_t, typename increment_t>
typename executor_t::policy_t::event_t _dot(
    executor_t &ex, index_t _N, container_0_t _vx, increment_t _incx,
    container_1_t _vy, increment_t _incy, container_2_t _rs);
/**
 * \brief ASUM Takes the sum of the absolute values
 * @param ex Executor
 * @param _vx BufferIterator
 * @param _incx Increment for the vector X
 */
template <typename executor_t, typename container_0_t, typename container_1_t,
          typename index_t, typename increment_t>
typename executor_t::policy_t::event_t _asum(executor_t &ex, index_t _N,
                                             container_0_t _vx,
                                             increment_t _incx,
                                             container_1_t _rs);
/**
 * \brief IAMAX finds the index of the first element having maximum
 * @param _vx BufferIterator
 * @param _incx Increment for the vector X
 */
template <typename executor_t, typename container_t, typename ContainerI,
          typename index_t, typename increment_t>
typename executor_t::policy_t::event_t _iamax(executor_t &ex, index_t _N,
                                              container_t _vx,
                                              increment_t _incx,
                                              ContainerI _rs);
/**
 * \brief IAMIN finds the index of the first element having minimum
 * @param _vx BufferIterator
 * @param _incx Increment for the vector X
 */
template <typename executor_t, typename container_t, typename ContainerI,
          typename index_t, typename increment_t>
typename executor_t::policy_t::event_t _iamin(executor_t &ex, index_t _N,
                                              container_t _vx,
                                              increment_t _incx,
                                              ContainerI _rs);

/**
 * \brief SWAP interchanges two vectors
 *
 * @param executor_t ex
 * @param _vx BufferIterator
 * @param _incx Increment for the vector X
 * @param _vy BufferIterator
 * @param _incy Increment for the vector Y
 */
template <typename executor_t, typename container_0_t, typename container_1_t,
          typename index_t, typename increment_t>
typename executor_t::policy_t::event_t _swap(executor_t &ex, index_t _N,
                                             container_0_t _vx,
                                             increment_t _incx,
                                             container_1_t _vy,
                                             increment_t _incy);

/**
 * \brief SCALAR  operation on a vector
 * @param executor_t ex
 * @param _vx BufferIterator
 * @param _incx Increment for the vector X
 */
template <typename executor_t, typename element_t, typename container_0_t,
          typename index_t, typename increment_t>
typename executor_t::policy_t::event_t _scal(executor_t &ex, index_t _N,
                                             element_t _alpha,
                                             container_0_t _vx,
                                             increment_t _incx);

/**
 * \brief NRM2 Returns the euclidian norm of a vector
 * @param ex Executor
 * @param _vx BufferIterator
 * @param _incx Increment for the vector X
 */
template <typename executor_t, typename container_0_t, typename container_1_t,
          typename index_t, typename increment_t>
typename executor_t::policy_t::event_t _nrm2(executor_t &ex, index_t _N,
                                             container_0_t _vx,
                                             increment_t _incx,
                                             container_1_t _rs);

/**
 * @brief _rot constructor given plane rotation
 * @param ex Executor
 * @param _vx BufferIterator
 * @param _incx Increment for the vector X
 * @param _vx BufferIterator
 * @param _incy Increment for the vector Y
 * @param _sin sine
 * @param _cos cosine
 * @param _N data size
 */
template <typename executor_t, typename container_0_t, typename container_1_t,
          typename element_t, typename index_t, typename increment_t>
typename executor_t::policy_t::event_t _rot(
    executor_t &ex, index_t _N, container_0_t _vx, increment_t _incx,
    container_1_t _vy, increment_t _incy, element_t _cos, element_t _sin);

/**
 * \brief Compute the inner product of two vectors with extended
    precision accumulation and result.
 *
 * @param ex Executor
 * @param _vx BufferIterator
 * @param _incx Increment for the vector X
 * @param _vx BufferIterator
 * @param _incy Increment for the vector Y
 */
template <typename executor_t, typename container_0_t, typename container_1_t,
          typename index_t, typename increment_t>
typename ValueType<container_0_t>::type _dot(executor_t &ex, index_t _N,
                                             container_0_t _vx,
                                             increment_t _incx,
                                             container_1_t _vy,
                                             increment_t _incy);
/**
 * \brief ICAMAX finds the index of the first element having maximum
 * @param _vx BufferIterator
 * @param _incx Increment for the vector X
 */
template <typename executor_t, typename container_t, typename index_t,
          typename increment_t>
index_t _iamax(executor_t &ex, index_t _N, container_t _vx, increment_t _incx);

/**
 * \brief ICAMIN finds the index of the first element having minimum
 * @param _vx BufferIterator
 * @param _incx Increment for the vector X
 */
template <typename executor_t, typename container_t, typename index_t,
          typename increment_t>
index_t _iamin(executor_t &ex, index_t _N, container_t _vx, increment_t _incx);

/**
 * \brief ASUM Takes the sum of the absolute values
 *
 * @param ex Executor
 * @param _vx BufferIterator
 * @param _incx Increment for the vector X
 */
template <typename executor_t, typename container_t, typename index_t,
          typename increment_t>
typename ValueType<container_t>::type _asum(executor_t &ex, index_t _N,
                                            container_t _vx, increment_t _incx);

/**
 * \brief NRM2 Returns the euclidian norm of a vector
 *
 * @param ex Executor
 * @param _vx BufferIterator
 * @param _incx Increment for the vector X
 */
template <typename executor_t, typename container_t, typename index_t,
          typename increment_t>
typename ValueType<container_t>::type _nrm2(executor_t &ex, index_t _N,
                                            container_t _vx, increment_t _incx);
}  // namespace internal

template <typename executor_t, typename container_0_t, typename container_1_t,
          typename element_t, typename index_t, typename increment_t>
typename executor_t::policy_t::event_t _axpy(
    executor_t &ex, index_t _N, element_t _alpha, container_0_t _vx,
    increment_t _incx, container_1_t _vy, increment_t _incy) {
#ifdef SYCL_BLAS_USE_USM
  return internal::_axpy(ex, _N, _alpha, _vx, _incx, _vy, _incy);
#else
  return internal::_axpy(ex, _N, _alpha,
                         ex.get_policy_handler().get_buffer(_vx), _incx,
                         ex.get_policy_handler().get_buffer(_vy), _incy);
#endif
}

/**
 * \brief COPY copies a vector, x, to a vector, y.
 *
 * @param ex Executor
 * @param _vx BufferIterator
 * @param _incx Increment for the vector X
 * @param _vy BufferIterator
 * @param _incy Increment for the vector Y
 */
template <typename executor_t, typename index_t, typename container_0_t,
          typename container_1_t, typename increment_t>
typename executor_t::policy_t::event_t _copy(executor_t &ex, index_t _N,
                                             container_0_t _vx,
                                             increment_t _incx,
                                             container_1_t _vy,
                                             increment_t _incy) {
#ifdef SYCL_BLAS_USE_USM
  return internal::_copy(ex, _N, _vx, _incx, _vy, _incy);
#else
  return internal::_copy(ex, _N, ex.get_policy_handler().get_buffer(_vx), _incx,
                         ex.get_policy_handler().get_buffer(_vy), _incy);
#endif
}

/**
 * \brief Compute the inner product of two vectors with extended precision
    accumulation.
 * @param ex Executor
 * @param _vx BufferIterator
 * @param _incx Increment for the vector X
 * @param _vx BufferIterator
 * @param _incy Increment for the vector Y
 */
template <typename executor_t, typename container_0_t, typename container_1_t,
          typename container_2_t, typename index_t, typename increment_t>
typename executor_t::policy_t::event_t _dot(
    executor_t &ex, index_t _N, container_0_t _vx, increment_t _incx,
    container_1_t _vy, increment_t _incy, container_2_t _rs) {
#ifdef SYCL_BLAS_USE_USM
  return internal::_dot(ex, _N, _vx, _incx, _vy, _incy, _rs);
#else
  return internal::_dot(ex, _N, ex.get_policy_handler().get_buffer(_vx), _incx,
                        ex.get_policy_handler().get_buffer(_vy), _incy,
                        ex.get_policy_handler().get_buffer(_rs));
#endif
}

/**
 * \brief ASUM Takes the sum of the absolute values
 * @param ex Executor
 * @param _vx BufferIterator
 * @param _incx Increment for the vector X
 */
template <typename executor_t, typename container_0_t, typename container_1_t,
          typename index_t, typename increment_t>
typename executor_t::policy_t::event_t _asum(executor_t &ex, index_t _N,
                                             container_0_t _vx,
                                             increment_t _incx,
                                             container_1_t _rs) {
#ifdef SYCL_BLAS_USE_USM
  return internal::_asum(ex, _N, _vx, _incx, _rs);
#else
  return internal::_asum(ex, _N, ex.get_policy_handler().get_buffer(_vx), _incx,
                         ex.get_policy_handler().get_buffer(_rs));
#endif
}

/**
 * \brief IAMAX finds the index of the first element having maximum
 * @param _vx BufferIterator
 * @param _incx Increment for the vector X
 */
template <typename executor_t, typename container_t, typename ContainerI,
          typename index_t, typename increment_t>
typename executor_t::policy_t::event_t _iamax(executor_t &ex, index_t _N,
                                              container_t _vx,
                                              increment_t _incx,
                                              ContainerI _rs) {
#ifdef SYCL_BLAS_USE_USM
  return internal::_iamax(ex, _N, _vx, _incx, _rs);
#else
  return internal::_iamax(ex, _N, ex.get_policy_handler().get_buffer(_vx),
                          _incx, ex.get_policy_handler().get_buffer(_rs));
#endif
}

/**
 * \brief IAMIN finds the index of the first element having minimum
 * @param _vx BufferIterator
 * @param _incx Increment for the vector X
 */
template <typename executor_t, typename container_t, typename ContainerI,
          typename index_t, typename increment_t>
typename executor_t::policy_t::event_t _iamin(executor_t &ex, index_t _N,
                                              container_t _vx,
                                              increment_t _incx,
                                              ContainerI _rs) {
#ifdef SYCL_BLAS_USE_USM
  return internal::_iamin(ex, _N, _vx, _incx, _rs);
#else
  return internal::_iamin(ex, _N, ex.get_policy_handler().get_buffer(_vx),
                          _incx, ex.get_policy_handler().get_buffer(_rs));
#endif
}

/**
 * \brief SWAP interchanges two vectors
 *
 * @param executor_t ex
 * @param _vx BufferIterator
 * @param _incx Increment for the vector X
 * @param _vy BufferIterator
 * @param _incy Increment for the vector Y
 */
template <typename executor_t, typename container_0_t, typename container_1_t,
          typename index_t, typename increment_t>
typename executor_t::policy_t::event_t _swap(executor_t &ex, index_t _N,
                                             container_0_t _vx,
                                             increment_t _incx,
                                             container_1_t _vy,
                                             increment_t _incy) {
#ifdef SYCL_BLAS_USE_USM
  return internal::_swap(ex, _N, _vx, _incx, _vy, _incy);
#else
  return internal::_swap(ex, _N, ex.get_policy_handler().get_buffer(_vx), _incx,
                         ex.get_policy_handler().get_buffer(_vy), _incy);
#endif
}

/**
 * \brief SCALAR operation on a vector
 * @param executor_t ex
 * @param _vx BufferIterator
 * @param _incx Increment for the vector X
 */
template <typename executor_t, typename element_t, typename container_0_t,
          typename index_t, typename increment_t>
typename executor_t::policy_t::event_t _scal(executor_t &ex, index_t _N,
                                             element_t _alpha,
                                             container_0_t _vx,
                                             increment_t _incx) {
#ifdef SYCL_BLAS_USE_USM
  return internal::_scal(ex, _N, _alpha, _vx, _incx);
#else
  return internal::_scal(ex, _N, _alpha,
                         ex.get_policy_handler().get_buffer(_vx), _incx);
#endif
}

/**
 * \brief NRM2 Returns the euclidian norm of a vector
 * @param ex Executor
 * @param _vx BufferIterator
 * @param _incx Increment for the vector X
 */
template <typename executor_t, typename container_0_t, typename container_1_t,
          typename index_t, typename increment_t>
typename executor_t::policy_t::event_t _nrm2(executor_t &ex, index_t _N,
                                             container_0_t _vx,
                                             increment_t _incx,
                                             container_1_t _rs) {
#ifdef SYCL_BLAS_USE_USM
  return internal::_nrm2(ex, _N, _vx, _incx, _rs);
#else
  return internal::_nrm2(ex, _N, ex.get_policy_handler().get_buffer(_vx), _incx,
                         ex.get_policy_handler().get_buffer(_rs));
#endif
}

/**
 * .
 * @brief _rot constructor given plane rotation
 *  *
 * @param ex Executor
 * @param _vx BufferIterator
 * @param _incx Increment for the vector X
 * @param _vx BufferIterator
 * @param _incy Increment for the vector Y
 * @param _sin sine
 * @param _cos cosine
 * @param _N data size
 *
 */
template <typename executor_t, typename container_0_t, typename container_1_t,
          typename element_t, typename index_t, typename increment_t>
typename executor_t::policy_t::event_t _rot(
    executor_t &ex, index_t _N, container_0_t _vx, increment_t _incx,
    container_1_t _vy, increment_t _incy, element_t _cos, element_t _sin) {
#ifdef SYCL_BLAS_USE_USM
  return internal::_rot(ex, _N, _vx, _incx, _vy, _incy, _cos, _sin);
#else
  return internal::_rot(ex, _N, ex.get_policy_handler().get_buffer(_vx), _incx,
                        ex.get_policy_handler().get_buffer(_vy), _incy, _cos,
                        _sin);
#endif
}

#ifndef SYCL_BLAS_USE_USM
/**
 * \brief Compute the inner product of two vectors with extended
    precision accumulation and result.
 *
 * @param ex Executor
 * @param _vx BufferIterator
 * @param _incx Increment for the vector X
 * @param _vx BufferIterator
 * @param _incy Increment for the vector Y
 */
template <typename executor_t, typename container_0_t, typename container_1_t,
          typename index_t, typename increment_t>
typename ValueType<container_0_t>::type _dot(executor_t &ex, index_t _N,
                                             container_0_t _vx,
                                             increment_t _incx,
                                             container_1_t _vy,
                                             increment_t _incy) {
  return internal::_dot(ex, _N, ex.get_policy_handler().get_buffer(_vx), _incx,
                        ex.get_policy_handler().get_buffer(_vy), _incy);
}

/**
 * \brief ICAMAX finds the index of the first element having maximum
 * @param _vx BufferIterator
 * @param _incx Increment for the vector X
 */
template <typename executor_t, typename container_t, typename index_t,
          typename increment_t>
index_t _iamax(executor_t &ex, index_t _N, container_t _vx, increment_t _incx) {
  return internal::_iamax(ex, _N, ex.get_policy_handler().get_buffer(_vx),
                          _incx);
}

/**
 * \brief ICAMIN finds the index of the first element having minimum
 * @param _vx BufferIterator
 * @param _incx Increment for the vector X
 */
template <typename executor_t, typename container_t, typename index_t,
          typename increment_t>
index_t _iamin(executor_t &ex, index_t _N, container_t _vx, increment_t _incx) {
  return internal::_iamin(ex, _N, ex.get_policy_handler().get_buffer(_vx),
                          _incx);
}

/**
 * \brief ASUM Takes the sum of the absolute values
 *
 * @param ex Executor
 * @param _vx BufferIterator
 * @param _incx Increment for the vector X
 */
template <typename executor_t, typename container_t, typename index_t,
          typename increment_t>
typename ValueType<container_t>::type _asum(executor_t &ex, index_t _N,
                                            container_t _vx,
                                            increment_t _incx) {
  return internal::_asum(ex, _N, ex.get_policy_handler().get_buffer(_vx),
                         _incx);
}

/**
 * \brief NRM2 Returns the euclidian norm of a vector
 *
 * @param ex Executor
 * @param _vx BufferIterator
 * @param _incx Increment for the vector X
 */
template <typename executor_t, typename container_t, typename index_t,
          typename increment_t>
typename ValueType<container_t>::type _nrm2(executor_t &ex, index_t _N,
                                            container_t _vx,
                                            increment_t _incx) {
  return internal::_nrm2(ex, _N, ex.get_policy_handler().get_buffer(_vx),
                         _incx);
}
#endif

}  // end namespace blas
#endif  // SYCL_BLAS_BLAS1_INTERFACE
