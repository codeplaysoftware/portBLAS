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
 * \brief Computes the inner product of two vectors with double precision
 * accumulation (Asynchronous version that returns an event)
 * @tparam executor_t Executor type
 * @tparam container_0_t Buffer Iterator
 * @tparam container_1_t Buffer Iterator
 * @tparam container_2_t Buffer Iterator
 * @tparam index_t Index type
 * @tparam increment_t Increment type
 * @param ex Executor
 * @param _N Input buffer sizes.
 * @param _vx Buffer holding input vector x
 * @param _incx Stride of vector x
 * @param _vy Buffer holding input vector y
 * @param _incy Stride of vector y
 * @param _rs output buffer
 * @return vector of events to wait for.
 */
template <typename executor_t, typename container_0_t, typename container_1_t,
          typename container_2_t, typename index_t, typename increment_t>
typename executor_t::policy_t::event_t _dot(
    executor_t &ex, index_t _N, container_0_t _vx, increment_t _incx,
    container_1_t _vy, increment_t _incy, container_2_t _rs);

/**
 * \brief Computes the inner product of two vectors with double precision
 * accumulation and adds a scalar to the result (Asynchronous version that
 * returns an event)
 * @tparam executor_t Executor type
 * @tparam container_0_t Buffer Iterator
 * @tparam container_1_t Buffer Iterator
 * @tparam container_2_t Buffer Iterator
 * @tparam index_t Index type
 * @tparam increment_t Increment type
 * @param ex Executor
 * @param _N Input buffer sizes. If size 0, the result will be sb.
 * @param sb scalar to add to the results of the inner product.
 * @param _vx Buffer holding input vector x
 * @param _incx Stride of vector x
 * @param _vy Buffer holding input vector y
 * @param _incy Stride of vector y
 * @param _rs output buffer
 * @return vector of events to wait for.
 */
template <typename executor_t, typename container_0_t, typename container_1_t,
        typename container_2_t, typename index_t, typename increment_t>
typename executor_t::policy_t::event_t _sdsdot(
        executor_t &ex, index_t _N, float sb, container_0_t _vx, increment_t _incx,
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
 * \brief Given the Cartesian coordinates (a, b) of a point, the rotg routines
 * return the parameters c, s, r, and z associated with the Givens rotation.
 * @tparam executor_t Executor type
 * @tparam container_0_t Buffer Iterator
 * @tparam container_1_t Buffer Iterator
 * @tparam container_2_t Buffer Iterator
 * @tparam container_3_t Buffer Iterator
 * @param ex Executor
 * @param a[in, out] On entry, Buffer holding the x-coordinate of the point. On
 * exit, the scalar z.
 * @param b[in, out] On entry, Buffer holding the y-coordinate of the point. On
 * exit, the scalar r.
 * @param c[out] Buffer holding the parameter c.
 * @param s[out] Buffer holding the parameter s.
 * @return vector of events to wait for.
 */
template <typename executor_t, typename container_0_t, typename container_1_t,
          typename container_2_t, typename container_3_t,
          std::enable_if_t<!is_sycl_scalar<container_0_t>, bool> = true>
typename executor_t::policy_t::event_t _rotg(executor_t &ex, container_0_t a,
                                             container_1_t b, container_2_t c,
                                             container_3_t s);

/**
 * \brief Synchronous version of rotg.
 * Given the Cartesian coordinates (a, b) of a point, the rotg routines
 * return the parameters c, s, r, and z associated with the Givens rotation.
 * @tparam executor_t Executor type
 * @tparam scalar_t Scalar type
 * @param ex Executor
 * @param a[in, out] On entry, x-coordinate of the point. On exit, the scalar z.
 * @param b[in, out] On entry, y-coordinate of the point. On exit, the scalar r.
 * @param c[out] scalar representing the output c.
 * @param s[out] scalar representing the output s.
 */
template <typename executor_t, typename scalar_t,
          std::enable_if_t<is_sycl_scalar<scalar_t>, bool> = true>
void _rotg(executor_t &ex, scalar_t &a, scalar_t &b, scalar_t &c, scalar_t &s);

/**
 * \brief Computes the inner product of two vectors with double precision
 * accumulation (synchronous version that returns the result directly)
 * @tparam executor_t Executor type
 * @tparam container_0_t Buffer Iterator
 * @tparam container_1_t Buffer Iterator
 * @tparam container_2_t Buffer Iterator
 * @tparam index_t Index type
 * @tparam increment_t Increment type
 * @param ex Executor
 * @param _N Input buffer sizes.
 * @param _vx Buffer holding input vector x
 * @param _incx Stride of vector x
 * @param _vy Buffer holding input vector y
 * @param _incy Stride of vector y
 * @param _rs output buffer
 * @return vector of events to wait for.
 */
template <typename executor_t, typename container_0_t, typename container_1_t,
          typename index_t, typename increment_t>
typename ValueType<container_0_t>::type _dot(executor_t &ex, index_t _N,
                                             container_0_t _vx,
                                             increment_t _incx,
                                             container_1_t _vy,
                                             increment_t _incy);

/**
 * \brief Computes the inner product of two vectors with double precision
 * accumulation and adds a scalar to the result (synchronous version that
 * returns the result directly)
 * @tparam executor_t Executor type
 * @tparam container_0_t Buffer Iterator
 * @tparam container_1_t Buffer Iterator
 * @tparam container_2_t Buffer Iterator
 * @tparam index_t Index type
 * @tparam increment_t Increment type
 * @param ex Executor
 * @param _N Input buffer sizes. If size 0, the result will be sb.
 * @param sb scalar to add to the results of the inner product.
 * @param _vx Buffer holding input vector x
 * @param _incx Stride of vector x
 * @param _vy Buffer holding input vector y
 * @param _incy Stride of vector y
 * @param _rs output buffer
 * @return vector of events to wait for.
 */
template<typename executor_t, typename container_0_t, typename container_1_t,
        typename index_t, typename increment_t>
typename ValueType<container_0_t>::type _sdsdot(executor_t& ex, index_t _N, float sb,
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
  return internal::_axpy(ex, _N, _alpha,
                         ex.get_policy_handler().get_buffer(_vx), _incx,
                         ex.get_policy_handler().get_buffer(_vy), _incy);
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
  return internal::_copy(ex, _N, ex.get_policy_handler().get_buffer(_vx), _incx,
                         ex.get_policy_handler().get_buffer(_vy), _incy);
}

/**
 * \brief Computes the inner product of two vectors with double precision
 * accumulation (Asynchronous version that returns an event)
 * @tparam executor_t Executor type
 * @tparam container_0_t Buffer Iterator
 * @tparam container_1_t Buffer Iterator
 * @tparam container_2_t Buffer Iterator
 * @tparam index_t Index type
 * @tparam increment_t Increment type
 * @param ex Executor
 * @param _N Input buffer sizes.
 * @param _vx Buffer holding input vector x
 * @param _incx Stride of vector x
 * @param _vy Buffer holding input vector y
 * @param _incy Stride of vector y
 * @param _rs output buffer
 * @return vector of events to wait for.
 */
template <typename executor_t, typename container_0_t, typename container_1_t,
          typename container_2_t, typename index_t, typename increment_t>
typename executor_t::policy_t::event_t _dot(
    executor_t &ex, index_t _N, container_0_t _vx, increment_t _incx,
    container_1_t _vy, increment_t _incy, container_2_t _rs) {
  return internal::_dot(ex, _N, ex.get_policy_handler().get_buffer(_vx), _incx,
                        ex.get_policy_handler().get_buffer(_vy), _incy,
                        ex.get_policy_handler().get_buffer(_rs));
}

/**
 * \brief Computes the inner product of two vectors with double precision
 * accumulation and adds a scalar to the result (Asynchronous version that
 * returns an event)
 * @tparam executor_t Executor type
 * @tparam container_0_t Buffer Iterator
 * @tparam container_1_t Buffer Iterator
 * @tparam container_2_t Buffer Iterator
 * @tparam index_t Index type
 * @tparam increment_t Increment type
 * @param ex Executor
 * @param _N Input buffer sizes. If size 0, the result will be sb.
 * @param sb scalar to add to the results of the inner product.
 * @param _vx Buffer holding input vector x
 * @param _incx Stride of vector x
 * @param _vy Buffer holding input vector y
 * @param _incy Stride of vector y
 * @param _rs output buffer
 * @return vector of events to wait for.
 */
template <typename executor_t, typename container_0_t, typename container_1_t,
        typename container_2_t, typename index_t, typename increment_t>
typename executor_t::policy_t::event_t _sdsdot(
        executor_t &ex, index_t _N, float sb, container_0_t _vx, increment_t _incx,
        container_1_t _vy, increment_t _incy, container_2_t _rs) {
    return internal::_sdsdot(ex, _N, sb, ex.get_policy_handler().get_buffer(_vx), _incx,
                          ex.get_policy_handler().get_buffer(_vy), _incy,
                          ex.get_policy_handler().get_buffer(_rs));
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
  return internal::_asum(ex, _N, ex.get_policy_handler().get_buffer(_vx), _incx,
                         ex.get_policy_handler().get_buffer(_rs));
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
  return internal::_iamax(ex, _N, ex.get_policy_handler().get_buffer(_vx),
                          _incx, ex.get_policy_handler().get_buffer(_rs));
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
  return internal::_iamin(ex, _N, ex.get_policy_handler().get_buffer(_vx),
                          _incx, ex.get_policy_handler().get_buffer(_rs));
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
  return internal::_swap(ex, _N, ex.get_policy_handler().get_buffer(_vx), _incx,
                         ex.get_policy_handler().get_buffer(_vy), _incy);
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
  return internal::_scal(ex, _N, _alpha,
                         ex.get_policy_handler().get_buffer(_vx), _incx);
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
  return internal::_nrm2(ex, _N, ex.get_policy_handler().get_buffer(_vx), _incx,
                         ex.get_policy_handler().get_buffer(_rs));
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
  return internal::_rot(ex, _N, ex.get_policy_handler().get_buffer(_vx), _incx,
                        ex.get_policy_handler().get_buffer(_vy), _incy, _cos,
                        _sin);
}

/**
 * \brief Given the Cartesian coordinates (a, b) of a point, the rotg routines
 * return the parameters c, s, r, and z associated with the Givens rotation.
 * @tparam executor_t Executor type
 * @tparam container_0_t Buffer Iterator
 * @tparam container_1_t Buffer Iterator
 * @tparam container_2_t Buffer Iterator
 * @tparam container_3_t Buffer Iterator
 * @param ex Executor
 * @param a[in, out] On entry, Buffer holding the x-coordinate of the point. On
 * exit, the scalar z.
 * @param b[in, out] On entry, Buffer holding the y-coordinate of the point. On
 * exit, the scalar r.
 * @param c[out] Buffer holding the parameter c.
 * @param s[out] Buffer holding the parameter s.
 * @return vector of events to wait for.
 */
template <typename executor_t, typename container_0_t, typename container_1_t,
          typename container_2_t, typename container_3_t,
          std::enable_if_t<!is_sycl_scalar<container_0_t>, bool> = true>
typename executor_t::policy_t::event_t _rotg(executor_t &ex, container_0_t a,
                                             container_1_t b, container_2_t c,
                                             container_3_t s) {
  return internal::_rotg(ex, a, b, c, s);
}

/**
 * \brief Synchronous version of rotg.
 * Given the Cartesian coordinates (a, b) of a point, the rotg routines
 * return the parameters c, s, r, and z associated with the Givens rotation.
 * @tparam executor_t Executor type
 * @tparam scalar_t Scalar type
 * @param ex Executor
 * @param a[in, out] On entry, x-coordinate of the point. On exit, the scalar z.
 * @param b[in, out] On entry, y-coordinate of the point. On exit, the scalar r.
 * @param c[out] scalar representing the output c.
 * @param s[out] scalar representing the output s.
 */
template <typename executor_t, typename scalar_t,
          std::enable_if_t<is_sycl_scalar<scalar_t>, bool> = true>
void _rotg(executor_t &ex, scalar_t &a, scalar_t &b, scalar_t &c, scalar_t &s) {
  internal::_rotg(ex, a, b, c, s);
}

/**
 * \brief Computes the inner product of two vectors with double precision
 * accumulation (synchronous version that returns the result directly)
 * @tparam executor_t Executor type
 * @tparam container_0_t Buffer Iterator
 * @tparam container_1_t Buffer Iterator
 * @tparam container_2_t Buffer Iterator
 * @tparam index_t Index type
 * @tparam increment_t Increment type
 * @param ex Executor
 * @param _N Input buffer sizes.
 * @param _vx Buffer holding input vector x
 * @param _incx Stride of vector x
 * @param _vy Buffer holding input vector y
 * @param _incy Stride of vector y
 * @param _rs output buffer
 * @return vector of events to wait for.
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
 * \brief Computes the inner product of two vectors with double precision
 * accumulation and adds a scalar to the result (synchronous version that
 * returns the result directly)
 * @tparam executor_t Executor type
 * @tparam container_0_t Buffer Iterator
 * @tparam container_1_t Buffer Iterator
 * @tparam container_2_t Buffer Iterator
 * @tparam index_t Index type
 * @tparam increment_t Increment type
 * @param ex Executor
 * @param _N Input buffer sizes. If size 0, the result will be sb.
 * @param sb scalar to add to the results of the inner product.
 * @param _vx Buffer holding input vector x
 * @param _incx Stride of vector x
 * @param _vy Buffer holding input vector y
 * @param _incy Stride of vector y
 * @param _rs output buffer
 * @return vector of events to wait for.
 */
template <typename executor_t, typename container_0_t, typename container_1_t,
        typename index_t, typename increment_t>
typename ValueType<container_0_t>::type _sdsdot(executor_t &ex, index_t _N, float sb,
                                             container_0_t _vx,
                                             increment_t _incx,
                                             container_1_t _vy,
                                             increment_t _incy) {
    return internal::_sdsdot(ex, _N, sb, ex.get_policy_handler().get_buffer(_vx), _incx,
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

}  // end namespace blas
#endif  // SYCL_BLAS_BLAS1_INTERFACE
