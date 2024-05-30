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
 *  portBLAS: BLAS implementation using SYCL
 *
 *  @filename blas1_interface.h
 *
 **************************************************************************/
#ifndef PORTBLAS_BLAS1_INTERFACE_H
#define PORTBLAS_BLAS1_INTERFACE_H
#include "blas_meta.h"

namespace blas {
namespace internal {
/**
 * \brief AXPY constant times a vector plus a vector.
 *
 * Implements AXPY \f$y = ax + y\f$
 *
 * @param sb_handle SB_Handle
 * @param _vx BufferIterator or USM pointer
 * @param _incx Increment for the vector X
 * @param _vy BufferIterator or USM pointer
 * @param _incy Increment for the vector Y
 * @param _dependencies Vector of events
 */
template <typename sb_handle_t, typename container_0_t, typename container_1_t,
          typename element_t, typename index_t, typename increment_t>
typename sb_handle_t::event_t _axpy(
    sb_handle_t &sb_handle, index_t _N, element_t _alpha, container_0_t _vx,
    increment_t _incx, container_1_t _vy, increment_t _incy,
    const typename sb_handle_t::event_t &_dependencies);

/**
 * \brief COPY copies a vector, x, to a vector, y.
 *
 * @param sb_handle SB_Handle
 * @param _vx BufferIterator or USM pointer
 * @param _incx Increment for the vector X
 * @param _vy BufferIterator or USM pointer
 * @param _incy Increment for the vector Y
 * @param _dependencies Vector of events
 */
template <typename sb_handle_t, typename index_t, typename container_0_t,
          typename container_1_t, typename increment_t>
typename sb_handle_t::event_t _copy(
    sb_handle_t &sb_handle, index_t _N, container_0_t _vx, increment_t _incx,
    container_1_t _vy, increment_t _incy,
    const typename sb_handle_t::event_t &_dependencies);

/**
 * \brief Computes the inner product of two vectors with double precision
 * accumulation (Asynchronous version that returns an event)
 * @tparam sb_handle_t SB_Handle type
 * @tparam container_0_t Buffer Iterator or USM pointer
 * @tparam container_1_t Buffer Iterator or USM pointer
 * @tparam container_2_t Buffer Iterator or USM pointer
 * @tparam index_t Index type
 * @tparam increment_t Increment type
 * @param sb_handle SB_Handle
 * @param _N Input buffer sizes.
 * @param _vx Memory object holding input vector x
 * @param _incx Stride of vector x (i.e. measured in elements of _vx)
 * @param _vy Memory object holding input vector y
 * @param _incy Stride of vector y (i.e. measured in elements of _vy)
 * @param _rs Output memory object
 * @param _dependencies Vector of events
 * @return Vector of events to wait for.
 */
template <typename sb_handle_t, typename container_0_t, typename container_1_t,
          typename container_2_t, typename index_t, typename increment_t>
typename sb_handle_t::event_t _dot(
    sb_handle_t &sb_handle, index_t _N, container_0_t _vx, increment_t _incx,
    container_1_t _vy, increment_t _incy, container_2_t _rs,
    const typename sb_handle_t::event_t &_dependencies);

/**
 * \brief Computes the inner product of two vectors with double precision
 * accumulation and adds a scalar to the result (Asynchronous version that
 * returns an event)
 * @tparam sb_handle_t SB_Handle type
 * @tparam container_0_t Buffer Iterator or USM pointer
 * @tparam container_1_t Buffer Iterator or USM pointer
 * @tparam container_2_t Buffer Iterator or USM pointer
 * @tparam index_t Index type
 * @tparam increment_t Increment type
 * @param sb_handle SB_Handle
 * @param _N Input buffer sizes. If size 0, the result will be sb.
 * @param sb Scalar to add to the results of the inner product.
 * @param _vx Memory object holding input vector x
 * @param _incx Stride of vector x (i.e. measured in elements of _vx)
 * @param _vy Memory object holding input vector y
 * @param _incy Stride of vector y (i.e. measured in elements of _vy)
 * @param _rs Output memory object
 * @param _dependencies Vector of events
 * @return Vector of events to wait for.
 */
template <typename sb_handle_t, typename container_0_t, typename container_1_t,
          typename container_2_t, typename index_t, typename increment_t>
typename sb_handle_t::event_t _sdsdot(
    sb_handle_t &sb_handle, index_t _N, float sb, container_0_t _vx,
    increment_t _incx, container_1_t _vy, increment_t _incy, container_2_t _rs,
    const typename sb_handle_t::event_t &_dependencies);

/**
 * \brief ASUM Takes the sum of the absolute values
 * @param sb_handle SB_Handle
 * @param _vx BufferIterator or USM pointer
 * @param _incx Increment for the vector X
 * @param _rs BufferIterator or USM pointer
 * @param _dependencies Vector of events
 */
template <typename sb_handle_t, typename container_0_t, typename container_1_t,
          typename index_t, typename increment_t>
typename sb_handle_t::event_t _asum(
    sb_handle_t &sb_handle, index_t _N, container_0_t _vx, increment_t _incx,
    container_1_t _rs, const typename sb_handle_t::event_t &_dependencies);

/*!
 * \brief Prototype for the internal implementation of the ASUM operation. See
 * documentation in the blas1_interface.hpp file for details.
 */
template <int localSize, int localMemSize, bool usmManagedMem = false,
          typename sb_handle_t, typename container_0_t, typename container_1_t,
          typename index_t, typename increment_t>
typename sb_handle_t::event_t _asum_impl(
    sb_handle_t &sb_handle, index_t _N, container_0_t _vx, increment_t _incx,
    container_1_t _rs, const index_t number_WG,
    const typename sb_handle_t::event_t &_dependencies);

template <int localSize, int localMemSize, bool is_max, bool single,
          typename sb_handle_t, typename container_0_t, typename container_1_t,
          typename index_t, typename increment_t>
typename sb_handle_t::event_t _iamax_iamin_impl(
    sb_handle_t &sb_handle, index_t _N, container_0_t _vx, increment_t _incx,
    container_1_t _rs, const index_t _nWG,
    const typename sb_handle_t::event_t &_dependencies);

/**
 * \brief IAMAX finds the index of the first element having maximum
 * @param _vx BufferIterator or USM pointer
 * @param _incx Increment for the vector X
 * @param _rs BufferIterator or USM pointer
 * @param _dependencies Vector of events
 */
template <typename sb_handle_t, typename container_t, typename ContainerI,
          typename index_t, typename increment_t>
typename sb_handle_t::event_t _iamax(
    sb_handle_t &sb_handle, index_t _N, container_t _vx, increment_t _incx,
    ContainerI _rs, const typename sb_handle_t::event_t &_dependencies);

/**
 * \brief IAMIN finds the index of the first element having minimum
 * @param _vx BufferIterator or USM pointer
 * @param _incx Increment for the vector X
 * @param _rs BufferIterator or USM pointer
 * @param _dependencies Vector of events
 */
template <typename sb_handle_t, typename container_t, typename ContainerI,
          typename index_t, typename increment_t>
typename sb_handle_t::event_t _iamin(
    sb_handle_t &sb_handle, index_t _N, container_t _vx, increment_t _incx,
    ContainerI _rs, const typename sb_handle_t::event_t &_dependencies);

/**
 * \brief SWAP interchanges two vectors
 *
 * @param sb_handle_t sb_handle
 * @param _vx BufferIterator or USM pointer
 * @param _incx Increment for the vector X
 * @param _vy BufferIterator or USM pointer
 * @param _incy Increment for the vector Y
 * @param _dependencies Vector of events
 */
template <typename sb_handle_t, typename container_0_t, typename container_1_t,
          typename index_t, typename increment_t>
typename sb_handle_t::event_t _swap(
    sb_handle_t &sb_handle, index_t _N, container_0_t _vx, increment_t _incx,
    container_1_t _vy, increment_t _incy,
    const typename sb_handle_t::event_t &_dependencies);

/**
 * \brief SCALAR operation on a vector
 * @param sb_handle_t sb_handle
 * @param _vx BufferIterator or USM pointer
 * @param _incx Increment for the vector X
 * @param _dependencies Vector of events
 */
template <typename sb_handle_t, typename element_t, typename container_0_t,
          typename index_t, typename increment_t>
typename sb_handle_t::event_t _scal(
    sb_handle_t &sb_handle, index_t _N, element_t _alpha, container_0_t _vx,
    increment_t _incx, const typename sb_handle_t::event_t &_dependencies);

/**
 * \brief SCALAR operation on a matrix. (this is a generalization of
 * vector-based _scal operator meant for internal use within the library, namely
 * for GEMM and inplace-Matcopy operators)
 * @param sb_handle_t sb_handle
 * @param _A Input/Output BufferIterator or USM pointer
 * @param _incA Increment for the matrix A
 * @param _lda Leading dimension for the matrix A
 * @param _M number of rows
 * @param _N number of columns
 * @param alpha scaling scalar
 * @param _dependencies Vector of events
 */
template <typename sb_handle_t, typename element_t, typename container_0_t,
          typename index_t, typename increment_t>
typename sb_handle_t::event_t _scal_matrix(
    sb_handle_t &sb_handle, index_t _M, index_t _N, element_t _alpha,
    container_0_t _A, index_t _lda, increment_t _incA,
    const typename sb_handle_t::event_t &_dependencies);

/*!
 * \brief Prototype for the internal implementation of the _scal_matrix
 * operator.
 */
template <bool has_inc, typename sb_handle_t, typename element_t,
          typename container_0_t, typename index_t, typename increment_t>
typename sb_handle_t::event_t _scal_matrix_impl(
    sb_handle_t &sb_handle, index_t _M, index_t _N, element_t _alpha,
    container_0_t _A, index_t _lda, increment_t _incA,
    const typename sb_handle_t::event_t &_dependencies);

/**
 * \brief NRM2 Returns the euclidian norm of a vector
 * @param sb_handle SB_Handle
 * @param _vx BufferIterator or USM pointer
 * @param _incx Increment for the vector X
 * @param _rs BufferIterator or USM pointer
 * @param _dependencies Vector of events
 */
template <typename sb_handle_t, typename container_0_t, typename container_1_t,
          typename index_t, typename increment_t>
typename sb_handle_t::event_t _nrm2(
    sb_handle_t &sb_handle, index_t _N, container_0_t _vx, increment_t _incx,
    container_1_t _rs, const typename sb_handle_t::event_t &_dependencies);

/*!
 * \brief Prototype for the internal implementation of the NRM2 operator. See
 * documentation in the blas1_interface.hpp file for details.
 */
template <int localSize, int localMemSize, bool usmManagedMem = false,
          typename sb_handle_t, typename container_0_t, typename container_1_t,
          typename index_t, typename increment_t>
typename sb_handle_t::event_t _nrm2_impl(
    sb_handle_t &sb_handle, index_t _N, container_0_t _vx, increment_t _incx,
    container_1_t _rs, const index_t number_WG,
    const typename sb_handle_t::event_t &_dependencies);

/*!
 * \brief Prototype for the internal implementation of the Dot operator. See
 * documentation in the blas1_interface.hpp file for details.
 */
template <int localSize, int localMemSize, bool usmManagedMem = false,
          typename sb_handle_t, typename container_0_t, typename container_1_t,
          typename container_2_t, typename index_t, typename increment_t>
typename sb_handle_t::event_t _dot_impl(
    sb_handle_t &sb_handle, index_t _N, container_0_t _vx, increment_t _incx,
    container_1_t _vy, increment_t _incy, container_2_t _rs,
    const index_t _number_wg,
    const typename sb_handle_t::event_t &_dependencies);

/**
 * @brief _rot constructor given plane rotation
 * @param sb_handle SB_Handle
 * @param _vx BufferIterator or USM pointer
 * @param _incx Increment for the vector X
 * @param _vx BufferIterator or USM pointer
 * @param _incy Increment for the vector Y
 * @param _sin sine
 * @param _cos cosine
 * @param _N data size
 * @param _dependencies Vector of events
 */
template <typename sb_handle_t, typename container_0_t, typename container_1_t,
          typename element_t, typename index_t, typename increment_t>
typename sb_handle_t::event_t _rot(
    sb_handle_t &sb_handle, index_t _N, container_0_t _vx, increment_t _incx,
    container_1_t _vy, increment_t _incy, element_t _cos, element_t _sin,
    const typename sb_handle_t::event_t &_dependencies);

/**
 * @brief Performs a modified Givens rotation of points.
 * Given two vectors x and y and a modified Givens transformation matrix, each
 * element of x and y is replaced as follows:
 *
 * [xi] = [h11 h12] * [xi]
 * [yi]   [h21 h22]   [yi]
 *
 * where h11, h12, h21 and h22 represent the modified Givens transformation
 * matrix.
 *
 * The value of the flag parameter can be used to modify the matrix as follows:
 *
 * -1.0: [h11 h12]     0.0: [1.0 h12]     1.0: [h11 1.0]     -2.0 = [1.0 0.0]
 *       [h21 h22]          [h21 1.0]          [-1.0 h22]           [0.0 1.0]
 *
 * @tparam sb_handle_t SB_Handle type
 * @tparam container_0_t Buffer Iterator or USM pointer
 * @tparam container_1_t Buffer Iterator or USM pointer
 * @tparam container_2_t Buffer Iterator or USM pointer
 * @tparam index_t Index type
 * @tparam increment_t Increment type
 * @param sb_handle SB_Handle
 * @param _N Input buffer sizes (for vx and vy).
 * @param[in, out] _vx Memory object holding input vector x
 * @param _incx Stride of vector x (i.e. measured in elements of _vx)
 * @param[in, out] _vy Memory object holding input vector y
 * @param _incy Stride of vector y (i.e. measured in elements of _vy)
 * @param[in] _param Buffer with the following layout: [flag, h11, h21, h12,
 * h22].
 * @param _dependencies Vector of events
 * @return Vector of events to wait for.
 */
template <typename sb_handle_t, typename container_0_t, typename container_1_t,
          typename container_2_t, typename index_t, typename increment_t>
typename sb_handle_t::event_t _rotm(
    sb_handle_t &sb_handle, index_t _N, container_0_t _vx, increment_t _incx,
    container_1_t _vy, increment_t _incy, container_2_t _param,
    const typename sb_handle_t::event_t &_dependencies);

/**
 * Given the Cartesian coordinates (x1, y1) of a point, the rotmg routines
 * compute the components of a modified Givens transformation matrix H that
 * zeros the y-component of the resulting point:
 *
 *                      [xi] = H * [xi * sqrt(d1) ]
 *                      [0 ]       [yi * sqrt(d2) ]
 *
 * Depending on the flag parameter, the components of H are set as follows:
 *
 * -1.0: [h11 h12]     0.0: [1.0 h12]     1.0: [h11 1.0]     -2.0 = [1.0 0.0]
 *       [h21 h22]          [h21 1.0]          [-1.0 h22]           [0.0 1.0]
 *
 * Rotmg may apply scaling operations to d1, d2 and x1 to avoid overflows.
 *
 * @tparam sb_handle_t SB_Handle type
 * @tparam container_0_t Buffer Iterator or USM pointer
 * @tparam container_1_t Buffer Iterator or USM pointer
 * @tparam container_2_t Buffer Iterator or USM pointer
 * @tparam container_3_t Buffer Iterator or USM pointer
 * @tparam container_4_t Buffer Iterator or USM pointer
 * @param sb_handle SB_Handle
 * @param _d1[in,out] On entry, memory object holding the scaling factor for
 * the x-coordinate. On exit, the re-scaled _d1.
 * @param _d2[in,out] On entry, memory object holding the scaling factor for
 * the y-coordinate. On exit, the re-scaled _d2.
 * @param _x1[in,out] On entry, memory object holding the x-coordinate. On
 * exit, the re-scaled _x1
 * @param _y1[in] Memory object holding the y-coordinate of the point.
 * @param _param[out] Buffer with the following layout: [flag, h11, h21, h12,
 * h22].
 * @param _dependencies Vector of events
 * @return Vector of events to wait for.
 */
template <typename sb_handle_t, typename container_0_t, typename container_1_t,
          typename container_2_t, typename container_3_t,
          typename container_4_t>
typename sb_handle_t::event_t _rotmg(
    sb_handle_t &sb_handle, container_0_t _d1, container_1_t _d2,
    container_2_t _x1, container_3_t _y1, container_4_t _param,
    const typename sb_handle_t::event_t &_dependencies);

/**
 * \brief Given the Cartesian coordinates (a, b) of a point, the rotg routines
 * return the parameters c, s, r, and z associated with the Givens rotation.
 * @tparam sb_handle_t SB_Handle type
 * @tparam container_0_t Buffer Iterator or USM pointer
 * @tparam container_1_t Buffer Iterator or USM pointer
 * @tparam container_2_t Buffer Iterator or USM pointer
 * @tparam container_3_t Buffer Iterator or USM pointer
 * @param sb_handle SB_Handle
 * @param a[in, out] On entry, memory object holding the x-coordinate of the
 * point. On exit, the scalar z.
 * @param b[in, out] On entry, memory object holding the y-coordinate of the
 * point. On exit, the scalar r.
 * @param c[out] Memory object holding the parameter c.
 * @param s[out] Memory object holding the parameter s.
 * @param _dependencies Vector of events
 * @return Vector of events to wait for.
 */
template <typename sb_handle_t, typename container_0_t, typename container_1_t,
          typename container_2_t, typename container_3_t,
          typename std::enable_if<!is_sycl_scalar<container_0_t>::value,
                                  bool>::type = true>
typename sb_handle_t::event_t _rotg(
    sb_handle_t &sb_handle, container_0_t a, container_1_t b, container_2_t c,
    container_3_t s, const typename sb_handle_t::event_t &_dependencies);

/**
 * \brief Synchronous version of rotg.
 * Given the Cartesian coordinates (a, b) of a point, the rotg routines
 * return the parameters c, s, r, and z associated with the Givens rotation.
 * @tparam sb_handle_t SB_Handle type
 * @tparam scalar_t Scalar type
 * @param sb_handle SB_Handle
 * @param a[in, out] On entry, x-coordinate of the point. On exit, the scalar
 * z.
 * @param b[in, out] On entry, y-coordinate of the point. On exit, the scalar
 * r.
 * @param c[out] scalar representing the output c.
 * @param s[out] scalar representing the output s.
 * @param _dependencies Vector of events
 */
template <
    typename sb_handle_t, typename scalar_t,
    typename std::enable_if<is_sycl_scalar<scalar_t>::value, bool>::type = true>
void _rotg(sb_handle_t &sb_handle, scalar_t &a, scalar_t &b, scalar_t &c,
           scalar_t &s, const typename sb_handle_t::event_t &_dependencies);

/**
 * \brief Computes the inner product of two vectors with double precision
 * accumulation (synchronous version that returns the result directly)
 * @tparam sb_handle_t SB_Handle type
 * @tparam container_0_t Buffer Iterator or USM pointer
 * @tparam container_1_t Buffer Iterator or USM pointer
 * @tparam index_t Index type
 * @tparam increment_t Increment type
 * @param sb_handle SB_Handle
 * @param _N Input buffer sizes.
 * @param _vx Memory object holding input vector x
 * @param _incx Stride of vector x (i.e. measured in elements of _vx)
 * @param _vy Memory object holding input vector y
 * @param _incy Stride of vector y (i.e. measured in elements of _vy)
 * @param _rs Output memory object
 * @param _dependencies Vector of events
 * @return Vector of events to wait for.
 */
template <typename sb_handle_t, typename container_0_t, typename container_1_t,
          typename index_t, typename increment_t>
typename ValueType<container_0_t>::type _dot(
    sb_handle_t &sb_handle, index_t _N, container_0_t _vx, increment_t _incx,
    container_1_t _vy, increment_t _incy,
    const typename sb_handle_t::event_t &_dependencies);

/**
 * \brief Computes the inner product of two vectors with double precision
 * accumulation and adds a scalar to the result (synchronous version that
 * returns the result directly)
 * @tparam sb_handle_t SB_Handle type
 * @tparam container_0_t Buffer Iterator or USM pointer
 * @tparam container_1_t Buffer Iterator or USM pointer
 * @tparam index_t Index type
 * @tparam increment_t Increment type
 * @param sb_handle SB_Handle
 * @param _N Input buffer sizes. If size 0, the result will be sb.
 * @param sb Scalar to add to the results of the inner product.
 * @param _vx Memory object holding input vector x
 * @param _incx Stride of vector x (i.e. measured in elements of _vx)
 * @param _vy Memory object holding input vector y
 * @param _incy Stride of vector y (i.e. measured in elements of _vy)
 * @param _rs Output memory object
 * @param _dependencies Vector of events
 * @return Vector of events to wait for.
 */
template <typename sb_handle_t, typename container_0_t, typename container_1_t,
          typename index_t, typename increment_t>
typename ValueType<container_0_t>::type _sdsdot(
    sb_handle_t &sb_handle, index_t _N, float sb, container_0_t _vx,
    increment_t _incx, container_1_t _vy, increment_t _incy,
    const typename sb_handle_t::event_t &_dependencies);
/**
 * \brief ICAMAX finds the index of the first element having maximum
 * @param _vx BufferIterator or USM pointer
 * @param _incx Increment for the vector X
 * @param _dependencies Vector of events
 */
template <typename sb_handle_t, typename container_t, typename index_t,
          typename increment_t>
index_t _iamax(sb_handle_t &sb_handle, index_t _N, container_t _vx,
               increment_t _incx,
               const typename sb_handle_t::event_t &_dependencies);

/**
 * \brief ICAMIN finds the index of the first element having minimum
 * @param _vx BufferIterator or USM pointer
 * @param _incx Increment for the vector X
 * @param _dependencies Vector of events
 */
template <typename sb_handle_t, typename container_t, typename index_t,
          typename increment_t>
index_t _iamin(sb_handle_t &sb_handle, index_t _N, container_t _vx,
               increment_t _incx,
               const typename sb_handle_t::event_t &_dependencies);

/**
 * \brief ASUM Takes the sum of the absolute values
 *
 * @param sb_handle SB_Handle
 * @param _vx BufferIterator or USM pointer
 * @param _incx Increment for the vector X
 * @param _dependencies Vector of events
 */
template <typename sb_handle_t, typename container_t, typename index_t,
          typename increment_t>
typename ValueType<container_t>::type _asum(
    sb_handle_t &sb_handle, index_t _N, container_t _vx, increment_t _incx,
    const typename sb_handle_t::event_t &_dependencies);

/**
 * \brief NRM2 Returns the euclidian norm of a vector
 *
 * @param sb_handle SB_Handle
 * @param _vx BufferIterator or USM pointer
 * @param _incx Increment for the vector X
 * @param _dependencies Vector of events
 */
template <typename sb_handle_t, typename container_t, typename index_t,
          typename increment_t>
typename ValueType<container_t>::type _nrm2(
    sb_handle_t &sb_handle, index_t _N, container_t _vx, increment_t _incx,
    const typename sb_handle_t::event_t &_dependencies);

}  // namespace internal

template <typename sb_handle_t, typename container_0_t, typename container_1_t,
          typename element_t, typename index_t, typename increment_t>
typename sb_handle_t::event_t _axpy(
    sb_handle_t &sb_handle, index_t _N, element_t _alpha, container_0_t _vx,
    increment_t _incx, container_1_t _vy, increment_t _incy,
    const typename sb_handle_t::event_t &_dependencies = {}) {
  return internal::_axpy(sb_handle, _N, _alpha, _vx, _incx, _vy, _incy,
                         _dependencies);
}

/**
 * \brief COPY copies a vector, x, to a vector, y.
 *
 * @param sb_handle SB_Handle
 * @param _vx BufferIterator or USM pointer
 * @param _incx Increment for the vector X
 * @param _vy BufferIterator or USM pointer
 * @param _incy Increment for the vector Y
 * @param _dependencies Vector of events
 */
template <typename sb_handle_t, typename index_t, typename container_0_t,
          typename container_1_t, typename increment_t>
typename sb_handle_t::event_t _copy(
    sb_handle_t &sb_handle, index_t _N, container_0_t _vx, increment_t _incx,
    container_1_t _vy, increment_t _incy,
    const typename sb_handle_t::event_t &_dependencies = {}) {
  return internal::_copy(sb_handle, _N, _vx, _incx, _vy, _incy, _dependencies);
}

/**
 * \brief Computes the inner product of two vectors with double precision
 * accumulation (Asynchronous version that returns an event)
 * @tparam sb_handle_t SB_Handle type
 * @tparam container_0_t Buffer Iterator or USM pointer
 * @tparam container_1_t Buffer Iterator or USM pointer
 * @tparam container_2_t Buffer Iterator or USM pointer
 * @tparam index_t Index type
 * @tparam increment_t Increment type
 * @param sb_handle SB_Handle
 * @param _N Input buffer sizes.
 * @param _vx Memory object holding input vector x
 * @param _incx Stride of vector x (i.e. measured in elements of _vx)
 * @param _vy Memory object holding input vector y
 * @param _incy Stride of vector y (i.e. measured in elements of _vy)
 * @param _rs Output memory object
 * @param _dependencies Vector of events
 * @return Vector of events to wait for.
 */
template <typename sb_handle_t, typename container_0_t, typename container_1_t,
          typename container_2_t, typename index_t, typename increment_t>
typename sb_handle_t::event_t _dot(
    sb_handle_t &sb_handle, index_t _N, container_0_t _vx, increment_t _incx,
    container_1_t _vy, increment_t _incy, container_2_t _rs,
    const typename sb_handle_t::event_t &_dependencies = {}) {
  return internal::_dot(sb_handle, _N, _vx, _incx, _vy, _incy, _rs,
                        _dependencies);
}

/**
 * \brief Computes the inner product of two vectors with double precision
 * accumulation and adds a scalar to the result (Asynchronous version that
 * returns an event)
 * @tparam sb_handle_t SB_Handle type
 * @tparam container_0_t Buffer Iterator or USM pointer
 * @tparam container_1_t Buffer Iterator or USM pointer
 * @tparam container_2_t Buffer Iterator or USM pointer
 * @tparam index_t Index type
 * @tparam increment_t Increment type
 * @param sb_handle SB_Handle
 * @param _N Input buffer sizes. If size 0, the result will be sb.
 * @param sb Scalar to add to the results of the inner product.
 * @param _vx Memory object holding input vector x
 * @param _incx Stride of vector x (i.e. measured in elements of _vx)
 * @param _vy Memory object holding input vector y
 * @param _incy Stride of vector y (i.e. measured in elements of _vy)
 * @param _rs Output memory object
 * @param _dependencies Vector of events
 * @return Vector of events to wait for.
 */
template <typename sb_handle_t, typename container_0_t, typename container_1_t,
          typename container_2_t, typename index_t, typename increment_t>
typename sb_handle_t::event_t _sdsdot(
    sb_handle_t &sb_handle, index_t _N, float sb, container_0_t _vx,
    increment_t _incx, container_1_t _vy, increment_t _incy, container_2_t _rs,
    const typename sb_handle_t::event_t &_dependencies = {}) {
  return internal::_sdsdot(sb_handle, _N, sb, _vx, _incx, _vy, _incy, _rs,
                           _dependencies);
}

/**
 * \brief ASUM Takes the sum of the absolute values
 * @param sb_handle SB_Handle
 * @param _vx BufferIterator or USM pointer
 * @param _incx Increment for the vector X
 * @param _rs BufferIterator or USM pointer
 * @param _dependencies Vector of events
 */
template <typename sb_handle_t, typename container_0_t, typename container_1_t,
          typename index_t, typename increment_t>
typename sb_handle_t::event_t _asum(
    sb_handle_t &sb_handle, index_t _N, container_0_t _vx, increment_t _incx,
    container_1_t _rs,
    const typename sb_handle_t::event_t &_dependencies = {}) {
  return internal::_asum(sb_handle, _N, _vx, _incx, _rs, _dependencies);
}

/**
 * \brief IAMAX finds the index of the first element having maximum
 * @param _vx BufferIterator or USM pointer
 * @param _incx Increment for the vector X
 * @param _dependencies Vector of events
 */
template <typename sb_handle_t, typename container_t, typename ContainerI,
          typename index_t, typename increment_t>
typename sb_handle_t::event_t _iamax(
    sb_handle_t &sb_handle, index_t _N, container_t _vx, increment_t _incx,
    ContainerI _rs, const typename sb_handle_t::event_t &_dependencies = {}) {
  return internal::_iamax(sb_handle, _N, _vx, _incx, _rs, _dependencies);
}

/**
 * \brief IAMIN finds the index of the first element having minimum
 * @param _vx BufferIterator or USM pointer
 * @param _incx Increment for the vector X
 * @param _rs BufferIterator or USM pointer
 * @param _dependencies Vector of events
 */
template <typename sb_handle_t, typename container_t, typename ContainerI,
          typename index_t, typename increment_t>
typename sb_handle_t::event_t _iamin(
    sb_handle_t &sb_handle, index_t _N, container_t _vx, increment_t _incx,
    ContainerI _rs, const typename sb_handle_t::event_t &_dependencies = {}) {
  return internal::_iamin(sb_handle, _N, _vx, _incx, _rs, _dependencies);
}

/**
 * \brief SWAP interchanges two vectors
 *
 * @param sb_handle_t sb_handle
 * @param _vx BufferIterator or USM pointer
 * @param _incx Increment for the vector X
 * @param _vy BufferIterator or USM pointer
 * @param _incy Increment for the vector Y
 * @param _dependencies Vector of events
 */
template <typename sb_handle_t, typename container_0_t, typename container_1_t,
          typename index_t, typename increment_t>
typename sb_handle_t::event_t _swap(
    sb_handle_t &sb_handle, index_t _N, container_0_t _vx, increment_t _incx,
    container_1_t _vy, increment_t _incy,
    const typename sb_handle_t::event_t &_dependencies = {}) {
  return internal::_swap(sb_handle, _N, _vx, _incx, _vy, _incy, _dependencies);
}

/**
 * \brief SCALAR operation on a vector
 * @param sb_handle_t sb_handle
 * @param _vx BufferIterator or USM pointer
 * @param _incx Increment for the vector X
 * @param _dependencies Vector of events
 */
template <typename sb_handle_t, typename element_t, typename container_0_t,
          typename index_t, typename increment_t>
typename sb_handle_t::event_t _scal(
    sb_handle_t &sb_handle, index_t _N, element_t _alpha, container_0_t _vx,
    increment_t _incx,
    const typename sb_handle_t::event_t &_dependencies = {}) {
  return internal::_scal(sb_handle, _N, _alpha, _vx, _incx, _dependencies);
}

/**
 * \brief NRM2 Returns the euclidian norm of a vector
 * @param sb_handle SB_Handle
 * @param _vx BufferIterator or USM pointer
 * @param _incx Increment for the vector X
 * @param _rs BufferIterator or USM pointer
 * @param _dependencies Vector of events
 */
template <typename sb_handle_t, typename container_0_t, typename container_1_t,
          typename index_t, typename increment_t>
typename sb_handle_t::event_t _nrm2(
    sb_handle_t &sb_handle, index_t _N, container_0_t _vx, increment_t _incx,
    container_1_t _rs,
    const typename sb_handle_t::event_t &_dependencies = {}) {
  return internal::_nrm2(sb_handle, _N, _vx, _incx, _rs, _dependencies);
}

/**
 * .
 * @brief _rot constructor given plane rotation
 *  *
 * @param sb_handle SB_Handle
 * @param _vx BufferIterator or USM pointer
 * @param _incx Increment for the vector X
 * @param _vx BufferIterator or USM pointer
 * @param _incy Increment for the vector Y
 * @param _sin sine
 * @param _cos cosine
 * @param _N data size
 * @param _dependencies Vector of events
 */
template <typename sb_handle_t, typename container_0_t, typename container_1_t,
          typename element_t, typename index_t, typename increment_t>
typename sb_handle_t::event_t _rot(
    sb_handle_t &sb_handle, index_t _N, container_0_t _vx, increment_t _incx,
    container_1_t _vy, increment_t _incy, element_t _cos, element_t _sin,
    const typename sb_handle_t::event_t &_dependencies = {}) {
  return internal::_rot(sb_handle, _N, _vx, _incx, _vy, _incy, _cos, _sin,
                        _dependencies);
}

/**
 * @brief Performs a modified Givens rotation of points.
 * Given two vectors x and y and a modified Givens transformation matrix, each
 * element of x and y is replaced as follows:
 *
 * [xi] = [h11 h12] * [xi]
 * [yi]   [h21 h22]   [yi]
 *
 * where h11, h12, h21 and h22 represent the modified Givens transformation
 * matrix.
 *
 * The value of the flag parameter can be used to modify the matrix as follows:
 *
 * -1.0: [h11 h12]     0.0: [1.0 h12]     1.0: [h11 1.0]     -2.0 = [1.0 0.0]
 *       [h21 h22]          [h21 1.0]          [-1.0 h22]           [0.0 1.0]
 *
 * @tparam sb_handle_t SB_Handle type
 * @tparam container_0_t Buffer Iterator or USM pointer
 * @tparam container_1_t Buffer Iterator or USM pointer
 * @tparam container_2_t Buffer Iterator or USM pointer
 * @tparam index_t Index type
 * @tparam increment_t Increment type
 * @param sb_handle SB_Handle
 * @param _N Input buffer sizes (for vx and vy).
 * @param[in, out] _vx Memory object holding input vector x
 * @param _incx Stride of vector x (i.e. measured in elements of _vx)
 * @param[in, out] _vy Memory object holding input vector y
 * @param _incy Stride of vector y (i.e. measured in elements of _vy)
 * @param[in] _param Buffer with the following layout: [flag, h11, h21, h12,
 * h22].
 * @param _dependencies Vector of events
 * @return Vector of events to wait for.
 */
template <typename sb_handle_t, typename container_0_t, typename container_1_t,
          typename container_2_t, typename index_t, typename increment_t>
typename sb_handle_t::event_t _rotm(
    sb_handle_t &sb_handle, index_t _N, container_0_t _vx, increment_t _incx,
    container_1_t _vy, increment_t _incy, container_2_t _param,
    const typename sb_handle_t::event_t &_dependencies = {}) {
  return internal::_rotm(sb_handle, _N, _vx, _incx, _vy, _incy, _param,
                         _dependencies);
}

/**
 * Given the Cartesian coordinates (x1, y1) of a point, the rotmg routines
 * compute the components of a modified Givens transformation matrix H that
 * zeros the y-component of the resulting point:
 *
 *                      [xi] = H * [xi * sqrt(d1) ]
 *                      [0 ]       [yi * sqrt(d2) ]
 *
 * Depending on the flag parameter, the components of H are set as follows:
 *
 * -1.0: [h11 h12]     0.0: [1.0 h12]     1.0: [h11 1.0]     -2.0 = [1.0 0.0]
 *       [h21 h22]          [h21 1.0]          [-1.0 h22]           [0.0 1.0]
 *
 * Rotmg may apply scaling operations to d1, d2 and x1 to avoid overflows.
 *
 * @tparam sb_handle_t SB_Handle type
 * @tparam container_0_t Buffer Iterator or USM pointer
 * @tparam container_1_t Buffer Iterator or USM pointer
 * @tparam container_2_t Buffer Iterator or USM pointer
 * @tparam container_3_t Buffer Iterator or USM pointer
 * @tparam container_4_t Buffer Iterator or USM pointer
 * @param sb_handle SB_Handle
 * @param _d1[in,out] On entry, memory object holding the scaling factor for
 * the x-coordinate. On exit, the re-scaled _d1.
 * @param _d2[in,out] On entry, memory object holding the scaling factor for
 * the y-coordinate. On exit, the re-scaled _d2.
 * @param _x1[in,out] On entry, memory object holding the x-coordinate. On
 * exit, the re-scaled _x1
 * @param _y1[in] Memory object holding the y-coordinate of the point.
 * @param _param[out] Buffer with the following layout: [flag, h11, h21, h12,
 * h22].
 * @param _dependencies Vector of events
 * @return Vector of events to wait for.
 */
template <typename sb_handle_t, typename container_0_t, typename container_1_t,
          typename container_2_t, typename container_3_t,
          typename container_4_t>
typename sb_handle_t::event_t _rotmg(
    sb_handle_t &sb_handle, container_0_t _d1, container_1_t _d2,
    container_2_t _x1, container_3_t _y1, container_4_t _param,
    const typename sb_handle_t::event_t &_dependencies = {}) {
  return internal::_rotmg(sb_handle, _d1, _d2, _x1, _y1, _param, _dependencies);
}

/**
 * \brief Given the Cartesian coordinates (a, b) of a point, the rotg routines
 * return the parameters c, s, r, and z associated with the Givens rotation.
 * @tparam sb_handle_t SB_Handle type
 * @tparam container_0_t Buffer Iterator or USM pointer
 * @tparam container_1_t Buffer Iterator or USM pointer
 * @tparam container_2_t Buffer Iterator or USM pointer
 * @tparam container_3_t Buffer Iterator or USM pointer
 * @param sb_handle SB_Handle
 * @param a[in, out] On entry, memory object holding the x-coordinate of the
 * point. On exit, the scalar z.
 * @param b[in, out] On entry, memory object holding the y-coordinate of the
 * point. On exit, the scalar r.
 * @param c[out] Memory object holding the parameter c.
 * @param s[out] Memory object holding the parameter s.
 * @param _dependencies Vector of events
 * @return Vector of events to wait for.
 */
template <typename sb_handle_t, typename container_0_t, typename container_1_t,
          typename container_2_t, typename container_3_t,
          typename std::enable_if<!is_sycl_scalar<container_0_t>::value,
                                  bool>::type = true>
typename sb_handle_t::event_t _rotg(
    sb_handle_t &sb_handle, container_0_t a, container_1_t b, container_2_t c,
    container_3_t s, const typename sb_handle_t::event_t &_dependencies = {}) {
  return internal::_rotg(sb_handle, a, b, c, s, _dependencies);
}

/**
 * \brief Synchronous version of rotg.
 * Given the Cartesian coordinates (a, b) of a point, the rotg routines
 * return the parameters c, s, r, and z associated with the Givens rotation.
 * @tparam sb_handle_t SB_Handle type
 * @tparam scalar_t Scalar type
 * @param sb_handle SB_Handle
 * @param a[in, out] On entry, x-coordinate of the point. On exit, the scalar
 * z.
 * @param b[in, out] On entry, y-coordinate of the point. On exit, the scalar
 * r.
 * @param c[out] scalar representing the output c.
 * @param s[out] scalar representing the output s.
 * @param _dependencies Vector of events
 */
template <
    typename sb_handle_t, typename scalar_t,
    typename std::enable_if<is_sycl_scalar<scalar_t>::value, bool>::type = true>
void _rotg(sb_handle_t &sb_handle, scalar_t &a, scalar_t &b, scalar_t &c,
           scalar_t &s,
           const typename sb_handle_t::event_t &_dependencies = {}) {
  internal::_rotg(sb_handle, a, b, c, s, _dependencies);
}

/**
 * \brief Computes the inner product of two vectors with double precision
 * accumulation (synchronous version that returns the result directly)
 * @tparam sb_handle_t SB_Handle type
 * @tparam container_0_t Buffer Iterator or USM pointer
 * @tparam container_1_t Buffer Iterator or USM pointer
 * @tparam container_2_t Buffer Iterator or USM pointer
 * @tparam index_t Index type
 * @tparam increment_t Increment type
 * @param sb_handle SB_Handle
 * @param _N Input buffer sizes.
 * @param _vx Memory object holding input vector x
 * @param _incx Stride of vector x (i.e. measured in elements of _vx)
 * @param _vy Memory object holding input vector y
 * @param _incy Stride of vector y (i.e. measured in elements of _vy)
 * @param _rs Output memory object
 * @param _dependencies Vector of events
 * @return Vector of events to wait for.
 */
template <typename sb_handle_t, typename container_0_t, typename container_1_t,
          typename index_t, typename increment_t>
typename ValueType<container_0_t>::type _dot(
    sb_handle_t &sb_handle, index_t _N, container_0_t _vx, increment_t _incx,
    container_1_t _vy, increment_t _incy,
    const typename sb_handle_t::event_t &_dependencies = {}) {
  return internal::_dot(sb_handle, _N, _vx, _incx, _vy, _incy, _dependencies);
}

/**
 * \brief Computes the inner product of two vectors with double precision
 * accumulation and adds a scalar to the result (synchronous version that
 * returns the result directly)
 * @tparam sb_handle_t SB_Handle type
 * @tparam container_0_t Buffer Iterator or USM pointer
 * @tparam container_1_t Buffer Iterator or USM pointer
 * @tparam container_2_t Buffer Iterator or USM pointer
 * @tparam index_t Index type
 * @tparam increment_t Increment type
 * @param sb_handle SB_Handle
 * @param _N Input buffer sizes. If size 0, the result will be sb.
 * @param sb Scalar to add to the results of the inner product.
 * @param _vx Memory object holding input vector x
 * @param _incx Stride of vector x (i.e. measured in elements of _vx)
 * @param _vy Memory object holding input vector y
 * @param _incy Stride of vector y (i.e. measured in elements of _vy)
 * @param _rs Output memory object
 * @param _dependencies Vector of events
 * @return Vector of events to wait for.
 */
template <typename sb_handle_t, typename container_0_t, typename container_1_t,
          typename index_t, typename increment_t>
typename ValueType<container_0_t>::type _sdsdot(
    sb_handle_t &sb_handle, index_t _N, float sb, container_0_t _vx,
    increment_t _incx, container_1_t _vy, increment_t _incy,
    const typename sb_handle_t::event_t &_dependencies = {}) {
  return internal::_sdsdot(sb_handle, _N, sb, _vx, _incx, _vy, _incy,
                           _dependencies);
}

/**
 * \brief ICAMAX finds the index of the first element having maximum
 * @param _vx BufferIterator or USM pointer
 * @param _incx Increment for the vector X
 * @param _dependencies Vector of events
 */
template <typename sb_handle_t, typename container_t, typename index_t,
          typename increment_t>
index_t _iamax(sb_handle_t &sb_handle, index_t _N, container_t _vx,
               increment_t _incx,
               const typename sb_handle_t::event_t &_dependencies = {}) {
  return internal::_iamax(sb_handle, _N, _vx, _incx, _dependencies);
}

/**
 * \brief ICAMIN finds the index of the first element having minimum
 * @param _vx BufferIterator or USM pointer
 * @param _incx Increment for the vector X
 * @param _dependencies Vector of events
 */
template <typename sb_handle_t, typename container_t, typename index_t,
          typename increment_t>
index_t _iamin(sb_handle_t &sb_handle, index_t _N, container_t _vx,
               increment_t _incx,
               const typename sb_handle_t::event_t &_dependencies = {}) {
  return internal::_iamin(sb_handle, _N, _vx, _incx, _dependencies);
}

/**
 * \brief ASUM Takes the sum of the absolute values
 *
 * @param sb_handle SB_Handle
 * @param _vx BufferIterator or USM pointer
 * @param _incx Increment for the vector X
 * @param _dependencies Vector of events
 */
template <typename sb_handle_t, typename container_t, typename index_t,
          typename increment_t>
typename ValueType<container_t>::type _asum(
    sb_handle_t &sb_handle, index_t _N, container_t _vx, increment_t _incx,
    const typename sb_handle_t::event_t &_dependencies = {}) {
  return internal::_asum(sb_handle, _N, _vx, _incx, _dependencies);
}

/**
 * \brief NRM2 Returns the euclidian norm of a vector
 *
 * @param sb_handle SB_Handle
 * @param _vx BufferIterator or USM pointer
 * @param _incx Increment for the vector X
 * @param _dependencies Vector of events
 */
template <typename sb_handle_t, typename container_t, typename index_t,
          typename increment_t>
typename ValueType<container_t>::type _nrm2(
    sb_handle_t &sb_handle, index_t _N, container_t _vx, increment_t _incx,
    const typename sb_handle_t::event_t &_dependencies = {}) {
  return internal::_nrm2(sb_handle, _N, _vx, _incx, _dependencies);
}

}  // end namespace blas
#endif  // PORTBLAS_BLAS1_INTERFACE
