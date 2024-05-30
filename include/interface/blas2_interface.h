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
 *  @filename blas2_interface.h
 *
 **************************************************************************/

#ifndef PORTBLAS_BLAS2_INTERFACE_H
#define PORTBLAS_BLAS2_INTERFACE_H

#include "operations/blas2_trees.h"
namespace blas {
namespace internal {
/*!
 @brief Generalised matrix vector product with a rectangular non-symmetric
 matrix.

 Generalised matrix vector product with a rectangular non-symmetric matrix, i.e.
 computing the mathematical operation:

 y = alpha*A*x + beta*y

 See the netlib blas interface documentation for more details of the high level
 interface: http://www.netlib.org/lapack/explore-html/db/d58/sgemv_8f.html

 */
template <typename sb_handle_t, typename index_t, typename element_t,
          typename container_0_t, typename container_1_t, typename increment_t,
          typename container_2_t>
typename sb_handle_t::event_t _gemv(
    sb_handle_t& sb_handle,  // sb_handle_t (sycl, parallel, serial, etc)
    char _trans,             // The transposition of the matrix ('n', 't', 'c')
    index_t _M,              // The size of dimension M of the matrix (rows)
    index_t _N,              // The size of dimension N of the matrix (columns)
    element_t _alpha,        // Scalar parameter Alpha
    container_0_t _mA,       // An array (LDA,N), with the first m*n elements
    index_t _lda,            // Specifies the first dimension of a, max(1, m)
    container_1_t _vx,  // An array of dimension at least: (1+(n-1)*abs(incx))
                        // when trans = 'n' and (1+(m-1)*abs(incx) otherwise,
                        // containing the vector "x"
    increment_t _incx,  // The increment for elements in x (nonzero).
    element_t _beta,    // Scalar parameter Beta
    container_2_t _vy,  // An array of dimension at least: (1+(m-1)*abs(incy))
                        // when trans = "n" and (1+(n-1)*abs(incy) otherwise,
    // containing the vector "y" (if beta is nonzero). When
    // finished, y is overwritten with the updated vector.
    increment_t _incy,  // The increment for elements in y (nonzero).
    const typename sb_handle_t::event_t& _dependencies  // Vector of events
);

/*!
 * @brief Prototype for the internal implementation of the GEMV operation. See
 * documentation in the blas2_interface.hpp file for details.
 */
template <uint32_t local_range, uint32_t cache_line_size,
          gemv_memory_t memory_type, transpose_type trn, typename sb_handle_t,
          typename index_t, typename element_t, typename container_t0,
          typename container_t1, typename increment_t, typename container_t2>
typename sb_handle_t::event_t _gemv_impl(
    sb_handle_t& sb_handle, index_t _M, index_t _N, element_t _alpha,
    container_t0 _mA, index_t _lda, container_t1 _vx, increment_t _incx,
    element_t _beta, container_t2 _vy, increment_t _incy,
    const typename sb_handle_t::event_t& _dependencies);

/*!
 @brief Generalised matrix vector product with a triangular symmetric matrix.

 Generalised matrix vector product with a triangular symmetric matrix, i.e.
 computing the mathematical operation:

 x = A*x

 See the netlib blas interface documentation for more details of the high level
 interface: http://www.netlib.org/lapack/explore-html/de/d45/strmv_8f.html

 */
template <typename sb_handle_t, typename index_t, typename container_0_t,
          typename container_1_t, typename increment_t>
typename sb_handle_t::event_t _trmv(
    sb_handle_t& sb_handle,  // sb_handle_t (sycl, parallel, serial, etc)
    char _Uplo,              // Whether the matrix is upper/lower ('u', 'l')
    char _trans,             // Whether the matrix is transposed ('n', 't', 'c')
    char _Diag,              // Whether the matrix is unit triangular ('u', 'n')
    index_t _N,              // >0 The order of matrix A
    container_0_t _mA,       // (_lda, _N) The input matrix
    index_t _lda,            // >max(1, _N) The first dimension of _mA
    container_1_t _vx,       // (1 + (_N-1)*abs(_incx)), output vector X
    increment_t _incx,       // !=0 The increment for the elements of X
    const typename sb_handle_t::event_t& _dependencies  // Vector of events
);

/**
 * @brief Linear system solver for triangular matrices.
 *
 * Linear system solver for triangular matrices, i.e., computing x s.t.
 *
 * op(A)*x = x
 *
 * See the netlib blas interface documentation for more details of the
 * interface: https://netlib.org/lapack/explore-html/d0/d2a/strsv_8f.html
 *
 * @param sb_handle SB_handle
 * @param _Uplo Specifies if A is upper or lower triangular
 * @param _trans Transposition operation applied to A ('n', 't', 'c')
 * @param _Diag Specifies if A unit triangular or not
 * @param _N Number of rows and columns of A
 * @param _mA A buffer (_LDA,_N) containing the coefficient of A
 * @param _lda Leading dimension _mA at least _N
 * @param _vx Buffer containing x of at least (1+(_N-1)*abs(_incx)) elements
 * @param _incx Increment for _vx (nonzero)
 * @param _dependencies Vector of events
 */
template <typename sb_handle_t, typename index_t, typename container_0_t,
          typename container_1_t, typename increment_t>
typename sb_handle_t::event_t _trsv(
    sb_handle_t& sb_handle, char _Uplo, char _trans, char _Diag, index_t _N,
    container_0_t _mA, index_t _lda, container_1_t _vx, increment_t _incx,
    const typename sb_handle_t::event_t& _dependencies = {});

template <uint32_t subgroup_size, uint32_t subgroups, uplo_type uplo,
          transpose_type trn, diag_type diag, typename sb_handle_t,
          typename index_t, typename container_t0, typename container_t1,
          typename increment_t>
typename sb_handle_t::event_t _trsv_impl(
    sb_handle_t& sb_handle, index_t _N, container_t0 _mA, index_t _lda,
    container_t1 _vx, increment_t _incx,
    const typename sb_handle_t::event_t& _dependencies);

/*!
 @brief Generalised matrix vector product with a square symmetric matrix,
 followed by a vector sum.

 Generalised matrix vector product with a square symmetric matrix, followed by
 a vector sum, i.e. computing the mathematical operation:

 x = alpha*A*x + beta*y

 See the netlib blas interface documentation for more details of the high level
 interface: http://www.netlib.org/lapack/explore-html/d2/d94/ssymv_8f.html

 */
template <typename sb_handle_t, typename index_t, typename element_t,
          typename container_0_t, typename container_1_t, typename increment_t,
          typename container_2_t>
typename sb_handle_t::event_t _symv(
    sb_handle_t& sb_handle,  // sb_handle_t (sycl, parallel, serial, etc)
    char _Uplo,              // Whether the matrix is upper/lower ('u', 'l')
    index_t _N,              // >0 The order of matrix A
    element_t _alpha,        // Scalar parameter alpha
    container_0_t _mA,       // (_lda, _N) The input matrix
    index_t _lda,            // >max(1, _N) The first dimension of _mA
    container_1_t _vx,       // (1 + (_N-1)*abs(_incx)), input vector X
    increment_t _incx,       // !=0 The increment for the elements of X
    element_t _beta,         // Scalar parameter beta
    container_2_t _vy,       // (1 + (_N-1)*abs(_incy)), output vector Y
    increment_t _incy,       // !=0 The increment for the elements of Y
    const typename sb_handle_t::event_t& _dependencies  // Vector of events
);

/*!
 * @brief Generalised vector product followed by a sum with a rectangular
 * non-symmetric matrix.
 *
 * Generalised vector product followed by a sum with a rectangular non-symmetric
 * matrix, i.e. computing the mathematical operation:
 *
 * A = alpha*x*yT + A
 *
 * See the netlib blas interface documentation for more details of the high
 * level interface:
 * http://www.netlib.org/lapack/explore-html/db/d5c/sger_8f.html
 *
 * @param sb_handle SB_handle
 * @param _M Number of rows in matrix A
 * @param _N Number of columns in matrix A
 * @param _alpha Scalar alpha
 * @param _vx Input vector having (1 + (_M-1)*abs(_incx)) elements
 * @param _incx Increment for vector X
 * @param _vy, Input vector having having (1 + (_N-1)*abs(_incy)) elements
 * @param _incy Increment for vector Y
 * @param _mA Input/output matrix A(_lda, n)
 * @param _lda Leading dimension of A
 * @param _dependencies Vector of events
 */
template <typename sb_handle_t, typename index_t, typename element_t,
          typename container_0_t, typename increment_t, typename container_1_t,
          typename container_2_t>
typename sb_handle_t::event_t _ger(
    sb_handle_t& sb_handle, index_t _M, index_t _N, element_t _alpha,
    container_0_t _vx, increment_t _incx, container_1_t _vy, increment_t _incy,
    container_2_t _mA, index_t _lda,
    const typename sb_handle_t::event_t& _dependencies);

/*!
 @brief Generalised vector squaring followed by a sum with a symmetric matrix.

 Generalised vector squaring followed by a sum with a symmetric matrix,
 i.e. computing the mathematical operation:

 A = alpha*x*xT + A

 See the netlib blas interface documentation for more details of the high level
 interface: http://www.netlib.org/lapack/explore-html/db/d99/ssyr2_8f.html

 */
template <typename sb_handle_t, typename index_t, typename element_t,
          typename container_0_t, typename increment_t, typename container_1_t>
typename sb_handle_t::event_t _syr(
    sb_handle_t& sb_handle,  // sb_handle_t (sycl, parallel, serial, etc)
    char _Uplo,              // Whether the matrix is upper/lower ('u', 'l')
    index_t _N,              // >0 The order of matrix A
    element_t _alpha,        // Scalar alpha
    container_0_t _vx,       // (1 + (_N-1)*abs(_incx)), input vector X
    increment_t _incx,       // !=0 The increment for the elements of X
    container_1_t _mA,       // (_lda, _N) The output matrix
    index_t _lda,            // >max(1, _N) The first dimension of _mA
    const typename sb_handle_t::event_t& _dependencies  // Vector of events
);

/**
 * @brief Generalised vector squaring followed by a sum with a packed symmetric
 * matrix.
 *
 * Generalised vector squaring followed by a sum with a packed symmetric matrix,
 * i.e. computing the mathematical operation:

 * A = alpha*x*xT + A

 * See the netlib blas interface documentation for more details of the high
 * level interface:
 * https://netlib.org/lapack/explore-html/d2/d9b/sspr_8f.html
 *
 * @param sb_handle sb_handle_t (sycl, parallel, serial, etc)
 * @param _Uplo Whether the matrix is upper/lower ('u', 'l')
 * @param _N >0 The order of matrix A
 * @param _alpha Scalar multiplier
 * @param _vx (1 + (_N-1)*abs(_incx)), input vector X
 * @param _incx !=0 The increment for the elements of X
 * @param _mPA (_lda, _N) The output matrix in packed format
 * @param _dependencies Vector of events
 */
template <typename sb_handle_t, typename index_t, typename element_t,
          typename container_0_t, typename increment_t, typename container_1_t>
typename sb_handle_t::event_t _spr(
    sb_handle_t& sb_handle, char _Uplo, index_t _N, element_t _alpha,
    container_0_t _vx, increment_t _incx, container_1_t _mPA,
    const typename sb_handle_t::event_t& _dependencies);

/**
 * @brief Generalised two vectors squaring followed by a sum with a packed
 * symmetric matrix.
 *
 * Generalised two vector squaring followed by a sum with a packed symmetric
 * matrix, i.e. computing the mathematical operation:

 * A = alpha*x*yT + alpha*y*xT + A

 * See the netlib blas interface documentation for more details of the high
 * level interface:
 * https://netlib.org/lapack/explore-html/db/d3e/sspr2_8f.html
 *
 * @param sb_handle sb_handle_t (sycl, parallel, serial, etc)
 * @param _Uplo Whether the matrix is upper/lower ('u', 'l')
 * @param _N >0 The order of matrix A
 * @param _alpha Scalar multiplier
 * @param _vx (1 + (_N-1)*abs(_incx)), input vector X
 * @param _incx !=0 The increment for the elements of X
 * @param _vy (1 + (_N-1)*abs(_incy)), input vector Y
 * @param _incy !=0 The increment for the elements of Y
 * @param _mPA (_lda, _N) The output matrix in packed format
 * @param _dependencies Vector of events
 */
template <typename sb_handle_t, typename index_t, typename element_t,
          typename container_t0, typename increment_t, typename container_t1,
          typename container_t2>
typename sb_handle_t::event_t _spr2(
    sb_handle_t& sb_handle, char _Uplo, index_t _N, element_t _alpha,
    container_t0 _vx, increment_t _incx, container_t1 _vy, increment_t _incy,
    container_t2 _mPA, const typename sb_handle_t::event_t& _dependencies);

/*!
 @brief Generalised vector products followed by a sum with a symmetric matrix.

 Generalised vector products followed by a sum with a symmetric matrix,
 i.e. computing the mathematical operation:

 A = alpha*x*yT + alpha*y*xT + A

 See the netlib blas interface documentation for more details of the high level
 interface: http://www.netlib.org/lapack/explore-html/d6/dac/ssyr_8f.html

 */
template <typename sb_handle_t, typename index_t, typename element_t,
          typename container_0_t, typename increment_t, typename container_1_t,
          typename container_2_t>
typename sb_handle_t::event_t _syr2(
    sb_handle_t& sb_handle,  // sb_handle_t (sycl, parallel, serial, etc)
    char _Uplo,              // Whether the matrix is upper/lower ('u', 'l')
    index_t _N,              // >0 The order of matrix A
    element_t _alpha,        // Scalar alpha
    container_0_t _vx,       // (1 + (_N-1)*abs(_incx)), input vector X
    increment_t _incx,       // !=0 The increment for the elements of X
    container_1_t _vy,       // (1 + (_N-1)*abs(_incx)), input vector Y
    increment_t _incy,       // !=0 The increment for the elements of Y
    container_2_t _mA,       // (_lda, _N) The output matrix
    index_t _lda,            // >max(1, _N) The first dimension of _mA
    const typename sb_handle_t::event_t& _dependencies  // Vector of events
);

/**
 * @brief Generalised matrix vector product with band matrices.
 *
 * Generalised matrix vector product with a band matrix, i.e. computing the
 * mathematical operation:
 *
 * y = alpha*op(A)*x + beta*y
 *
 * See the netlib blas interface documentation for more details of the
 * interface: https://netlib.org/lapack/explore-html/d6/d46/sgbmv_8f.html
 *
 * @param sb_handle SB_handle
 * @param _trans Transposition operation applied to A ('n', 't', 'c')
 * @param _M Number of rows of A
 * @param _N Number of columns of A
 * @param _KL Number of A sub-diagonals
 * @param _KU Number of A super-diagonals
 * @param _alpha Scalar parameter alpha
 * @param _mA Buffer (_LDA,_N) containing the coefficient of A in the Band
 *            Matrix format
 * @param _lda Leading dimension _mA at least (_KL + _KU + 1)
 * @param _vx Buffer containing x of at least (1+(_N-1)*abs(_incx)) elements
 *            when trans = 'n' and (1+(_M-1)*abs(_incx) otherwise
 * @param _incx Increment for _vx (nonzero)
 * @param _beta Scalar parameter beta
 * @param _vy Buffer containing y of at least (1+(_M-1)*abs(_incy)) elements
 *            when trans = 'n' and (1+(_N-1)*abs(_incy) otherwise
 * @param _incy Increment for _vy
 * @param _dependencies Vector of events
 */
template <typename sb_handle_t, typename index_t, typename element_t,
          typename container_0_t, typename container_1_t, typename increment_t,
          typename container_2_t>
typename sb_handle_t::event_t _gbmv(
    sb_handle_t& sb_handle, char _trans, index_t _M, index_t _N, index_t _KL,
    index_t _KU, element_t _alpha, container_0_t _mA, index_t _lda,
    container_1_t _vx, increment_t _incx, element_t _beta, container_2_t _vy,
    increment_t _incy, const typename sb_handle_t::event_t& _dependencies);

template <uint32_t local_range, transpose_type trn, typename sb_handle_t,
          typename index_t, typename element_t, typename container_t0,
          typename container_t1, typename increment_t, typename container_t2>
typename sb_handle_t::event_t _gbmv_impl(
    sb_handle_t& sb_handle, index_t _M, index_t _N, index_t _KL, index_t _KU,
    element_t _alpha, container_t0 _mA, index_t _lda, container_t1 _vx,
    increment_t _incx, element_t _beta, container_t2 _vy, increment_t _incy,
    const typename sb_handle_t::event_t& _dependencies);

/**
 * @brief Matrix vector product with symmetric band matrices.
 *
 * Matrix vector product with a symmetric band matrix, i.e. computing the
 * mathematical operation:
 *
 * y = alpha*A*x + beta*y
 *
 * See the netlib blas interface documentation for more details of the
 * interface: https://netlib.org/lapack/explore-html/d3/da1/ssbmv_8f.html
 *
 * @param sb_handle SB_handle
 * @param _Uplo Specifies if A is upper or lower triangular
 * @param _N Number of rows and columns of A
 * @param _K Number of A super-diagonals
 * @param _alpha Scalar parameter alpha
 * @param _mA Buffer (_LDA,_N) containing the coefficient of A in the Band
 *            Matrix format
 * @param _lda Leading dimension _mA at least (_K + 1)
 * @param _vx Buffer containing x of at least (1+(_N-1)*abs(_incx)) elements
 * @param _incx Increment for _vx (nonzero)
 * @param _beta Scalar parameter beta
 * @param _vy Buffer containing y of at least (1+(_N-1)*abs(_incy)) elements
 * @param _incy Increment for _vy
 * @param _dependencies Vector of events
 */
template <typename sb_handle_t, typename index_t, typename element_t,
          typename container_0_t, typename container_1_t, typename increment_t,
          typename container_2_t>
typename sb_handle_t::event_t _sbmv(
    sb_handle_t& sb_handle, char _Uplo, index_t _N, index_t _K,
    element_t _alpha, container_0_t _mA, index_t _lda, container_1_t _vx,
    increment_t _incx, element_t _beta, container_2_t _vy, increment_t _incy,
    const typename sb_handle_t::event_t& _dependencies);

template <uint32_t local_range, uplo_type uplo, typename sb_handle_t,
          typename index_t, typename element_t, typename container_t0,
          typename container_t1, typename increment_t, typename container_t2>
typename sb_handle_t::event_t _sbmv_impl(
    sb_handle_t& sb_handle, index_t _N, index_t _K, element_t _alpha,
    container_t0 _mA, index_t _lda, container_t1 _vx, increment_t _incx,
    element_t _beta, container_t2 _vy, increment_t _incy,
    const typename sb_handle_t::event_t& _dependencies);

/**
 * @brief Matrix vector product with symmetric packed matrices.
 *
 * Matrix vector product with a symmetric packed matrix, i.e. computing the
 * mathematical operation:
 *
 * y = alpha*A*x + beta*y
 *
 * See the netlib blas interface documentation for more details of the
 * interface: https://netlib.org/lapack/explore-html/d8/d68/sspmv_8f.html
 *
 * @param sb_handle SB_handle
 * @param _Uplo Specifies if A is upper or lower triangular
 * @param _N Number of rows and columns of A
 * @param _alpha Scalar parameter alpha
 * @param _mA Buffer containing the coefficient of A in the Packed Triangular
 *            Matrix format
 * @param _vx Buffer containing x of at least (1+(_N-1)*abs(_incx)) elements
 * @param _incx Increment for _vx (nonzero)
 * @param _beta Scalar parameter beta
 * @param _vy Buffer containing y of at least (1+(_N-1)*abs(_incy)) elements
 * @param _incy Increment for _vy
 */
template <typename sb_handle_t, typename index_t, typename element_t,
          typename container_0_t, typename container_1_t, typename increment_t,
          typename container_2_t>
typename sb_handle_t::event_t _spmv(
    sb_handle_t& sb_handle, char _Uplo, index_t _N, element_t _alpha,
    container_0_t _mA, container_1_t _vx, increment_t _incx, element_t _beta,
    container_2_t _vy, increment_t _incy,
    const typename sb_handle_t::event_t& _dependencies);

template <uint32_t local_range_x, uint32_t local_range_y, uplo_type uplo,
          typename sb_handle_t, typename index_t, typename element_t,
          typename container_t0, typename container_t1, typename increment_t,
          typename container_t2>
typename sb_handle_t::event_t _spmv_impl(
    sb_handle_t& sb_handle, index_t _N, element_t _alpha, container_t0 _mA,
    container_t1 _vx, increment_t _incx, element_t _beta, container_t2 _vy,
    increment_t _incy, const typename sb_handle_t::event_t& _dependencies);

/**
 * @brief Matrix vector product with triangular band matrices.
 *
 * Matrix vector product with a triangular band matrix, i.e. computing the
 * mathematical operation:
 *
 * x = op(A)*x
 *
 * See the netlib blas interface documentation for more details of the
 * interface: https://netlib.org/lapack/explore-html/d6/d7d/stbmv_8f.html
 *
 * @param sb_handle SB_handle
 * @param _Uplo Specifies if A is upper or lower triangular
 * @param _trans Transposition operation applied to A ('n', 't', 'c')
 * @param _Diag Specifies if A unit triangular or not
 * @param _N Number of rows and columns of A
 * @param _K Number of A super-diagonals
 * @param _mA Buffer (_LDA,_N) containing the coefficient of A in the Band
 *            Matrix format
 * @param _lda Leading dimension _mA at least (_K + 1)
 * @param _vx Buffer containing x of at least (1+(_N-1)*abs(_incx)) elements
 * @param _incx Increment for _vx (nonzero)
 * @param _dependencies Vector of events
 */
template <typename sb_handle_t, typename index_t, typename container_0_t,
          typename container_1_t, typename increment_t>
typename sb_handle_t::event_t _tbmv(
    sb_handle_t& sb_handle, char _Uplo, char _trans, char _Diag, index_t _N,
    index_t _K, container_0_t _mA, index_t _lda, container_1_t _vx,
    increment_t _incx, const typename sb_handle_t::event_t& _dependencies);

template <uint32_t local_range, uplo_type uplo, transpose_type trn,
          diag_type diag, typename sb_handle_t, typename index_t,
          typename container_t0, typename container_t1, typename increment_t>
typename sb_handle_t::event_t _tbmv_impl(
    sb_handle_t& sb_handle, index_t _N, index_t _K, container_t0 _mA,
    index_t _lda, container_t1 _vx, increment_t _incx,
    const typename sb_handle_t::event_t& _dependencies);

/**
 * @brief Matrix vector product with triangular packed matrices.
 *
 * Matrix vector product with a triangular band matrix, i.e. computing the
 * mathematical operation:
 *
 * x = op(A)*x
 *
 * See the netlib blas interface documentation for more details of the
 * interface: https://netlib.org/lapack/explore-html/db/db1/stpmv_8f.html
 *
 * @param sb_handle SB_handle
 * @param _Uplo Specifies if A is upper or lower triangular
 * @param _trans Transposition operation applied to A ('n', 't', 'c')
 * @param _Diag Specifies if A unit triangular or not
 * @param _N Number of rows and columns of A
 * @param _ma buffer containing the coefficient of a in the packed triangular
 *            matrix format
 * @param _vx Buffer containing x of at least (1+(_N-1)*abs(_incx)) elements
 * @param _incx Increment for _vx (nonzero)
 * @param _dependencies Vector of events
 */
template <typename sb_handle_t, typename index_t, typename container_0_t,
          typename container_1_t, typename increment_t>
typename sb_handle_t::event_t _tpmv(
    sb_handle_t& sb_handle, char _Uplo, char _trans, char _Diag, index_t _N,
    container_0_t _mA, container_1_t _vx, increment_t _incx,
    const typename sb_handle_t::event_t& _dependencies);

template <uint32_t local_range_x, uint32_t local_range_y, uplo_type uplo,
          transpose_type trn, diag_type diag, typename sb_handle_t,
          typename index_t, typename container_t0, typename container_t1,
          typename increment_t>
typename sb_handle_t::event_t _tpmv_impl(
    sb_handle_t& sb_handle, index_t _N, container_t0 _mA, container_t1 _vx,
    increment_t _incx, const typename sb_handle_t::event_t& _dependencies);

/**
 * @brief Linear system solver for triangular band matrices.
 *
 * Linear system solver for triangular band matrices, i.e., computing x s.t.
 *
 * op(A)*x = b
 *
 * See the netlib blas interface documentation for more details of the
 * interface: https://netlib.org/lapack/explore-html/d0/d1f/stbsv_8f.html
 *
 * @param sb_handle SB_handle
 * @param _Uplo Specifies if A is upper or lower triangular
 * @param _trans Transposition operation applied to A ('n', 't', 'c')
 * @param _Diag Specifies if A unit triangular or not
 * @param _N Number of rows and columns of A
 * @param _K Number of A super-diagonals
 * @param _mA Buffer (_LDA,_N) containing the coefficient of A in the Band
 *            Matrix format
 * @param _lda Leading dimension _mA at least (_K + 1)
 * @param _vx Buffer containing x of at least (1+(_N-1)*abs(_incx)) elements
 * @param _incx Increment for _vx (nonzero)
 * @param _dependencies Vector of events
 */
template <typename sb_handle_t, typename index_t, typename container_0_t,
          typename container_1_t, typename increment_t>
typename sb_handle_t::event_t _tbsv(
    sb_handle_t& sb_handle, char _Uplo, char _trans, char _Diag, index_t _N,
    index_t _K, container_0_t _mA, index_t _lda, container_1_t _vx,
    increment_t _incx, const typename sb_handle_t::event_t& _dependencies);

template <uint32_t subgroup_size, uint32_t subgroups, uplo_type uplo,
          transpose_type trn, diag_type diag, typename sb_handle_t,
          typename index_t, typename container_t0, typename container_t1,
          typename increment_t>
typename sb_handle_t::event_t _tbsv_impl(
    sb_handle_t& sb_handle, index_t _N, index_t _K, container_t0 _mA,
    index_t _lda, container_t1 _vx, increment_t _incx,
    const typename sb_handle_t::event_t& _dependencies);

/**
 * @brief Linear system solver for triangular packed matrices.
 *
 * Linear system solver for triangular packed matrices, i.e., computing x s.t.
 *
 * op(A)*x = b
 *
 * See the netlib blas interface documentation for more details of the
 * interface: https://netlib.org/lapack/explore-html/d0/d7c/stpsv_8f.html
 *
 * @param sb_handle SB_handle
 * @param _Uplo Specifies if A is upper or lower triangular
 * @param _trans Transposition operation applied to A ('n', 't', 'c')
 * @param _Diag Specifies if A unit triangular or not
 * @param _N Number of rows and columns of A
 * @param _mA Buffer containing the coefficient of A in the Packed Triangular
 *            Matrix format
 * @param _vx Buffer containing x of at least (1+(_N-1)*abs(_incx)) elements
 * @param _incx Increment for _vx (nonzero)
* @param _dependencies Vector of events
 */
template <typename sb_handle_t, typename index_t, typename container_0_t,
          typename container_1_t, typename increment_t>
typename sb_handle_t::event_t _tpsv(sb_handle_t& sb_handle, char _Uplo,
                                    char _trans, char _Diag, index_t _N,
                                    container_0_t _mA, container_1_t _vx,
                                    increment_t _incx,
                                    const typename sb_handle_t::event_t& _dependencies);

template <uint32_t subgroup_size, uint32_t subgroups, uplo_type uplo,
          transpose_type trn, diag_type diag, typename sb_handle_t,
          typename index_t, typename container_t0, typename container_t1,
          typename increment_t>
typename sb_handle_t::event_t _tpsv_impl(sb_handle_t& sb_handle, index_t _N,
                                         container_t0 _mA, container_t1 _vx,
                                         increment_t _incx,
                                         const typename sb_handle_t::event_t& _dependencies);

}  // namespace internal

/*!
 @brief Generalised matrix vector product with a rectangular non-symmetric
 matrix.

 Generalised matrix vector product with a rectangular non-symmetric matrix, i.e.
 computing the mathematical operation:

 y = alpha*A*x + beta*y

 See the netlib blas interface documentation for more details of the high level
 interface: http://www.netlib.org/lapack/explore-html/db/d58/sgemv_8f.html

 */
template <typename sb_handle_t, typename index_t, typename element_t,
          typename container_0_t, typename container_1_t, typename increment_t,
          typename container_2_t>
typename sb_handle_t::event_t inline _gemv(
    sb_handle_t& sb_handle,  // sb_handle_t (sycl, parallel, serial, etc)
    char _trans,             // The transposition of the matrix ('n', 't', 'c')
    index_t _M,              // The size of dimension M of the matrix (rows)
    index_t _N,              // The size of dimension N of the matrix (columns)
    element_t _alpha,        // Scalar parameter Alpha
    container_0_t _mA,       // An array (LDA,N), with the first m*n elements
    index_t _lda,            // Specifies the first dimension of a, max(1, m)
    container_1_t _vx,  // An array of dimension at least: (1+(n-1)*abs(incx))
                        // when trans = 'n' and (1+(m-1)*abs(incx) otherwise,
                        // containing the vector "x"
    increment_t _incx,  // The increment for elements in x (nonzero).
    element_t _beta,    // Scalar parameter Beta
    container_2_t _vy,  // An array of dimension at least: (1+(m-1)*abs(incy))
                        // when trans = "n" and (1+(n-1)*abs(incy) otherwise,
    // containing the vector "y" (if beta is nonzero). When
    // finished, y is overwritten with the updated vector.
    increment_t _incy,  // The increment for elements in y (nonzero).
    const typename sb_handle_t::event_t& _dependencies = {}  // Vector of events
) {
  return internal::_gemv(sb_handle, _trans, _M, _N, _alpha, _mA, _lda, _vx,
                         _incx, _beta, _vy, _incy, _dependencies);
}

/*!
 @brief Generalised matrix vector product with a triangular symmetric matrix.

 Generalised matrix vector product with a triangular symmetric matrix, i.e.
 computing the mathematical operation:

 x = A*x

 See the netlib blas interface documentation for more details of the high level
 interface: http://www.netlib.org/lapack/explore-html/de/d45/strmv_8f.html

 */
template <typename sb_handle_t, typename index_t, typename container_0_t,
          typename container_1_t, typename increment_t>
typename sb_handle_t::event_t inline _trmv(
    sb_handle_t& sb_handle,  // sb_handle_t (sycl, parallel, serial, etc)
    char _Uplo,              // Whether the matrix is upper/lower ('u', 'l')
    char _trans,             // Whether the matrix is transposed ('n', 't', 'c')
    char _Diag,              // Whether the matrix is unit triangular ('u', 'n')
    index_t _N,              // >0 The order of matrix A
    container_0_t _mA,       // (_lda, _N) The input matrix
    index_t _lda,            // >max(1, _N) The first dimension of _mA
    container_1_t _vx,       // (1 + (_N-1)*abs(_incx)), output vector X
    increment_t _incx,       // !=0 The increment for the elements of X
    const typename sb_handle_t::event_t& _dependencies = {}  // Vector of events
) {
  return internal::_trmv(sb_handle, _Uplo, _trans, _Diag, _N, _mA, _lda, _vx,
                         _incx, _dependencies);
}

/**
 * @brief Linear system solver for triangular matrices.
 *
 * Linear system solver for triangular matrices, i.e., computing x s.t.
 *
 * op(A)*x = x
 *
 * See the netlib blas interface documentation for more details of the
 * interface: https://netlib.org/lapack/explore-html/d0/d2a/strsv_8f.html
 *
 * @param sb_handle SB_handle
 * @param _Uplo Specifies if A is upper or lower triangular
 * @param _trans Transposition operation applied to A ('n', 't', 'c')
 * @param _Diag Specifies if A unit triangular or not
 * @param _N Number of rows and columns of A
 * @param _mA A buffer (_LDA,_N) containing the coefficient of A
 * @param _lda Leading dimension _mA at least _N
 * @param _vx Buffer containing x of at least (1+(_N-1)*abs(_incx)) elements
 * @param _incx Increment for _vx (nonzero)
 * @param _dependencies Vector of events
 */
template <typename sb_handle_t, typename index_t, typename container_0_t,
          typename container_1_t, typename increment_t>
typename sb_handle_t::event_t inline _trsv(
    sb_handle_t& sb_handle, char _Uplo, char _trans, char _Diag, index_t _N,
    container_0_t _mA, index_t _lda, container_1_t _vx, increment_t _incx,
    const typename sb_handle_t::event_t& _dependencies = {}) {
  return internal::_trsv(sb_handle, _Uplo, _trans, _Diag, _N, _mA, _lda, _vx,
                         _incx, _dependencies);
}

/*!
 @brief Generalised matrix vector product with a rectangular symmetric
 matrix, followed by a vector sum.

 Generalised matrix vector product with a rectangular symmetric
 matrix, followed by a vector sum, i.e.
 computing the mathematical operation:

 x = alpha*A*x + beta*y

 See the netlib blas interface documentation for more details of the high level
 interface: http://www.netlib.org/lapack/explore-html/d2/d94/ssymv_8f.html

 */
template <typename sb_handle_t, typename index_t, typename element_t,
          typename container_0_t, typename container_1_t, typename increment_t,
          typename container_2_t>
typename sb_handle_t::event_t inline _symv(
    sb_handle_t& sb_handle,  // sb_handle_t (sycl, parallel, serial, etc)
    char _Uplo,              // Whether the matrix is upper/lower ('u', 'l')
    index_t _N,              // >0 The order of matrix A
    element_t _alpha,        // Scalar parameter alpha
    container_0_t _mA,       // (_lda, _N) The input matrix
    index_t _lda,            // >max(1, _N) The first dimension of _mA
    container_1_t _vx,       // (1 + (_N-1)*abs(_incx)), input vector X
    increment_t _incx,       // !=0 The increment for the elements of X
    element_t _beta,         // Scalar parameter beta
    container_2_t _vy,       // (1 + (_N-1)*abs(_incy)), output vector Y
    increment_t _incy,       // !=0 The increment for the elements of Y
    const typename sb_handle_t::event_t& _dependencies = {}  // Vector of events
) {
  return internal::_symv(sb_handle, _Uplo, _N, _alpha, _mA, _lda, _vx, _incx,
                         _beta, _vy, _incy, _dependencies);
}

/*!
 * @brief Generalised vector product followed by a sum with a rectangular
 * non-symmetric matrix.
 *
 * Generalised vector product followed by a sum with a rectangular non-symmetric
 * matrix, i.e.
 * computing the mathematical operation:
 *
 * A = alpha*x*yT + A
 *
 * See the netlib blas interface documentation for more details of the high
 * level interface:
 * http://www.netlib.org/lapack/explore-html/db/d5c/sger_8f.html
 *
 * @param sb_handle SB_handle
 * @param _M Number of rows in matrix A
 * @param _N Number of columns in matrix A
 * @param _alpha Scalar alpha
 * @param _vx Input vector having (1 + (_M-1)*abs(_incx)) elements
 * @param _incx Increment for vector X
 * @param _vy, Input vector having having (1 + (_N-1)*abs(_incy)) elements
 * @param _incy Increment for vector Y
 * @param _mA Input/output matrix A(_lda, n)
 * @param _lda Leading dimension of A
 * @param _dependencies Vector of events
 */
template <typename sb_handle_t, typename index_t, typename element_t,
          typename container_0_t, typename increment_t, typename container_1_t,
          typename container_2_t>
typename sb_handle_t::event_t inline _ger(
    sb_handle_t& sb_handle, index_t _M, index_t _N, element_t _alpha,
    container_0_t _vx, increment_t _incx, container_1_t _vy, increment_t _incy,
    container_2_t _mA, index_t _lda,
    const typename sb_handle_t::event_t& _dependencies = {}) {
  return internal::_ger(sb_handle, _M, _N, _alpha, _vx, _incx, _vy, _incy, _mA,
                        _lda, _dependencies);
}

/*!
 @brief Generalised vector product sum.

  Generalised vector squaring followed by a sum with a rectangular symmetric
 matrix, i.e.
 computing the mathematical operation:

 A = alpha*x*xT + A

 See the netlib blas interface documentation for more details of the high level
 interface: http://www.netlib.org/lapack/explore-html/db/d99/ssyr2_8f.html

 */
template <typename sb_handle_t, typename index_t, typename element_t,
          typename container_0_t, typename increment_t, typename container_1_t>
typename sb_handle_t::event_t inline _syr(
    sb_handle_t& sb_handle,  // sb_handle_t (sycl, parallel, serial, etc)
    char _Uplo,              // Whether the matrix is upper/lower ('u', 'l')
    index_t _N,              // >0 The order of matrix A
    element_t _alpha,        // Scalar alpha
    container_0_t _vx,       // (1 + (_N-1)*abs(_incx)), input vector X
    increment_t _incx,       // !=0 The increment for the elements of X
    container_1_t _mA,       // (_lda, _N) The output matrix
    index_t _lda,            // >max(1, _N) The first dimension of _mA
    const typename sb_handle_t::event_t& _dependencies = {}  // Vector of events
) {
  return internal::_syr(sb_handle, _Uplo, _N, _alpha, _vx, _incx, _mA, _lda,
                        _dependencies);
}

/**
 * @brief Generalised vector squaring followed by a sum with a packed symmetric
 * matrix.
 *
 * Generalised vector squaring followed by a sum with a packed symmetric matrix,
 * i.e. computing the mathematical operation:

 * A = alpha*x*xT + A

 * See the netlib blas interface documentation for more details of the high
 * level interface:
 * https://netlib.org/lapack/explore-html/d2/d9b/sspr_8f.html
 *
 * @param sb_handle sb_handle_t (sycl, parallel, serial, etc)
 * @param _Uplo Whether the matrix is upper/lower ('u', 'l')
 * @param _N >0 The order of matrix A
 * @param _alpha Scalar multiplier
 * @param _vx (1 + (_N-1)*abs(_incx)), input vector X
 * @param _incx !=0 The increment for the elements of X
 * @param _mPA (_lda, _N) The output matrix in packed format
 * @param _dependencies Vector of events
 */
template <typename sb_handle_t, typename index_t, typename element_t,
          typename container_0_t, typename increment_t, typename container_1_t>
typename sb_handle_t::event_t inline _spr(
    sb_handle_t& sb_handle, char _Uplo, index_t _N, element_t _alpha,
    container_0_t _vx, increment_t _incx, container_1_t _mPA,
    const typename sb_handle_t::event_t& _dependencies = {}) {
  return internal::_spr(sb_handle, _Uplo, _N, _alpha, _vx, _incx, _mPA,
                        _dependencies);
}

/**
 * @brief Generalised two vectors squaring followed by a sum with a packed
 * symmetric matrix.
 *
 * Generalised two vector squaring followed by a sum with a packed symmetric
 * matrix, i.e. computing the mathematical operation:

 * A = alpha*x*yT + alpha*y*xT + A

 * See the netlib blas interface documentation for more details of the high
 * level interface:
 * https://netlib.org/lapack/explore-html/db/d3e/sspr2_8f.html
 *
 * @param sb_handle sb_handle_t (sycl, parallel, serial, etc)
 * @param _Uplo Whether the matrix is upper/lower ('u', 'l')
 * @param _N >0 The order of matrix A
 * @param _alpha Scalar multiplier
 * @param _vx (1 + (_N-1)*abs(_incx)), input vector X
 * @param _incx !=0 The increment for the elements of X
 * @param _vy (1 + (_N-1)*abs(_incy)), input vector Y
 * @param _incy !=0 The increment for the elements of Y
 * @param _mPA (_lda, _N) The output matrix in packed format
 * @param _dependencies Vector of events
 */
template <typename sb_handle_t, typename index_t, typename element_t,
          typename container_t0, typename increment_t, typename container_t1,
          typename container_t2>
typename sb_handle_t::event_t inline _spr2(
    sb_handle_t& sb_handle, char _Uplo, index_t _N, element_t _alpha,
    container_t0 _vx, increment_t _incx, container_t1 _vy, increment_t _incy,
    container_t2 _mPA,
    const typename sb_handle_t::event_t& _dependencies = {}) {
  return internal::_spr2(sb_handle, _Uplo, _N, _alpha, _vx, _incx, _vy, _incy,
                         _mPA, _dependencies);
}

/*!
 @brief Generalised vector product followed by a sum with a rectangular
symmetric matrix.

Generalised vector product followed by a sum with a rectangular symmetric
 matrix, i.e.
 computing the mathematical operation:

 A = alpha*x*yT + alpha*y*xT + A

 See the netlib blas interface documentation for more details of the high level
 interface: http://www.netlib.org/lapack/explore-html/d6/dac/ssyr_8f.html

 */
template <typename sb_handle_t, typename index_t, typename element_t,
          typename container_0_t, typename increment_t, typename container_1_t,
          typename container_2_t>
typename sb_handle_t::event_t inline _syr2(
    sb_handle_t& sb_handle,  // sb_handle_t (sycl, parallel, serial, etc)
    char _Uplo,              // Whether the matrix is upper/lower ('u', 'l')
    index_t _N,              // >0 The order of matrix A
    element_t _alpha,        // Scalar alpha
    container_0_t _vx,       // (1 + (_N-1)*abs(_incx)), input vector X
    increment_t _incx,       // !=0 The increment for the elements of X
    container_1_t _vy,       // (1 + (_N-1)*abs(_incx)), input vector Y
    increment_t _incy,       // !=0 The increment for the elements of Y
    container_2_t _mA,       // (_lda, _N) The output matrix
    index_t _lda,            // >max(1, _N) The first dimension of _mA
    const typename sb_handle_t::event_t& _dependencies = {}  // Vector of events
) {
  return internal::_syr2(sb_handle, _Uplo, _N, _alpha, _vx, _incx, _vy, _incy,
                         _mA, _lda, _dependencies);
}

/**
 * @brief Generalised matrix vector product with band matrices.
 *
 * Generalised matrix vector product with a band matrix, i.e. computing the
 * mathematical operation:
 *
 * y = alpha*op(A)*x + beta*y
 *
 * See the netlib blas interface documentation for more details of the
 * interface: https://netlib.org/lapack/explore-html/d6/d46/sgbmv_8f.html
 *
 * @param sb_handle SB_handle
 * @param _trans Transposition operation applied to A ('n', 't', 'c')
 * @param _M Number of rows of A
 * @param _N Number of columns of A
 * @param _KL Number of A sub-diagonals
 * @param _KU Number of A super-diagonals
 * @param _alpha Scalar parameter alpha
 * @param _mA Buffer (_LDA,_N) containing the coefficient of A in the Band
 *            Matrix format
 * @param _lda Leading dimension _mA at least (_KL + _KU + 1)
 * @param _vx Buffer containing x of at least (1+(_N-1)*abs(_incx)) elements
 *            when trans = 'n' and (1+(_M-1)*abs(_incx) otherwise
 * @param _incx Increment for _vx (nonzero)
 * @param _beta Scalar parameter beta
 * @param _vy Buffer containing y of at least (1+(_M-1)*abs(_incy)) elements
 *            when trans = 'n' and (1+(_N-1)*abs(_incy) otherwise
 * @param _incy Increment for _vy
 * @param _dependencies Vector of events
 */
template <typename sb_handle_t, typename index_t, typename element_t,
          typename container_0_t, typename container_1_t, typename increment_t,
          typename container_2_t>
typename sb_handle_t::event_t inline _gbmv(
    sb_handle_t& sb_handle, char _trans, index_t _M, index_t _N, index_t _KL,
    index_t _KU, element_t _alpha, container_0_t _mA, index_t _lda,
    container_1_t _vx, increment_t _incx, element_t _beta, container_2_t _vy,
    increment_t _incy,
    const typename sb_handle_t::event_t& _dependencies = {}) {
  return internal::_gbmv(sb_handle, _trans, _M, _N, _KL, _KU, _alpha, _mA, _lda,
                         _vx, _incx, _beta, _vy, _incy, _dependencies);
}

/**
 * @brief Matrix vector product with symmetric band matrices.
 *
 * Matrix vector product with a symmetric band matrix, i.e. computing the
 * mathematical operation:
 *
 * y = alpha*A*x + beta*y
 *
 * See the netlib blas interface documentation for more details of the
 * interface: https://netlib.org/lapack/explore-html/d3/da1/ssbmv_8f.html
 *
 * @param sb_handle SB_handle
 * @param _Uplo Specifies if A is upper or lower triangular
 * @param _N Number of rows and columns of A
 * @param _K Number of A super-diagonals
 * @param _alpha Scalar parameter alpha
 * @param _mA Buffer (_LDA,_N) containing the coefficient of A in the Band
 *            Matrix format
 * @param _lda Leading dimension _mA at least (_K + 1)
 * @param _vx Buffer containing x of at least (1+(_N-1)*abs(_incx)) elements
 * @param _incx Increment for _vx (nonzero)
 * @param _beta Scalar parameter beta
 * @param _vy Buffer containing y of at least (1+(_N-1)*abs(_incy)) elements
 * @param _incy Increment for _vy
 * @param _dependencies Vector of events
 */
template <typename sb_handle_t, typename index_t, typename element_t,
          typename container_0_t, typename container_1_t, typename increment_t,
          typename container_2_t>
typename sb_handle_t::event_t _sbmv(
    sb_handle_t& sb_handle, char _Uplo, index_t _N, index_t _K,
    element_t _alpha, container_0_t _mA, index_t _lda, container_1_t _vx,
    increment_t _incx, element_t _beta, container_2_t _vy, increment_t _incy,
    const typename sb_handle_t::event_t& _dependencies = {}) {
  return internal::_sbmv(sb_handle, _Uplo, _N, _K, _alpha, _mA, _lda, _vx,
                         _incx, _beta, _vy, _incy, _dependencies);
}

/**
 * @brief Matrix vector product with symmetric packed matrices.
 *
 * Matrix vector product with a symmetric packed matrix, i.e. computing the
 * mathematical operation:
 *
 * y = alpha*A*x + beta*y
 *
 * See the netlib blas interface documentation for more details of the
 * interface: https://netlib.org/lapack/explore-html/d8/d68/sspmv_8f.html
 *
 * @param sb_handle SB_handle
 * @param _Uplo Specifies if A is upper or lower triangular
 * @param _N Number of rows and columns of A
 * @param _alpha Scalar parameter alpha
 * @param _mA Buffer containing the coefficient of A in the Packed Triangular
 *            Matrix format
 * @param _lda Leading dimension _mA at least (_K + 1)
 * @param _vx Buffer containing x of at least (1+(_N-1)*abs(_incx)) elements
 * @param _incx Increment for _vx (nonzero)
 * @param _beta Scalar parameter beta
 * @param _vy Buffer containing y of at least (1+(_N-1)*abs(_incy)) elements
 * @param _incy Increment for _vy
 */

template <typename sb_handle_t, typename index_t, typename element_t,
          typename container_0_t, typename container_1_t, typename increment_t,
          typename container_2_t>
typename sb_handle_t::event_t _spmv(
    sb_handle_t& sb_handle, char _Uplo, index_t _N, element_t _alpha,
    container_0_t _mA, container_1_t _vx, increment_t _incx, element_t _beta,
    container_2_t _vy, increment_t _incy,
    const typename sb_handle_t::event_t& _dependencies = {}) {
  return internal::_spmv(sb_handle, _Uplo, _N, _alpha, _mA, _vx, _incx, _beta,
                         _vy, _incy, _dependencies);
}

/**
 * @brief Matrix vector product with triangular band matrices.
 *
 * Matrix vector product with a triangular band matrix, i.e. computing the
 * mathematical operation:
 *
 * x = op(A)*x
 *
 * See the netlib blas interface documentation for more details of the
 * interface: https://netlib.org/lapack/explore-html/d6/d7d/stbmv_8f.html
 *
 * @param sb_handle SB_handle
 * @param _Uplo Specifies if A is upper or lower triangular
 * @param _trans Transposition operation applied to A ('n', 't', 'c')
 * @param _Diag Specifies if A unit triangular or not
 * @param _N Number of rows and columns of A
 * @param _K Number of A super-diagonals
 * @param _mA Buffer (_LDA,_N) containing the coefficient of A in the Band
 *            Matrix format
 * @param _lda Leading dimension _mA at least (_K + 1)
 * @param _vx Buffer containing x of at least (1+(_N-1)*abs(_incx)) elements
 * @param _incx Increment for _vx (nonzero)
 * @param _dependencies Vector of events
 */
template <typename sb_handle_t, typename index_t, typename container_0_t,
          typename container_1_t, typename increment_t>
typename sb_handle_t::event_t _tbmv(
    sb_handle_t& sb_handle, char _Uplo, char _trans, char _Diag, index_t _N,
    index_t _K, container_0_t _mA, index_t _lda, container_1_t _vx,
    increment_t _incx,
    const typename sb_handle_t::event_t& _dependencies = {}) {
  return internal::_tbmv(sb_handle, _Uplo, _trans, _Diag, _N, _K, _mA, _lda,
                         _vx, _incx, _dependencies);
}

/**
 * @brief Matrix vector product with triangular packed matrices.
 *
 * Matrix vector product with a triangular band matrix, i.e. computing the
 * mathematical operation:
 *
 * x = op(A)*x
 *
 * See the netlib blas interface documentation for more details of the
 * interface: https://netlib.org/lapack/explore-html/db/db1/stpmv_8f.html
 *
 * @param sb_handle SB_handle
 * @param _Uplo Specifies if A is upper or lower triangular
 * @param _trans Transposition operation applied to A ('n', 't', 'c')
 * @param _Diag Specifies if A unit triangular or not
 * @param _N Number of rows and columns of A
 * @param _ma buffer containing the coefficient of a in the packed triangular
 *            matrix format
 * @param _vx Buffer containing x of at least (1+(_N-1)*abs(_incx)) elements
 * @param _incx Increment for _vx (nonzero)
 * @param _dependencies Vector of events
 */
template <typename sb_handle_t, typename index_t, typename container_0_t,
          typename container_1_t, typename increment_t>
typename sb_handle_t::event_t _tpmv(
    sb_handle_t& sb_handle, char _Uplo, char _trans, char _Diag, index_t _N,
    container_0_t _mA, container_1_t _vx, increment_t _incx,
    const typename sb_handle_t::event_t& _dependencies = {}) {
  return internal::_tpmv(sb_handle, _Uplo, _trans, _Diag, _N, _mA, _vx, _incx,
                         _dependencies);
}

/**
 * @brief Linear system solver for triangular band matrices.
 *
 * Linear system solver for triangular band matrices, i.e., computing x s.t.
 *
 * op(A)*x = b
 *
 * See the netlib blas interface documentation for more details of the
 * interface: https://netlib.org/lapack/explore-html/d0/d1f/stbsv_8f.html
 *
 * @param sb_handle SB_handle
 * @param _Uplo Specifies if A is upper or lower triangular
 * @param _trans Transposition operation applied to A ('n', 't', 'c')
 * @param _Diag Specifies if A unit triangular or not
 * @param _N Number of rows and columns of A
 * @param _K Number of A super-diagonals
 * @param _mA Buffer (_LDA,_N) containing the coefficient of A in the Band
 *            Matrix format
 * @param _lda Leading dimension _mA at least (_K + 1)
 * @param _vx Buffer containing x of at least (1+(_N-1)*abs(_incx)) elements
 * @param _incx Increment for _vx (nonzero)
 * @param _dependencies Vector of events
 */
template <typename sb_handle_t, typename index_t, typename container_0_t,
          typename container_1_t, typename increment_t>
typename sb_handle_t::event_t _tbsv(
    sb_handle_t& sb_handle, char _Uplo, char _trans, char _Diag, index_t _N,
    index_t _K, container_0_t _mA, index_t _lda, container_1_t _vx,
    increment_t _incx,
    const typename sb_handle_t::event_t& _dependencies = {}) {
  return internal::_tbsv(sb_handle, _Uplo, _trans, _Diag, _N, _K, _mA, _lda,
                         _vx, _incx, _dependencies);
}

/**`
 * @brief Linear system solver for triangular packed matrices.
 *
 * Linear system solver for triangular packed matrices, i.e., computing x s.t.
 *
 * op(A)*x = b
 *
 * See the netlib blas interface documentation for more details of the
 * interface: https://netlib.org/lapack/explore-html/d0/d7c/stpsv_8f.html
 *
 * @param sb_handle SB_handle
 * @param _Uplo Specifies if A is upper or lower triangular
 * @param _trans Transposition operation applied to A ('n', 't', 'c')
 * @param _Diag Specifies if A unit triangular or not
 * @param _N Number of rows and columns of A
 * @param _mA Buffer containing the coefficient of A in the Packed Triangular
 *            Matrix format
 * @param _vx Buffer containing x of at least (1+(_N-1)*abs(_incx)) elements
 * @param _incx Increment for _vx (nonzero)
 */
template <typename sb_handle_t, typename index_t, typename container_0_t,
          typename container_1_t, typename increment_t>
typename sb_handle_t::event_t _tpsv(sb_handle_t& sb_handle, char _Uplo,
                                    char _trans, char _Diag, index_t _N,
                                    container_0_t _mA, container_1_t _vx,
                                    increment_t _incx,
                                    const typename sb_handle_t::event_t& _dependencies = {}) {
  return internal::_tpsv(sb_handle, _Uplo, _trans, _Diag, _N, _mA, _vx, _incx, _dependencies);
}
}  // namespace blas

#endif  // PORTBLAS_BLAS2_INTERFACE
