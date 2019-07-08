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
 *  @filename blas2_interface.h
 *
 **************************************************************************/

#ifndef SYCL_BLAS_BLAS2_INTERFACE_H
#define SYCL_BLAS_BLAS2_INTERFACE_H
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
template <typename executor_t, typename index_t, typename element_t,
          typename container_0_t, typename container_1_t, typename increment_t,
          typename container_2_t>
typename executor_t::policy_t::event_t _gemv(
    executor_t& ex,     // executor_t (sycl, parallel, serial, etc)
    char _trans,        // The transposition of the matrix ('n', 't', 'c')
    index_t _M,         // The size of dimension M of the matrix (rows)
    index_t _N,         // The size of dimension N of the matrix (columns)
    element_t _alpha,   // Scalar parameter Alpha
    container_0_t _mA,  // An array (LDA,N), with the first m*n elements
    index_t _lda,       // Specifies the first dimension of a, max(1, m)
    container_1_t _vx,  // An array of dimension at least: (1+(n-1)*abs(incx))
                        // when trans = 'n' and (1+(m-1)*abs(incx) otherwise,
                        // containing the vector "x"
    increment_t _incx,  // The increment for elements in x (nonzero).
    element_t _beta,    // Scalar parameter Beta
    container_2_t _vy,  // An array of dimension at least: (1+(m-1)*abs(incy))
                        // when trans = "n" and (1+(n-1)*abs(incy) otherwise,
    // containing the vector "y" (if beta is nonzero). When
    // finished, y is overwritten with the updated vector.
    increment_t _incy  // The increment for elements in y (nonzero).
);

/*!
 * @brief Prototype for the internal implementation of the GEMV operation. See
 * documentation in the blas2_interface.hpp file for details.
 */
template <uint32_t local_range, uint32_t cache_line_size,
          gemv_memory_t memory_type, transpose_type trn, typename Executor,
          typename index_t, typename element_t, typename container_t0,
          typename container_t1, typename increment_t, typename container_t2>
typename Executor::policy_t::event_t _gemv_impl(
    Executor& ex, index_t _M, index_t _N, element_t _alpha, container_t0 _mA,
    index_t _lda, container_t1 _vx, increment_t _incx, element_t _beta,
    container_t2 _vy, increment_t _incy);

/*!
 @brief Generalised matrix vector product with a triangular symmetric matrix.

 Generalised matrix vector product with a triangular symmetric matrix, i.e.
 computing the mathematical operation:

 x = A*x

 See the netlib blas interface documentation for more details of the high level
 interface: http://www.netlib.org/lapack/explore-html/de/d45/strmv_8f.html

 */
template <typename executor_t, typename index_t, typename container_0_t,
          typename container_1_t, typename increment_t>
typename executor_t::policy_t::event_t _trmv(
    executor_t& ex,     // executor_t (sycl, parallel, serial, etc)
    char _Uplo,         // Whether the matrix is upper/lower ('u', 'l')
    char _trans,        // Whether the matrix is transposed ('n', 't', 'c')
    char _Diag,         // Whether the matrix is unit triangular ('u', 'n')
    index_t _N,         // >0 The order of matrix A
    container_0_t _mA,  // (_lda, _N) The input matrix
    index_t _lda,       // >max(1, _N) The first dimension of _mA
    container_1_t _vx,  // (1 + (_N-1)*abs(_incx)), output vector X
    increment_t _incx   // !=0 The increment for the elements of X
);

/*!
 @brief Generalised matrix vector product with a square symmetric matrix,
 followed by a vector sum.

 Generalised matrix vector product with a square symmetric matrix, followed by
 a vector sum, i.e. computing the mathematical operation:

 x = alpha*A*x + beta*y

 See the netlib blas interface documentation for more details of the high level
 interface: http://www.netlib.org/lapack/explore-html/d2/d94/ssymv_8f.html

 */
template <typename executor_t, typename index_t, typename element_t,
          typename container_0_t, typename container_1_t, typename increment_t,
          typename container_2_t>
typename executor_t::policy_t::event_t _symv(
    executor_t& ex,     // executor_t (sycl, parallel, serial, etc)
    char _Uplo,         // Whether the matrix is upper/lower ('u', 'l')
    index_t _N,         // >0 The order of matrix A
    element_t _alpha,   // Scalar parameter alpha
    container_0_t _mA,  // (_lda, _N) The input matrix
    index_t _lda,       // >max(1, _N) The first dimension of _mA
    container_1_t _vx,  // (1 + (_N-1)*abs(_incx)), input vector X
    increment_t _incx,  // !=0 The increment for the elements of X
    element_t _beta,    // Scalar parameter beta
    container_2_t _vy,  // (1 + (_N-1)*abs(_incy)), output vector Y
    increment_t _incy   // !=0 The increment for the elements of Y
);

/*!
 @brief Generalised vector product followed by a sum with a rectangular
 non-symmetric matrix.

 Generalised vector product followed by a sum with a rectangular non-symmetric
 matrix, i.e. computing the mathematical operation:

 A = alpha*x*yT + A

 See the netlib blas interface documentation for more details of the high level
 interface: http://www.netlib.org/lapack/explore-html/db/d5c/sger_8f.html

 */
template <typename executor_t, typename index_t, typename element_t,
          typename container_0_t, typename increment_t, typename container_1_t,
          typename container_2_t>
typename executor_t::policy_t::event_t _ger(
    executor_t& ex,     // executor_t (sycl, parallel, serial, etc)
    index_t _M,         // The rows in matrix A
    index_t _N,         // The cols of matrix A
    element_t _alpha,   // Scalar alpha
    container_0_t _vx,  // >(1 + (_M-1)*abs(_incx)), input vector X
    increment_t _incx,  // Increment for vector X
    container_1_t _vy,  // >(1 + (_N-1)*abs(_incy)), input vector Y
    increment_t _incy,  // Increment for vector Y
    container_2_t _mA,  // (_lda, n) array containing A, the output
    index_t _lda        // >max(1, m), Leading dimension of A
);

/*!
 @brief Generalised vector squaring followed by a sum with a symmetric matrix.

 Generalised vector squaring followed by a sum with a symmetric matrix,
 i.e. computing the mathematical operation:

 A = alpha*x*xT + A

 See the netlib blas interface documentation for more details of the high level
 interface: http://www.netlib.org/lapack/explore-html/db/d99/ssyr2_8f.html

 */
template <typename executor_t, typename index_t, typename element_t,
          typename container_0_t, typename increment_t, typename container_1_t>
typename executor_t::policy_t::event_t _syr(
    executor_t& ex,     // executor_t (sycl, parallel, serial, etc)
    char _Uplo,         // Whether the matrix is upper/lower ('u', 'l')
    index_t _N,         // >0 The order of matrix A
    element_t _alpha,   // Scalar alpha
    container_0_t _vx,  // (1 + (_N-1)*abs(_incx)), input vector X
    increment_t _incx,  // !=0 The increment for the elements of X
    container_1_t _mA,  // (_lda, _N) The output matrix
    index_t _lda        // >max(1, _N) The first dimension of _mA
);

/*!
 @brief Generalised vector products followed by a sum with a symmetric matrix.

 Generalised vector products followed by a sum with a symmetric matrix,
 i.e. computing the mathematical operation:

 A = alpha*x*yT + alpha*y*xT + A

 See the netlib blas interface documentation for more details of the high level
 interface: http://www.netlib.org/lapack/explore-html/d6/dac/ssyr_8f.html

 */
template <typename executor_t, typename index_t, typename element_t,
          typename container_0_t, typename increment_t, typename container_1_t,
          typename container_2_t>
typename executor_t::policy_t::event_t _syr2(
    executor_t& ex,     // executor_t (sycl, parallel, serial, etc)
    char _Uplo,         // Whether the matrix is upper/lower ('u', 'l')
    index_t _N,         // >0 The order of matrix A
    element_t _alpha,   // Scalar alpha
    container_0_t _vx,  // (1 + (_N-1)*abs(_incx)), input vector X
    increment_t _incx,  // !=0 The increment for the elements of X
    container_1_t _vy,  // (1 + (_N-1)*abs(_incx)), input vector Y
    increment_t _incy,  // !=0 The increment for the elements of Y
    container_2_t _mA,  // (_lda, _N) The output matrix
    index_t _lda        // >max(1, _N) The first dimension of _mA
);
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
template <typename executor_t, typename index_t, typename element_t,
          typename container_0_t, typename container_1_t, typename increment_t,
          typename container_2_t>
typename executor_t::policy_t::event_t inline _gemv(
    executor_t& ex,     // executor_t (sycl, parallel, serial, etc)
    char _trans,        // The transposition of the matrix ('n', 't', 'c')
    index_t _M,         // The size of dimension M of the matrix (rows)
    index_t _N,         // The size of dimension N of the matrix (columns)
    element_t _alpha,   // Scalar parameter Alpha
    container_0_t _mA,  // An array (LDA,N), with the first m*n elements
    index_t _lda,       // Specifies the first dimension of a, max(1, m)
    container_1_t _vx,  // An array of dimension at least: (1+(n-1)*abs(incx))
                        // when trans = 'n' and (1+(m-1)*abs(incx) otherwise,
                        // containing the vector "x"
    increment_t _incx,  // The increment for elements in x (nonzero).
    element_t _beta,    // Scalar parameter Beta
    container_2_t _vy,  // An array of dimension at least: (1+(m-1)*abs(incy))
                        // when trans = "n" and (1+(n-1)*abs(incy) otherwise,
    // containing the vector "y" (if beta is nonzero). When
    // finished, y is overwritten with the updated vector.
    increment_t _incy  // The increment for elements in y (nonzero).
) {
  return internal::_gemv(ex, _trans, _M, _N, _alpha,
                         ex.get_policy_handler().get_buffer(_mA), _lda,
                         ex.get_policy_handler().get_buffer(_vx), _incx, _beta,
                         ex.get_policy_handler().get_buffer(_vy), _incy);
}

/*!
 @brief Generalised matrix vector product with a triangular symmetric matrix.

 Generalised matrix vector product with a triangular symmetric matrix, i.e.
 computing the mathematical operation:

 x = A*x

 See the netlib blas interface documentation for more details of the high level
 interface: http://www.netlib.org/lapack/explore-html/de/d45/strmv_8f.html

 */
template <typename executor_t, typename index_t, typename container_0_t,
          typename container_1_t, typename increment_t>
typename executor_t::policy_t::event_t inline _trmv(
    executor_t& ex,     // executor_t (sycl, parallel, serial, etc)
    char _Uplo,         // Whether the matrix is upper/lower ('u', 'l')
    char _trans,        // Whether the matrix is transposed ('n', 't', 'c')
    char _Diag,         // Whether the matrix is unit triangular ('u', 'n')
    index_t _N,         // >0 The order of matrix A
    container_0_t _mA,  // (_lda, _N) The input matrix
    index_t _lda,       // >max(1, _N) The first dimension of _mA
    container_1_t _vx,  // (1 + (_N-1)*abs(_incx)), output vector X
    increment_t _incx   // !=0 The increment for the elements of X
) {
  return internal::_trmv(ex, _Uplo, _trans, _Diag, _N,
                         ex.get_policy_handler().get_buffer(_mA), _lda,
                         ex.get_policy_handler().get_buffer(_vx), _incx);
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
template <typename executor_t, typename index_t, typename element_t,
          typename container_0_t, typename container_1_t, typename increment_t,
          typename container_2_t>
typename executor_t::policy_t::event_t inline _symv(
    executor_t& ex,     // executor_t (sycl, parallel, serial, etc)
    char _Uplo,         // Whether the matrix is upper/lower ('u', 'l')
    index_t _N,         // >0 The order of matrix A
    element_t _alpha,   // Scalar parameter alpha
    container_0_t _mA,  // (_lda, _N) The input matrix
    index_t _lda,       // >max(1, _N) The first dimension of _mA
    container_1_t _vx,  // (1 + (_N-1)*abs(_incx)), input vector X
    increment_t _incx,  // !=0 The increment for the elements of X
    element_t _beta,    // Scalar parameter beta
    container_2_t _vy,  // (1 + (_N-1)*abs(_incy)), output vector Y
    increment_t _incy   // !=0 The increment for the elements of Y
) {
  return internal::_symv(ex, _Uplo, _N, _alpha,
                         ex.get_policy_handler().get_buffer(_mA), _lda,
                         ex.get_policy_handler().get_buffer(_vx), _incx, _beta,
                         ex.get_policy_handler().get_buffer(_vy), _incy);
}

/*!
 @brief Generalised vector product followed by a sum with a rectangular
 non-symmetric matrix.

 Generalised vector product followed by a sum with a rectangular non-symmetric
 matrix, i.e.
 computing the mathematical operation:

 A = alpha*x*yT + A

 See the netlib blas interface documentation for more details of the high level
 interface: http://www.netlib.org/lapack/explore-html/db/d5c/sger_8f.html

 */
template <typename executor_t, typename index_t, typename element_t,
          typename container_0_t, typename increment_t, typename container_1_t,
          typename container_2_t>
typename executor_t::policy_t::event_t inline _ger(
    executor_t& ex,     // executor_t (sycl, parallel, serial, etc)
    index_t _M,         // The rows in matrix M
    index_t _N,         // The rows of matrix N
    element_t _alpha,   // Scalar alpha
    container_0_t _vx,  // >(1 + (_M-1)*abs(_incx)), input vector X
    increment_t _incx,  // Increment for vector X
    container_1_t _vy,  // >(1 + (_N-1)*abs(_incy)), input vector Y
    increment_t _incy,  // Increment for vector Y
    container_2_t _mA,  // (_lda, n) array containing A, the output
    index_t _lda        // >max(1, m), Leading dimension of A
) {
  return internal::_ger(ex, _M, _N, _alpha,
                        ex.get_policy_handler().get_buffer(_vx), _incx,
                        ex.get_policy_handler().get_buffer(_vy), _incy,
                        ex.get_policy_handler().get_buffer(_mA), _lda);
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
template <typename executor_t, typename index_t, typename element_t,
          typename container_0_t, typename increment_t, typename container_1_t>
typename executor_t::policy_t::event_t inline _syr(
    executor_t& ex,     // executor_t (sycl, parallel, serial, etc)
    char _Uplo,         // Whether the matrix is upper/lower ('u', 'l')
    index_t _N,         // >0 The order of matrix A
    element_t _alpha,   // Scalar alpha
    container_0_t _vx,  // (1 + (_N-1)*abs(_incx)), input vector X
    increment_t _incx,  // !=0 The increment for the elements of X
    container_1_t _mA,  // (_lda, _N) The output matrix
    index_t _lda        // >max(1, _N) The first dimension of _mA
) {
  return internal::_syr(ex, _Uplo, _N, _alpha,
                        ex.get_policy_handler().get_buffer(_vx), _incx,
                        ex.get_policy_handler().get_buffer(_mA), _lda);
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
template <typename executor_t, typename index_t, typename element_t,
          typename container_0_t, typename increment_t, typename container_1_t,
          typename container_2_t>
typename executor_t::policy_t::event_t inline _syr2(
    executor_t& ex,     // executor_t (sycl, parallel, serial, etc)
    char _Uplo,         // Whether the matrix is upper/lower ('u', 'l')
    index_t _N,         // >0 The order of matrix A
    element_t _alpha,   // Scalar alpha
    container_0_t _vx,  // (1 + (_N-1)*abs(_incx)), input vector X
    increment_t _incx,  // !=0 The increment for the elements of X
    container_1_t _vy,  // (1 + (_N-1)*abs(_incx)), input vector Y
    increment_t _incy,  // !=0 The increment for the elements of Y
    container_2_t _mA,  // (_lda, _N) The output matrix
    index_t _lda        // >max(1, _N) The first dimension of _mA
) {
  return internal::_syr2(ex, _Uplo, _N, _alpha,
                         ex.get_policy_handler().get_buffer(_vx), _incx,
                         ex.get_policy_handler().get_buffer(_vy), _incy,
                         ex.get_policy_handler().get_buffer(_mA), _lda);
}
}  // namespace blas

#endif  // SYCL_BLAS_BLAS2_INTERFACE
