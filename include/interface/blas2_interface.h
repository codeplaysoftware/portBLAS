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
 @brief Generalised matrix vector product with rectangular non-symmetric
 matrices.

 Generalised matrix vector product with rectangular non-symmetric matrices, i.e.
 computing the mathematical operation:

 y = alpha*A*x + beta*y

 See the netlib blas interface documentation for more details of the high level
 interface: http://www.netlib.org/lapack/explore-html/db/d58/sgemv_8f.html

 */
template <typename Executor, typename IndexType, typename T,
          typename ContainerT0, typename ContainerT1, typename IncrementType,
          typename ContainerT2>
typename Executor::Policy::event_type _gemv_legacy(
    Executor& ex,         // Executor (sycl, parallel, serial, etc)
    char _trans,          // The transposition of the matrix ('n', 't', 'c')
    IndexType _M,         // The size of dimension M of the matrix (rows)
    IndexType _N,         // The size of dimension N of the matrix (columns)
    T _alpha,             // Scalar parameter Alpha
    ContainerT0 _mA,      // An array (LDA,N), with the first m*n elements
    IndexType _lda,       // Specifies the first dimension of a, max(1, m)
    ContainerT1 _vx,      // An array of dimension at least: (1+(n-1)*abs(incx))
                          // when trans = 'n' and (1+(m-1)*abs(incx) otherwise,
                          // containing the vector "x"
    IncrementType _incx,  // The increment for elements in x (nonzero).
    T _beta,              // Scalar parameter Beta
    ContainerT2 _vy,      // An array of dimension at least: (1+(m-1)*abs(incy))
                          // when trans = "n" and (1+(n-1)*abs(incy) otherwise,
    // containing the vector "y" (if beta is nonzero). When
    // finished, y is overwritten with the updated vector.
    IncrementType _incy  // The increment for elements in y (nonzero).
);

/*!
 @brief Generalised matrix vector product with rectangular non-symmetric
 matrices.

 Generalised matrix vector product with rectangular non-symmetric matrices, i.e.
 computing the mathematical operation:

 y = alpha*A*x + beta*y

 See the netlib blas interface documentation for more details of the high level
 interface: http://www.netlib.org/lapack/explore-html/db/d58/sgemv_8f.html

 */
template <typename Executor, typename IndexType, typename T,
          typename ContainerT0, typename ContainerT1, typename IncrementType,
          typename ContainerT2>
typename Executor::Policy::event_type _gemv(
    Executor& ex,         // Executor (sycl, parallel, serial, etc)
    char _trans,          // The transposition of the matrix ('n', 't', 'c')
    IndexType _M,         // The size of dimension M of the matrix (rows)
    IndexType _N,         // The size of dimension N of the matrix (columns)
    T _alpha,             // Scalar parameter Alpha
    ContainerT0 _mA,      // An array (LDA,N), with the first m*n elements
    IndexType _lda,       // Specifies the first dimension of a, max(1, m)
    ContainerT1 _vx,      // An array of dimension at least: (1+(n-1)*abs(incx))
                          // when trans = 'n' and (1+(m-1)*abs(incx) otherwise,
                          // containing the vector "x"
    IncrementType _incx,  // The increment for elements in x (nonzero).
    T _beta,              // Scalar parameter Beta
    ContainerT2 _vy,      // An array of dimension at least: (1+(m-1)*abs(incy))
                          // when trans = "n" and (1+(n-1)*abs(incy) otherwise,
    // containing the vector "y" (if beta is nonzero). When
    // finished, y is overwritten with the updated vector.
    IncrementType _incy  // The increment for elements in y (nonzero).
);

template <typename Executor, typename IndexType, typename ContainerT0,
          typename ContainerT1, typename IncrementType>
typename Executor::Policy::event_type _trmv(Executor& ex, char _Uplo,
                                            char _trans, char _Diag,
                                            IndexType _N, ContainerT0 _mA,
                                            IndexType _lda, ContainerT1 _vx,
                                            IncrementType _incx);
template <typename Executor, typename IndexType, typename T,
          typename ContainerT0, typename ContainerT1, typename IncrementType,
          typename ContainerT2>
typename Executor::Policy::event_type _symv(
    Executor& ex, char _Uplo, IndexType _N, T _alpha, ContainerT0 _mA,
    IndexType _lda, ContainerT1 _vx, IncrementType _incx, T _beta,
    ContainerT2 _vy, IncrementType _incy);
template <typename Executor, typename IndexType, typename T,
          typename ContainerT0, typename IncrementType, typename ContainerT1,
          typename ContainerT2>
typename Executor::Policy::event_type _ger(Executor& ex, IndexType _M,
                                           IndexType _N, T _alpha,
                                           ContainerT0 _vx, IncrementType _incx,
                                           ContainerT1 _vy, IncrementType _incy,
                                           ContainerT2 _mA, IndexType _lda);
template <typename Executor, typename IndexType, typename T,
          typename ContainerT0, typename IncrementType, typename ContainerT1>
typename Executor::Policy::event_type _syr(Executor& ex, char _Uplo,
                                           IndexType _N, T _alpha,
                                           ContainerT0 _vx, IncrementType _incx,
                                           ContainerT1 _mA, IndexType _lda);
template <typename Executor, typename IndexType, typename T,
          typename ContainerT0, typename IncrementType, typename ContainerT1,
          typename ContainerT2>
typename Executor::Policy::event_type _syr2(
    Executor& ex, char _Uplo, IndexType _N, T _alpha, ContainerT0 _vx,
    IncrementType _incx, ContainerT1 _vy, IncrementType _incy, ContainerT2 _mA,
    IndexType _lda);
}  // namespace internal
/*!
 @brief Generalised matrix vector product with rectangular non-symmetric
 matrices.

 Generalised matrix vector product with rectangular non-symmetric matrices, i.e.
 computing the mathematical operation:

 y = alpha*A*x + beta*y

 See the netlib blas interface documentation for more details of the high level
 interface: http://www.netlib.org/lapack/explore-html/db/d58/sgemv_8f.html

 */
template <typename Executor, typename IndexType, typename T,
          typename ContainerT0, typename ContainerT1, typename IncrementType,
          typename ContainerT2>
typename Executor::Policy::event_type inline _gemv_legacy(
    Executor& ex,         // Executor (sycl, parallel, serial, etc)
    char _trans,          // The transposition of the matrix ('n', 't', 'c')
    IndexType _M,         // The size of dimension M of the matrix (rows)
    IndexType _N,         // The size of dimension N of the matrix (columns)
    T _alpha,             // Scalar parameter Alpha
    ContainerT0 _mA,      // An array (LDA,N), with the first m*n elements
    IndexType _lda,       // Specifies the first dimension of a, max(1, m)
    ContainerT1 _vx,      // An array of dimension at least: (1+(n-1)*abs(incx))
                          // when trans = 'n' and (1+(m-1)*abs(incx) otherwise,
                          // containing the vector "x"
    IncrementType _incx,  // The increment for elements in x (nonzero).
    T _beta,              // Scalar parameter Beta
    ContainerT2 _vy,      // An array of dimension at least: (1+(m-1)*abs(incy))
                          // when trans = "n" and (1+(n-1)*abs(incy) otherwise,
    // containing the vector "y" (if beta is nonzero). When
    // finished, y is overwritten with the updated vector.
    IncrementType _incy  // The increment for elements in y (nonzero).
) {
  // TODO: Here we can use some heuristics to select localn global, local, and
  // scratch size per device
  return internal::_gemv_legacy(
      ex, _trans, _M, _N, _alpha, ex.get_policy_handler().get_buffer(_mA), _lda,
      ex.get_policy_handler().get_buffer(_vx), _incx, _beta,
      ex.get_policy_handler().get_buffer(_vy), _incy);
}

/*!
 @brief Generalised matrix vector product with rectangular non-symmetric
 matrices.

 Generalised matrix vector product with rectangular non-symmetric matrices, i.e.
 computing the mathematical operation:

 y = alpha*A*x + beta*y

 See the netlib blas interface documentation for more details of the high level
 interface: http://www.netlib.org/lapack/explore-html/db/d58/sgemv_8f.html

 */
template <typename Executor, typename IndexType, typename T,
          typename ContainerT0, typename ContainerT1, typename IncrementType,
          typename ContainerT2>
typename Executor::Policy::event_type inline _gemv(
    Executor& ex,         // Executor (sycl, parallel, serial, etc)
    char _trans,          // The transposition of the matrix ('n', 't', 'c')
    IndexType _M,         // The size of dimension M of the matrix (rows)
    IndexType _N,         // The size of dimension N of the matrix (columns)
    T _alpha,             // Scalar parameter Alpha
    ContainerT0 _mA,      // An array (LDA,N), with the first m*n elements
    IndexType _lda,       // Specifies the first dimension of a, max(1, m)
    ContainerT1 _vx,      // An array of dimension at least: (1+(n-1)*abs(incx))
                          // when trans = 'n' and (1+(m-1)*abs(incx) otherwise,
                          // containing the vector "x"
    IncrementType _incx,  // The increment for elements in x (nonzero).
    T _beta,              // Scalar parameter Beta
    ContainerT2 _vy,      // An array of dimension at least: (1+(m-1)*abs(incy))
                          // when trans = "n" and (1+(n-1)*abs(incy) otherwise,
    // containing the vector "y" (if beta is nonzero). When
    // finished, y is overwritten with the updated vector.
    IncrementType _incy  // The increment for elements in y (nonzero).
) {
  return internal::_gemv(ex, _trans, _M, _N, _alpha,
                         ex.get_policy_handler().get_buffer(_mA), _lda,
                         ex.get_policy_handler().get_buffer(_vx), _incx, _beta,
                         ex.get_policy_handler().get_buffer(_vy), _incy);
}

template <typename Executor, typename IndexType, typename ContainerT0,
          typename ContainerT1, typename IncrementType>
typename Executor::Policy::event_type inline _trmv(
    Executor& ex, char _Uplo, char _trans, char _Diag, IndexType _N,
    ContainerT0 _mA, IndexType _lda, ContainerT1 _vx, IncrementType _incx) {
  // TODO: Here we can use some heuristics to select localn global, local, and
  // scratch size per device
  return internal::_trmv(ex, _Uplo, _trans, _Diag, _N,
                         ex.get_policy_handler().get_buffer(_mA), _lda,
                         ex.get_policy_handler().get_buffer(_vx), _incx);
}
template <typename Executor, typename IndexType, typename T,
          typename ContainerT0, typename ContainerT1, typename IncrementType,
          typename ContainerT2>
typename Executor::Policy::event_type inline _symv(
    Executor& ex, char _Uplo, IndexType _N, T _alpha, ContainerT0 _mA,
    IndexType _lda, ContainerT1 _vx, IncrementType _incx, T _beta,
    ContainerT2 _vy, IncrementType _incy) {
  // TODO: Here we can use some heuristics to select localn global, local, and
  // scratch size per device
  return internal::_symv(ex, _Uplo, _N, _alpha,
                         ex.get_policy_handler().get_buffer(_mA), _lda,
                         ex.get_policy_handler().get_buffer(_vx), _incx, _beta,
                         ex.get_policy_handler().get_buffer(_vy), _incy);
}
template <typename Executor, typename IndexType, typename T,
          typename ContainerT0, typename IncrementType, typename ContainerT1,
          typename ContainerT2>
typename Executor::Policy::event_type inline _ger(
    Executor& ex, IndexType _M, IndexType _N, T _alpha, ContainerT0 _vx,
    IncrementType _incx, ContainerT1 _vy, IncrementType _incy, ContainerT2 _mA,
    IndexType _lda) {
  // TODO: Here we can use some heuristics to select localn global, local, and
  // scratch size per device
  return internal::_ger(ex, _M, _N, _alpha,
                        ex.get_policy_handler().get_buffer(_vx), _incx,
                        ex.get_policy_handler().get_buffer(_vy), _incy,
                        ex.get_policy_handler().get_buffer(_mA), _lda);
}
template <typename Executor, typename IndexType, typename T,
          typename ContainerT0, typename IncrementType, typename ContainerT1>
typename Executor::Policy::event_type inline _syr(
    Executor& ex, char _Uplo, IndexType _N, T _alpha, ContainerT0 _vx,
    IncrementType _incx, ContainerT1 _mA, IndexType _lda) {
  // TODO: Here we can use some heuristics to select localn global, local, and
  // scratch size per device
  return internal::_syr(ex, _Uplo, _N, _alpha,
                        ex.get_policy_handler().get_buffer(_vx), _incx,
                        ex.get_policy_handler().get_buffer(_mA), _lda);
}
template <typename Executor, typename IndexType, typename T,
          typename ContainerT0, typename IncrementType, typename ContainerT1,
          typename ContainerT2>
typename Executor::Policy::event_type inline _syr2(
    Executor& ex, char _Uplo, IndexType _N, T _alpha, ContainerT0 _vx,
    IncrementType _incx, ContainerT1 _vy, IncrementType _incy, ContainerT2 _mA,
    IndexType _lda) {
  // TODO: Here we can use some heuristics to select localn global, local, and
  // scratch size per device
  return internal::_syr2(ex, _Uplo, _N, _alpha,
                         ex.get_policy_handler().get_buffer(_vx), _incx,
                         ex.get_policy_handler().get_buffer(_vy), _incy,
                         ex.get_policy_handler().get_buffer(_mA), _lda);
}
}  // namespace blas

#endif  // SYCL_BLAS_BLAS2_INTERFACE
