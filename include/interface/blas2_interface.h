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

template <typename executor_t, typename index_t, typename container_0_t,
          typename container_1_t, typename increment_t>
typename executor_t::policy_t::event_t _trmv(executor_t& ex, char _Uplo,
                                             char _trans, char _Diag,
                                             index_t _N, container_0_t _mA,
                                             index_t _lda, container_1_t _vx,
                                             increment_t _incx);
template <typename executor_t, typename index_t, typename element_t,
          typename container_0_t, typename container_1_t, typename increment_t,
          typename container_2_t>
typename executor_t::policy_t::event_t _symv(
    executor_t& ex, char _Uplo, index_t _N, element_t _alpha, container_0_t _mA,
    index_t _lda, container_1_t _vx, increment_t _incx, element_t _beta,
    container_2_t _vy, increment_t _incy);
template <typename executor_t, typename index_t, typename element_t,
          typename container_0_t, typename increment_t, typename container_1_t,
          typename container_2_t>
typename executor_t::policy_t::event_t _ger(
    executor_t& ex, index_t _M, index_t _N, element_t _alpha, container_0_t _vx,
    increment_t _incx, container_1_t _vy, increment_t _incy, container_2_t _mA,
    index_t _lda);
template <typename executor_t, typename index_t, typename element_t,
          typename container_0_t, typename increment_t, typename container_1_t>
typename executor_t::policy_t::event_t _syr(executor_t& ex, char _Uplo,
                                            index_t _N, element_t _alpha,
                                            container_0_t _vx,
                                            increment_t _incx,
                                            container_1_t _mA, index_t _lda);
template <typename executor_t, typename index_t, typename element_t,
          typename container_0_t, typename increment_t, typename container_1_t,
          typename container_2_t>
typename executor_t::policy_t::event_t _syr2(
    executor_t& ex, char _Uplo, index_t _N, element_t _alpha, container_0_t _vx,
    increment_t _incx, container_1_t _vy, increment_t _incy, container_2_t _mA,
    index_t _lda);
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
  // TODO: Here we can use some heuristics to select localn global, local, and
  // scratch size per device
  return internal::_gemv(ex, _trans, _M, _N, _alpha,
                         ex.get_policy_handler().get_buffer(_mA), _lda,
                         ex.get_policy_handler().get_buffer(_vx), _incx, _beta,
                         ex.get_policy_handler().get_buffer(_vy), _incy);
}

template <typename executor_t, typename index_t, typename container_0_t,
          typename container_1_t, typename increment_t>
typename executor_t::policy_t::event_t inline _trmv(
    executor_t& ex, char _Uplo, char _trans, char _Diag, index_t _N,
    container_0_t _mA, index_t _lda, container_1_t _vx, increment_t _incx) {
  // TODO: Here we can use some heuristics to select localn global, local, and
  // scratch size per device
  return internal::_trmv(ex, _Uplo, _trans, _Diag, _N,
                         ex.get_policy_handler().get_buffer(_mA), _lda,
                         ex.get_policy_handler().get_buffer(_vx), _incx);
}
template <typename executor_t, typename index_t, typename element_t,
          typename container_0_t, typename container_1_t, typename increment_t,
          typename container_2_t>
typename executor_t::policy_t::event_t inline _symv(
    executor_t& ex, char _Uplo, index_t _N, element_t _alpha, container_0_t _mA,
    index_t _lda, container_1_t _vx, increment_t _incx, element_t _beta,
    container_2_t _vy, increment_t _incy) {
  // TODO: Here we can use some heuristics to select localn global, local, and
  // scratch size per device
  return internal::_symv(ex, _Uplo, _N, _alpha,
                         ex.get_policy_handler().get_buffer(_mA), _lda,
                         ex.get_policy_handler().get_buffer(_vx), _incx, _beta,
                         ex.get_policy_handler().get_buffer(_vy), _incy);
}
template <typename executor_t, typename index_t, typename element_t,
          typename container_0_t, typename increment_t, typename container_1_t,
          typename container_2_t>
typename executor_t::policy_t::event_t inline _ger(
    executor_t& ex, index_t _M, index_t _N, element_t _alpha, container_0_t _vx,
    increment_t _incx, container_1_t _vy, increment_t _incy, container_2_t _mA,
    index_t _lda) {
  // TODO: Here we can use some heuristics to select localn global, local, and
  // scratch size per device
  return internal::_ger(ex, _M, _N, _alpha,
                        ex.get_policy_handler().get_buffer(_vx), _incx,
                        ex.get_policy_handler().get_buffer(_vy), _incy,
                        ex.get_policy_handler().get_buffer(_mA), _lda);
}
template <typename executor_t, typename index_t, typename element_t,
          typename container_0_t, typename increment_t, typename container_1_t>
typename executor_t::policy_t::event_t inline _syr(
    executor_t& ex, char _Uplo, index_t _N, element_t _alpha, container_0_t _vx,
    increment_t _incx, container_1_t _mA, index_t _lda) {
  // TODO: Here we can use some heuristics to select localn global, local, and
  // scratch size per device
  return internal::_syr(ex, _Uplo, _N, _alpha,
                        ex.get_policy_handler().get_buffer(_vx), _incx,
                        ex.get_policy_handler().get_buffer(_mA), _lda);
}
template <typename executor_t, typename index_t, typename element_t,
          typename container_0_t, typename increment_t, typename container_1_t,
          typename container_2_t>
typename executor_t::policy_t::event_t inline _syr2(
    executor_t& ex, char _Uplo, index_t _N, element_t _alpha, container_0_t _vx,
    increment_t _incx, container_1_t _vy, increment_t _incy, container_2_t _mA,
    index_t _lda) {
  // TODO: Here we can use some heuristics to select localn global, local, and
  // scratch size per device
  return internal::_syr2(ex, _Uplo, _N, _alpha,
                         ex.get_policy_handler().get_buffer(_vx), _incx,
                         ex.get_policy_handler().get_buffer(_vy), _incy,
                         ex.get_policy_handler().get_buffer(_mA), _lda);
}
}  // namespace blas

#endif  // SYCL_BLAS_BLAS2_INTERFACE
