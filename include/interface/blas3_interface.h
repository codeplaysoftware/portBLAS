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
 *  @filename blas3_interface.h
 *
 **************************************************************************/

#ifndef SYCL_BLAS_BLAS3_INTERFACE_H
#define SYCL_BLAS_BLAS3_INTERFACE_H
namespace blas {
namespace internal {
/*!
 * @brief This is a top-level wrapper for GemmFactory, which provides a
 *        "standard" BLAS gemm interface.
 *
 * See netlib.org/blas for details.
 */
template <typename Executor, typename ContainerT0, typename ContainerT1,
          typename ContainerT2, typename T, typename IndexType>
typename Executor::Policy::event_type _gemm(
    Executor& ex, char _TransA, char _TransB, IndexType _M, IndexType _N,
    IndexType _K, T _alpha, ContainerT0 _A, IndexType _lda, ContainerT1 _B,
    IndexType _ldb, T _beta, ContainerT2 _C, IndexType _ldc);

template <typename Executor, typename ContainerT0, typename ContainerT1,
          typename ContainerT2, typename T, typename IndexType>
typename Executor::Policy::event_type _gemm_batched(
    Executor& ex, char _TransA, char _TransB, IndexType _M, IndexType _N,
    IndexType _K, T _alpha, ContainerT0 _A, IndexType _lda, ContainerT1 _B,
    IndexType _ldb, T _beta, ContainerT2 _C, IndexType _ldc,
    IndexType batch_size);
}  // namespace internal
template <typename Executor, typename ContainerT0, typename ContainerT1,
          typename ContainerT2, typename T, typename IndexType>
typename Executor::Policy::event_type _gemm(
    Executor& ex, char _TransA, char _TransB, IndexType _M, IndexType _N,
    IndexType _K, T _alpha, ContainerT0 _A, IndexType _lda, ContainerT1 _B,
    IndexType _ldb, T _beta, ContainerT2 _C, IndexType _ldc) {
  return internal::_gemm(ex, _TransA, _TransB, _M, _N, _K, _alpha,
                         ex.get_policy_handler().get_buffer(_A), _lda,
                         ex.get_policy_handler().get_buffer(_B), _ldb, _beta,
                         ex.get_policy_handler().get_buffer(_C), _ldc);
}

template <typename Executor, typename ContainerT0, typename ContainerT1,
          typename ContainerT2, typename T, typename IndexType>
typename Executor::Policy::event_type _gemm_batched(
    Executor& ex, char _TransA, char _TransB, IndexType _M, IndexType _N,
    IndexType _K, T _alpha, ContainerT0 _A, IndexType _lda, ContainerT1 _B,
    IndexType _ldb, T _beta, ContainerT2 _C, IndexType _ldc,
    IndexType batch_size) {
  return internal::_gemm_batched(ex, _TransA, _TransB, _M, _N, _K, _alpha,
                                 ex.get_policy_handler().get_buffer(_A), _lda,
                                 ex.get_policy_handler().get_buffer(_B), _ldb,
                                 _beta, ex.get_policy_handler().get_buffer(_C),
                                 _ldc, batch_size);
}
}  // namespace blas
#endif  // SYCL_BLAS_BLAS3_INTERFACE
