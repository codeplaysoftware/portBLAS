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
template <typename executor_t, typename container_0_t, typename container_1_t,
          typename container_2_t, typename element_t, typename index_t>
typename executor_t::policy_t::event_t _gemm(executor_t& ex, char _TransA,
                                             char _TransB, index_t _M,
                                             index_t _N, index_t _K,
                                             element_t _alpha, container_0_t a_,
                                             index_t _lda, container_1_t b_,
                                             index_t _ldb, element_t _beta,
                                             container_2_t _C, index_t _ldc);

template <typename executor_t, typename container_0_t, typename container_1_t,
          typename container_2_t, typename element_t, typename index_t>
typename executor_t::policy_t::event_t _gemm_batched(
    executor_t& ex, char _TransA, char _TransB, index_t _M, index_t _N,
    index_t _K, element_t _alpha, container_0_t a_, index_t _lda,
    container_1_t b_, index_t _ldb, element_t _beta, container_2_t _C,
    index_t _ldc, index_t batch_size);
}  // namespace internal
template <typename executor_t, typename container_0_t, typename container_1_t,
          typename container_2_t, typename element_t, typename index_t>
typename executor_t::policy_t::event_t _gemm(executor_t& ex, char _TransA,
                                             char _TransB, index_t _M,
                                             index_t _N, index_t _K,
                                             element_t _alpha, container_0_t a_,
                                             index_t _lda, container_1_t b_,
                                             index_t _ldb, element_t _beta,
                                             container_2_t _C, index_t _ldc) {
  return internal::_gemm(ex, _TransA, _TransB, _M, _N, _K, _alpha,
                         ex.get_policy_handler().get_buffer(a_), _lda,
                         ex.get_policy_handler().get_buffer(b_), _ldb, _beta,
                         ex.get_policy_handler().get_buffer(_C), _ldc);
}

template <typename executor_t, typename container_0_t, typename container_1_t,
          typename container_2_t, typename element_t, typename index_t>
typename executor_t::policy_t::event_t _gemm_batched(
    executor_t& ex, char _TransA, char _TransB, index_t _M, index_t _N,
    index_t _K, element_t _alpha, container_0_t a_, index_t _lda,
    container_1_t b_, index_t _ldb, element_t _beta, container_2_t _C,
    index_t _ldc, index_t batch_size) {
  return internal::_gemm_batched(ex, _TransA, _TransB, _M, _N, _K, _alpha,
                                 ex.get_policy_handler().get_buffer(a_), _lda,
                                 ex.get_policy_handler().get_buffer(b_), _ldb,
                                 _beta, ex.get_policy_handler().get_buffer(_C),
                                 _ldc, batch_size);
}
}  // namespace blas
#endif  // SYCL_BLAS_BLAS3_INTERFACE
