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
 *  @filename reduction_interface.h
 *
 **************************************************************************/

#ifndef SYCL_BLAS_EXTENSION_INTERFACE_H
#define SYCL_BLAS_EXTENSION_INTERFACE_H

#include "operations/extension/reduction.h"
#include "operations/extension/transpose.h"
#include "sb_handle/sycl_blas_handle.h"

namespace blas {

namespace extension {

namespace internal {

/**
 * \brief COPY Matrix in to out with scaling factor of alpha
 *
 * Copy Matrix to out_memory, it can be out of place if out_memory is different
 * from in_memory. The increment within the same column can be different from 1
 * and specified by inc and inc_out argument.
 *
 *
 * @tparam sb_handle_t SB_Handle type
 * @tparam element_t Scaling factor type
 * @tparam index_t Index type
 * @tparam in_t Buffer Iterator
 * @tparam out_t Buffer Iterator
 * @param sb_handle SB_Handle
 * @param trans compute matrix transpose or not.
 * @param m rows of matrix
 * @param n cols of matrix
 * @param alpha Scaling factor
 * @param in_memory BufferIterator of input
 * @param ld_in leading dimension of in_matrix
 * @param inc_in internal increment for the in_matrix
 * @param matrix_out BufferIterator of output
 * @param ld_out leading dimention of out_matrix
 * @param inc_out internal increment for the out_matrix
 */
template <bool in_place, typename sb_handle_t, typename element_t,
          typename index_t, typename in_t, typename out_t>
typename sb_handle_t::event_t _matcopy(sb_handle_t& sb_handle, char trans,
                                       index_t m, index_t n, element_t alpha,
                                       in_t in_memory, index_t ld_in,
                                       index_t inc_in, out_t out_memory,
                                       index_t ld_out, index_t inc_out);

template <typename operator_t, typename element_t, typename sb_handle_t,
          typename input_t, typename output_t, typename index_t>
typename sb_handle_t::event_t _reduction(sb_handle_t& sb_handle,
                                         input_t buffer_in, index_t ld,
                                         output_t buffer_out, index_t rows,
                                         index_t cols,
                                         reduction_dim_t reduction_dim);

}  // namespace internal

/**
 * \brief COPY in_matrix to out_matrix with scaling factor of alpha
 *
 * @tparam sb_handle_t SB_Handle type
 * @tparam element_t Scaling factor type
 * @tparam index_t Index type
 * @tparam in_t Buffer Iterator
 * @tparam out_t Buffer Iterator
 * @param sb_handle SB_Handle
 * @param trans compute matrix transpose or not.
 * @param m rows of matrix
 * @param n cols of matrix
 * @param alpha Scaling factor
 * @param in_memory BufferIterator of input
 * @param ld_in leading dimension of in_matrix
 * @param matrix_out BufferIterator of output
 * @param ld_out leading dimention of out_matrix
 */
template <typename sb_handle_t, typename element_t, typename index_t,
          typename in_t, typename out_t>
typename sb_handle_t::event_t _omatcopy(sb_handle_t& sb_handle, char trans,
                                        index_t m, index_t n, element_t alpha,
                                        in_t in_memory, index_t ld_in,
                                        out_t out_memory, index_t ld_out) {
  return internal::_matcopy<false>(sb_handle, trans, m, n, alpha, in_memory,
                                   ld_in, static_cast<index_t>(1), out_memory,
                                   ld_out, static_cast<index_t>(1));
}

/**
 * \brief Copy out of place of in_matrix to out_matrix with increment between cols
 * element different from 1.
 *
 * The increment within the same column can be different from 1 and specified
 * by inc_in and inc_out arguments.
 *
 * @tparam sb_handle_t SB_Handle type
 * @tparam element_t Scaling factor type
 * @tparam index_t Index type
 * @tparam in_t Buffer Iterator
 * @tparam out_t Buffer Iterator
 * @param sb_handle SB_Handle
 * @param trans compute matrix transpose or not.
 * @param m rows of matrix
 * @param n cols of matrix
 * @param alpha Scaling factor
 * @param in_memory BufferIterator of input
 * @param ld_in leading dimension of in_matrix
 * @param inc_in internal increment for the in_matrix
 * @param matrix_out BufferIterator of output
 * @param ld_out leading dimention of out_matrix
 * @param inc_out internal increment for the out_matrix
 */
template <typename sb_handle_t, typename element_t, typename index_t,
          typename in_t, typename out_t>
typename sb_handle_t::event_t _omatcopy2(sb_handle_t& sb_handle, char trans,
                                         index_t m, index_t n, element_t alpha,
                                         in_t in_memory, index_t ld_in,
                                         index_t inc_in, out_t out_memory,
                                         index_t ld_out, index_t inc_out) {
  return internal::_matcopy<false>(sb_handle, trans, m, n, alpha, in_memory,
                                   ld_in, inc_in, out_memory, ld_out, inc_out);
}

template <typename operator_t, typename element_t, typename sb_handle_t,
          typename input_t, typename output_t, typename index_t>
typename sb_handle_t::event_t _reduction(sb_handle_t& sb_handle,
                                         input_t buffer_in, index_t ld,
                                         output_t buffer_out, index_t rows,
                                         index_t cols,
                                         reduction_dim_t reduction_dim) {
  return internal::_reduction<operator_t, element_t>(
      sb_handle, buffer_in, ld, buffer_out, rows, cols, reduction_dim);
}
}  // namespace extension

}  // namespace blas

#endif  // SYCL_BLAS_EXTENSION_INTERFACE_H
