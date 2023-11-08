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
 *  @filename reduction_interface.h
 *
 **************************************************************************/

#ifndef PORTBLAS_EXTENSION_INTERFACE_H
#define PORTBLAS_EXTENSION_INTERFACE_H

#include "operations/extension/reduction.h"
#include "operations/extension/transpose.h"
#include "sb_handle/portblas_handle.h"

namespace blas {

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
 * @tparam in_t Buffer Iterator or USM Pointer
 * @tparam out_t Buffer Iterator or USM Pointer
 * @param sb_handle SB_Handle
 * @param trans compute matrix transpose or not.
 * @param m rows of matrix
 * @param n cols of matrix
 * @param alpha Scaling factor
 * @param in_memory BufferIterator or USM Pointer of input
 * @param ld_in leading dimension of in_matrix
 * @param inc_in internal increment for the in_matrix
 * @param matrix_out BufferIterator or USM Pointer of output
 * @param ld_out leading dimention of out_matrix
 * @param inc_out internal increment for the out_matrix
 */
template <bool in_place, typename sb_handle_t, typename element_t,
          typename index_t, typename in_t, typename out_t>
typename sb_handle_t::event_t _matcopy(
    sb_handle_t& sb_handle, char trans, index_t m, index_t n, element_t alpha,
    in_t in_memory, index_t ld_in, index_t inc_in, out_t out_memory,
    index_t ld_out, index_t inc_out,
    const typename sb_handle_t::event_t& _dependencies);

template <typename sb_handle_t, typename element_t, typename index_t,
          typename container_0_t, typename container_1_t,
          typename container_2_t>
typename sb_handle_t::event_t _omatadd(
    sb_handle_t& sb_handle, char trans_a, char trans_b, index_t m, index_t n,
    element_t alpha, container_0_t a, index_t lda, element_t beta,
    container_1_t b, index_t ldb, container_2_t c, index_t ldc,
    const typename sb_handle_t::event_t& _dependencies);

template <bool in_place, typename element_t, typename sb_handle_t,
          typename index_t, typename in_t, typename out_t>
typename sb_handle_t::event_t _transpose(
    sb_handle_t& sb_handle, index_t m, index_t n, in_t A, index_t ld_a, out_t B,
    index_t ld_b, const typename sb_handle_t::event_t& _dependencies);

template <bool in_place, typename sb_handle_t, typename element_t,
          typename index_t, typename in_t, typename out_t>
typename sb_handle_t::event_t _matcopy_batch(
    sb_handle_t& sb_handle, char trans, index_t m, index_t n, element_t alpha,
    in_t in_memory, index_t ld_in, index_t stride_in, out_t out_memory,
    index_t ld_out, index_t stride_out, index_t batch_size,
    const typename sb_handle_t::event_t& _dependencies);

template <uint32_t TileSize, int TilePerWG, typename sb_handle_t,
          typename element_t, typename index_t, typename in_t, typename out_t>
typename sb_handle_t::event_t _matcopy_batch_impl(
    sb_handle_t& sb_handle, index_t m, index_t n, element_t alpha, in_t memory,
    index_t ld_in, index_t in_stride, out_t out_memory, index_t ld_out,
    index_t out_stride, index_t batch_size,
    const typename sb_handle_t::event_t& _dependencies);

template <typename sb_handle_t, typename element_t, typename index_t,
          typename container_0_t, typename container_1_t,
          typename container_2_t>
typename sb_handle_t::event_t _omatadd_batch(
    sb_handle_t& sb_handle, char trans_a, char trans_b, index_t m, index_t n,
    element_t alpha, container_0_t a, index_t lda, index_t stride_a,
    element_t beta, container_1_t b, index_t ldb, index_t stride_b,
    container_2_t c, index_t ldc, index_t stride_c, index_t batch_size,
    const typename sb_handle_t::event_t& _dependencies);

template <uint32_t TileSize, int TilePerWG, typename sb_handle_t,
          typename element_t, typename index_t, typename container_0_t,
          typename container_1_t, typename container_2_t>
typename sb_handle_t::event_t _omatadd_batch_impl(
    sb_handle_t& sb_handle, index_t m, index_t n, element_t alpha,
    container_0_t A, index_t lda, index_t stride_a, element_t beta,
    container_1_t B, index_t ldb, index_t stride_b, container_2_t C,
    index_t ldc, index_t stride_c, index_t batch_size,
    const typename sb_handle_t::event_t& _dependencies);

template <typename operator_t, typename element_t, typename sb_handle_t,
          typename input_t, typename output_t, typename index_t>
typename sb_handle_t::event_t _reduction(
    sb_handle_t& sb_handle, input_t buffer_in, index_t ld, output_t buffer_out,
    index_t rows, index_t cols, reduction_dim_t reduction_dim,
    const typename sb_handle_t::event_t& _dependencies);

template <int Tile_size, int wg_size, int cl_size, bool local_memory,
          typename sb_handle_t, typename container_0_t, typename container_1_t,
          typename element_t, typename index_t>
typename sb_handle_t::event_t _transpose_outplace_impl(
    sb_handle_t& sb_handle, index_t _M, index_t _N, element_t _alpha,
    container_0_t in_, index_t _ld_in, index_t _inc_in, index_t _stride_in,
    container_1_t out_, index_t _ld_out, index_t _inc_out, index_t _stride_out,
    index_t _batch_size, const typename sb_handle_t::event_t& _dependencies);

template <bool both_trans, int Tile_size, int wg_size, int cl_size,
          bool local_memory, typename sb_handle_t, typename container_0_t,
          typename container_1_t, typename container_2_t, typename element_t,
          typename index_t>
typename sb_handle_t::event_t _transpose_add_impl(
    sb_handle_t& sb_handle, index_t _M, index_t _N, element_t _alpha,
    container_0_t a_, index_t _lda, index_t _nrows_a, index_t _ncols_a,
    index_t _stride_a, element_t _beta, container_1_t b_, index_t _ldb,
    index_t _nrows_b, index_t _ncols_b, index_t _stride_b, container_2_t c_,
    index_t _ldc, index_t _stride_c, index_t _batch_size,
    const typename sb_handle_t::event_t& _dependencies);

template <typename sb_handle_t, typename container_0_t, typename container_1_t,
          typename element_t, typename index_t>
typename sb_handle_t::event_t _axpy_batch(
    sb_handle_t& sb_handle, index_t _N, element_t _alpha, container_0_t _vx,
    index_t _incx, index_t _stride_x, container_1_t _vy, index_t _incy,
    index_t _stride_y, index_t _batch_size,
    const typename sb_handle_t::event_t& _dependencies);

}  // namespace internal

/**
 * \brief COPY in_matrix to out_matrix with scaling factor of alpha
 *
 * @tparam sb_handle_t SB_Handle type
 * @tparam element_t Scaling factor type
 * @tparam index_t Index type
 * @tparam in_t Buffer Iterator or USM Pointer
 * @tparam out_t Buffer Iterator or USM Pointer
 * @param sb_handle SB_Handle
 * @param trans compute matrix transpose or not.
 * @param m rows of matrix
 * @param n cols of matrix
 * @param alpha Scaling factor
 * @param in_memory BufferIterator or USM Pointer of input
 * @param ld_in leading dimension of in_matrix
 * @param matrix_out BufferIterator or USM Pointer of output
 * @param ld_out leading dimention of out_matrix
 */
template <typename sb_handle_t, typename element_t, typename index_t,
          typename in_t, typename out_t>
typename sb_handle_t::event_t _omatcopy(
    sb_handle_t& sb_handle, char trans, index_t m, index_t n, element_t alpha,
    in_t in_memory, index_t ld_in, out_t out_memory, index_t ld_out,
    const typename sb_handle_t::event_t& _dependencies = {}) {
  return internal::_matcopy<false>(
      sb_handle, trans, m, n, alpha, in_memory, ld_in, static_cast<index_t>(1),
      out_memory, ld_out, static_cast<index_t>(1), _dependencies);
}

/**
 * \brief Copy out of place of in_matrix to out_matrix with increment between
 * cols element different from 1.
 *
 * The increment within the same column can be different from 1 and specified
 * by inc_in and inc_out arguments.
 *
 * @tparam sb_handle_t SB_Handle type
 * @tparam element_t Scaling factor type
 * @tparam index_t Index type
 * @tparam in_t Buffer Iterator or USM Pointer
 * @tparam out_t Buffer Iterator or USM Pointer
 * @param sb_handle SB_Handle
 * @param trans compute matrix transpose or not.
 * @param m rows of matrix
 * @param n cols of matrix
 * @param alpha Scaling factor
 * @param in_memory BufferIterator or USM Pointer of input
 * @param ld_in leading dimension of in_matrix
 * @param inc_in internal increment for the in_matrix
 * @param matrix_out BufferIterator or USM Pointer of output
 * @param ld_out leading dimention of out_matrix
 * @param inc_out internal increment for the out_matrix
 */
template <typename sb_handle_t, typename element_t, typename index_t,
          typename in_t, typename out_t>
typename sb_handle_t::event_t _omatcopy2(
    sb_handle_t& sb_handle, char trans, index_t m, index_t n, element_t alpha,
    in_t in_memory, index_t ld_in, index_t inc_in, out_t out_memory,
    index_t ld_out, index_t inc_out,
    const typename sb_handle_t::event_t& _dependencies = {}) {
  return internal::_matcopy<false>(sb_handle, trans, m, n, alpha, in_memory,
                                   ld_in, inc_in, out_memory, ld_out, inc_out,
                                   _dependencies);
}

/**
 * \brief Computes scaled addition of two matrices A & B with or without
 * transpose and copying results back to an output matrix C.
 *
 * @tparam sb_handle_t SB_Handle type
 * @tparam element_t Undelying element data type of the matrix container
 * @tparam index_t Index type
 * @tparam container_t Inputs/Output Container Type
 * @param trans_a Apply or not matrix transpose to A.
 * @param trans_b Apply or not matrix transpose to B.
 * @param m Number of rows in output matrix C
 * @param n Number of columns in output matrix C
 * @param alpha Scaling factor of matrix A
 * @param A Container Input matrix A
 * @param lda Matrix A leading dimension
 * @param beta scaling factor of matrix B
 * @param B Container Input matrix B
 * @param ldb Matrix B leading dimension
 * @param C Container Output matrix C
 * @param ldc Matrix C leading dimension
 */
template <typename sb_handle_t, typename element_t, typename index_t,
          typename container_0_t, typename container_1_t,
          typename container_2_t>
typename sb_handle_t::event_t _omatadd(
    sb_handle_t& sb_handle, char trans_a, char trans_b, index_t m, index_t n,
    element_t alpha, container_0_t A, index_t lda, element_t beta,
    container_1_t B, index_t ldb, container_2_t C, index_t ldc,
    const typename sb_handle_t::event_t& _dependencies = {}) {
  return internal::_omatadd(sb_handle, trans_a, trans_b, m, n, alpha, A, lda,
                            beta, B, ldb, C, ldc, _dependencies);
}
/**
 * \brief COPY batch of matrices inplace with scaling factor of alpha
 *
 * @tparam sb_handle_t SB_Handle type
 * @tparam element_t Scaling factor type
 * @tparam index_t Index type
 * @tparam in_out_t input/output type
 * @param sb_handle SB_Handle
 * @param trans compute matrix transpose or not
 * @param m rows of matrix
 * @param n cols of matrix
 * @param alpha Scaling factor
 * @param memory container of input & output matrices
 * @param ld_in leading dimension at input
 * @param ld_out leading dimention at output
 * @param stride stride distance between matrices inside batch
 * @param batch_size number of matrices to compute
 */
template <typename sb_handle_t, typename element_t, typename index_t,
          typename in_out_t>
typename sb_handle_t::event_t _imatcopy_batch(sb_handle_t& sb_handle,
                                              char trans, index_t m, index_t n,
                                              element_t alpha, in_out_t memory,
                                              index_t ld_in, index_t ld_out,
                                              index_t stride,
                                              index_t batch_size) {
  return internal::_matcopy_batch<true>(sb_handle, trans, m, n, alpha, memory,
                                        ld_in, stride, memory, ld_out, stride,
                                        batch_size);
}

/**
 * \brief COPY batch of matrices outplace from in_memory to out_memory with
 * scaling factor of alpha
 *
 * @tparam sb_handle_t SB_Handle type
 * @tparam element_t Scaling factor type
 * @tparam index_t Index type
 * @tparam in_t container input type
 * @tparam out_t container output type
 * @param sb_handle SB_Handle
 * @param trans compute matrix transpose or not
 * @param m rows of matrix
 * @param n cols of matrix
 * @param alpha Scaling factor
 * @param in_memory input matrix container
 * @param ld_in leading dimension of input
 * @param stride_in stride distance between matrices inside batch
 * @param out_memory output matrix container
 * @param ld_out leading dimention of output
 * @param stride_out stride distance between matrices inside batch
 * @param batch_size number of matrices to compute
 */
template <typename sb_handle_t, typename element_t, typename index_t,
          typename in_t, typename out_t>
typename sb_handle_t::event_t _omatcopy_batch(
    sb_handle_t& sb_handle, char trans, index_t m, index_t n, element_t alpha,
    in_t in_memory, index_t ld_in, index_t stride_in, out_t out_memory,
    index_t ld_out, index_t stride_out, index_t batch_size,
    const typename sb_handle_t::event_t& _dependencies = {}) {
  return internal::_matcopy_batch<false>(
      sb_handle, trans, m, n, alpha, in_memory, ld_in, stride_in, out_memory,
      ld_out, stride_out, batch_size, _dependencies);
}

/**
 * \brief Batch Computation of scaled addition of two matrices A & B with or
 * without transpose and copying results back to an output matrix C.
 *
 * @tparam sb_handle_t SB_Handle type
 * @tparam element_t Undelying element data type of the matrix container
 * @tparam index_t Index type
 * @tparam container_t Inputs/Output Container Type
 * @param trans_a Apply or not matrix transpose to A.
 * @param trans_b Apply or not matrix transpose to B.
 * @param m Number of rows in output matrix C
 * @param n Number of columns in output matrix C
 * @param alpha Scaling factor of matrix A
 * @param A Container Input matrix A
 * @param lda Matrix A leading dimension
 * @param stride_a stride distance between two matrices inside A container
 * @param beta scaling factor of matrix B
 * @param B Container Input matrix B
 * @param ldb Matrix B leading dimension
 * @param stride_b stride distance between two matrices inside B container
 * @param C Container Output matrix C
 * @param ldc Matrix C leading dimension
 * @param stride_c stride distance between two matrices inside C container
 * @param batch_size number of matrices to compute in this batch
 */
template <typename sb_handle_t, typename element_t, typename index_t,
          typename container_0_t, typename container_1_t,
          typename container_2_t>
typename sb_handle_t::event_t _omatadd_batch(
    sb_handle_t& sb_handle, char trans_a, char trans_b, index_t m, index_t n,
    element_t alpha, container_0_t a, index_t lda, index_t stride_a,
    element_t beta, container_1_t b, index_t ldb, index_t stride_b,
    container_2_t c, index_t ldc, index_t stride_c, index_t batch_size,
    const typename sb_handle_t::event_t& _dependencies = {}) {
  return internal::_omatadd_batch(sb_handle, trans_a, trans_b, m, n, alpha, a,
                                  lda, stride_a, beta, b, ldb, stride_b, c, ldc,
                                  stride_c, batch_size, _dependencies);
}

/**
 * \brief Compute a batch of AXPY operation all together
 *
 * Implements AXPY \f$y = ax + y\f$
 *
 * @param sb_handle SB_Handle
 * @param _alpha scalar
 * @param _vx BufferIterator or USM pointer
 * @param _incx Increment for the vector X
 * @param _stride_x Stride distance of two consecutive vectors in X
 * @param _vy BufferIterator or USM pointer
 * @param _incy Increment for the vector Y
 * @param _stride_y Stride distance of two consecutive vectors in Y
 * @param _batch_size number of axpy operations to compute
 * @param _dependencies Vector of events
 */
template <typename sb_handle_t, typename container_0_t, typename container_1_t,
          typename element_t, typename index_t>
typename sb_handle_t::event_t _axpy_batch(
    sb_handle_t& sb_handle, index_t _N, element_t _alpha, container_0_t _vx,
    index_t _incx, index_t _stride_x, container_1_t _vy, index_t _incy,
    index_t _stride_y, index_t _batch_size,
    const typename sb_handle_t::event_t& _dependencies = {}) {
  return internal::_axpy_batch(sb_handle, _N, _alpha, _vx, _incx, _stride_x,
                               _vy, _incy, _stride_y, _batch_size,
                               _dependencies);
}

namespace extension {
/**
 * \brief Transpose a Matrix in-place
 *
 * Provided matrix A serves as input with leading dimension ld_in as well as
 * output with leading dimension ld_out to which it's transposed.
 *
 * @tparam element_t Undelying element data type of the matrix container
 * @tparam sb_handle_t SB_Handle type
 * @tparam index_t Index type
 * @tparam in_t Input Container Type
 * @tparam out_t Output Container Type
 * @param sb_handle sb_handle
 * @param m Rows of matrix (input)
 * @param n Columns of matrix (input)
 * @param A Input-Output matrix container
 * @param ld_in leading dimension of A at input
 * @param ld_out leading dimention of A at output
 */
template <typename element_t, typename sb_handle_t, typename index_t,
          typename in_t, typename out_t>
typename sb_handle_t::event_t _transpose(
    sb_handle_t& sb_handle, index_t m, index_t n, in_t A, index_t ld_in,
    index_t ld_out, const typename sb_handle_t::event_t& _dependencies = {}) {
  return blas::internal::_transpose<true, element_t>(sb_handle, m, n, A, ld_in,
                                                     A, ld_out, _dependencies);
}

/**
 * \brief Transpose a Matrix out-of-place
 *
 * Input matrix A with leading dimension ld_a gets transposed and written back
 * to matrix B with leading dimension ld_b.
 *
 * @tparam element_t Undelying element data type of the matrix container
 * @tparam sb_handle_t SB_Handle type
 * @tparam index_t Index type
 * @tparam in_t Input Container Type
 * @tparam out_t Output Container Type
 * @param sb_handle sb_handle
 * @param m Rows of matrix A
 * @param n Columns of matrix A
 * @param A Input matrix container
 * @param ld_a leading dimension of A
 * @param B Output matrix container
 * @param ld_b leading dimention of B
 */
template <typename element_t, typename sb_handle_t, typename index_t,
          typename in_t, typename out_t>
typename sb_handle_t::event_t _transpose(
    sb_handle_t& sb_handle, index_t m, index_t n, in_t A, index_t ld_a, out_t B,
    index_t ld_b, const typename sb_handle_t::event_t& _dependencies = {}) {
  return blas::internal::_transpose<false, element_t>(sb_handle, m, n, A, ld_a,
                                                      B, ld_b, _dependencies);
}

template <typename operator_t, typename element_t, typename sb_handle_t,
          typename input_t, typename output_t, typename index_t>
typename sb_handle_t::event_t _reduction(
    sb_handle_t& sb_handle, input_t buffer_in, index_t ld, output_t buffer_out,
    index_t rows, index_t cols, reduction_dim_t reduction_dim,
    const typename sb_handle_t::event_t& _dependencies = {}) {
  return blas::internal::_reduction<operator_t, element_t>(
      sb_handle, buffer_in, ld, buffer_out, rows, cols, reduction_dim,
      _dependencies);
}

}  // namespace extension
}  // namespace blas

#endif  // PORTBLAS_EXTENSION_INTERFACE_H
