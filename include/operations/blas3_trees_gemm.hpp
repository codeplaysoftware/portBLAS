/***************************************************************************
 *
 *  @license
 *  Copyright (C) 2017 Codeplay Software Limited
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
 *  @filename blas3_trees_gemm.hpp
 *
 **************************************************************************/

#ifndef BLAS3_TREES_GEMM_HPP
#define BLAS3_TREES_GEMM_HPP


#include <CL/sycl.hpp>


namespace blas {


// const int cl_size = 64;


template <typename T, typename GlobalPointerType>
void _gemm_v2(
    int item_id, int m, int n, int k, T alpha, GlobalPointerType A, int lda,
    GlobalPointerType B, int ldb, T beta, GlobalPointerType C, int ldc) {
  using value_type = T;
  if (item_id >= m*n) {
    return;
  }

  const int row = item_id % m;
  const int col = item_id / m;

  A = A + row;
  B = B + col*ldb;
  C = C + row + col*ldc;

  value_type reg_res = {};

  while (k > 0) {
    reg_res += A[0] * B[0];
    --k;
    A = A + lda;
    B = B + 1;
  }

  C[0] = alpha * reg_res + beta * C[0];
}


template <bool> inline bool do_check(bool cond) { return cond; }
template <> inline bool do_check<false>(bool) { return true; }


template <bool check_m_limit, bool check_n_limit, bool check_k_limit,
          int cl_elems, int block_rows, int block_cols, int wg_size,
          typename T, typename GlobalPointerType, typename LocalPointerType>
inline void extract_input_blocks(
  int item_id, int m, int n, int k,
  GlobalPointerType A, int lda, GlobalPointerType B, int ldb,
  LocalPointerType s1, LocalPointerType s3) {
  #pragma unroll
  for (int i = 0; i < block_cols * cl_elems / wg_size; ++i) {
    const bool in_range = do_check<check_k_limit>(item_id % cl_elems < k) &&
                         do_check<check_n_limit>(wg_size/cl_elems*i < n);
    s1[i*wg_size] = in_range ? B[i*(wg_size/cl_elems)*ldb] : T(0);
  }
  #pragma unroll
  for (int i = 0; i < block_rows * cl_elems / wg_size; ++i) {
    const bool in_range =
        do_check<check_n_limit>(0 < m) &&
        do_check<check_k_limit>(item_id/block_rows+i*(wg_size/block_rows) < k);
    s3[i*wg_size] = in_range ? A[i*(wg_size/block_rows)*lda] : T(0);
  }
}


template <int cl_elems, int item_rows, int item_cols, int wg_rows, int wg_cols,
          int block_rows, int block_cols,
          typename T, typename LocalPointerType>
inline void compute_block_gemm(
    LocalPointerType s2, LocalPointerType s4,
    T reg_a[item_rows], T &reg_b, T reg_res[item_rows][item_cols]) {
  for (int i = 0; i < cl_elems; ++i) {
    #pragma unroll
    for (int j = 0; j < item_rows; ++j) {
      reg_a[j] = s4[j*wg_rows + i*block_rows];
    }
    #pragma unroll
    for (int j = 0; j < item_cols; ++j) {
      reg_b = s2[i + j*cl_elems];
      #pragma unroll
      for (int l = 0; l < item_rows; ++l) {
        reg_res[l][j] += reg_a[l] * reg_b;
      }
    }
  }
}


template <bool check_m_limit, bool check_n_limit, int cl_elems, int block_rows,
          int block_cols, int wg_size, int item_rows, int item_cols,
          int wg_rows, int wg_cols, typename T, typename GlobalPointerType,
          typename LocalPointerType>
inline void compute_panel_gemm(
    cl::sycl::nd_item<1> id, int item_id,
    int m, int mc, int n, int nc, int k, T alpha,
    GlobalPointerType A, int lda, GlobalPointerType B, int ldb,
    T beta, GlobalPointerType C, int ldc,
    LocalPointerType s1, LocalPointerType s2,
    LocalPointerType s3, LocalPointerType s4,
    T reg_a[item_rows], T &reg_b, T reg_res[item_rows][item_cols]) {
  int ofs = 1;

  while (k >= cl_elems) {
    extract_input_blocks
      <check_m_limit, check_n_limit, false, cl_elems, block_rows, block_cols,
       wg_size, T>
      (item_id, m, n, k, A, lda, B, ldb, s1, s3);
    id.barrier(cl::sycl::access::fence_space::local_space);
    compute_block_gemm<cl_elems, item_rows, item_cols, wg_rows, wg_cols,
                       block_rows, block_cols>
      (s2, s4, reg_a, reg_b, reg_res);
    A = A + cl_elems*lda;
    B = B + cl_elems;
    k -= cl_elems;
    s1 = s1 + ofs*block_cols*cl_elems;
    s2 = s2 + ofs*block_cols*cl_elems;
    s3 = s3 + ofs*block_rows*cl_elems;
    s4 = s4 + ofs*block_rows*cl_elems;
    ofs = -ofs;
  }

  if (k > 0) {
    extract_input_blocks
      <check_m_limit, check_n_limit, true, cl_elems, block_rows, block_cols,
       wg_size, T>
      (item_id, m, n, k, A, lda, B, ldb, s1, s3);
    id.barrier(cl::sycl::access::fence_space::local_space);
    compute_block_gemm<cl_elems, item_rows, item_cols, wg_rows, wg_cols,
                       block_rows, block_cols>
      (s2, s4, reg_a, reg_b, reg_res);
  }

  #pragma unroll
  for (int i = 0; i < item_cols; ++i) {
    #pragma unroll
    for (int j = 0; j < item_rows; ++j) {
      const bool in_range = do_check<check_m_limit>(j*wg_rows < mc) &&
                            do_check<check_n_limit>(i < nc);
      if (in_range) {
        C[j*wg_rows] = alpha*reg_res[j][i] + beta*C[j*wg_rows];
      }
    }
    C = C + ldc;
  }
}


template <int cl, int item_rows, int item_cols, int wg_rows, int wg_cols,
          typename T, typename GlobalPointerType, typename LocalPointerType>
void _gemm_v19(
    cl::sycl::nd_item<1> id, int wg_id, int item_id, int m, int n, int k,
    T alpha, GlobalPointerType A, int lda, GlobalPointerType B, int ldb,
    T beta, GlobalPointerType C, int ldc, LocalPointerType scratch) {
  const auto cl_elems = cl / sizeof(T);

  const auto wg_size = wg_rows * wg_cols;

  const auto block_rows = wg_rows * item_rows;
  const auto block_cols = wg_cols * item_cols;

  const auto wg_per_col = (m - 1) / block_rows + 1;
  const auto wg_row = (wg_id % wg_per_col) * block_cols;
  const auto wg_col = (wg_id / wg_per_col) * block_cols;

  const auto item_row = item_id % wg_rows;
  const auto item_col = (item_id / wg_rows) * item_cols;

  const auto row = wg_row + item_row;
  const auto col = wg_col + item_col;

  const auto c_row = item_id % cl_elems;
  const auto c_col = item_id / cl_elems;
  const auto c_inc = wg_size / cl_elems;

  T reg_res[item_rows][item_cols] = {};
  T reg_a[item_rows];
  T reg_b;

  C = C + row + col*ldc;
  const auto mc = m - row;
  const auto nc = n - col;

  const bool internal = m - wg_row >= block_rows && n - wg_col >= block_cols;

  B = B + c_row + (wg_col + c_col)*ldb;
  n = n - wg_col - c_col;
  A = A + wg_row + item_id%block_rows + (item_id/block_rows)*lda;
  m = m - wg_row - item_id%block_rows;

  LocalPointerType s1 = scratch + c_row + c_col*cl_elems;
  LocalPointerType s2 = scratch + item_col*cl_elems;
  LocalPointerType s3 = scratch + 2*block_cols*cl_elems + item_id;
  LocalPointerType s4 = scratch + 2*block_cols*cl_elems + item_row;

  if (internal) {
    compute_panel_gemm
      <false, false, cl_elems, block_rows, block_cols, wg_size,
       item_rows, item_cols, wg_rows, wg_cols>
      (id, item_id, m, mc, n, nc, k, alpha, A, lda, B, ldb, beta, C,
       ldc, s1, s2, s3, s4, reg_a, reg_b, reg_res);
  } else {
    compute_panel_gemm
      <true, true, cl_elems, block_rows, block_cols, wg_size,
       item_rows, item_cols, wg_rows, wg_cols>
      (id, item_id, m, mc, n, nc, k, alpha, A, lda, B, ldb, beta, C,
       ldc, s1, s2, s3, s4, reg_a, reg_b, reg_res);
  }

}

}  // namespace blas


#endif  // BLAS3_TREES_GEMM_HPP

