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


const int cl_size = 64;


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
          int cl_elems, int b_size, int c_inc, int wg_size, typename T>
inline void extract_input_blocks(
  int c_row, int item_id, int m, int n, int k,
  cl::sycl::global_ptr<T> A, int lda, cl::sycl::global_ptr<T> B, int ldb,
  cl::sycl::local_ptr<T> s1, cl::sycl::local_ptr<T> s3) {
  #pragma unroll
  for (int i = 0; i < b_size/c_inc; ++i) {
    const bool in_range = do_check<check_k_limit>(c_row < k) &&
                         do_check<check_n_limit>(c_inc*i < n);
    s1[c_inc*i*cl_elems] = in_range ? B[c_inc*i*ldb] : T(0);
  }
  #pragma unroll
  for (int i = 0; i < b_size/c_inc; ++i) {
    const bool in_range =
        do_check<check_n_limit>(0 < m) &&
        do_check<check_k_limit>(item_id/b_size + i*(wg_size/b_size) < k);
    s3[i*wg_size] = in_range ? A[i*(wg_size/b_size)*lda] : T(0);
  }
}


template <int cl_elems, int wsize, int rsize, int csize, int b_size,
          typename T>
inline void compute_block_gemm(
    cl::sycl::local_ptr<T> s2, cl::sycl::local_ptr<T> s4,
    T reg_a[wsize/rsize], T &reg_b, T reg_res[wsize/rsize][cl_elems/csize]) {
  for (int i = 0; i < cl_elems; ++i) {
    #pragma unroll
      for (int j = 0; j < wsize/rsize; ++j) {
      reg_a[j] = s4[j*rsize*cl_elems + i*b_size];
    }
    #pragma unroll
    for (int j = 0; j < cl_elems/csize; ++j) {
      reg_b = s2[i + j*cl_elems];
      #pragma unroll
      for (int l = 0; l < wsize/rsize; ++l) {
        reg_res[l][j] += reg_a[l] * reg_b;
      }
    }
  }
}


template <bool check_m_limit, bool check_n_limit, int cl_elems, int b_size,
          int c_inc, int wg_size, int wsize, int rsize, int csize, typename T>
inline void compute_panel_gemm(
    cl::sycl::nd_item<1> id, int c_row, int item_id,
    int m, int mc, int n, int nc, int k, T alpha,
    cl::sycl::global_ptr<T> A, int lda, cl::sycl::global_ptr<T> B, int ldb,
    T beta, cl::sycl::global_ptr<T> C, int ldc,
    cl::sycl::local_ptr<T> s1, cl::sycl::local_ptr<T> s2,
    cl::sycl::local_ptr<T> s3, cl::sycl::local_ptr<T> s4,
    T reg_a[wsize/rsize], T &reg_b, T reg_res[wsize/rsize][cl_elems/csize]) {
  int ofs = 1;

  while (k >= cl_elems) {
    extract_input_blocks
      <check_m_limit, check_n_limit, false, cl_elems, b_size, c_inc, wg_size>
      (c_row, item_id, m, n, k, A, lda, B, ldb, s1, s3);
    id.barrier(cl::sycl::access::fence_space::local_space);
    compute_block_gemm<cl_elems, wsize, rsize, csize, b_size>
      (s2, s4, reg_a, reg_b, reg_res);
    A = A + cl_elems*lda;
    B = B + cl_elems;
    k -= cl_elems;
    s1 = s1 + ofs*b_size*cl_elems;
    s2 = s2 + ofs*b_size*cl_elems;
    s3 = s3 + ofs*b_size*cl_elems;
    s4 = s4 + ofs*b_size*cl_elems;
    ofs = -ofs;
  }

  if (k > 0) {
    extract_input_blocks
      <check_m_limit, check_n_limit, true, cl_elems, b_size, c_inc, wg_size>
      (c_row, item_id, m, n, k, A, lda, B, ldb, s1, s3);
    id.barrier(cl::sycl::access::fence_space::local_space);
    compute_block_gemm<cl_elems, wsize, rsize, csize, b_size>
      (s2, s4, reg_a, reg_b, reg_res);
  }

  #pragma unroll
  for (int i = 0; i < cl_elems/csize; ++i) {
    #pragma unroll
    for (int j = 0; j < wsize/rsize; ++j) {
      const bool in_range = do_check<check_m_limit>(j*rsize*cl_elems < mc) &&
                            do_check<check_n_limit>(i < nc);
      if (in_range) {
        C[j*rsize*cl_elems] = alpha*reg_res[j][i] + beta*C[j*rsize*cl_elems];
      }
    }
    C = C + ldc;
  }
}


template <int rsize, int csize, int wsize, typename T>
void _gemm_v17(
    cl::sycl::nd_item<1> id, int wg_id, int item_id, int m, int n, int k,
    T alpha, cl::sycl::global_ptr<T> A, int lda, cl::sycl::global_ptr<T> B,
    int ldb, T beta, cl::sycl::global_ptr<T> C, int ldc,
    cl::sycl::local_ptr<T> scratch) {

  const int cl_elems = cl_size / sizeof(T);
  const int wg_size = rsize * csize * wsize * cl_elems;
  const int b_size = cl_elems * wsize;

  const int wg_per_col = (m - 1) / b_size + 1;
  const int wg_row = (wg_id % wg_per_col) * b_size;
  const int wg_col = (wg_id / wg_per_col) * b_size;

  const int b_row = item_id / (b_size * csize);
  const int b_row_id = item_id % (b_size * csize);

  const int item_row = b_row*cl_elems + b_row_id % cl_elems;
  const int item_col = b_row_id / cl_elems * (cl_elems / csize);

  const int row = wg_row + item_row;
  const int col = wg_col + item_col;

  const int c_row = item_id % cl_elems;
  const int c_col = item_id / cl_elems;
  const int c_inc = wg_size / cl_elems;

  T reg_res[wsize/rsize][cl_elems/csize] = {};
  T reg_a[wsize/rsize];
  T reg_b;

  C = C + row + col*ldc;
  const auto mc = m - row;
  const auto nc = n - col;

  const bool internal = m - wg_row >= b_size && n - wg_col >= b_size;

  B = B + c_row + (wg_col + c_col)*ldb;
  n = n - wg_col - c_col;
  A = A + wg_row + item_id%b_size + (item_id/b_size)*lda;
  m = m - wg_row - item_id%b_size;

  cl::sycl::local_ptr<T> s1 = scratch + c_row + c_col*cl_elems;
  cl::sycl::local_ptr<T> s2 = scratch + item_col*cl_elems;
  cl::sycl::local_ptr<T> s3 = scratch + 2*b_size*cl_elems + item_id;
  cl::sycl::local_ptr<T> s4 = scratch + 2*b_size*cl_elems + item_row;

  if (internal) {
    compute_panel_gemm
      <false, false, cl_elems, b_size, c_inc, wg_size, wsize, rsize, csize>
      (id, c_row, item_id, m, mc, n, nc, k, alpha, A, lda, B, ldb, beta, C,
       ldc, s1, s2, s3, s4, reg_a, reg_b, reg_res);
  } else {
    compute_panel_gemm
      <true, true, cl_elems, b_size, c_inc, wg_size, wsize, rsize, csize>
      (id, c_row, item_id, m, mc, n, nc, k, alpha, A, lda, B, ldb, beta, C,
       ldc, s1, s2, s3, s4, reg_a, reg_b, reg_res);
  }

}


}  // namespace blas


#endif  // BLAS3_TREES_GEMM_HPP

