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


#include <utils/vec.hpp>
#include <utils/matrix.hpp>


namespace blas {


namespace vector {


template <int vec_size, typename VecType>
inline void _gemv(const VecType *A, const VecType &B, VecType &C) {
  using element_type = typename VecType::element_type;
  for_vec_elem<0, vec_size>::map(B, [&] (int i, element_type el) {
    C += el * A[i];
  });
}


}  // namespace vector


namespace thread {


template <int scratch_width, int thr_size, int vec_size,
          typename VecType, typename ScratchType>
inline void _gemm(const VecType *A, ScratchType B, VecType *C) {
  VecType vecB;
  #pragma unroll
  for (int i = 0; i < scratch_width; ++i) {
    #pragma unroll
    for (int j = 0; j < thr_size; ++j) {
      vecB.load(i*thr_size + j, B);
      vector::_gemv<vec_size>(A + j*vec_size, vecB, C[i]);
    }
  }
}


}  // namespace thread


template <typename T, typename MatrixTypeA, typename MatrixTypeB,
          typename MatrixTypeC>
void _gemm_host_v1(T alpha, const MatrixTypeA A, const MatrixTypeB B, T beta,
                   MatrixTypeC C) {
  using value_type = decltype(C(0, 0));
  if (beta != T(1)) {
    for_each(C, [beta](int, int, value_type& v) { v *= beta; });
  }
  for (int j = 0; j < C.get_num_cols(); ++j) {
    for (int k = 0; k < A.get_num_cols(); ++k) {
      auto tmp = alpha * B(k, j);
      for (int i = 0; i < C.get_num_rows(); ++i) {
        C(i, j) += tmp * A(i, k);
      }
    }
  }
}


template <typename T, typename MatrixTypeA, typename MatrixTypeB,
          typename MatrixTypeC>
void _gemm_host_v2(T alpha, const MatrixTypeA &A, const MatrixTypeB &B, T beta,
                   MatrixTypeC &C, int block_size = 64) {
  using value_type = decltype(C(0, 0));
  if (beta != T(1)) {
    for_each(C, [beta](int, int, value_type& v) { v *= beta; });
  }
  const auto t = range(0, block_size);
  for (auto j = t; C.get_num_cols() > j; ++j) {
    for (auto k = t; A.get_num_cols() > k; ++k) {
      for (auto i = t; C.get_num_rows() > i; ++i) {
        _gemm_host_v1(alpha, A(i, k), B(k, j), T(1), C(i, j));
      }
    }
  }
}


template <typename T, typename MatrixTypeA, typename MatrixTypeB,
          typename MatrixTypeC>
void _gemm_v1(int item_id, T alpha, const MatrixTypeA A, const MatrixTypeB B,
              T beta, MatrixTypeC C) {
  using value_type = decltype(C(0, 0));
  if (item_id >= C.get_num_rows() * C.get_num_cols()) {
    return;
  }
  const int col = item_id / C.get_num_rows();
  const int row = item_id % C.get_num_rows();
  C(row, col) *= beta;
  for (int k = 0; k < A.get_num_cols(); ++k) {
    C(row, col) += alpha * A(row, k) * B(k, col);
  }
}


template <typename T, typename MatrixTypeA, typename MatrixTypeB,
          typename MatrixTypeC>
void _gemm_v2(int item_id, T alpha, const MatrixTypeA A, const MatrixTypeB B,
              T beta, MatrixTypeC C) {
  using value_type = typename MatrixTypeC::value_type;
  if (item_id >= C.get_num_rows() * C.get_num_cols()) {
    return;
  }
  const int col = item_id / C.get_num_rows();
  const int row = item_id % C.get_num_rows();
  value_type tmp(0);
  for (int k = 0; k < A.get_num_cols(); ++k) {
    tmp += A(row, k) * B(k, col);
  }
  C(row, col) = alpha * tmp + beta * C(row, col);
}


template <typename T, typename MatrixTypeA, typename MatrixTypeB,
          typename MatrixTypeC>
void _gemm_v3(int wg_id, int wg_size, int item_id, T alpha,
              const MatrixTypeA A, const MatrixTypeB B, T beta,
              MatrixTypeC C) {
  using value_type = typename MatrixTypeC::value_type;
  const int wg_per_col = (C.get_num_rows() + wg_size - 1) / wg_size;
  const int col = wg_id / wg_per_col;
  const int row = (wg_id % wg_per_col) * wg_size + item_id;
  if (col >= C.get_num_cols() || row >= C.get_num_rows()) {
    return;
  }
  value_type tmp(0);
  for (int k = 0; k < A.get_num_cols(); ++k) {
    tmp += A(row, k) * B(k, col);
  }
  C(row, col) = alpha * tmp + beta * C(row, col);
}


template <int wsize, typename T, typename MatrixTypeA, typename MatrixTypeB,
          typename MatrixTypeC>
void _gemm_v4(int wg_id, int wg_size, int item_id, T alpha,
              const MatrixTypeA A, const MatrixTypeB B, T beta,
              MatrixTypeC C) {
  using value_type = typename MatrixTypeC::value_type;
  const int wg_per_col = (C.get_num_rows() + wg_size - 1) / wg_size;
  const int col = (wg_id / wg_per_col) * wsize;
  const int row = (wg_id % wg_per_col) * wg_size + item_id;
  if (col >= C.get_num_cols() || row >= C.get_num_rows()) {
    return;
  }
  value_type tmp[wsize] = {};
  value_type a;
  for (int k = 0; k < A.get_num_cols(); ++k) {
    a = A(row, k);
    #pragma unroll
    for (int j = 0; j < wsize; ++j) {
      if (col + j < C.get_num_cols()) {
        tmp[j] += a * B(k, col + j);
      }
    }
  }
  #pragma unroll
  for (int j = 0; j < wsize; ++j) {
    if (col + j < C.get_num_cols()) {
      C(row, col + j) = alpha * tmp[j] + beta * C(row, col + j);
    }
  }
}


template <typename T, typename MatrixTypeA, typename MatrixTypeB,
          typename MatrixTypeC>
void _gemm_v5(int wg_id, int wg_size, int item_id, T alpha,
              const MatrixTypeA A, const MatrixTypeB B, T beta,
              MatrixTypeC C) {
  using value_type = typename MatrixTypeC::value_type;
  const int wg_per_col = (C.get_num_rows() + wg_size - 1) / wg_size;
  const int col = (wg_id / wg_per_col) * 4;
  const int row = (wg_id % wg_per_col) * wg_size + item_id;
  if (col >= C.get_num_cols() || row >= C.get_num_rows()) {
    return;
  }

  value_type t0 = 0, t1 = 0, t2 = 0, t3 = 0;
  value_type a;
  for (int k = 0; k < A.get_num_cols(); ++k) {
    a = A(row, k);
    if (col + 0 < C.get_num_cols()) {
      t0 += a * B(k, col + 0);
    }
    if (col + 1 < C.get_num_cols()) {
      t1 += a * B(k, col + 1);
    }
    if (col + 2 < C.get_num_cols()) {
      t2 += a * B(k, col + 2);
    }
    if (col + 3 < C.get_num_cols()) {
      t3 += a * B(k, col + 3);
    }
  }

  if (col + 0 < C.get_num_cols()) {
    C(row, col + 0) = alpha * t0 + beta * C(row, col + 0);
  }
  if (col + 1 < C.get_num_cols()) {
    C(row, col + 1) = alpha * t1 + beta * C(row, col + 1);
  }
  if (col + 2 < C.get_num_cols()) {
    C(row, col + 2) = alpha * t2 + beta * C(row, col + 2);
  }
  if (col + 3 < C.get_num_cols()) {
    C(row, col + 3) = alpha * t3 + beta * C(row, col + 3);
  }
}


template <typename T, typename MatrixTypeA, typename MatrixTypeB,
          typename MatrixTypeC>
void _gemm_v6(int wg_id, int wg_size, int item_id, T alpha,
              const MatrixTypeA A, const MatrixTypeB B, T beta,
              MatrixTypeC C) {
  using value_type = typename MatrixTypeC::value_type;
  const int wg_per_col = (C.get_num_rows() + wg_size - 1) / wg_size;
  const int col = (wg_id / wg_per_col) * 4;
  const int row = (wg_id % wg_per_col) * wg_size + item_id;
  if (col >= C.get_num_cols() || row >= C.get_num_rows()) {
    return;
  }

  value_type tmp[4] = {};
  value_type a;
  for (int k = 0; k < A.get_num_cols(); ++k) {
    a = A(row, k);
    if (col + 0 < C.get_num_cols()) {
      tmp[0] += a * B(k, col + 0);
    }
    if (col + 1 < C.get_num_cols()) {
      tmp[1] += a * B(k, col + 1);
    }
    if (col + 2 < C.get_num_cols()) {
      tmp[2] += a * B(k, col + 2);
    }
    if (col + 3 < C.get_num_cols()) {
      tmp[3] += a * B(k, col + 3);
    }
  }

  if (col + 0 < C.get_num_cols()) {
    C(row, col + 0) = alpha * tmp[0] + beta * C(row, col + 0);
  }
  if (col + 1 < C.get_num_cols()) {
    C(row, col + 1) = alpha * tmp[1] + beta * C(row, col + 1);
  }
  if (col + 2 < C.get_num_cols()) {
    C(row, col + 2) = alpha * tmp[2] + beta * C(row, col + 2);
  }
  if (col + 3 < C.get_num_cols()) {
    C(row, col + 3) = alpha * tmp[3] + beta * C(row, col + 3);
  }
}


template <int start, int end>
struct static_for {
  template <typename UnaryOperator>
  static inline void loop(UnaryOperator op) {
    op(start);
    static_for<start+1, end>::loop(op);
  }
};


template <int end>
struct static_for<end, end> {
  template <typename UnaryOperator>
  static inline void loop(UnaryOperator) {}
};


template <int wsize, typename T, typename MatrixTypeA, typename MatrixTypeB,
          typename MatrixTypeC>
void _gemm_v7(int wg_id, int wg_size, int item_id, T alpha,
              const MatrixTypeA A, const MatrixTypeB B, T beta,
              MatrixTypeC C) {
  using value_type = typename MatrixTypeC::value_type;
  const int wg_per_col = (C.get_num_rows() + wg_size - 1) / wg_size;
  const int col = (wg_id / wg_per_col) * wsize;
  const int row = (wg_id % wg_per_col) * wg_size + item_id;
  if (col >= C.get_num_cols() || row >= C.get_num_rows()) {
    return;
  }
  value_type tmp[wsize] = {};
  value_type a;
  for (int k = 0; k < A.get_num_cols(); ++k) {
    a = A(row, k);
    static_for<0, wsize>::loop([&](int j) {
      if (col + j < C.get_num_cols()) {
        tmp[j] += a * B(k, col + j);
      }
    });
  }
  static_for<0, wsize>::loop([&](int j) {
    if (col + j < C.get_num_cols()) {
      C(row, col + j) = alpha * tmp[j] + beta * C(row, col + j);
    }
  });
}


template <typename T, typename MatrixTypeA, typename MatrixTypeB,
          typename MatrixTypeC>
void _gemm_v8(int wg_id, int wg_size, int item_id, T alpha,
              const MatrixTypeA A, const MatrixTypeB B, T beta,
              MatrixTypeC C) {
  using value_type = typename MatrixTypeC::value_type;
  const int wg_per_col = (C.get_num_rows() + wg_size - 1) / wg_size;
  const int col = (wg_id / wg_per_col) * 8;
  const int row = (wg_id % wg_per_col) * wg_size + item_id;
  if (col >= C.get_num_cols() || row >= C.get_num_rows()) {
    return;
  }

  value_type t0 = 0, t1 = 0, t2 = 0, t3 = 0;
  value_type t4 = 0, t5 = 0, t6 = 0, t7 = 0;
  value_type a;
  for (int k = 0; k < A.get_num_cols(); ++k) {
    a = A(row, k);
    if (col + 0 < C.get_num_cols()) {
      t0 += a * B(k, col + 0);
    }
    if (col + 1 < C.get_num_cols()) {
      t1 += a * B(k, col + 1);
    }
    if (col + 2 < C.get_num_cols()) {
      t2 += a * B(k, col + 2);
    }
    if (col + 3 < C.get_num_cols()) {
      t3 += a * B(k, col + 3);
    }
    if (col + 4 < C.get_num_cols()) {
      t4 += a * B(k, col + 4);
    }
    if (col + 5 < C.get_num_cols()) {
      t5 += a * B(k, col + 5);
    }
    if (col + 6 < C.get_num_cols()) {
      t6 += a * B(k, col + 6);
    }
    if (col + 7 < C.get_num_cols()) {
      t7 += a * B(k, col + 7);
    }
  }

  if (col + 0 < C.get_num_cols()) {
    C(row, col + 0) = alpha * t0 + beta * C(row, col + 0);
  }
  if (col + 1 < C.get_num_cols()) {
    C(row, col + 1) = alpha * t1 + beta * C(row, col + 1);
  }
  if (col + 2 < C.get_num_cols()) {
    C(row, col + 2) = alpha * t2 + beta * C(row, col + 2);
  }
  if (col + 3 < C.get_num_cols()) {
    C(row, col + 3) = alpha * t3 + beta * C(row, col + 3);
  }
  if (col + 4 < C.get_num_cols()) {
    C(row, col + 4) = alpha * t4 + beta * C(row, col + 4);
  }
  if (col + 5 < C.get_num_cols()) {
    C(row, col + 5) = alpha * t5 + beta * C(row, col + 5);
  }
  if (col + 6 < C.get_num_cols()) {
    C(row, col + 6) = alpha * t6 + beta * C(row, col + 6);
  }
  if (col + 7 < C.get_num_cols()) {
    C(row, col + 7) = alpha * t7 + beta * C(row, col + 7);
  }
}


template <int wsize, typename T, typename MatrixTypeA, typename MatrixTypeB,
          typename MatrixTypeC>
void _gemm_v9(int wg_id, int wg_size, int item_id, int tpr, T alpha,
              const MatrixTypeA A, const MatrixTypeB B, T beta,
              MatrixTypeC C) {
  using value_type = typename MatrixTypeC::value_type;
  const int tpc = wg_size / tpr;
  const int wg_per_col = (C.get_num_rows() + tpc - 1) / tpc;
  const int col = (wg_id / wg_per_col * tpr + item_id % tpr) * wsize;
  const int row = (wg_id % wg_per_col) * tpc + item_id / tpr;
  if (col >= C.get_num_cols() || row >= C.get_num_rows()) {
    return;
  }
  value_type tmp[wsize] = {};
  value_type a;
  for (int k = 0; k < A.get_num_cols(); ++k) {
    a = A(row, k);
    static_for<0, wsize>::loop([&](int j) {
      if (col + j < C.get_num_cols()) {
        tmp[j] += a * B(k, col + j);
      }
    });
  }
  static_for<0, wsize>::loop([&](int j) {
    if (col + j < C.get_num_cols()) {
      C(row, col + j) = alpha * tmp[j] + beta * C(row, col + j);
    }
  });
}


template <int wsize, typename T, typename MatrixTypeA, typename MatrixTypeB,
          typename MatrixTypeC>
void _gemm_v10(int wg_id, int wg_size, int item_id, int tpr, T alpha,
               const MatrixTypeA A, const MatrixTypeB B, T beta,
               MatrixTypeC C) {
  using value_type = typename MatrixTypeC::value_type;
  const int tpc = wg_size / tpr;
  const int wg_per_col = (C.get_num_rows() + tpc - 1) / tpc;
  const int col = (wg_id / wg_per_col * tpr + item_id / tpc) * wsize;
  const int row = (wg_id % wg_per_col) * tpc + item_id % tpc;
  if (col >= C.get_num_cols() || row >= C.get_num_rows()) {
    return;
  }
  value_type tmp[wsize] = {};
  value_type a;
  for (int k = 0; k < A.get_num_cols(); ++k) {
    a = A(row, k);
    static_for<0, wsize>::loop([&](int j) {
      if (col + j < C.get_num_cols()) {
        tmp[j] += a * B(k, col + j);
      }
    });
  }
  static_for<0, wsize>::loop([&](int j) {
    if (col + j < C.get_num_cols()) {
      C(row, col + j) = alpha * tmp[j] + beta * C(row, col + j);
    }
  });
}


template <int wsize, typename T, typename MatrixTypeA, typename MatrixTypeB,
          typename MatrixTypeC, typename MatrixTypeS>
void _gemm_v11(cl::sycl::nd_item<1> id, int wg_id, int wg_size, int item_id,
               int tpr, T alpha, const MatrixTypeA A, const MatrixTypeB B,
               T beta, MatrixTypeC C, MatrixTypeS scratch) {
  using value_type = typename MatrixTypeC::value_type;
  const int tpc = wg_size / tpr;
  const int wg_per_col = (C.get_num_rows() + tpc - 1) / tpc;
  const int wg_col = wg_id / wg_per_col * tpr * wsize;
  const int it_col = item_id / tpc * wsize;
  const int col = wg_col + it_col;
  const int row = (wg_id % wg_per_col) * tpc + item_id % tpc;
  if (col >= C.get_num_cols() || row >= C.get_num_rows()) {
    return;
  }
  value_type tmp[wsize] = {};
  value_type a;
  const int sh_size = scratch.get_num_rows();
  for (int sh = 0; sh < A.get_num_cols(); sh += sh_size) {
    // read B to shared
    for (int k = item_id % tpc; k < sh_size; k += tpc) {
      static_for<0, wsize>::loop([&](int j) {
        bool t = sh + k < B.get_num_rows() && col + j < B.get_num_cols();
        scratch(k, it_col + j) = t ? B(sh + k, col + j) : value_type(0);
      });
    }
    id.barrier(cl::sycl::access::fence_space::local_space);
    // use shared to do the computations
    for (int k = 0; k < sh_size; ++k) {
      a = A(row, sh + k);
      static_for<0, wsize>::loop([&](int j) {
        tmp[j] += a * scratch(k, it_col + j);
      });
    }
  }
  static_for<0, wsize>::loop([&](int j) {
    if (col + j < C.get_num_cols()) {
      C(row, col + j) = alpha * tmp[j] + beta * C(row, col + j);
    }
  });
}


const int cl_size = 64;

template <typename MatrixType>
void print_matrix(const MatrixType &M) {
  for (int i = 0; i < M.get_num_rows(); ++i) {
    for (int j = 0; j < M.get_num_cols(); ++j) {
      printf("% 1.1e ", M(i, j));
    }
    printf("\n");
  }
  printf("\n");
}

template <int wsize, typename T, typename MatrixTypeA, typename MatrixTypeB,
          typename MatrixTypeC, typename MatrixTypeS>
void _gemm_v12(cl::sycl::nd_item<1> id, int wg_id, int wg_size, int item_id,
               T alpha, const MatrixTypeA A, const MatrixTypeB B,
               T beta, MatrixTypeC C, MatrixTypeS scratch) {
  using value_type = typename MatrixTypeC::value_type;
  const int cl_elems = cl_size / sizeof(value_type);
  const int b_size = cl_elems * wsize;

  const int wg_per_col = (C.get_num_rows() - 1) / b_size + 1;
  const int wg_row = (wg_id % wg_per_col) * b_size;
  const int wg_col = (wg_id / wg_per_col) * b_size;

  const int b_row = item_id / b_size;
  const int b_row_id = item_id % b_size;

  const int item_row = b_row*cl_elems + b_row_id % cl_elems;
  const int item_col = b_row_id / cl_elems * cl_elems;

  const int row = wg_row + item_row;
  const int col = wg_col + item_col;

  const int c_row = item_id % cl_elems;
  const int c_col = item_id / cl_elems;
  const int c_inc = wg_size / cl_elems;

  value_type reg_res[wsize][cl_elems] = {};
  value_type reg_a[wsize];
  value_type reg_b;

  for (int bk = 0; bk < A.get_num_cols(); bk += cl_elems) {
    id.barrier(cl::sycl::access::fence_space::local_space);
    for (int i = c_col; i < b_size; i += c_inc) {
      const int in_range = bk + c_row < B.get_num_rows() &&
                           wg_col + i < B.get_num_cols();
      scratch(c_row, i) = in_range ? B(bk + c_row, wg_col + i) : 0;
    }
    id.barrier(cl::sycl::access::fence_space::local_space);
    for (int i = 0; i < cl_elems; ++i) {
      static_for<0, wsize>::loop([&](int j) {
        const int in_range = row + j*cl_elems < A.get_num_rows() &&
                             bk + i < A.get_num_cols();
        reg_a[j] = in_range ? A(row + j*cl_elems, bk+i) : value_type(0);
      });
      static_for<0, cl_elems>::loop([&](int j) {
        reg_b = scratch(i, item_col + j);
        static_for<0, wsize>::loop([&](int k) {
          reg_res[k][j] += reg_a[k] * reg_b;
        });
      });
    }
  }

  static_for<0, cl_elems>::loop([&](int i) {
    static_for<0, wsize>::loop([&](int j) {
      const int in_range = row + j*cl_elems < C.get_num_rows() &&
                           col + i < C.get_num_cols();
      if (in_range) {
        C(row + j*cl_elems, col + i) =
          alpha * reg_res[j][i] + beta * C(row + j*cl_elems, col+i);
      }
    });
  });
}


template <int rsize, int wsize, typename T, typename MatrixTypeA,
          typename MatrixTypeB, typename MatrixTypeC, typename MatrixTypeS>
void _gemm_v13(cl::sycl::nd_item<1> id, int wg_id, int wg_size, int item_id,
               T alpha, const MatrixTypeA A, const MatrixTypeB B,
               T beta, MatrixTypeC C, MatrixTypeS scratch) {
  using value_type = typename MatrixTypeC::value_type;
  const int cl_elems = cl_size / sizeof(value_type);
  const int b_size = cl_elems * wsize;

  const int wg_per_col = (C.get_num_rows() - 1) / b_size + 1;
  const int wg_row = (wg_id % wg_per_col) * b_size;
  const int wg_col = (wg_id / wg_per_col) * b_size;

  const int b_row = item_id / b_size;
  const int b_row_id = item_id % b_size;

  const int item_row = b_row*cl_elems + b_row_id % cl_elems;
  const int item_col = b_row_id / cl_elems * cl_elems;

  const int row = wg_row + item_row;
  const int col = wg_col + item_col;

  const int c_row = item_id % cl_elems;
  const int c_col = item_id / cl_elems;
  const int c_inc = wg_size / cl_elems;

  /*
  printf("Thread (%d, %d): wg_row = %d; wg_col = %d; "
         "item_row = %d; item_col = %d\n",
         wg_id, item_id, wg_row, wg_col, item_row, item_col);

  id.barrier(cl::sycl::access::fence_space::local_space);

  if (wg_id + item_id == 0) {
    print_matrix(A);
    print_matrix(B);
    print_matrix(C);
  }
  */

  value_type reg_res[wsize/rsize][cl_elems] = {};
  value_type reg_a[wsize/rsize];
  value_type reg_b;

  for (int bk = 0; bk < A.get_num_cols(); bk += cl_elems) {
    id.barrier(cl::sycl::access::fence_space::local_space);
    for (int i = c_col; i < b_size; i += c_inc) {
      const int in_range = bk + c_row < B.get_num_rows() &&
                           wg_col + i < B.get_num_cols();
      scratch(c_row, i) = in_range ? B(bk + c_row, wg_col + i) : 0;
    }
    id.barrier(cl::sycl::access::fence_space::local_space);
    /*
    if (item_id == 0) {
      print_matrix(scratch);
    }
    */
    for (int i = 0; i < cl_elems; ++i) {
      static_for<0, wsize/rsize>::loop([&](int j) {
        const int in_range = row + j*rsize*cl_elems < A.get_num_rows() &&
                             bk + i < A.get_num_cols();
        reg_a[j] = in_range ? A(row + j*rsize*cl_elems, bk+i) : value_type(0);
      });
      static_for<0, cl_elems>::loop([&](int j) {
        reg_b = scratch(i, item_col + j);
        static_for<0, wsize/rsize>::loop([&](int k) {
          reg_res[k][j] += reg_a[k] * reg_b;
        });
      });
      /*
      id.barrier(cl::sycl::access::fence_space::local_space);

      if (item_id < 2) {
        printf("Thread (%d, %d): %2.2f %2.2f\n",
               wg_id, item_id, reg_res[0][0], reg_res[0][1]);
      }

      id.barrier(cl::sycl::access::fence_space::local_space);
      */

    }
  }

  /*
  if (wg_id + item_id == 0) {
    print_matrix(A);
    print_matrix(B);
    print_matrix(C);
  }
  */

  static_for<0, cl_elems>::loop([&](int i) {
    static_for<0, wsize/rsize>::loop([&](int j) {
      const int in_range = row + j*rsize*cl_elems < C.get_num_rows() &&
                           col + i < C.get_num_cols();
      if (in_range) {
        C(row + j*rsize*cl_elems, col + i) =
          alpha * reg_res[j][i] + beta * C(row + j*rsize*cl_elems, col+i);
      }
    });
  });

  /*
  if (wg_id + item_id == 0) {
    print_matrix(C);
  }
  */
}


template <int rsize, int csize, int wsize, typename T>
void _gemm_v14(
    cl::sycl::nd_item<1> id, int wg_id, int item_id, int m, int n, int k,
    T alpha, cl::sycl::global_ptr<T> A, int lda, cl::sycl::global_ptr<T> B,
    int ldb, T beta, cl::sycl::global_ptr<T> C, int ldc,
    cl::sycl::local_ptr<T> scratch, int lds) {

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

  B = B + c_row + (wg_col + c_col)*ldb;
  n = n - wg_col;
  A = A + row;
  m = m - row;
  C = C + row + col*ldc;

  auto s1 = scratch + c_row + c_col*lds;
  auto s2 = scratch + item_col*lds;
  int ofs = cl_elems;

  while (k > 0) {
    static_for<0, b_size/c_inc>::loop([&](int i) {
      const int in_range = c_row < k && c_inc*i + c_col < n;
      s1[c_inc*i*lds] = in_range ? B[c_inc*i*ldb] : T(0);
    });
    id.barrier(cl::sycl::access::fence_space::local_space);
    for (int i = 0; i < cl_elems; ++i) {
      static_for<0, wsize/rsize>::loop([&](int j) {
        const int in_range = j*rsize*cl_elems < m && i < k;
        reg_a[j] = in_range ? A[j*rsize*cl_elems] : T(0);
      });
      static_for<0, cl_elems/csize>::loop([&](int j) {
        reg_b = s2[i + j*lds];
        static_for<0, wsize/rsize>::loop([&](int l) {
          reg_res[l][j] += reg_a[l] * reg_b;
        });
      });
      A = A + lda;
    }
    B = B + cl_elems;
    k -= cl_elems;
    s1 = s1 + ofs;
    s2 = s2 + ofs;
    ofs = -ofs;
  }

  static_for<0, cl_elems/csize>::loop([&](int i) {
    static_for<0, wsize/rsize>::loop([&](int j) {
      const int in_range = j*rsize*cl_elems < m && item_col + i < n;
      if (in_range) {
        C[j*rsize*cl_elems] = alpha*reg_res[j][i] + beta*C[j*rsize*cl_elems];
      }
    });
    C = C + ldc;
  });

}


template <int rsize, int csize, int wsize, typename T>
void _gemm_v15(
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

  B = B + c_row + (wg_col + c_col)*ldb;
  n = n - wg_col - c_col;
  A = A + wg_row + item_id%b_size + (item_id/b_size)*lda;
  m = m - wg_row - item_id%b_size;

  auto s1 = scratch + c_row + c_col*cl_elems;
  auto s2 = scratch + item_col*cl_elems;
  auto s3 = scratch + 2*b_size*cl_elems + item_id;
  auto s4 = scratch + 2*b_size*cl_elems + item_row;
  int ofs = 1;

  while (k >= cl_elems) {
    static_for<0, b_size/c_inc>::loop([&](int i) {
      const int in_range = c_inc*i < n;
      s1[c_inc*i*cl_elems] = in_range ? B[c_inc*i*ldb] : T(0);
    });
    static_for<0, b_size/c_inc>::loop([&](int i) {
      const int in_range = 0 < m;
      s3[i*wg_size] = in_range ? A[i*(wg_size/b_size)*lda] : T(0);
    });
    id.barrier(cl::sycl::access::fence_space::local_space);
    for (int i = 0; i < cl_elems; ++i) {
      static_for<0, wsize/rsize>::loop([&](int j) {
        reg_a[j] = s4[j*rsize*cl_elems + i*b_size];
      });
      static_for<0, cl_elems/csize>::loop([&](int j) {
        reg_b = s2[i + j*cl_elems];
        static_for<0, wsize/rsize>::loop([&](int l) {
          reg_res[l][j] += reg_a[l] * reg_b;
        });
      });
    }
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
    static_for<0, b_size/c_inc>::loop([&](int i) {
      const int in_range = c_row < k && c_inc*i < n;
      s1[c_inc*i*cl_elems] = in_range ? B[c_inc*i*ldb] : T(0);
    });
    static_for<0, b_size/c_inc>::loop([&](int i) {
      const int in_range = 0 < m && item_id/b_size + i*(wg_size/b_size) < k;
      s3[i*wg_size] = in_range ? A[i*(wg_size/b_size)*lda] : T(0);
    });
    id.barrier(cl::sycl::access::fence_space::local_space);
    for (int i = 0; i < cl_elems; ++i) {
      static_for<0, wsize/rsize>::loop([&](int j) {
        reg_a[j] = s4[j*rsize*cl_elems + i*b_size];
      });
      static_for<0, cl_elems/csize>::loop([&](int j) {
        reg_b = s2[i + j*cl_elems];
        static_for<0, wsize/rsize>::loop([&](int l) {
          reg_res[l][j] += reg_a[l] * reg_b;
        });
      });
    }
  }

  static_for<0, cl_elems/csize>::loop([&](int i) {
    static_for<0, wsize/rsize>::loop([&](int j) {
      const int in_range = j*rsize*cl_elems < mc && i < nc;
      if (in_range) {
        C[j*rsize*cl_elems] = alpha*reg_res[j][i] + beta*C[j*rsize*cl_elems];
      }
    });
    C = C + ldc;
  });

}


template <bool> inline bool do_check(bool cond) { return cond; }
template <> inline bool do_check<false>(bool) { return true; }


template <bool check_m_limit, bool check_n_limit, bool check_k_limit,
          int cl_elems, int b_size, int c_inc, int wg_size, typename T>
inline void extract_input_blocks(
  int c_row, int item_id, int m, int n, int k,
  cl::sycl::global_ptr<T> A, int lda, cl::sycl::global_ptr<T> B, int ldb,
  cl::sycl::local_ptr<T> s1, cl::sycl::local_ptr<T> s3) {
  static_for<0, b_size/c_inc>::loop([&](int i) {
    const bool in_range = do_check<check_k_limit>(c_row < k) &&
                         do_check<check_n_limit>(c_inc*i < n);
    s1[c_inc*i*cl_elems] = in_range ? B[c_inc*i*ldb] : T(0);
  });
  static_for<0, b_size/c_inc>::loop([&](int i) {
    const bool in_range =
        do_check<check_n_limit>(0 < m) &&
        do_check<check_k_limit>(item_id/b_size + i*(wg_size/b_size) < k);
    s3[i*wg_size] = in_range ? A[i*(wg_size/b_size)*lda] : T(0);
  });
}


template <int cl_elems, int wsize, int rsize, int csize, int b_size,
          typename T>
inline void compute_block_gemm(
    cl::sycl::local_ptr<T> s2, cl::sycl::local_ptr<T> s4,
    T reg_a[wsize/rsize], T &reg_b, T reg_res[wsize/rsize][cl_elems/csize]) {
  for (int i = 0; i < cl_elems; ++i) {
    static_for<0, wsize/rsize>::loop([&](int j) {
      reg_a[j] = s4[j*rsize*cl_elems + i*b_size];
    });
    static_for<0, cl_elems/csize>::loop([&](int j) {
      reg_b = s2[i + j*cl_elems];
      static_for<0, wsize/rsize>::loop([&](int l) {
        reg_res[l][j] += reg_a[l] * reg_b;
      });
    });
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

  static_for<0, cl_elems/csize>::loop([&](int i) {
    static_for<0, wsize/rsize>::loop([&](int j) {
      const bool in_range = do_check<check_m_limit>(j*rsize*cl_elems < mc) &&
                            do_check<check_n_limit>(i < nc);
      if (in_range) {
        C[j*rsize*cl_elems] = alpha*reg_res[j][i] + beta*C[j*rsize*cl_elems];
      }
    });
    C = C + ldc;
  });
}


template <int rsize, int csize, int wsize, typename T>
void _gemm_v16(
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

  B = B + c_row + (wg_col + c_col)*ldb;
  n = n - wg_col - c_col;
  A = A + wg_row + item_id%b_size + (item_id/b_size)*lda;
  m = m - wg_row - item_id%b_size;

  cl::sycl::local_ptr<T> s1 = scratch + c_row + c_col*cl_elems;
  cl::sycl::local_ptr<T> s2 = scratch + item_col*cl_elems;
  cl::sycl::local_ptr<T> s3 = scratch + 2*b_size*cl_elems + item_id;
  cl::sycl::local_ptr<T> s4 = scratch + 2*b_size*cl_elems + item_row;

  compute_panel_gemm
    <true, true, cl_elems, b_size, c_inc, wg_size, wsize, rsize, csize>
    (id, c_row, item_id, m, mc, n, nc, k, alpha, A, lda, B, ldb, beta, C, ldc,
     s1, s2, s3, s4, reg_a, reg_b, reg_res);
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

