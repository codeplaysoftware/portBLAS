/***************************************************************************
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


#include <string>
#include <type_traits>


namespace blas {


template <typename T> struct type_string {
  static const char *get_value() { return "unknown"; }
};


#define ENABLE_TYPE_STRING(_type) \
template <> struct type_string<_type> { \
  static const char *get_value() { return #_type; }\
};


ENABLE_TYPE_STRING(float)
ENABLE_TYPE_STRING(double)


#undef ENABLE_TYPE_STRING


template <int WgSize, bool TransA, bool TransB, typename T>
class ReferenceGemmFactory {
public:
  using value_type = T;

  static const int version = 2;
  static const int wg_size = WgSize;
  static const bool trans_a = TransA;
  static const bool trans_b = TransB;

  static inline std::string get_type_string() noexcept
  {
    return std::string("ReferenceGemmFactory<") + std::to_string(wg_size) +
           ", " + type_string<value_type>::get_value() + ">";
  }

  static inline cl::sycl::nd_range<1> get_nd_range(int m, int n) noexcept
  {
    const cl::sycl::range<1> nwg((m*n - 1) / wg_size + 1);
    const cl::sycl::range<1> wgs(wg_size);
    return cl::sycl::nd_range<1>(nwg*wgs, wgs);
  }

  template <typename GlobalPointerType>
  static inline void run(
      int item_id, int m, int n, int k, T alpha, GlobalPointerType A, int lda,
      GlobalPointerType B, int ldb, T beta, GlobalPointerType C, int ldc)
      noexcept
  {
    if (item_id >= m*n) {
      return;
    }

    const int row = item_id % m;
    const int col = item_id / m;

    A = A + row * (trans_a ? lda : 1);
    B = B + col * (trans_b ? 1 : ldb);
    C = C + row + col*ldc;

    value_type reg_res = {};

    while (k > 0) {
      reg_res += A[0] * B[0];
      --k;
      A = A + (trans_a ? 1 : lda);
      B = B + (trans_b ? ldb : 1);
    }

    C[0] = alpha * reg_res + beta * C[0];
  }
};


template <bool> inline bool do_check(bool cond) { return cond; }
template <> inline bool do_check<false>(bool) { return true; }


template <int ItemRows = 8, int ItemCols = 8, int WgRows = 16, int WgCols = 16,
          int TlRows = 1, int TlCols = 1>
struct Tile {
  static const int item_rows = ItemRows;
  static const int item_cols = ItemCols;
  static const int wg_rows = WgRows;
  static const int wg_cols = WgCols;
  static const int tl_rows = TlRows;
  static const int tl_cols = TlCols;

  static inline std::string get_type_string() noexcept
  {
    return std::string("Tile<") + std::to_string(item_rows) + ", " +
           std::to_string(item_cols) + ", " + std::to_string(wg_rows) + ", " +
           std::to_string(wg_cols) + ", " + std::to_string(tl_rows) + ", " +
           std::to_string(tl_cols) + ">";
  }
};


template <bool DoubleBuffer, bool NbcA, bool NbcB,
          int ClSize, typename TileType,
          bool TransA, bool TransB, typename T>
class GemmFactory {
public:
  using tile_type = TileType;
  using value_type = T;

  static const int version = 19;

  // enable easier access to tile dimensions
  static const int item_rows = tile_type::item_rows;
  static const int item_cols = tile_type::item_cols;
  static const int wg_rows = tile_type::wg_rows;
  static const int wg_cols = tile_type::wg_cols;
  static const int tl_rows = tile_type::tl_rows;
  static const int tl_cols = tile_type::tl_cols;

  static const bool double_buffer = DoubleBuffer;
  static const bool nbc_a = NbcA;
  static const bool nbc_b = NbcB;
  static const bool trans_a = TransA;
  static const bool trans_b = TransB;

  static const int cl_size = ClSize;
  static const int cl_elems = cl_size / sizeof(T);
  static const int wg_size = wg_rows * wg_cols;
  static const int block_rows = wg_rows * item_rows;
  static const int block_cols = wg_cols * item_cols;
  static const int big_tile_rows = tl_rows * block_rows;
  static const int big_tile_cols = tl_cols * block_cols;

  static_assert(wg_size % cl_elems == 0,
                "Work group size should be a multiple "
                "of elements in a cache line\n"
                " --- this is ensured iff:"
                " cl_size | sizeof(T) * wg_rows * wg_cols");

  static_assert(wg_size % block_rows == 0,
                "Work group size should be a multiple "
                "of the number of rows in a block\n"
                " --- this is ensured iff: item_rows | wg_cols");

  static_assert(wg_size % block_cols == 0,
                "Work group size should be a multiple "
                "of the number of columns in a block\n"
                " --- this is ensured iff: item_cols | wg_rows");

  static const int ldsa = block_rows + nbc_a;
  static const int ldsb = cl_elems + nbc_b;
  static const int scratch_size =
      (double_buffer+1) * (ldsa * cl_elems + ldsb * block_cols);

  static inline std::string get_type_string() noexcept
  {
    return std::string("GemmFactory<") + std::to_string(double_buffer) +
           ", " + std::to_string(nbc_a) + ", " + std::to_string(nbc_b) +
           ", " + std::to_string(cl_size) + ", " +
           tile_type::get_type_string() + ", " +
           type_string<value_type>::get_value() + ">";
  }

  static inline cl::sycl::nd_range<1> get_nd_range(int m, int n) noexcept
  {
    const cl::sycl::range<1> nwg(
        ((m - 1) / big_tile_rows + 1) * ((n - 1) / big_tile_cols + 1) *
         tl_rows * tl_cols);
    const cl::sycl::range<1> wgs(wg_size);
    return cl::sycl::nd_range<1>(nwg*wgs, wgs);
  }

  template <typename InputPointerType, typename OutputPointerType,
            typename ScratchPointerType>
  static inline void run(
      cl::sycl::nd_item<1> id, int wg_id, int item_id, int m, int n, int k,
      T alpha, InputPointerType A, int lda, InputPointerType B, int ldb,
      T beta, OutputPointerType C, int ldc, ScratchPointerType scratch)
      noexcept
  {
    const auto tile_size = tl_rows * tl_cols;
    const auto tile_id = wg_id / tile_size;
    const auto tile_local_id = wg_id % tile_size;
    const auto tiles_per_col = (m - 1) / big_tile_rows + 1;
    const auto tile_row = (tile_id % tiles_per_col) * tl_rows;
    const auto tile_col = (tile_id / tiles_per_col) * tl_cols;
    const auto wg_row = (tile_row + tile_local_id % tl_rows) * block_rows;
    const auto wg_col = (tile_col + tile_local_id / tl_rows) * block_rows;
    if (wg_row >= m || wg_col >= n) {
      return;
    }

    const auto item_row = item_id % wg_rows;
    const auto item_col = (item_id / wg_rows) * item_cols;

    const auto row = wg_row + item_row;
    const auto col = wg_col + item_col;

    T reg_res[item_rows][item_cols] = {};
    T reg_a[item_rows];
    T reg_b;

    C = C + row + col*ldc;
    const auto mc = m - row;
    const auto nc = n - col;

    const bool internal = m - wg_row >= block_rows && n - wg_col >= block_cols;

    B = B + (trans_b ?
        (item_id/block_cols) * ldb + (wg_col + item_id%block_cols) :
        item_id%cl_elems + (wg_col + item_id/cl_elems) * ldb);
    n = n - wg_col - (trans_b ? item_id%block_cols : item_id/cl_elems);
    A = A + (trans_a ?
        (wg_row + item_id/cl_elems) * lda + (item_id%cl_elems) :
        (wg_row + item_id%block_rows) + (item_id/block_rows) * lda);
    m = m - wg_row - (trans_a ? item_id/cl_elems : item_id%block_rows);

    ScratchPointerType s1 = scratch + (trans_b ?
        item_id/block_cols + (item_id%block_cols)*ldsb :
        item_id%cl_elems + (item_id/cl_elems)*ldsb);
    ScratchPointerType s2 = scratch + item_col*ldsb;
    const auto ofs = (double_buffer+1)*block_cols*ldsb;
    ScratchPointerType s3 = scratch + ofs + (trans_a ?
        item_id/cl_elems + (item_id%cl_elems)*ldsa :
        item_id%block_rows + (item_id/block_rows)*ldsa);
    ScratchPointerType s4 = scratch + ofs + item_row;

    if (internal) {
      compute_panel_gemm
        <double_buffer, false, false>
        (id, item_id, m, mc, n, nc, k, alpha, A, lda, B, ldb, beta, C,
         ldc, s1, s2, s3, s4, reg_a, reg_b, reg_res);
    } else {
      compute_panel_gemm
        <double_buffer, true, true>
        (id, item_id, m, mc, n, nc, k, alpha, A, lda, B, ldb, beta, C,
         ldc, s1, s2, s3, s4, reg_a, reg_b, reg_res);
    }
  }

private:
  template <bool double_buffer,
            bool check_m_limit, bool check_n_limit,
            typename InputPointerType, typename OutputPointerType,
            typename ScratchPointerType>
  static inline void compute_panel_gemm(
      cl::sycl::nd_item<1> id, int item_id,
      int m, int mc, int n, int nc, int k, T alpha,
      InputPointerType A, int lda, InputPointerType B, int ldb,
      T beta, OutputPointerType C, int ldc,
      ScratchPointerType s1, ScratchPointerType s2,
      ScratchPointerType s3, ScratchPointerType s4,
      T (&reg_a)[item_rows], T &reg_b, T (&reg_res)[item_rows][item_cols])
      noexcept
  {
    int ofs = 1;

    while (k >= cl_elems) {
      extract_input_blocks
        <check_m_limit, check_n_limit, false>
        (item_id, m, n, k, A, lda, B, ldb, s1, s3);
      id.barrier(cl::sycl::access::fence_space::local_space);
      compute_block_gemm
        (s2, s4, reg_a, reg_b, reg_res);
      A = A + cl_elems * (trans_a ? 1 : lda);
      B = B + cl_elems * (trans_b ? ldb : 1);
      k -= cl_elems;
      sync_smem<double_buffer, block_cols*ldsb, block_cols*ldsb,
                ldsa*cl_elems, ldsa*cl_elems>
          (id, ofs, s1, s2, s3, s4);
    }

    if (k > 0) {
      extract_input_blocks
        <check_m_limit, check_n_limit, true>
        (item_id, m, n, k, A, lda, B, ldb, s1, s3);
      id.barrier(cl::sycl::access::fence_space::local_space);
      compute_block_gemm
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

  /*!
   * @brief Extract a block of A, and a conformant block of B.
   *
   * @see GemmFactory::extract_block()
   */
  template <bool check_m_limit, bool check_n_limit, bool check_k_limit,
            typename InputPointerType, typename ScratchPointerType>
  static inline void extract_input_blocks(
      int item_id, int m, int n, int k,
      InputPointerType A, int lda, InputPointerType B, int ldb,
      ScratchPointerType sB, ScratchPointerType sA) noexcept
  {
    extract_block
      <check_m_limit, check_k_limit, trans_a, block_rows, cl_elems, ldsa>
      (item_id, A, lda, sA, [&](int ir, int cr) { return cr < m; },
       [&](int ic, int cc) { return cc < k - ic; });
    extract_block
      <check_k_limit, check_n_limit, trans_b, cl_elems, block_cols, ldsb>
      (item_id, B, ldb, sB, [&](int ir, int cr) { return cr < k - ir; },
       [&](int ic, int cc) { return cc < n; });
  }

  /*!
   * @brief Extract a block of a matrix from global to shared memory, and
   *        optionally transpose it on the fly.
   *
   * This is a collective operation on all items in a work group.
   *
   * @tparam check_row_limit  iff true, check the row out-of-bound condition
   * @tparam check_col_limit  iff true, check the column out-of-bound condition
   * @tparam trans  iff true, transpose the matrix
   * @tparam rows  number of rows in the block
   * @tparam cols  number of columns in the block
   * @tparam lds  leading dimension of the block in shared memory
   * @tparam InputPointerType  pointer type of the input matrix
   * @tparam ScratchPointerType  pointer type of the memory used to store the
   *                             extracted block
   * @tparam RowPredicate  row out-of-bound condition type
   * @tparam ColPredicate  column out-of-bound condition type
   *
   * @param item_id  id of the work item which called this method
   * @param ptr  pointer to the input matrix with proper item-dependent offset,
   *             see GemmFactory::run() for details
   * @param ld  the leading dimension of the input matrix
   * @param scratch  the pointer to memory where the output block is stored,
   *                 with proper item-dependent offset, see GemmFactory::run()
   *                 for details
   * @param in_row  a predicate which checks whether a row index is within
   *                matrix bounds
   * @param in_col  a predicate which checks whether a col index is within
   *                matrix bounds
   */
  template <bool check_row_limit, bool check_col_limit, bool trans,
            int rows, int cols, int lds,
            typename InputPointerType, typename ScratchPointerType,
            typename RowPredicate, typename ColPredicate>
  static inline typename std::enable_if<!trans>::type
  extract_block(
      int item_id, InputPointerType ptr, int ld, ScratchPointerType scratch,
      RowPredicate in_row, ColPredicate in_col)
  {
    const int bs = rows * cols;
    #pragma unroll
    for (int i = 0; i < (bs - 1) / wg_size + 1; ++i) {
      if (!do_check<bs % wg_size>(item_id + i*wg_size < bs)) continue;
      const int col_ofs = i * (wg_size/rows);
      const bool in_range =
          do_check<check_row_limit>(in_row(item_id%rows, 0)) &&
          do_check<check_col_limit>(in_col(item_id/rows, col_ofs));
      scratch[col_ofs * lds] = in_range ? ptr[col_ofs * ld] : T(0);
    }
  }

  template <bool check_row_limit, bool check_col_limit, bool trans,
            int rows, int cols, int lds,
            typename InputPointerType, typename ScratchPointerType,
            typename RowPredicate, typename ColPredicate>
  static inline typename std::enable_if<trans>::type
  extract_block(
      int item_id, InputPointerType ptr, int ld, ScratchPointerType scratch,
      RowPredicate in_row, ColPredicate in_col)
  {
    const int bs = rows * cols;
    #pragma unroll
    for (int i = 0; i < (bs - 1) / wg_size + 1; ++i) {
      if (!do_check<bs % wg_size>(item_id + i*wg_size < bs)) continue;
      const int row_ofs = i * (wg_size/cols);
      const bool in_range =
          do_check<check_row_limit>(in_row(item_id/cols, row_ofs)) &&
          do_check<check_col_limit>(in_col(item_id%cols, 0));
      scratch[row_ofs] = in_range ? ptr[row_ofs * ld] : T(0);
    }
  }

  /*!
   * @brief Compute a small matrix-matrix product `reg_res += A*B`.
   *
   * @tparam InputPointerType  pointer type for A and B
   *
   * @param B  pointer to matrix A with proper item-dependent offset,
   *           see GemmFactory::run() for details
   * @param A  pointer to matrix B with proper item-dependent offset,
   *           see GemmFactory::run() for details
   * @param reg_a  temporary register array used to prefetch columns of A
   * @param reg_b  temporary register used to prefetch elements of B
   * @param reg_res  2D register array used to store the result C
   */
  template <typename InputPointerType>
  static inline void compute_block_gemm(
      InputPointerType B, InputPointerType A,
      T (&reg_a)[item_rows], T &reg_b, T (&reg_res)[item_rows][item_cols])
      noexcept
  {
    for (int i = 0; i < cl_elems; ++i) {
      #pragma unroll
      for (int j = 0; j < item_rows; ++j) {
        reg_a[j] = A[j*wg_rows];
      }
      #pragma unroll
      for (int j = 0; j < item_cols; ++j) {
        reg_b = B[j*ldsb];
        #pragma unroll
        for (int l = 0; l < item_rows; ++l) {
          reg_res[l][j] += reg_a[l] * reg_b;
        }
      }
      A = A + ldsa;
      B = B + 1;
    }
  }

  /*!
   * @brief Synchronize multiple shared memory blocks using a barrier or
   *        double buffering.
   *
   * @tparam db  if true, use double buffering, otherwise use barrier
   * @tparam o  size of the first memory block
   * @tparam os  sizes of other memory blocks
   * @tparam P  type of first memory block
   * @tparam Ps  types of other memory blocks
   *
   * @param id  nd_item used to call barrier sync
   * @param ofs_sign  if 1, use next block for double buffering, if -1 use
   *                  previous block
   * @param s  pointer to first memory block
   * @param ss  pointers to other memory blocks
   */
  template <bool db, int o, int... os, typename P, typename... Ps>
  static inline typename std::enable_if<db>::type
  sync_smem(cl::sycl::nd_item<1> id, int &ofs_sign, P &s, Ps &...ss) noexcept
  {
    s = s + ofs_sign*o;
    sync_smem<db, os...>(id, ofs_sign, ss...);
  }

  template <bool db>
  static inline typename std::enable_if<db>::type
  sync_smem(cl::sycl::nd_item<1>, int &ofs_sign) noexcept
  { ofs_sign = -ofs_sign; }


  template <bool db, int..., typename... Ps>
  static inline typename std::enable_if<!db>::type
  sync_smem(cl::sycl::nd_item<1> id, int&, Ps&...) noexcept
  {
    id.barrier(cl::sycl::access::fence_space::local_space);
  }

};


}  // namespace blas


#endif  // BLAS3_TREES_GEMM_HPP

