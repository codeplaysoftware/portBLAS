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
class GemmFactoryV2 {
public:
  using value_type = T;

  static const int version = 2;
  static const int wg_size = WgSize;
  static const bool trans_a = TransA;
  static const bool trans_b = TransB;

  static inline std::string get_type_string() noexcept
  {
    return std::string("GemmFactoryV2<") + std::to_string(wg_size) + ", " +
           type_string<value_type>::get_value() + ">";
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


template <int ItemRows, int ItemCols, int WgRows, int WgCols>
struct Tile {
  static const int item_rows = ItemRows;
  static const int item_cols = ItemCols;
  static const int wg_rows = WgRows;
  static const int wg_cols = WgCols;

  static inline std::string get_type_string() noexcept
  {
    return std::string("Tile<") + std::to_string(item_rows) + ", " +
           std::to_string(item_cols) + ", " + std::to_string(wg_rows) + ", " +
           std::to_string(wg_cols) + ">";
  }
};


template <bool DoubleBuffer, int ClSize, typename TileType,
          bool TransA, bool TransB, typename T>
class GemmFactoryV19 {
public:
  using tile_type = TileType;
  using value_type = T;

  static const int version = 19;

  // enable easier access to tile dimensions
  static const int item_rows = tile_type::item_rows;
  static const int item_cols = tile_type::item_cols;
  static const int wg_rows = tile_type::wg_rows;
  static const int wg_cols = tile_type::wg_cols;

  static const bool double_buffer = DoubleBuffer;
  static const bool trans_a = TransA;
  static const bool trans_b = TransB;

  static const int cl_size = ClSize;
  static const int cl_elems = cl_size / sizeof(T);
  static const int wg_size = wg_rows * wg_cols;
  static const int block_rows = wg_rows * item_rows;
  static const int block_cols = wg_cols * item_cols;

  static_assert(wg_size % cl_elems == 0,
                "Work group size should be a multiple "
                "of elements in a cache line\n"
                " --- this is ensured iff:"
                " cl_size | sizeof(T) * wg_rows * wg_cols");

  static_assert(wg_size % block_rows == 0,
                "Work group size should be a multiple "
                "of the number of rows in a block\n"
                " --- this is ensured iff: item_rows | wg_cols");

  static const int scratch_size =
      (double_buffer+1) * cl_elems * (block_rows + block_cols);

  static inline std::string get_type_string() noexcept
  {
    return std::string("GemmFactoryV19<") + std::to_string(double_buffer) +
           ", " + std::to_string(cl_size) + ", " +
           tile_type::get_type_string() + ", " +
           type_string<value_type>::get_value() + ">";
  }

  static inline cl::sycl::nd_range<1> get_nd_range(int m, int n) noexcept
  {
    const cl::sycl::range<1> nwg(
        ((m - 1) / block_rows + 1) * ((n - 1) / block_cols + 1));
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
    const auto wg_per_col = (m - 1) / block_rows + 1;
    const auto wg_row = (wg_id % wg_per_col) * block_rows;
    const auto wg_col = (wg_id / wg_per_col) * block_cols;

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

    B = B + item_id%cl_elems * (trans_b ? ldb : 1)
          + (wg_col + item_id/cl_elems) * (trans_b ? 1 : ldb);
    n = n - wg_col - item_id/cl_elems;
    A = A + (wg_row + item_id%block_rows) * (trans_a ? lda : 1)
          + (item_id/block_rows) * (trans_a ? 1 : lda);
    m = m - wg_row - item_id%block_rows;

    ScratchPointerType s1 =
        scratch + item_id%cl_elems + (item_id/cl_elems)*cl_elems;
    ScratchPointerType s2 = scratch + item_col*cl_elems;
    const auto ofs = (double_buffer+1)*block_cols*cl_elems;
    ScratchPointerType s3 =
        scratch + ofs + item_id%block_rows + (item_id/block_rows)*block_rows;
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
      sync_smem<double_buffer, block_cols*cl_elems, block_cols*cl_elems,
                block_rows*cl_elems, block_rows*cl_elems>
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

  template <bool check_m_limit, bool check_n_limit, bool check_k_limit,
            typename InputPointerType, typename ScratchPointerType>
  static inline void extract_input_blocks(
    int item_id, int m, int n, int k,
    InputPointerType A, int lda, InputPointerType B, int ldb,
    ScratchPointerType s1, ScratchPointerType s3) noexcept
  {
    const int bsb = block_cols * cl_elems;
    #pragma unroll
    for (int i = 0; i < (bsb-1) / wg_size + 1; ++i) {
      if (!do_check<bsb % wg_size>(item_id + i*wg_size < bsb))
        continue;
      const bool in_range = do_check<check_k_limit>(item_id % cl_elems < k) &&
                            do_check<check_n_limit>(wg_size/cl_elems*i < n);
      s1[i*wg_size] = in_range ? B[i*(wg_size/cl_elems) * (trans_b ? 1:ldb)]
                               : T(0);
    }
    const int bsa = block_rows * cl_elems;
    #pragma unroll
    for (int i = 0; i < (bsa-1) / wg_size + 1; ++i) {
      if (!do_check<bsa % wg_size>(item_id + i*wg_size < bsa))
        continue;
      const bool in_range =
          do_check<check_n_limit>(0 < m) &&
          do_check<check_k_limit>
            (item_id/block_rows+i*(wg_size/block_rows) < k);
      s3[i*wg_size] = in_range ? A[i*(wg_size/block_rows) * (trans_a ? 1:lda)]
                               : T(0);
    }
  }

  template <typename ScratchPointerType>
  static inline void compute_block_gemm(
      ScratchPointerType s2, ScratchPointerType s4,
      T (&reg_a)[item_rows], T &reg_b, T (&reg_res)[item_rows][item_cols])
      noexcept
  {
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

  template <bool db>
  static inline typename std::enable_if<db>::type
  sync_smem(cl::sycl::nd_item<1>, int &ofs_sign) noexcept
  { ofs_sign = -ofs_sign; }


  template <bool db, int o, int... os, typename P, typename... Ps>
  static inline typename std::enable_if<db>::type
  sync_smem(cl::sycl::nd_item<1> id, int &ofs_sign, P &s, Ps &...ss) noexcept
  {
    s = s + ofs_sign*o;
    sync_smem<db, os...>(id, ofs_sign, ss...);
  }

  template <bool db, int..., typename... Ps>
  static inline typename std::enable_if<!db>::type
  sync_smem(cl::sycl::nd_item<1> id, int&, Ps&...) noexcept
  {
    id.barrier(cl::sycl::access::fence_space::local_space);
  }

};


}  // namespace blas


#endif  // BLAS3_TREES_GEMM_HPP

