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
 *  @filename blas2_trees.h
 *
 **************************************************************************/

#ifndef SYCL_BLAS_BLAS2_TREES_H
#define SYCL_BLAS_BLAS2_TREES_H
namespace blas {

/*!
 * @brief Determines the storage format of a matrix.
 */
enum class matrix_format_t : uint32_t { full = 1, banded = 2, packed = 0 };

/*!
 * @brief Determines the memory type of the GEMV kernel.
 * It can either use local memory or not
 */
enum class gemv_memory_t : int { local = 0, no_local = 1 };

/*!
 * @brief Gemv is a templated class whose instantiations provide different
 * implementations of the the GEMV kernel function.
 *
 * The class is constructed using the make_gemv function below.
 *
 * @tparam local_range  specifies the number of threads per work group used by
 *                      the kernel
 * @tparam is_transposed  specifies whether the input matrix should be
 * transposed
 * @tparam cache_line_size  specifies the size in bytes of the cache line. This
 *                          value will determine the dimensions of tiles loaded
 *                          into local memory in the transposed local memory
 *                          version of the kernel
 * @tparam memory_type  specifies whether the kernel should use local shared
 *                      memory or not
 * @tparam work_per_thread  (not implemented) would specify the multiplier of
 *                          work done per each work item
 * @param lhs_        the output buffer of the kernel
 * @param matrix_a_   the input matrix a
 * @param vector_x_   the input vector x
 * @param wgs_per_nc  the number of work groups per non-contracting dimension
 * @param wgs_per_c   the number of work groups per contracting dimension
 *
 */
template <typename lhs_t, typename matrix_t, typename vector_t,
          uint32_t local_range, bool is_transposed, int cache_line_size,
          int work_per_thread>
struct Gemv {
  using value_t = typename std::remove_cv<typename vector_t::value_t>::type;
  using index_t = typename vector_t::index_t;
  lhs_t lhs_;
  matrix_t matrix_a_;
  vector_t vector_x_;
  index_t wgs_per_nc_;
  index_t wgs_per_c_;

  Gemv(lhs_t &_l, matrix_t &_matrix, vector_t &_vector, index_t &_wgs_per_nc,
       index_t &_wgs_per_c);
  bool valid_thread(cl::sycl::nd_item<1> ndItem) const;
  value_t eval(cl::sycl::nd_item<1> ndItem);
  template <typename local_memory_t>
  value_t eval(local_memory_t local_mem, cl::sycl::nd_item<1> ndItem);
  void bind(cl::sycl::handler &h);
  void adjust_access_displacement();

 private:
  template <typename ScratchPointerType>
  void extract_input_block(ScratchPointerType scratch, const index_t &local_id,
                           const index_t &group_id, const index_t &lda,
                           index_t mat_tile_id);
};

/*!
 * @brief Contructs an instance of the Gemv class used to launch Gemv operation
 * kernels
 */
template <uint32_t local_range, bool is_transposed, int cache_line_size,
          int work_per_thread, typename lhs_t, typename matrix_t,
          typename vector_t>
Gemv<lhs_t, matrix_t, vector_t, local_range, is_transposed, cache_line_size,
     work_per_thread>
make_gemv(lhs_t &lhs_, matrix_t &matrix_, vector_t &vector_,
          typename vector_t::index_t wgs_per_nc_,
          typename vector_t::index_t wgs_per_c_) {
  return Gemv<lhs_t, matrix_t, vector_t, local_range, is_transposed,
              cache_line_size, work_per_thread>(lhs_, matrix_, vector_,
                                                wgs_per_nc_, wgs_per_c_);
}

template <typename rhs_t>
struct SumMatrixColumns {
  using value_t = typename rhs_t::value_t;
  using index_t = typename rhs_t::index_t;

  rhs_t rhs_;

  SumMatrixColumns(rhs_t &_r);

  index_t get_size() const;

  bool valid_thread(cl::sycl::nd_item<1> ndItem) const;

  value_t eval(index_t i);

  value_t eval(cl::sycl::nd_item<1> ndItem);
  void bind(cl::sycl::handler &h);
  void adjust_access_displacement();
};

/**
 * @fn make_sum_matrix_columns
 * @brief Constructs a SumMatrixColumns structure.
 */
template <typename rhs_t>
SumMatrixColumns<rhs_t> make_sum_matrix_columns(rhs_t &rhs_) {
  return SumMatrixColumns<rhs_t>(rhs_);
}

/**
 * @struct GemvCol
 * @brief Tree node representing a Gemv, with parallel expressed across columns
 * *
 */
template <bool Lower, bool Diag, bool Upper, bool Unit, typename lhs_t,
          typename matrix_t, typename vector_t>
struct GemvCol {
  using value_t = typename vector_t::value_t;
  using index_t = typename vector_t::index_t;
  lhs_t lhs_;
  matrix_t matrix_;
  vector_t vector_;
  index_t nWG_row_;
  index_t nWG_col_;
  index_t local_memory_size_;

  GemvCol(lhs_t &_l, matrix_t &_matrix, vector_t &_vector, index_t &_nWG_row,
          index_t &_nWG_col, index_t &_shrMemSize);
  index_t get_size() const;
  bool valid_thread(cl::sycl::nd_item<1> ndItem) const;
  value_t eval(index_t i);
  value_t eval(cl::sycl::nd_item<1> ndItem);
  template <typename sharedT>
  value_t eval(sharedT shrMem, cl::sycl::nd_item<1> ndItem);
  void bind(cl::sycl::handler &h);
  void adjust_access_displacement();
};

// template <typename lhs_t, typename matrix_t,typename vector_t>
template <bool Lower = true, bool Diag = true, bool Upper = true,
          bool Unit = false, typename lhs_t, typename matrix_t,
          typename vector_t>
GemvCol<Lower, Diag, Upper, Unit, lhs_t, matrix_t, vector_t> make_gemv_col(
    lhs_t &lhs_, matrix_t &matrix_, vector_t &vector_,
    typename vector_t::index_t nWG_row_, typename vector_t::index_t nWG_col_,
    typename vector_t::index_t local_memory_size_) {
  return GemvCol<Lower, Diag, Upper, Unit, lhs_t, matrix_t, vector_t>(
      lhs_, matrix_, vector_, nWG_row_, nWG_col_, local_memory_size_);
}

/**
 * @struct GemvRow
 * @brief Tree node representing a row-based/row-parallel generalised matrix_
 * vector_ multiplication.
 */
template <int interLoop, bool Lower, bool Diag, bool Upper, bool Unit,
          typename lhs_t, typename matrix_t, typename vector_t>
struct GemvRow {
  using value_t = typename vector_t::value_t;
  using index_t = typename vector_t::index_t;

  lhs_t lhs_;
  matrix_t matrix_;
  vector_t vector_;
  index_t nWG_row_;
  index_t nWG_col_;
  index_t local_memory_size_;

  GemvRow(lhs_t &_l, matrix_t &_matrix, vector_t &_vector, index_t &_nWG_row,
          index_t &_nWG_col, index_t &_shrMemSize);
  index_t get_size() const;
  bool valid_thread(cl::sycl::nd_item<1> ndItem) const;
  value_t eval(index_t i);
  value_t eval(cl::sycl::nd_item<1> ndItem);
  template <typename sharedT>
  value_t eval(sharedT shrMem, cl::sycl::nd_item<1> ndItem);
  void bind(cl::sycl::handler &h);
  void adjust_access_displacement();
};
/*!
 @brief Generator/factory for row-based GEMV trees.

 make_gemv_row(
    lhs_t &lhs_,
     matrix_t &matrix_,
    vector_t &vector_,
    typename vector_t::index_t nWG_row_,
    typename vector_t::index_t nWG_col_,
    typename vector_t::index_t local_memory_size_
 )
 */
template <int interLoop = 1, bool Lower = true, bool Diag = true,
          bool Upper = true, bool Unit = false, typename lhs_t,
          typename matrix_t, typename vector_t>
GemvRow<interLoop, Lower, Diag, Upper, Unit, lhs_t, matrix_t, vector_t>
make_gemv_row(lhs_t &lhs_, matrix_t &matrix_, vector_t &vector_,
              typename vector_t::index_t nWG_row_,
              typename vector_t::index_t nWG_col_,
              typename vector_t::index_t local_memory_size_) {
  return GemvRow<interLoop, Lower, Diag, Upper, Unit, lhs_t, matrix_t,
                 vector_t>(lhs_, matrix_, vector_, nWG_row_, nWG_col_,
                           local_memory_size_);
}

/**
 * @struct Gbmv
 * @brief Tree node representing a band matrix_ vector_ multiplication.
 */
template <typename lhs_t, typename matrix_t, typename vector_t,
          uint32_t local_range, bool is_transposed>
struct Gbmv {
  using value_t = typename vector_t::value_t;
  using index_t = typename vector_t::index_t;

  lhs_t lhs_;
  matrix_t matrix_;
  index_t kl_;
  index_t ku_;
  vector_t vector_;
  value_t alpha_, beta_;

  Gbmv(lhs_t &_l, matrix_t &_matrix, index_t &_kl, index_t &_ku,
       vector_t &_vector, value_t _alpha, value_t _beta);
  index_t get_size() const;
  bool valid_thread(cl::sycl::nd_item<1> ndItem) const;
  value_t eval(index_t i);
  value_t eval(cl::sycl::nd_item<1> ndItem);
  template <typename sharedT>
  value_t eval(sharedT shrMem, cl::sycl::nd_item<1> ndItem);
  void bind(cl::sycl::handler &h);
  void adjust_access_displacement();
};
/*!
 @brief Generator/factory for GBMV trees.
 */
template <uint32_t local_range, bool is_transposed, typename lhs_t,
          typename matrix_t, typename vector_t>
Gbmv<lhs_t, matrix_t, vector_t, local_range, is_transposed> make_gbmv(
    typename vector_t::index_t kl_, typename vector_t::index_t ku_,
    typename vector_t::value_t alpha_, matrix_t &matrix_, vector_t &vector_,
    typename vector_t::value_t beta_, lhs_t &lhs_) {
  return Gbmv<lhs_t, matrix_t, vector_t, local_range, is_transposed>(
      lhs_, matrix_, kl_, ku_, vector_, alpha_, beta_);
}

/**
 * @struct Sbmv
 * @brief Tree node representing a symmetric band matrix_ vector_
 * multiplication.
 */
template <typename lhs_t, typename matrix_t, typename vector_t,
          uint32_t local_range, bool uplo>
struct Sbmv {
  using value_t = typename vector_t::value_t;
  using index_t = typename vector_t::index_t;

  lhs_t lhs_;
  matrix_t matrix_;
  index_t k_;
  vector_t vector_;
  value_t alpha_, beta_;

  Sbmv(lhs_t &_l, matrix_t &_matrix, index_t &_k, vector_t &_vector,
       value_t _alpha, value_t _beta);
  index_t get_size() const;
  bool valid_thread(cl::sycl::nd_item<1> ndItem) const;
  value_t eval(index_t i);
  value_t eval(cl::sycl::nd_item<1> ndItem);
  template <typename sharedT>
  value_t eval(sharedT shrMem, cl::sycl::nd_item<1> ndItem);
  void bind(cl::sycl::handler &h);
  void adjust_access_displacement();
};
/*!
 @brief Generator/factory for SBMV trees.
 */
template <uint32_t local_range, bool uplo, typename lhs_t, typename matrix_t,
          typename vector_t>
Sbmv<lhs_t, matrix_t, vector_t, local_range, uplo> make_sbmv(
    typename vector_t::index_t k_, typename vector_t::value_t alpha_,
    matrix_t &matrix_, vector_t &vector_, typename vector_t::value_t beta_,
    lhs_t &lhs_) {
  return Sbmv<lhs_t, matrix_t, vector_t, local_range, uplo>(
      lhs_, matrix_, k_, vector_, alpha_, beta_);
}

/**
 * @struct Xpmv
 * @brief Tree node representing a symmetric/triangular packed matrix_ vector_
 * multiplication.
 *
 * If the matrix is triangular, it computes
 *                          lhs_ = matrix_ * vector_
 *
 * If the matrix is symmetric, it computes
 *                 lhs_ = alpha_ * matrix_ * vector_ + beta_ * lhs_
 *
 * The class is constructed using the make_xpmv function below.
 *
 * @tparam local_range_x  specifies the work group x-dimension
 * @tparam local_range_y  specifies the work group y-dimension
 * @tparam is_upper       specifies whether the triangular input matrix is upper
 * @tparam is_transposed  specifies whether the input matrix should be
 * transposed
 * @tparam is_unit  specifies whether considering the input matrix
 * main-diagonal filled with ones
 * @param lhs_      the output vector
 * @param matrix_   the input matrix
 * @param vector_   the input vector
 * @param alpha_    factor used only if is_symmetric == true
 * @param beta_     factor used only if is_symmetric == true
 */
template <typename lhs_t, typename matrix_t, typename vector_t,
          uint32_t local_range_x, uint32_t local_range_y, bool is_symmetric,
          bool is_upper, bool is_transposed, bool is_unit>
struct Xpmv {
  using value_t = typename vector_t::value_t;
  using index_t = typename vector_t::index_t;

  lhs_t lhs_;
  matrix_t matrix_;
  vector_t vector_;
  value_t alpha_, beta_;

  Xpmv(lhs_t &_l, matrix_t &_matrix, vector_t &_vector, value_t _alpha,
       value_t _beta);
  index_t get_size() const;
  bool valid_thread(cl::sycl::nd_item<1> ndItem) const;
  template <typename sharedT>
  value_t eval(sharedT shrMem, cl::sycl::nd_item<1> ndItem);
  void bind(cl::sycl::handler &h);
  void adjust_access_displacement();
};
/*!
 @brief Generator/factory for XPMV trees.
 */
template <uint32_t local_range_x, uint32_t local_range_y, bool is_symmetric,
          bool is_upper, bool is_transposed, bool is_unit, typename lhs_t,
          typename matrix_t, typename vector_t>
Xpmv<lhs_t, matrix_t, vector_t, local_range_x, local_range_y, is_symmetric,
     is_upper, is_transposed, is_unit>
make_xpmv(typename vector_t::value_t alpha_, matrix_t &matrix_,
          vector_t &vector_, typename vector_t::value_t beta_, lhs_t &lhs_) {
  return Xpmv<lhs_t, matrix_t, vector_t, local_range_x, local_range_y,
              is_symmetric, is_upper, is_transposed, is_unit>(
      lhs_, matrix_, vector_, alpha_, beta_);
}

/**
 * @struct Tbmv
 * @brief Tree node representing a triangular band matrix_ vector_
 * multiplication.
 */
template <typename lhs_t, typename matrix_t, typename vector_t,
          uint32_t local_range, bool is_upper, bool is_transposed, bool is_unit>
struct Tbmv {
  using value_t = typename vector_t::value_t;
  using index_t = typename vector_t::index_t;

  lhs_t lhs_;
  matrix_t matrix_;
  index_t k_;
  vector_t vector_;

  Tbmv(lhs_t &_l, matrix_t &_matrix, index_t &_k, vector_t &_vector);
  index_t get_size() const;
  bool valid_thread(cl::sycl::nd_item<1> ndItem) const;
  value_t eval(index_t i);
  value_t eval(cl::sycl::nd_item<1> ndItem);
  template <typename sharedT>
  value_t eval(sharedT shrMem, cl::sycl::nd_item<1> ndItem);
  void bind(cl::sycl::handler &h);
  void adjust_access_displacement();
};
/*!
 @brief Generator/factory for TBMV trees.
 */
template <uint32_t local_range, bool is_upper, bool is_transposed, bool is_unit,
          typename lhs_t, typename matrix_t, typename vector_t>
Tbmv<lhs_t, matrix_t, vector_t, local_range, is_upper, is_transposed, is_unit>
make_tbmv(lhs_t &lhs_, matrix_t &matrix_, typename vector_t::index_t k_,
          vector_t &vector_) {
  return Tbmv<lhs_t, matrix_t, vector_t, local_range, is_upper, is_transposed,
              is_unit>(lhs_, matrix_, k_, vector_);
}

/**
 * @struct Txsv
 * @brief Tree node representing a linear system solver for triangular matrices,
 * i.e., it computes lhs_  such that lhs_ = matrix_ * lhs_
 *
 * The class is constructed using the make_txsv function below.
 *
 * @tparam matrix_format  specifies how the matrix is stored, full, packed, or
 * banded
 * @tparam subgroup_size  specifies the subgroup size of the device
 * @tparam subgroups      specifies the number of subgroups within a work group
 * @tparam is_upper       specifies whether the triangular input matrix is upper
 * @tparam is_transposed  specifies whether the input matrix should be
 * transposed
 * @tparam is_unit  specifies whether considering the input matrix
 * @param lhs_      the input/output vector
 * @param matrix_   the input matrix
 * @param k_        the number of extra-diagonal in case of banded matrices
 * @param sync_     vector of size 2 utilized for work group synchronization
 * purposes
 *
 */
template <typename vector_t, typename matrix_t, typename sync_t,
          matrix_format_t matrix_format, uint32_t subgroup_size,
          uint32_t subgroups, bool is_upper, bool is_transposed, bool is_unit>
struct Txsv {
  using value_t = typename vector_t::value_t;
  using index_t = typename vector_t::index_t;

  vector_t lhs_;
  matrix_t matrix_;
  index_t k_;
  sync_t sync_;

  Txsv(vector_t &_l, matrix_t &_matrix, index_t &_k, sync_t &_sync);
  template <typename local_memory_t>
  value_t eval(local_memory_t local_mem, cl::sycl::nd_item<1> ndItem);
  void bind(cl::sycl::handler &h);
  void adjust_access_displacement();
  value_t read_matrix(const index_t &row, const index_t &col) const;
};
/*!
 @brief Generator/factory for TXSV trees.
 */
template <matrix_format_t matrix_format, uint32_t subgroup_size,
          uint32_t subgroups, bool is_upper, bool is_transposed, bool is_unit,
          typename vector_t, typename matrix_t, typename sync_t>
Txsv<vector_t, matrix_t, sync_t, matrix_format, subgroup_size, subgroups,
     is_upper, is_transposed, is_unit>
make_txsv(vector_t &lhs_, matrix_t &matrix_, typename vector_t::index_t k_,
          sync_t &sync_) {
  return Txsv<vector_t, matrix_t, sync_t, matrix_format, subgroup_size,
              subgroups, is_upper, is_transposed, is_unit>(lhs_, matrix_, k_,
                                                           sync_);
}

/**** GER BY ROWS M ROWS x N BLOCK USING PROPERLY THE SHARED MEMORY ****/
// template <typename lhs_t,typename rhs_1_t,typename rhs_2_t>
template <bool Single, bool Lower, bool Diag, bool Upper, typename lhs_t,
          typename rhs_1_t, typename rhs_2_t>
struct GerRow {
  using value_t = typename rhs_2_t::value_t;
  using index_t = typename rhs_2_t::index_t;
  lhs_t lhs_;
  rhs_1_t rhs_1_;
  rhs_2_t rhs_2_;
  index_t nWG_row_;
  index_t nWG_col_;
  index_t local_memory_size_;
  value_t scalar_;
  GerRow(lhs_t &_l, value_t _scl, rhs_1_t &_r1, rhs_2_t &_r2, index_t &_nWG_row,
         index_t &_nWG_col, index_t &_shrMemSize);
  index_t get_size() const;
  bool valid_thread(cl::sycl::nd_item<1> ndItem) const;
  value_t eval(index_t i);
  value_t eval(cl::sycl::nd_item<1> ndItem);
  template <typename sharedT>
  value_t eval(sharedT shrMem, cl::sycl::nd_item<1> ndItem);
  void bind(cl::sycl::handler &h);
  void adjust_access_displacement();
};

template <bool Single = true, bool Lower = true, bool Diag = true,
          bool Upper = true, typename lhs_t, typename rhs_1_t, typename rhs_2_t>
GerRow<Single, Lower, Diag, Upper, lhs_t, rhs_1_t, rhs_2_t> make_ger_row(
    lhs_t &lhs_, typename lhs_t::value_t scalar_, rhs_1_t &rhs_1_,
    rhs_2_t &rhs_2_, typename rhs_2_t::index_t nWG_row_,
    typename rhs_2_t::index_t nWG_col_,
    typename rhs_2_t::index_t local_memory_size_) {
  return GerRow<Single, Lower, Diag, Upper, lhs_t, rhs_1_t, rhs_2_t>(
      lhs_, scalar_, rhs_1_, rhs_2_, nWG_row_, nWG_col_, local_memory_size_);
}

/**** GER BY COLUMNS M ROWS x N BLOCK USING PROPERLY THE SHARED MEMORY ****/
// template <typename lhs_t,typename rhs_1_t,typename rhs_2_t>
template <bool Single, bool Lower, bool Diag, bool Upper, typename lhs_t,
          typename rhs_1_t, typename rhs_2_t>
struct GerCol {
  using value_t = typename rhs_2_t::value_t;
  using index_t = typename rhs_2_t::index_t;

  lhs_t lhs_;
  rhs_1_t rhs_1_;
  rhs_2_t rhs_2_;
  index_t nWG_row_;
  index_t nWG_col_;
  index_t local_memory_size_;
  value_t scalar_;

  GerCol(lhs_t &_l, value_t _scl, rhs_1_t &_r1, rhs_2_t &_r2, index_t &_nWG_row,
         index_t &_nWG_col, index_t &_shrMemSize);
  index_t get_size() const;
  bool valid_thread(cl::sycl::nd_item<1> ndItem) const;
  value_t eval(index_t i);
  value_t eval(cl::sycl::nd_item<1> ndItem);
  template <typename sharedT>
  value_t eval(sharedT shrMem, cl::sycl::nd_item<1> ndItem);
  void bind(cl::sycl::handler &h);
  void adjust_access_displacement();
};

// template <typename lhs_t,typename rhs_1_t,typename rhs_2_t>
template <bool Single = true, bool Lower = true, bool Diag = true,
          bool Upper = true, typename lhs_t, typename rhs_1_t, typename rhs_2_t>
GerCol<Single, Lower, Diag, Upper, lhs_t, rhs_1_t, rhs_2_t> make_ger_col(
    lhs_t &lhs_, typename lhs_t::value_t scalar_, rhs_1_t &rhs_1_,
    rhs_2_t &rhs_2_, typename rhs_2_t::index_t nWG_row_,
    typename rhs_2_t::index_t nWG_col_,
    typename rhs_2_t::index_t local_memory_size_) {
  return GerCol<Single, Lower, Diag, Upper, lhs_t, rhs_1_t, rhs_2_t>(
      lhs_, scalar_, rhs_1_, rhs_2_, nWG_row_, nWG_col_, local_memory_size_);
}

/**** SPR N COLS x (N + 1)/2 ROWS FOR PACKED MATRIX ****/
/* This class performs rank 1/2 update for symmetric packed matrices. For more
 * details on matrix refer to the explanation here:
 * https://spec.oneapi.io/versions/1.1-rev-1/elements/oneMKL/source/domains/matrix-storage.html#matrix-storage
 */
template <bool Single, bool isUpper, typename lhs_t, typename rhs_1_t,
          typename rhs_2_t>
struct Spr {
  using value_t = typename rhs_1_t::value_t;
  using index_t = typename rhs_1_t::index_t;

  lhs_t lhs_;
  rhs_1_t rhs_1_;
  rhs_2_t rhs_2_;
  value_t alpha_;
  index_t N_, incX_1_, incX_2_;
  // cl::sycl::sqrt(float) gives incorrect results when
  // the operand becomes big. The sqrt_overflow_limit was
  // identified empirically by testing the spr operator
  // for matrix sizes up to 16384x16384 on the integrated
  // Intel GPU. To make the experiment generic and to reduce
  // the chances of failing tests on different hardware we opt
  // for a more naive limit. (1048576 = 1024 * 1024)
  static constexpr index_t sqrt_overflow_limit = 1048576;

  Spr(lhs_t &_l, index_t N_, value_t _alpha, rhs_1_t &_r1, index_t _incX_1,
      rhs_2_t &_r2, index_t _incX_2);
  index_t get_size() const;
  bool valid_thread(cl::sycl::nd_item<1> ndItem) const;
  value_t eval(cl::sycl::nd_item<1> ndItem);
  void bind(cl::sycl::handler &h);
  void adjust_access_displacement();
  // Row-Col index calculation for Upper Packed Matrix
  template <bool Upper>
  SYCL_BLAS_ALWAYS_INLINE static typename std::enable_if<Upper>::type
  compute_row_col(const int64_t id, const index_t size, index_t &row,
                  index_t &col) {
    int64_t internal = 1 + 8 * id;
    float val = internal * 1.f;
    float sqrt = 0.f;
    float divisor = id >= sqrt_overflow_limit ? size * 1.f : 1.f;
    val = internal / (divisor * divisor);
    sqrt = cl::sycl::sqrt(val) * divisor;
    col = static_cast<index_t>((-1 + sqrt) / 2);
    row = id - col * (col + 1) / 2;
    // adjust the row/col if out of bounds
    if (row > col) {
      int diff = row - col;
      col += diff;
      row -= col;
    } else if (row < 0) {
      col--;
      row = id - col * (col + 1) / 2;
    }
  }

  // Row-Col index calculation for Lower Packed Matrix
  template <bool Upper>
  SYCL_BLAS_ALWAYS_INLINE static typename std::enable_if<!Upper>::type
  compute_row_col(const int64_t id, const index_t size, index_t &row,
                  index_t &col) {
    index_t temp = 2 * size + 1;
    int64_t internal = temp * temp - 8 * id;
    float val = internal * 1.f;
    float sqrt = 0.f;
    float divisor = internal >= sqrt_overflow_limit ? 2.f * size : 1.f;
    val = internal / (divisor * divisor);
    sqrt = cl::sycl::sqrt(val) * divisor;
    col = static_cast<index_t>((temp - sqrt) / 2);
    row = id - (col * (temp - col)) / 2 + col;
    // adjust row-col if out of bounds
    if (row < 0 || col < 0 || row >= size || col >= size || row < col) {
      index_t diff = id < size || row < col ? -1 : row >= size ? 1 : 0;
      col += diff;
      row = id - (col * (temp - col)) / 2 + col;
    }
  }
};

template <bool Single, bool isUpper, typename lhs_t, typename rhs_1_t,
          typename rhs_2_t>
Spr<Single, isUpper, lhs_t, rhs_1_t, rhs_2_t> make_spr(
    lhs_t &lhs_, typename rhs_1_t::index_t _N, typename lhs_t::value_t alpha_,
    rhs_1_t &rhs_1_, typename rhs_1_t::index_t incX_1, rhs_2_t &rhs_2_,
    typename rhs_1_t::index_t incX_2) {
  return Spr<Single, isUpper, lhs_t, rhs_1_t, rhs_2_t>(lhs_, _N, alpha_, rhs_1_,
                                                       incX_1, rhs_2_, incX_2);
}

}  // namespace blas
#endif  // BLAS2_TREES_H
