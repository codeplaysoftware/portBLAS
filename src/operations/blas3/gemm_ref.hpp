/***************************************************************************
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
 *  @filename gemm_ref.hpp
 *
 **************************************************************************/

#ifndef SYCL_BLAS_BLAS3_REF_GEMM_HPP
#define SYCL_BLAS_BLAS3_REF_GEMM_HPP

#include "gemm_common.hpp"

namespace blas {

/*!
 * @brief This factory generates reference GEMM implementations.
 *
 * These implementations use a naive approach of mapping one value of the
 * output matrix to each work item, and are highly memory bound.
 * They should only be used as a reference in performance testing, or to check
 * correctness of other implementations.
 * Refer to GemmFactory for details about how to use this. Note that there is
 * no local_memory value, as these functions do not use local memory.
 *
 * @tparam WgSize  the number of items in a work group
 * @tparam TransA  iff true, A will be transposed on the fly
 * @tparam TransB  iff true, B will be transposed on the fly
 * @tparam element_t  the type of matrix elements
 */
template <typename input_t, typename output_t, bool DoubleBuffer, bool NbcA,
          bool NbcB, int ClSize, typename tile_type, bool TransA, bool TransB,
          typename element_t, bool is_beta_zero, int GemmMemoryType,
          int GemmAlgorithm, int VectorSize>
SYCL_BLAS_INLINE Gemm<input_t, output_t, DoubleBuffer, NbcA, NbcB, ClSize,
                      tile_type, TransA, TransB, element_t, is_beta_zero,
                      GemmMemoryType, GemmAlgorithm, VectorSize>::
    Gemm(input_t A, input_t B, output_t C, element_t alpha, element_t beta,
         typename std::make_signed<typename input_t::index_t>::type batch_size)
    : a_(A),
      b_(B),
      c_(C),
      alpha_(alpha),
      beta_(beta),
      m_(a_.get_size_row()),
      n_(b_.get_size_col()),
      k_(a_.get_size_col()),
      lda_(a_.getSizeL()),
      ldb_(b_.getSizeL()),
      ldc_(c_.getSizeL()),
      batch_size_(batch_size) {}
template <typename input_t, typename output_t, bool DoubleBuffer, bool NbcA,
          bool NbcB, int ClSize, typename tile_type, bool TransA, bool TransB,
          typename element_t, bool is_beta_zero, int GemmMemoryType,
          int GemmAlgorithm, int VectorSize>
SYCL_BLAS_INLINE std::string
Gemm<input_t, output_t, DoubleBuffer, NbcA, NbcB, ClSize, tile_type, TransA,
     TransB, element_t, is_beta_zero, GemmMemoryType, GemmAlgorithm,
     VectorSize>::get_type_string() noexcept {
  std::ostringstream str{};
  str << "ReferenceGemmFactory<" << wg_size << ", "
      << type_string<value_t>::get_value() << ">";
  return str.str();
}
/*!
 *@brief gt_workgroup_cluster. This function is used to find the optimum
 *number of work_group required to execute each GEMM.
 *
 */
template <typename input_t, typename output_t, bool DoubleBuffer, bool NbcA,
          bool NbcB, int ClSize, typename tile_type, bool TransA, bool TransB,
          typename element_t, bool is_beta_zero, int GemmMemoryType,
          int GemmAlgorithm, int VectorSize>
SYCL_BLAS_INLINE
    typename Gemm<input_t, output_t, DoubleBuffer, NbcA, NbcB, ClSize,
                  tile_type, TransA, TransB, element_t, is_beta_zero,
                  GemmMemoryType, GemmAlgorithm, VectorSize>::index_t
    Gemm<input_t, output_t, DoubleBuffer, NbcA, NbcB, ClSize, tile_type, TransA,
         TransB, element_t, is_beta_zero, GemmMemoryType, GemmAlgorithm,
         VectorSize>::get_workgroup_cluster(index_t m, index_t n) noexcept {
  return ((m * n - 1) / wg_size + 1);
}
/*!
 *@brief get_num_workgroup_cluster. This function is used to extend the number
 *of work_group cluster, in order to make sure that atleast 4
 *gemm operations is available per work group. The number 4
 *is used based on empirical research.
 *
 */
template <typename input_t, typename output_t, bool DoubleBuffer, bool NbcA,
          bool NbcB, int ClSize, typename tile_type, bool TransA, bool TransB,
          typename element_t, bool is_beta_zero, int GemmMemoryType,
          int GemmAlgorithm, int VectorSize>
SYCL_BLAS_INLINE typename Gemm<input_t, output_t, DoubleBuffer, NbcA, NbcB,
                               ClSize, tile_type, TransA, TransB, element_t,
                               is_beta_zero, GemmMemoryType, GemmAlgorithm,
                               VectorSize>::index_t
Gemm<input_t, output_t, DoubleBuffer, NbcA, NbcB, ClSize, tile_type, TransA,
     TransB, element_t, is_beta_zero, GemmMemoryType, GemmAlgorithm,
     VectorSize>::get_num_workgroup_cluster(index_t m, index_t n,
                                            index_t compute_units) noexcept {
  constexpr index_t num_gemm_per_compute_units = 4;
  return (
      (num_gemm_per_compute_units * compute_units - 1) /
          Gemm<input_t, output_t, DoubleBuffer, NbcA, NbcB, ClSize, tile_type,
               TransA, TransB, element_t, is_beta_zero, GemmMemoryType,
               GemmAlgorithm, VectorSize>::get_workgroup_cluster(m, n) +
      1);
}

template <typename input_t, typename output_t, bool DoubleBuffer, bool NbcA,
          bool NbcB, int ClSize, typename tile_type, bool TransA, bool TransB,
          typename element_t, bool is_beta_zero, int GemmMemoryType,
          int GemmAlgorithm, int VectorSize>
SYCL_BLAS_INLINE cl::sycl::nd_range<1>
Gemm<input_t, output_t, DoubleBuffer, NbcA, NbcB, ClSize, tile_type, TransA,
     TransB, element_t, is_beta_zero, GemmMemoryType, GemmAlgorithm,
     VectorSize>::get_nd_range(index_t m, index_t n,
                               index_t compute_units) noexcept {
  const cl::sycl::range<1> nwg(
      Gemm<input_t, output_t, DoubleBuffer, NbcA, NbcB, ClSize, tile_type,
           TransA, TransB, element_t, is_beta_zero, GemmMemoryType,
           GemmAlgorithm, VectorSize>::get_workgroup_cluster(m, n) *
      Gemm<input_t, output_t, DoubleBuffer, NbcA, NbcB, ClSize, tile_type,
           TransA, TransB, element_t, is_beta_zero, GemmMemoryType,
           GemmAlgorithm,
           VectorSize>::get_num_workgroup_cluster(m, n, compute_units));
  const cl::sycl::range<1> wgs(wg_size);
  return cl::sycl::nd_range<1>(nwg * wgs, wgs);
}
template <typename input_t, typename output_t, bool DoubleBuffer, bool NbcA,
          bool NbcB, int ClSize, typename tile_type, bool TransA, bool TransB,
          typename element_t, bool is_beta_zero, int GemmMemoryType,
          int GemmAlgorithm, int VectorSize>
SYCL_BLAS_INLINE
    typename Gemm<input_t, output_t, DoubleBuffer, NbcA, NbcB, ClSize,
                  tile_type, TransA, TransB, element_t, is_beta_zero,
                  GemmMemoryType, GemmAlgorithm, VectorSize>::index_t
    Gemm<input_t, output_t, DoubleBuffer, NbcA, NbcB, ClSize, tile_type, TransA,
         TransB, element_t, is_beta_zero, GemmMemoryType, GemmAlgorithm,
         VectorSize>::get_size() const {
  return m_ * n_;
}

template <typename input_t, typename output_t, bool DoubleBuffer, bool NbcA,
          bool NbcB, int ClSize, typename tile_type, bool TransA, bool TransB,
          typename element_t, bool is_beta_zero, int GemmMemoryType,
          int GemmAlgorithm, int VectorSize>
SYCL_BLAS_INLINE bool
Gemm<input_t, output_t, DoubleBuffer, NbcA, NbcB, ClSize, tile_type, TransA,
     TransB, element_t, is_beta_zero, GemmMemoryType, GemmAlgorithm,
     VectorSize>::valid_thread(cl::sycl::nd_item<1> ndItem) const {
  return true;
}

template <typename input_t, typename output_t, bool DoubleBuffer, bool NbcA,
          bool NbcB, int ClSize, typename tile_type, bool TransA, bool TransB,
          typename element_t, bool is_beta_zero, int GemmMemoryType,
          int GemmAlgorithm, int VectorSize>
SYCL_BLAS_INLINE void
Gemm<input_t, output_t, DoubleBuffer, NbcA, NbcB, ClSize, tile_type, TransA,
     TransB, element_t, is_beta_zero, GemmMemoryType, GemmAlgorithm,
     VectorSize>::eval(cl::sycl::nd_item<1> id) noexcept {
  const index_t wg_batch_id = id.get_group(0) / get_workgroup_cluster(m_, n_);
  // This will disable all workgroups that dont have any batch to work on
  if (wg_batch_id >= batch_size_) {
    return;
  }
  const index_t batch_stride =
      id.get_group_range(0) / get_workgroup_cluster(m_, n_);

  const index_t a_size = trans_a ? m_ * lda_ : k_ * lda_;
  const index_t b_size = trans_b ? ldb_ * k_ : n_ * ldb_;
  const index_t c_size = ldc_ * n_;

  auto orig_A = a_.get_pointer() + (wg_batch_id * a_size);
  auto orig_B = b_.get_pointer() + (wg_batch_id * b_size);
  auto orig_C = c_.get_pointer() + (wg_batch_id * c_size);

  index_t item_id = (id.get_group(0) % get_workgroup_cluster(m_, n_)) *
                        (id.get_local_range(0)) +
                    id.get_local_id(0);
  if (item_id >= m_ * n_) {
    return;
  }

  const index_t row = item_id % m_;
  const index_t col = item_id / m_;

  orig_A = orig_A + row * (trans_a ? lda_ : 1);
  orig_B = orig_B + col * (trans_b ? 1 : ldb_);
  orig_C = orig_C + row + col * ldc_;

  do {
    auto A = orig_A;
    auto B = orig_B;
    auto C = orig_C;
    value_t reg_res = {};
    while (k_ > 0) {
      reg_res = cl::sycl::mad(A[0], B[0], reg_res);
      --k_;
      A = A + (trans_a ? 1 : lda_);
      B = B + (trans_b ? ldb_ : 1);
    }
    // when C is uninitialized the element of the C can be NaN, and Nan*0
    // will be NaN
    if (is_beta_zero) {
      C[0] = alpha_ * reg_res;
    } else {
      C[0] = alpha_ * reg_res + beta_ * C[0];
    }

    orig_A += (a_size * batch_stride);
    orig_B += (b_size * batch_stride);
    orig_C += (c_size * batch_stride);
    k_ = a_.get_size_col();
    // batch_size_ must be signed as the negative value has meaning here.
    batch_size_ -= batch_stride;
  } while (batch_size_ > wg_batch_id);
}

template <typename input_t, typename output_t, bool DoubleBuffer, bool NbcA,
          bool NbcB, int ClSize, typename tile_type, bool TransA, bool TransB,
          typename element_t, bool is_beta_zero, int GemmMemoryType,
          int GemmAlgorithm, int VectorSize>
SYCL_BLAS_INLINE void
Gemm<input_t, output_t, DoubleBuffer, NbcA, NbcB, ClSize, tile_type, TransA,
     TransB, element_t, is_beta_zero, GemmMemoryType, GemmAlgorithm,
     VectorSize>::bind(cl::sycl::handler &h) {
  a_.bind(h);
  b_.bind(h);
  c_.bind(h);
}

template <typename input_t, typename output_t, bool DoubleBuffer, bool NbcA,
          bool NbcB, int ClSize, typename tile_type, bool TransA, bool TransB,
          typename element_t, bool is_beta_zero, int GemmMemoryType,
          int GemmAlgorithm, int VectorSize>
SYCL_BLAS_INLINE void
Gemm<input_t, output_t, DoubleBuffer, NbcA, NbcB, ClSize, tile_type, TransA,
     TransB, element_t, is_beta_zero, GemmMemoryType, GemmAlgorithm,
     VectorSize>::adjust_access_displacement() {
  a_.adjust_access_displacement();
  b_.adjust_access_displacement();
  c_.adjust_access_displacement();
}

}  // namespace blas

#endif  // SYCL_BLAS_BLAS3_REF_GEMM_HPP
