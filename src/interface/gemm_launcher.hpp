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
 *  @filename gemm_launcher.hpp
 *
 **************************************************************************/

#ifndef SYCL_BLAS_BLAS3_LAUNCHER_HPP
#define SYCL_BLAS_BLAS3_LAUNCHER_HPP

#include "interface/gemm_launcher.h"

namespace blas {

/*!
 * @brief Wrapper around Gemm. Creates the views, then makes and launches Gemm
 */
template <int WgSize, bool DoubleBuffer, bool ConflictA, bool ConflictB,
          int ClSize, typename TileT, bool TransA, bool TransB,
          int GemmMemoryType, int GemmAlgorithm, bool is_beta_zero,
          int VectorSize>
template <typename Executor, typename container_t0, typename container_t1,
          typename container_t2, typename element_t, typename index_t>
typename Executor::policy_t::event_t
Gemm_Launcher<WgSize, DoubleBuffer, ConflictA, ConflictB, ClSize, TileT, TransA,
              TransB, GemmMemoryType, GemmAlgorithm, is_beta_zero,
              VectorSize>::_select_gemm(Executor& ex, index_t _M, index_t _N,
                                        index_t _K, element_t _alpha,
                                        container_t0 a_, index_t _lda,
                                        container_t1 b_, index_t _ldb,
                                        element_t _beta, container_t2 _C,
                                        index_t _ldc, index_t batch_size) {
  auto buffer_a = make_matrix_view<col_major>(ex, a_, _M, _K, _lda);
  auto buffer_b = make_matrix_view<col_major>(ex, b_, _K, _N, _ldb);
  auto buffer_c = make_matrix_view<col_major>(ex, _C, _M, _N, _ldc);
  auto gemm =
      make_gemm<DoubleBuffer, ConflictA, ConflictB, ClSize, TileT, TransA,
                TransB, GemmMemoryType, GemmAlgorithm, is_beta_zero,
                VectorSize>(buffer_a, buffer_b, buffer_c, element_t(_alpha),
                            element_t(_beta), batch_size);
  return ex.execute(gemm);
}

}  // namespace blas

#endif  // SYCL_BLAS_BLAS3_LAUNCHER_HPP
