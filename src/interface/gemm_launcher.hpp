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

#include <algorithm>
#include <cctype>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>

#include "executors/executor.h"
#include "interface/gemm_launcher.h"
#include "operations/blas3_trees.h"
#include "operations/blas_constants.hpp"
#include "types/access_types.h"

namespace blas {

/*!
 * @brief Select the correct transpose version of GemmFactory, depending on the
 *        runtime values of transpose.
 */
template <int WgSize, bool DoubleBuffer, bool ConflictA, bool ConflictB,
          int ClSize, typename TileT, bool TransA, bool TransB, int GemmType,
          bool is_beta_zero>
template <typename Executor, typename ContainerT0, typename ContainerT1,
          typename ContainerT2, typename T, typename IndexType>
typename Executor::Policy::event_type Gemm_Launcher<
    WgSize, DoubleBuffer, ConflictA, ConflictB, ClSize, TileT, TransA, TransB,
    GemmType, is_beta_zero>::_select_gemm(Executor& ex, IndexType _M,
                                          IndexType _N, IndexType _K, T _alpha,
                                          ContainerT0 _A, IndexType _lda,
                                          ContainerT1 _B, IndexType _ldb,
                                          T _beta, ContainerT2 _C,
                                          IndexType _ldc,
                                          IndexType batch_size) {
  auto buffer_a = make_matrix_view(ex, _A, _M, _K, _lda, Access::ColMajor());
  auto buffer_b = make_matrix_view(ex, _B, _K, _N, _ldb, Access::ColMajor());
  auto buffer_c = make_matrix_view(ex, _C, _M, _N, _ldc, Access::ColMajor());
  auto gemm = make_gemm<DoubleBuffer, ConflictA, ConflictB, ClSize, TileT,
                        TransA, TransB, GemmType, is_beta_zero>(
      buffer_a, buffer_b, buffer_c, T(_alpha), T(_beta), batch_size);
  return ex.execute(gemm);
}

}  // namespace blas

#endif  // SYCL_BLAS_BLAS3_LAUNCHER_HPP
