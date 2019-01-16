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
 *  @filename gemm_launcher.h
 *
 **************************************************************************/

#ifndef SYCL_BLAS_BLAS3_GEMM_LAUNCHER_H
#define SYCL_BLAS_BLAS3_GEMM_LAUNCHER_H

#include <algorithm>
#include <cctype>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>

#include "executors/executor.h"
#include "operations/blas3_trees.h"

namespace blas {

/*!
 * @brief Select the correct transpose version of GemmFactory, depending on the
 *        runtime values of transpose.
 */
template <int WgSize, bool DoubleBuffer, bool ConflictA, bool ConflictB,
          int ClSize, typename TileT, bool TransA, bool TransB, int GemmType,
          bool is_beta_zero>
struct Gemm_Launcher {
  template <typename Executor, typename ContainerT0, typename ContainerT1,
            typename ContainerT2, typename T, typename IndexType>
  static typename Executor::Policy::event_type _select_gemm(
      Executor& ex, IndexType _M, IndexType _N, IndexType _K, T _alpha,
      ContainerT0 _A, IndexType _lda, ContainerT1 _B, IndexType _ldb, T _beta,
      ContainerT2 _C, IndexType _ldc, IndexType batch_size);
};

}  // namespace blas

#endif  // SYCL_BLAS_BLAS3_GEMM_LAUNCHER_H
