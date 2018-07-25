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
 *  @filename blas2_interface.hpp
 *
 **************************************************************************/

#ifndef BLAS2_INTERFACE_HPP
#define BLAS2_INTERFACE_HPP

#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>

#include <sycl-blas/interface/blas_interface_sycl.hpp>

#include <sycl-blas/executors/executor_sycl.hpp>
#include <sycl-blas/operations/blas1_trees.hpp>

namespace blas {

/**** MATRIX VECTOR PRODUCT ****/

#define OPT 2  // ACTIVE CASE FOR THE COLUMN ACCESS

/*! _gemv.
 * @brief Implementation of the General Matrix Vector product.
 */
template <typename Executor, typename ContainerT0, typename ContainerT1,
          typename ContainerT2, typename T, typename IndexType,
          typename IncrementType>
typename Executor::Return_Type _gemv(Executor& ex, char _Trans, IndexType _M,
                                     IndexType _N, T _alpha, ContainerT0 _mA,
                                     IndexType _lda, ContainerT1 _vx,
                                     IncrementType _incx, T _beta,
                                     ContainerT2 _vy, IncrementType _incy) {
  typename Executor::Return_Type ret;
  _Trans = tolower(_Trans);

  if ((_Trans != 'n') && (_Trans != 't') && (_Trans != 'c'))
    std::cout << "Erroneous parameter" << std::endl;
  int accessOpr = (_Trans == 'n');

  IndexType M = (_Trans == 'n') ? _M : _N;
  IndexType N = (_Trans == 'n') ? _N : _M;

  auto mA = make_matrix_view(ex, _mA, M, N, _lda, accessOpr);
  auto vx = make_vector_view(ex, _vx, _incx, N);
  auto vy = make_vector_view(ex, _vy, _incy, M);

  if (mA.getAccess()) {
    auto scalOp1 = make_op<ScalarOp, prdOp2_struct>(_beta, vy);
    auto redRowMatVectOp = make_redRowMatVct(mA, vx, 1);
    auto scalOp2 = make_op<ScalarOp, prdOp2_struct>(_alpha, redRowMatVectOp);
    auto addOp = make_op<BinaryOp, addOp2_struct>(scalOp1, scalOp2);
    auto assignOp = make_op<Assign>(vy, addOp);
#ifdef BLAS_EXPERIMENTAL
    ret = ex.execute(assignOp, M);
#else
    ret = ex.execute(assignOp);
#endif                    // BLAS_EXPERIMENTAL
  } else if (OPT == 1) {  // Sure solution
    auto scalOp1 = make_op<ScalarOp, prdOp2_struct>(_beta, vy);
    auto prdRowMatVectOp = make_prdRowMatVct(mA, vx);
    auto scalOp2 = make_op<ScalarOp, prdOp2_struct>(_alpha, prdRowMatVectOp);
    auto addOp = make_op<BinaryOp, addOp2_struct>(scalOp1, scalOp2);
    auto assignOp = make_op<Assign>(vy, addOp);
#ifdef BLAS_EXPERIMENTAL
    ret = ex.execute(assignOp, M);
#else
    ret = ex.execute(assignOp);
#endif                    // BLAS_EXPERIMENTAL
  } else if (OPT == 2) {  // First improvement

    IndexType nThr = 2;
    auto scalOp1 = make_op<ScalarOp, prdOp2_struct>(_beta, vy);
    auto prdRowMatVectOp =
        make_prdRowMatVctMult(vy, _alpha, mA, vx, scalOp1, nThr);
    auto localSize = 32;  // NOT FINAL VALUE
    auto nWG = (M + localSize - 1) / localSize;
    auto gridSize = localSize * nThr * nWG;
    ret = ex.execute(prdRowMatVectOp, localSize * nThr, gridSize,
                     localSize * nThr);
  } else if (OPT == 3) {  // Unstable implementation

    IndexType nThr = 2;
    auto valT1 = blas::helper::make_sycl_iteator_buffer<T>(nThr * M);
    auto scalOp1 = make_op<ScalarOp, prdOp2_struct>(_beta, vy);
    auto mat1 = make_matrix_view(ex, valT1, M, nThr, nThr, 1);

#ifdef BLAS_EXPERIMENTAL
    auto val1 = RHS1(valT1, 0, 1, nThr * M);
    auto prdRowMatVectOp = make_prdRowMatVctMultShm(val1, mA, vx, nThr);
#else
    auto prdRowMatVectOp = make_prdRowMatVctMultShm(mat1, mA, vx, nThr);
#endif                    // BLAS_EXPERIMENTAL
    auto localSize = 32;  // NOT FINAL VALUE
    auto nWG = (M + localSize - 1) / localSize;
    auto gridSize = localSize * nThr * nWG;
    ret =
        ex.execute(prdRowMatVectOp, localSize, gridSize, (N + nThr - 1) / nThr);

    auto addPrdOp = make_addPrdRowMatVctMultShm(vy, _alpha, mat1, scalOp1);
#ifdef BLAS_EXPERIMENTAL
    ret = ex.execute(addPrdOp, M);
#else
    ret = ex.execute(addPrdOp);
#endif  // BLAS_EXPERIMENTAL
  }

  return ret;
}

/**** RANK 1 MODIFICATION ****/

template <typename Executor, typename T, typename ContainerT0,
          typename ContainerT1, typename ContainerT2, typename IndexType,
          typename IncrementType>
typename Executor::Return_Type _ger(Executor& ex, IndexType _M, IndexType _N,
                                    T _alpha, ContainerT0 _vx,
                                    IncrementType _incx, ContainerT1 _vy,
                                    IncrementType _incy, ContainerT2 _mA,
                                    IndexType _lda) {
  int accessOpr = true;
  auto mA = make_matrix_view(ex, _mA, _M, _N, _lda, accessOpr);
  auto vx = make_vector_view(ex, _vx, _incx, _M);
  auto vy = make_vector_view(ex, _vy, _incy, _N);

  auto modifOp = make_modifRank1(mA, vx, vy);
  auto scalOp = make_op<ScalarOp, prdOp2_struct>(_alpha, modifOp);
  auto addOp = make_op<BinaryOp, addOp2_struct>(mA, scalOp);
  auto assignOp = make_op<Assign>(mA, addOp);
  auto ret = ex.execute(assignOp);

  return ret;
}

}  // namespace blas

#endif  // BLAS2_INTERFACE_HPP
