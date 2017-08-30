/***************************************************************************
 *
 *  @license
 *  Copyright (C) 2016 Codeplay Software Limited
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
 *  @filename blas2_interface_sycl.hpp
 *
 **************************************************************************/

#ifndef BLAS2_INTERFACE_SYCL_HPP
#define BLAS2_INTERFACE_SYCL_HPP

#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>

#include <executors/executor_sycl.hpp>
#include <operations/blas2_trees.hpp>

namespace blas {

/**** MATRIX VECTOR PRODUCT ****/

#define OPT 2  // ACTIVE CASE FOR THE COLUMN ACCESS

/*! _gemv.
 * @brief Implementation of the General Matrix Vector product.
 */
template <typename Device, typename T, typename ContainerT>
void _gemv(Device &dev, std::string _Trans, size_t _M, size_t _N, T _alpha,
           matrix_view<T, ContainerT> _mA, size_t _lda,
           vector_view<T, ContainerT> _vx, size_t _incx, T _beta,
           vector_view<T, ContainerT> _vy, size_t _incy) {
  if ((_Trans[0] != 'n') && (_Trans[0] != 't') && (_Trans[0] != 'c') &&
      (_Trans[0] != 'N') && (_Trans[0] != 'T') && (_Trans[0] != 'C'))
    std::cout << "Erroneous parameter" << std::endl;
  int accessOpr = ((_Trans[0] == 'n') || (_Trans[0] == 'N'));
  size_t M = _M;
  size_t N = _N;
  auto my_mA =
      matrix_view<T, ContainerT>(_mA, _M, _N, accessOpr, _lda, _mA.getDisp());
  auto my_vx = vector_view<T, ContainerT>(_vx, _vx.getDisp(), _incx, N);
  auto my_vy = vector_view<T, ContainerT>(_vy, _vy.getDisp(), _incy, M);
#ifdef VERBOSE
  std::cout << "alpha = " << _alpha << " , beta = " << _beta << std::endl;
  my_mA.printH("MA");
  my_vx.printH("VX");
  my_vy.printH("VY");
#endif  // VERBOSE
  if (my_mA.getAccess()) {
#ifdef VERBOSE
    std::cout << "ROWS_2" << std::setprecision(15) << "M = " << _M
              << " N = " << _N << std::endl;
#endif  // VERBOSE
    auto scalExpr1 = make_expr<ScalarExpr, prdOp2_struct>(_beta, my_vy);
    auto redRowMatVectExpr = make_redRowMatVctExpr(my_mA, my_vx);
    auto scalExpr2 =
        make_expr<ScalarExpr, prdOp2_struct>(_alpha, redRowMatVectExpr);
    auto addExpr = make_expr<BinaryExpr, addOp2_struct>(scalExpr1, scalExpr2);
    auto assignExpr = make_expr<AssignExpr>(my_vy, addExpr);
#ifdef BLAS_EXPERIMENTAL
    blas::execute(dev, assignExpr, M);
#endif  // BLAS_EXPERIMENTAL
    blas::execute(dev, assignExpr);
  }
#if 0
  else if (OPT == 1) {  // Sure solution
#ifdef VERBOSE
    std::cout << "COLS_2" << std::endl;
#endif  // VERBOSE
    auto scalExpr1 = make_expr<ScalarExpr, prdOp2_struct>(_beta, my_vy);
    auto prdRowMatVectExpr = make_prdRowMatVctExpr(my_mA, my_vx);
    auto scalExpr2 = make_expr<ScalarExpr, prdOp2_struct>(_alpha, prdRowMatVectExpr);
    auto addExpr = make_expr<BinaryExpr, addOp2_struct>(scalExpr1, scalExpr2);
    auto assignExpr = make_expr<AssignExpr>(my_vy, addExpr);
#ifdef BLAS_EXPERIMENTAL
    blas::execute(assignExpr, M);
#endif  // BLAS_EXPERIMENTAL
    blas::execute(assignExpr);
  } else if (OPT == 2) {  // First improvement
#ifdef VERBOSE
    std::cout << "COLS_2" << std::endl;
#endif  // VERBOSE
    auto nThr = 2;
    auto scalExpr1 = make_op<ScalarExpr, prdOp2_struct>(_beta, my_vy);
    auto prdRowMatVectExpr = make_prdRowMatVctMultExpr(my_vy, _alpha, my_mA, my_vx, scalExpr1);
    /* auto localSize = 32;  // NOT FINAL VALUE */
    /* auto nWG = (M + localSize - 1) / localSize; */
    /* auto gridSize = localSize * nThr * nWG; */
    blas::execute(prdRowMatVectExpr);
  } else if (OPT == 3) {  // Unstable implementation
#ifdef VERBOSE
    std::cout << "COLS_2" << std::endl;
#endif  // VERBOSE
    auto nThr = 2;
    ContainerT valT1(nThr * M);
    auto mat1 = matrix_view<T, ContainerT>(valT1, M, nThr);
    auto scalExpr1 = make_op<ScalarExpr, prdOp2_struct>(_beta, my_vy);
#ifdef BLAS_EXPERIMENTAL
    auto val1 = vector_view<T, ContainerT>(valT1, 0, 1, nThr * M);
    auto mat1 = matrix_view<T, ContainerT>(valT1, M, nThr);
    auto scalExpr1 = make_op<ScalarExpr, prdOp2_struct>(_beta, my_vy);
    auto prdRowMatVectExpr = make_prdRowMatVctMultShmExpr(val1, my_mA, my_vx);
#endif  // BLAS_EXPERIMENTAL
    auto prdRowMatVectExpr = make_prdRowMatVctMultShmExpr(mat1, my_mA, my_vx);
    /* auto localSize = 32;  // NOT FINAL VALUE */
    /* auto nWG = (M + localSize - 1) / localSize; */
    /* auto gridSize = localSize * nThr * nWG; */
    blas::execute(prdRowMatVectExpr);
#ifdef VERBOSE
    mat1.printH("MAT1");
#endif  // VERBOSE
    auto addPrdExpr = make_addPrdRowMatVctMultShm(my_vy, _alpha, mat1, scalExpr1);
#ifdef BLAS_EXPERIMENTAL
    blas::execute(addPrdExpr, M);
#endif  // BLAS_EXPERIMENTAL
    blas::execute(addPrdExpr);
#ifdef VERBOSE
    val1.printH("VAL1");
#endif  // VERBOSE
  }
#ifdef VERBOSE
  my_vy.printH("VY");
#endif
#endif  // 0
}

/**** RANK 1 MODIFICATION ****/

template <typename Device, typename T, typename ContainerT>
void _ger(Device &dev, size_t _M, size_t _N, T _alpha,
          vector_view<T, ContainerT> _vx, size_t _incx,
          vector_view<T, ContainerT> _vy, size_t _incy,
          matrix_view<T, ContainerT> _mA, size_t _lda) {
  int accessOpr = true;
  size_t M = _M;
  size_t N = _N;
  auto my_mA =
      matrix_view<T, ContainerT>(_mA, _M, _N, accessOpr, _lda, _mA.getDisp());
  auto my_vx = vector_view<T, ContainerT>(_vx, _vx.getDisp(), _incx, M);
  auto my_vy = vector_view<T, ContainerT>(_vy, _vy.getDisp(), _incy, N);
#ifdef VERBOSE
  std::cout << "alpha = " << _alpha << std::endl;
  my_mA.printH("MA");
  my_vx.printH("VX");
  my_vy.printH("VY");
#endif
  auto modifExpr = make_modifRank1Expr(my_mA, my_vx, my_vy);
  auto scalExpr = make_expr<ScalarExpr, prdOp2_struct>(_alpha, modifExpr);
  auto addExpr = make_expr<BinaryExpr, addOp2_struct>(my_mA, scalExpr);
  auto assignExpr = make_expr<AssignExpr>(my_mA, addExpr);
  blas::execute(dev, assignExpr);
#ifdef VERBOSE
  my_vy.printH("VY");
#endif
}

}  // namespace blas

#endif  // BLAS2_INTERFACE_SYCL_HPP
