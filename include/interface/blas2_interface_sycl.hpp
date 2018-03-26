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
#include <operations/blas1_trees.hpp>

namespace blas {

/**** MATRIX VECTOR PRODUCT ****/

#define OPT 2  // ACTIVE CASE FOR THE COLUMN ACCESS

/*! _gemv.
 * @brief Implementation of the General Matrix Vector product.
 */
template <typename ExecutorType, typename T>
cl::sycl::event _gemv(Executor<ExecutorType>& ex, char _Trans, size_t _M,
                      size_t _N, T _alpha, T* _mA, size_t _lda, T* _vx,
                      size_t _incx, T _beta, T* _vy, size_t _incy) {
  cl::sycl::event event;
  _Trans = tolower(_Trans);

  if ((_Trans != 'n') && (_Trans != 't') && (_Trans != 'c'))
    std::cout << "Erroneous parameter" << std::endl;
  int accessOpr = (_Trans == 'n');

  size_t M = (_Trans == 'n') ? _M : _N;
  size_t N = (_Trans == 'n') ? _N : _M;
  auto _mA_container = ex.get_buffer(_mA);
  using RHS =
      matrix_view<T, typename Executor<ExecutorType>::template ContainerT<T> >;

  RHS my_mA(_mA_container, M, N, accessOpr, _lda, ex.get_offset(_mA));
  using RHS1 =
      vector_view<T, typename Executor<ExecutorType>::template ContainerT<T> >;
  auto _vx_container = ex.get_buffer(_vx);
  RHS1 my_vx(_vx_container, ex.get_offset(_vx), _incx, N);
  auto _vy_container = ex.get_buffer(_vy);
  RHS1 my_vy(_vy_container, ex.get_offset(_vy), _incy, M);
#ifdef VERBOSE
  std::cout << "alpha = " << _alpha << " , beta = " << _beta << std::endl;
  my_mA.printH("MA");
  my_vx.printH("VX");
  my_vy.printH("VY");
#endif  // VERBOSE
  if (my_mA.getAccess()) {
#ifdef VERBOSE
    std::cout << "ROWS_2" << std::setprecision(15) << "M = " << M
              << " N = " << N << std::endl;
#endif  // VERBOSE
    auto scalOp1 = make_op<ScalarOp, prdOp2_struct>(_beta, my_vy);
    auto redRowMatVectOp = make_redRowMatVct(my_mA, my_vx, 1);
    auto scalOp2 = make_op<ScalarOp, prdOp2_struct>(_alpha, redRowMatVectOp);
    auto addOp = make_op<BinaryOp, addOp2_struct>(scalOp1, scalOp2);
    auto assignOp = make_op<Assign>(my_vy, addOp);
#ifdef BLAS_EXPERIMENTAL
    event = ex.execute(assignOp, M);
#else
    event = ex.execute(assignOp);
#endif                    // BLAS_EXPERIMENTAL
  } else if (OPT == 1) {  // Sure solution
#ifdef VERBOSE
    std::cout << "COLS_2" << std::endl;
#endif  // VERBOSE
    auto scalOp1 = make_op<ScalarOp, prdOp2_struct>(_beta, my_vy);
    auto prdRowMatVectOp = make_prdRowMatVct(my_mA, my_vx);
    auto scalOp2 = make_op<ScalarOp, prdOp2_struct>(_alpha, prdRowMatVectOp);
    auto addOp = make_op<BinaryOp, addOp2_struct>(scalOp1, scalOp2);
    auto assignOp = make_op<Assign>(my_vy, addOp);
#ifdef BLAS_EXPERIMENTAL
    event = ex.execute(assignOp, M);
#else
    event = ex.execute(assignOp);
#endif                    // BLAS_EXPERIMENTAL
  } else if (OPT == 2) {  // First improvement
#ifdef VERBOSE
    std::cout << "COLS_2" << std::endl;
#endif  // VERBOSE
    auto nThr = 2;
    auto scalOp1 = make_op<ScalarOp, prdOp2_struct>(_beta, my_vy);
    auto prdRowMatVectOp =
        make_prdRowMatVctMult(my_vy, _alpha, my_mA, my_vx, scalOp1, nThr);
    auto localSize = 32;  // NOT FINAL VALUE
    auto nWG = (M + localSize - 1) / localSize;
    auto gridSize = localSize * nThr * nWG;
    event = ex.execute(prdRowMatVectOp, localSize * nThr, gridSize,
                       localSize * nThr);
  } else if (OPT == 3) {  // Unstable implementation
#ifdef VERBOSE
    std::cout << "COLS_2" << std::endl;
#endif  // VERBOSE
    auto nThr = 2;
    auto val_ptr = ex.template allocate<T>(nThr * M);
    auto valT1 = ex.get_buffer(val_ptr);
    auto scalOp1 = make_op<ScalarOp, prdOp2_struct>(_beta, my_vy);
    RHS mat1(valT1, M, nThr);
#ifdef BLAS_EXPERIMENTAL
    auto val1 = RHS1(valT1, 0, 1, nThr * M);
    auto prdRowMatVectOp = make_prdRowMatVctMultShm(val1, my_mA, my_vx, nThr);
#else
    auto prdRowMatVectOp = make_prdRowMatVctMultShm(mat1, my_mA, my_vx, nThr);
#endif                    // BLAS_EXPERIMENTAL
    auto localSize = 32;  // NOT FINAL VALUE
    auto nWG = (M + localSize - 1) / localSize;
    auto gridSize = localSize * nThr * nWG;
    event =
        ex.execute(prdRowMatVectOp, localSize, gridSize, (N + nThr - 1) / nThr);
#ifdef VERBOSE
    mat1.printH("MAT1");
#endif  // VERBOSE
    auto addPrdOp = make_addPrdRowMatVctMultShm(my_vy, _alpha, mat1, scalOp1);
#ifdef BLAS_EXPERIMENTAL
    event = ex.execute(addPrdOp, M);
#else
    event = ex.execute(addPrdOp);
#endif  // BLAS_EXPERIMENTAL
#ifdef VERBOSE
    val1.printH("VAL1");
#endif  // VERBOSE
    ex.template deallocate<T>(val_ptr);
  }
#ifdef VERBOSE
  my_vy.printH("VY");
#endif
  return event;
}

/**** RANK 1 MODIFICATION ****/

template <typename ExecutorType, typename T>
cl::sycl::event _ger(Executor<ExecutorType>& ex, size_t _M, size_t _N, T _alpha,
                     T* _vx, size_t _incx, T* _vy, size_t _incy, T* _mA,
                     size_t _lda) {
  int accessOpr = true;
  auto _mA_container = ex.get_buffer(_mA);
  using RHS =
      matrix_view<T, typename Executor<ExecutorType>::template ContainerT<T> >;
  RHS my_mA(_mA_container, _M, _N, accessOpr, _lda, ex.get_offset(_mA));
  using RHS1 =
      vector_view<T, typename Executor<ExecutorType>::template ContainerT<T> >;
  auto _vx_container = ex.get_buffer(_vx);
  RHS1 my_vx(_vx_container, ex.get_offset(_vx), _incx, _M);
  auto _vy_container = ex.get_buffer(_vy);
  RHS1 my_vy(_vy_container, ex.get_offset(_vy), _incy, _N);

#ifdef VERBOSE
  std::cout << "alpha = " << _alpha << std::endl;
  my_mA.printH("MA");
  my_vx.printH("VX");
  my_vy.printH("VY");
#endif
  auto modifOp = make_modifRank1(my_mA, my_vx, my_vy);
  auto scalOp = make_op<ScalarOp, prdOp2_struct>(_alpha, modifOp);
  auto addOp = make_op<BinaryOp, addOp2_struct>(my_mA, scalOp);
  auto assignOp = make_op<Assign>(my_mA, addOp);
  auto event = ex.execute(assignOp);
#ifdef VERBOSE
  my_vy.printH("VY");
#endif
  return event;
}

}  // namespace blas

#endif  // BLAS2_INTERFACE_SYCL_HPP
