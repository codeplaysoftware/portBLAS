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
 *  @filename blas3_interface_sycl.hpp
 *
 **************************************************************************/

#ifndef BLAS3_INTERFACE_SYCL_HPP
#define BLAS3_INTERFACE_SYCL_HPP

#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>

// #define VERBOSE 1

#include <executors/executor_sycl.hpp>
#include <operations/blas3_trees.hpp>

namespace blas {

// dgemm (TRANSA, TRANSB, M, N, K, ALPHA, A, LDA, B, LDB, BETA, C, LDC)

template <typename Device, typename T, typename ContainerT>
void _gemm(Device &dev, std::string _TransA, std::string _TransB, size_t _M,
           size_t _N, size_t _K, T _alpha, matrix_view<T, ContainerT> _mA,
           size_t _lda, matrix_view<T, ContainerT> _mB, size_t _ldb, T _beta,
           matrix_view<T, ContainerT> _mC, size_t _ldc) {
  if ((_TransA[0] != 'n') && (_TransA[0] != 't') && (_TransA[0] != 'c') &&
      (_TransA[0] != 'N') && (_TransA[0] != 'T') && (_TransA[0] != 'C') &&
      (_TransB[0] != 'n') && (_TransB[0] != 't') && (_TransB[0] != 'c') &&
      (_TransB[0] != 'N') && (_TransB[0] != 'T') && (_TransB[0] != 'C'))
    std::cout << "Erroneous parameter" << std::endl;
  bool accessOprA = ((_TransA[0] == 'n') || (_TransA[0] == 'N'));
  bool accessOprB = ((_TransB[0] == 'n') || (_TransB[0] == 'N'));
#ifdef BLAS_EXPERIMENTAL
  printf("M = %ld , K = %ld , N = %ld\n", _M, _K, _N);
  size_t M = (accessOpr ? _M : _N);
  size_t N = (accessOpr ? _N : _M);
#endif  // BLAS_EXPERIMENTAL
  size_t M = _M;
  size_t K = _K;
  size_t N = _N;
  auto my_mA = matrix_view<T, ContainerT>(_mA, (accessOprA) ? M : K,
                                          (accessOprA) ? K : M, accessOprA,
                                          _lda, _mA.getDisp());
  auto my_mB = matrix_view<T, ContainerT>(_mB, (accessOprB) ? K : N,
                                          (accessOprB) ? N : K, accessOprB,
                                          _ldb, _mB.getDisp());
#ifdef BLAS_EXPERIMENTAL
  auto my_mB =
      matrix_view<T, ContainerT>(_mB, _K, _N, accessOprB, _ldb, _mB.getDisp());
#endif  // BLAS_EXPERIMENTAL
  auto my_mC =
      matrix_view<T, ContainerT>(_mC, _M, _N, true, _ldc, _mC.getDisp());
#ifdef VERBOSE
  std::cout << "alpha = " << _alpha << " , beta = " << _beta << std::endl;
  my_mA.printH("MA");
  my_mB.printH("MB");
  my_mC.printH("MC");
#endif  // VERBOSE
  if (my_mA.getAccess() && my_mB.getAccess()) {
    printf("A*B NO IMPLEMENTED\n");
  } else if (my_mA.getAccess() && !(my_mB.getAccess())) {
    printf("A*B^t NO IMPLEMENTED\n");
  } else if (!(my_mA.getAccess()) && my_mB.getAccess()) {
    // EFFICIENT IMPLEMENTATION FOR THE A^tB PRODUCT
    // IN WHICH, A IS ACCESSED BY ROWS AND B BY COLUMNS.
    // THUS, ALL THE THREADS COMPUTE A DOT PRODUCT MAKING
    // A COALESCENT ACCESS TO THE DATA
    auto scalExpr1 = make_expr<ScalarExpr, prdOp2_struct>(_beta, my_mC);
#ifdef BLAS_EXPERIMENTAL
    auto assignExpr = make_expr<AssignExpr>(my_mC, scalExpr1);
    blas::execute(dev, assignExpr);
#endif  // BLAS_EXPERIMENTAL
    auto prdRowMatColMattExpr = make_prdRowMatColMatExpr(my_mA, my_mB);
#ifdef BLAS_EXPERIMENTAL
    auto assignExpr = make_expr<AssignExpr>(my_mC, prdRowMatColMattExpr);
    blas::execute(dev, assignExpr);
#endif  // BLAS_EXPERIMENTAL
    auto scalExpr2 =
        make_expr<ScalarExpr, prdOp2_struct>(_alpha, prdRowMatColMattExpr);
    auto addExpr = make_expr<BinaryExpr, addOp2_struct>(scalExpr1, scalExpr2);
    auto assignExpr = make_expr<AssignExpr>(my_mC, addExpr);
    blas::execute(dev, assignExpr);
  } else {
    printf("A^t*B^t NO IMPLEMENTED\n");
  }
#ifdef VERBOSE
  my_mC.printH("MC");
#endif  // VERBOSE
}

}  // namespace blas

#endif  // BLAS3_INTERFACE_SYCL_HPP
