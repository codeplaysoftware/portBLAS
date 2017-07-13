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
 *  @filename blas3_trees.hpp
 *
 **************************************************************************/

#ifndef BLAS3_TREE_EXPR_HPP
#define BLAS3_TREE_EXPR_HPP

#include <stdexcept>
#include <vector>

#include <operations/blas_operators.hpp>
#include <views/view_sycl.hpp>

namespace blas {

/*! PrdRowMatColMat.
 * @brief CLASSICAL DOT PRODUCT GEMM
 * Each thread computes a dot product
 * If the matrix is column-major the accesses are coalescent.
*/
template <class RHS1, class RHS2>
struct PrdRowMatColMatExpr {
  using value_type = typename RHS2::value_type;

  RHS1 r1;
  RHS2 r2;

  PrdRowMatColMatExpr(RHS1 &_r1, RHS2 &_r2) : r1(_r1), r2(_r2){};

  size_t getSize() const {
    return (r1.getAccessOpr() ? r1.getSizeR() : r1.getSizeC()) *
           (r2.getAccessOpr() ? r2.getSizeC() : r2.getSizeR());
  }
};

template <class RHS1, class RHS2>
PrdRowMatColMatExpr<RHS1, RHS2> make_prdRowMatColMatExpr(RHS1 &r1, RHS2 &r2) {
  return PrdRowMatColMatExpr<RHS1, RHS2>(r1, r2);
}

}  // namespace blas

#endif  // BLAS3_TREES_HPP
