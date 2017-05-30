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

#ifndef BLAS3_TREES_HPP
#define BLAS3_TREES_HPP

#include <iostream>
#include <operations/blas3_trees.hpp>
#include <operations/blas_operators.hpp>
#include <stdexcept>
#include <vector>
#include <views/view_sycl.hpp>

namespace blas {

/*! PrdRowMatColMat.
 * @brief CLASSICAL DOT PRODUCT GEMM
 * Each thread computes a dot product
 * If the matrix is column-major the accesses are coalescent.
*/
template <class RHS1, class RHS2>
struct PrdRowMatColMat {
  RHS1 r1;
  RHS2 r2;

  using value_type = typename RHS2::value_type;

  PrdRowMatColMat(RHS1 &_r1, RHS2 &_r2) : r1(_r1), r2(_r2){};

  value_type eval(size_t k) {
    auto dim1 = (r2.getAccessOpr()) ? r2.getSizeR() : r2.getSizeC();
    auto dim2 = (r2.getAccessOpr()) ? r2.getSizeC() : r2.getSizeR();
    auto row = (r2.getAccess()) ? (k / dim2) : (k % dim2);
    auto col = (r2.getAccess()) ? (k % dim2) : (k / dim2);

    auto val = iniAddOp1_struct::eval(r1.eval(0));
    for (size_t j = 0; j < dim1; j++) {
      val += r1.eval(row, j) * r2.eval(j, col);
    }
    return val;
  }

  value_type eval(cl::sycl::nd_item<1> ndItem) {
    return eval(ndItem.get_global(0));
  }

  size_t getSize() {
    return (r1.getAccessOpr() ? r1.getSizeR() : r1.getSizeC()) *
           (r2.getAccessOpr() ? r2.getSizeC() : r2.getSizeR());
  }
};

template <class RHS1, class RHS2>
PrdRowMatColMat<RHS1, RHS2> make_prdRowMatColMat(RHS1 &r1, RHS2 &r2) {
  return PrdRowMatColMat<RHS1, RHS2>(r1, r2);
}

}  // namespace blas

#endif  // BLAS3_TREES_HPP
