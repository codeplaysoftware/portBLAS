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
 *  @filename blas2_trees.hpp
 *
 **************************************************************************/

#ifndef BLAS2_TREE_EXPR_HPP
#define BLAS2_TREE_EXPR_HPP

#include <stdexcept>
#include <vector>

#include <operations/blas_operators.hpp>
#include <views/view_sycl.hpp>

namespace blas {

/*! PrdRowMatVct.
 * @brief CLASSICAL DOT PRODUCT GEMV
 * Each thread computes a dot product, If
 * the matrix is column-major the accesses are coalescent.
 */
template <class RHS1_, class RHS2_>
struct PrdRowMatVctExpr {
  using RHS1 = RHS1_;
  using RHS2 = RHS2_;
  using value_type = typename RHS2::value_type;

  RHS1 r1;
  RHS2 r2;
  size_t mult;

  PrdRowMatVctExpr(RHS1 &_r1, RHS2 &_r2) : r1(_r1), r2(_r2){};

  size_t getSize() const { return r1.getSizeR(); }
};

template <class RHS1, class RHS2>
PrdRowMatVctExpr<RHS1, RHS2> make_prdRowMatVctExpr(RHS1 &r1, RHS2 &r2) {
  return PrdRowMatVctExpr<RHS1, RHS2>(r1, r2);
}

/** PrdRowMatVctMult
 * @brief MULTITHREAD DOT PRODUCT GEMV
 * P threads compute a dot product
 * If the matrix is column-major the accesses are coalescent.
 */
template <class LHS, class RHS1, class RHS2, class RHS3>
struct PrdRowMatVctMultExpr {
  using value_type = typename RHS2::value_type;

  value_type scl;
  LHS l;

  RHS1 r1;
  RHS2 r2;
  RHS3 r3;
  size_t nThr;

  PrdRowMatVctMultExpr(LHS &_l, value_type _scl, RHS1 &_r1, RHS2 &_r2,
                       RHS3 &_r3, size_t _nThr)
      : l(_l), scl(_scl), r1(_r1), r2(_r2), r3(_r3), nThr{_nThr} {};

  size_t getSize() const { return r1.getSizeR(); }
};

template <class LHS, class RHS1, class RHS2, class RHS3>
PrdRowMatVctMultExpr<LHS, RHS1, RHS2, RHS3> make_prdRowMatVctMultExpr(
    LHS &l, typename LHS::value_type scl, RHS1 &r1, RHS2 &r2, RHS3 &r3,
    size_t nThr) {
  return PrdRowMatVctMultExpr<LHS, RHS1, RHS2, RHS3>(l, scl, r1, r2, r3, nThr);
}

/*! PrdRowMatCvtMultShm.
 * @brief TWO KERNELS DOT PRODUCT GEMV
 * FIRST KERNEL: THE LOCAL COMPUTATIONS ARE MADE
 * The common data are copied to the scratch vector,
 * and later the computation begins.
 */
template <class LHS, class RHS1, class RHS2>
struct PrdRowMatVctMultShmExpr {
  using value_type = typename RHS2::value_type;

  LHS l;
  RHS1 r1;
  RHS2 r2;
  size_t nThr;

  PrdRowMatVctMultShmExpr(LHS &_l, RHS1 &_r1, RHS2 &_r2, size_t _nThr)
      : l(_l), r1(_r1), r2(_r2), nThr{_nThr} {};

  size_t getSize() const { return r1.getSizeR(); }
};

template <class LHS, class RHS1, class RHS2>
PrdRowMatVctMultShmExpr<LHS, RHS1, RHS2> make_prdRowMatVctMultShmExpr(
    LHS &l, RHS1 &r1, RHS2 &r2, size_t nThr) {
  return PrdRowMatVctMultShmExpr<LHS, RHS1, RHS2>(l, r1, r2, nThr);
}

/*! AddPrdRowMatVctMultShm.
 * @brief SECOND KERNEL: REDUCTION OF THE LOCAL COMPUTATIONS
 */
template <class LHS, class RHS1, class RHS2>
struct AddPrdRowMatVctMultShmExpr {
  using value_type = typename RHS2::value_type;

  LHS l;
  value_type scl;
  RHS1 r1;
  RHS2 r2;

  AddPrdRowMatVctMultShmExpr(LHS &_l, value_type _scl, RHS1 &_r1, RHS2 &_r2)
      : l(_l), scl(_scl), r1(_r1), r2(_r2){};

  size_t getSize() const { return r1.getSizeR(); }
};

template <class LHS, class RHS1, class RHS2>
AddPrdRowMatVctMultShmExpr<LHS, RHS1, RHS2> make_addPrdRowMatVctMultShmExpr(
    LHS &l, typename RHS1::value_type &scl, RHS1 &r1, RHS2 &r2) {
  return AddPrdRowMatVctMultShmExpr<LHS, RHS1, RHS2>(l, scl, r1, r2);
}

/*! RedRowMatVct.
 * @brief CLASSICAL AXPY GEMV
 */
template <class RHS1, class RHS2>
struct RedRowMatVctExpr {
  using value_type = typename RHS2::value_type;

  RHS1 r1;
  RHS2 r2;
  size_t warpSize;

  RedRowMatVctExpr(RHS1 &_r1, RHS2 &_r2, size_t _warpSize)
      : r1(_r1), r2(_r2), warpSize(_warpSize){};

  size_t getSize() const { return r1.getSizeR(); }
};

template <class RHS1, class RHS2>
RedRowMatVctExpr<RHS1, RHS2> make_redRowMatVctExpr(RHS1 &r1, RHS2 &r2,
                                                   size_t warpSize) {
  return RedRowMatVctExpr<RHS1, RHS2>(r1, r2, warpSize);
}

/*! ModifRank1.
 * @brief RANK 1 UPDATE
 */
template <class RHS1, class RHS2, class RHS3>
struct ModifRank1Expr {
  RHS1 r1;
  RHS2 r2;
  RHS3 r3;

  using value_type = typename RHS2::value_type;

  ModifRank1Expr(RHS1 &_r1, RHS2 &_r2, RHS3 &_r3) : r1(_r1), r2(_r2), r3(_r3){};

  size_t getSize() const { return r1.getSize(); }
};

template <class RHS1, class RHS2, class RHS3>
ModifRank1Expr<RHS1, RHS2, RHS3> make_modifRank1Expr(RHS1 &r1, RHS2 &r2,
                                                     RHS3 &r3) {
  return ModifRank1Expr<RHS1, RHS2, RHS3>(r1, r2, r3);
}

}  // namespace blas

#endif  // BLAS2_TREES_HPP
