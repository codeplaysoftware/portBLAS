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
 *  @filename blas2_tree_evaluator.hpp
 *
 **************************************************************************/

#ifndef BLAS2_TREE_EVALUATOR_HPP
#define BLAS2_TREE_EVALUATOR_HPP

#include <stdexcept>
#include <vector>

#include <evaluators/blas_tree_evaluator_base.hpp>
#include <executors/blas_packet_traits_sycl.hpp>
#include <operations/blas2_trees.hpp>
#include <operations/blas_operators.hpp>
#include <views/view_sycl.hpp>

namespace blas {

/*! PrdRowMatVct.
 * @brief CLASSICAL DOT PRODUCT GEMV
 * Each thread computes a dot product, If
 * the matrix is column-major the accesses are coalescent.
 */

template <class RHS1, class RHS2, typename Device>
struct Evaluator<PrdRowMatVctExpr<RHS1, RHS2>, Device> {
  using Expression = PrdRowMatVctExpr<RHS1, RHS2>;
  using value_type = typename Expression::value_type;
  using cont_type = typename Evaluator<RHS1, Device>::cont_type;
  /* static constexpr bool supported = RHS1::supported && RHS2::supported; */

  Evaluator<RHS1, Device> r1;
  Evaluator<RHS2, Device> r2;

  Evaluator(Expression &expr)
      : r1(Evaluator<RHS1, Device>(expr.r1)),
        r2(Evaluator<RHS2, Device>(expr.r2)) {}

  size_t getSize() const { return r1.getSizeR(); }
  cont_type data() { return r1.data(); }

  bool eval_subexpr_if_needed(cont_type *cont, Device &dev) {
    r1.eval_subexpr_if_needed(NULL, dev);
    r2.eval_subexpr_if_needed(NULL, dev);
    return true;
  }

  value_type eval(size_t i) {
    auto dim = r2.getSize();

    auto val =
        functor_traits<iniAddOp1_struct, value_type, Device>::eval(r2.eval(0));
    for (size_t j = 0; j < dim; j++) {
      val += r1.eval(i, j) * r2.eval(j);
    }
    return val;
  }

  value_type eval(cl::sycl::nd_item<1> ndItem) {
    return eval(ndItem.get_global(0));
  }
};

/** PrdRowMatVctMult
 * @brief MULTITHREAD DOT PRODUCT GEMV
 * P threads compute a dot product
 * If the matrix is column-major the accesses are coalescent.
 */
template <class LHS, class RHS1, class RHS2, class RHS3, typename Device>
struct Evaluator<PrdRowMatVctMultExpr<LHS, RHS1, RHS2, RHS3>, Device> {
  using Expression = PrdRowMatVctMultExpr<LHS, RHS1, RHS2, RHS3>;
  using value_type = typename Expression::value_type;
  using cont_type = typename Evaluator<LHS, Device>::cont_type;
  /* static constexpr bool supported = LHS::supported && RHS1::supported &&
   * RHS2::supported && RHS3::supported; */

  value_type scl;
  Evaluator<LHS, Device> l;
  Evaluator<RHS1, Device> r1;
  Evaluator<RHS2, Device> r2;
  Evaluator<RHS3, Device> r3;

  size_t nThr = 0;

  Evaluator(Expression &expr)
      : scl(expr.scl),
        l(Evaluator<LHS, Device>(expr.l)),
        r1(Evaluator<RHS1, Device>(expr.r1)),
        r2(Evaluator<RHS2, Device>(expr.r2)),
        r3(Evaluator<RHS3, Device>(expr.r3)),
        nThr(2) {}

  size_t getSize() const { return r1.getSizeR(); }
  cont_type data() { return l.data(); }

  bool eval_subexpr_if_needed(cont_type *cont, Device &dev) {
    l.eval_subexpr_if_needed(NULL, dev);
    r1.eval_subexpr_if_needed(l.data(), dev);
    r2.eval_subexpr_if_needed(l.data(), dev);
    r3.eval_subexpr_if_needed(l.data(), dev);
    return true;
  }

  value_type eval(size_t i) {
    auto dim = r2.getSize();

    auto val =
        functor_traits<iniAddOp1_struct, value_type, Device>::eval(r2.eval(0));
    for (size_t j = 0; j < dim; j++) {
      val += r1.eval(i, j) * r2.eval(j);
    }
    l.eval(i) += scl * val;
    return val;
  }

  template <typename sharedT>
  value_type eval(sharedT scratch, cl::sycl::nd_item<1> ndItem) {
    size_t localid = ndItem.get_local(0);
    size_t localSz = ndItem.get_local_range(0);
    size_t groupid = ndItem.get_group(0);

    size_t dimR = r1.getSizeR();
    size_t dimC = r1.getSizeC();

    size_t rowSz = (localSz / nThr);  // number of rows per each workgroup
    size_t rowid = groupid * rowSz + localid % rowSz;  // rowid of the thread

    size_t colid = localid / rowSz;  // first column on which thread works

    // Local computations
    auto val =
        functor_traits<iniAddOp1_struct, value_type, Device>::eval(r2.eval(0));
    if (rowid < dimR) {
      for (size_t j = colid; j < dimC; j += nThr) {
        val += r1.eval(rowid, j) * r2.eval(j);
      }
    }

    scratch[localid] = val;
    // This barrier is mandatory to be sure the data is on the shared memory
    ndItem.barrier(cl::sycl::access::fence_space::local_space);

    // Reduction inside the block
    for (size_t offset = nThr >> 1; offset > 0; offset >>= 1) {
      if ((rowid < dimR) && (colid < offset)) {
        scratch[localid] += scratch[localid + offset * rowSz];
      }
      // This barrier is mandatory to be sure the data are on the shared memory
      ndItem.barrier(cl::sycl::access::fence_space::local_space);
    }
    // The result is stored in lhs
    if ((rowid < dimR) && (colid == 0)) {
      l.eval(rowid) = scl * scratch[localid] + r3.eval(rowid);
    }
    return val;
  }
};

/*! PrdRowMatCvtMultShm.
 * @brief TWO KERNELS DOT PRODUCT GEMV
 * FIRST KERNEL: THE LOCAL COMPUTATIONS ARE MADE
 * The common data are copied to the scratch vector,
 * and later the computation begins.
 */
template <class LHS, class RHS1, class RHS2, typename Device>
struct Evaluator<PrdRowMatVctMultShmExpr<LHS, RHS1, RHS2>, Device> {
  using Expression = PrdRowMatVctMultShmExpr<LHS, RHS1, RHS2>;
  using value_type = typename Expression::value_type;
  using cont_type = typename Evaluator<LHS, Device>::cont_type;
  /* static constexpr bool supported = LHS::supported && RHS1::supported &&
   * RHS2::supported; */

  Evaluator<LHS, Device> l;
  Evaluator<RHS1, Device> r1;
  Evaluator<RHS2, Device> r2;

  size_t nThr = 0;

  Evaluator(Expression &expr)
      : l(Evaluator<LHS, Device>(expr.l)),
        r1(Evaluator<RHS1, Device>(expr.r1)),
        r2(Evaluator<RHS2, Device>(expr.r2)) {
    nThr = 2;
  }

  size_t getSize() const { return r1.getSizeR(); }
  cont_type data() { return l.data(); }

  bool eval_subexpr_if_needed(cont_type *cont, Device &dev) {
    l.eval_subexpr_if_needed(NULL, dev);
    r1.eval_subexpr_if_needed(l.data(), dev);
    r2.eval_subexpr_if_needed(l.data(), dev);
    return true;
  }

  value_type eval(size_t i) {
    auto dim = r2.getSize();

    auto val = iniAddOp1_struct::eval(r2.eval(0));
    for (size_t j = 0; j < dim; j++) {
      val += r1.eval(i, j) * r2.eval(j);
    }
    l.eval(i) += val;
    return val;
  }

  template <typename sharedT>
  value_type eval(sharedT scratch, cl::sycl::nd_item<1> ndItem) {
    size_t localid = ndItem.get_local(0);
    size_t localSz = ndItem.get_local_range(0);
    size_t groupid = ndItem.get_group(0);
    size_t groupSz = ndItem.get_num_groups(0);

    size_t dimR = r1.getSizeR();
    size_t dimC = r1.getSizeC();

    size_t blqSz = (groupSz + nThr - 1) / nThr;  // number of "real" workgroups
    size_t blqidR = groupid % blqSz;  // 1st row id of the current workgroup
    size_t blqidC = groupid / blqSz;  // col bloq id of the current workgroup

    size_t rowSz =
        (dimR < localSz) ? dimR : localSz;  // number of rows per each workgroup
    size_t colSz =
        (dimC + nThr - 1) / nThr;  // number of columns per each thread

    size_t rowid = blqidR * rowSz + localid;  // rowid of the current thread
    size_t colid = blqidC * colSz;  // first column of the current thread

    size_t k;

    // Copying  to the scratch
    k = localid;
    for (size_t j = colid + localid; j < colid + colSz; j += rowSz) {
      if ((rowid < dimR) && (j < dimC)) {
        scratch[k] = r2.eval(j);
      }
      k += rowSz;
    }

    // This barrier is mandatory to be sure the data are on the shared memory
    ndItem.barrier(cl::sycl::access::fence_space::local_space);

    // Local computation
    auto val =
        functor_traits<iniAddOp1_struct, value_type, Device>::eval(r2.eval(0));
    k = 0;
    for (size_t j = colid; j < colid + colSz; j++) {
      if ((rowid < dimR) && (j < dimC)) {
        val += r1.eval(rowid, j) * scratch[k++];
      }
    }
    // The result is stored in lhs
    if (rowid < dimR) {
      l.eval(rowid, blqidC) = val;
    }

    return val;
  }
};

/*! AddPrdRowMatVctMultShm.
 * @brief SECOND KERNEL: REDUCTION OF THE LOCAL COMPUTATIONS
 */
template <class LHS, class RHS1, class RHS2, typename Device>
struct Evaluator<AddPrdRowMatVctMultShmExpr<LHS, RHS1, RHS2>, Device> {
  using Expression = AddPrdRowMatVctMultShmExpr<LHS, RHS1, RHS2>;
  using value_type = typename Expression::value_type;
  using cont_type = typename Evaluator<LHS, Device>::cont_type;
  /* static constexpr bool supported = LHS::supported && RHS1::supported &&
   * RHS2::supported; */

  value_type scl;
  Evaluator<LHS, Device> l;
  Evaluator<RHS1, Device> r1;
  Evaluator<RHS2, Device> r2;

  Evaluator(Expression &expr)
      : scl(expr.scl),
        l(Evaluator<LHS, Device>(expr.l)),
        r1(Evaluator<RHS1, Device>(expr.r1)),
        r2(Evaluator<RHS2, Device>(expr.r2)) {}

  size_t getSize() const { return r1.getSizeR(); }
  cont_type data() { return l.data(); }

  bool eval_subexpr_if_needed(cont_type *cont, Device &dev) {
    l.eval_subexpr_if_needed(NULL, dev);
    r1.eval_subexpr_if_needed(l.data(), dev);
    r2.eval_subexpr_if_needed(l.data(), dev);
    return true;
  }

  value_type eval(size_t i) {
    auto dimC = r1.getSizeC();

    auto val =
        functor_traits<iniAddOp1_struct, value_type, Device>::eval(r2.eval(0));
    for (size_t j = 0; j < dimC; j++) {
      val += r1.eval(i, j);
    }
    l.eval(i) = scl * val + r2.eval(i);
    return val;
  }

  value_type eval(cl::sycl::nd_item<1> ndItem) {
    return eval(ndItem.get_global(0));
  }
};

/*! RedRowMatVct.
 * @brief CLASSICAL AXPY GEMV
 */
template <class RHS1, class RHS2, typename Device>
struct Evaluator<RedRowMatVctExpr<RHS1, RHS2>, Device> {
  using Expression = RedRowMatVctExpr<RHS1, RHS2>;
  using value_type = typename Expression::value_type;
  using cont_type = typename Evaluator<RHS1, Device>::cont_type;
  /* static constexpr bool supported = RHS1::supported && RHS2::supported; */

  value_type scl;
  Evaluator<RHS1, Device> r1;
  Evaluator<RHS2, Device> r2;

  Evaluator(Expression &expr)
      : scl(expr.scl),
        r1(Evaluator<RHS1, Device>(expr.r1)),
        r2(Evaluator<RHS2, Device>(expr.r2)) {}

  size_t getSize() const { return r1.getSizeR(); }
  cont_type data() { return r1.data(); }

  bool eval_subexpr_if_needed(cont_type *cont, Device &dev) {
    r1.eval_subexpr_if_needed(NULL, dev);
    r2.eval_subexpr_if_needed(NULL, dev);
    return true;
  }

  value_type eval(size_t i) {
    auto dim = r2.getSize();
    auto valWG =
        functor_traits<iniAddOp1_struct, value_type, Device>::eval(r2.eval(0));
    for (size_t j = 0; j < dim; j++) {
      valWG += r1.eval(i, j) * r2.eval(j);
    }
    return valWG;
  }

  value_type eval(cl::sycl::nd_item<1> ndItem) {
    return eval(ndItem.get_global(0));
  }

#if BLAS_EXPERIMENTAL
  template <typename sharedT>
  value_type eval(sharedT scratch, cl::sycl::nd_item<1> ndItem) {
    size_t Pieces = 2;

    size_t localid = ndItem.get_local(0);
    size_t localSz = ndItem.get_local_range(0);
    size_t groupid = ndItem.get_group(0);
    size_t groupSz = ndItem.get_num_groups(0);
    size_t globalid = ndItem.get_global(0);
    size_t globalSz = ndItem.get_global_range(0);

    size_t dimR = r1.getSizeR();
    size_t dimC = r1.getSizeC();

    size_t blqSz = groupSz;  // number of workgroups
    // row blq id of the current workgroup
    size_t blqidR = (groupid + (Pieces * blqSz) - 1) / (Pieces * blqSz);
    size_t blqidC =
        groupid % (Pieces * blqSz);  // 1st col id of the current workgroup

    // number of columns per each workgroup
    size_t colSz = (dimC < (Pieces * localSz)) ? dimC : Pieces * localSz;
    // number of rows per each thread
    size_t rowSz = (dimR + blqidR - 1) / blqidR;

    size_t colid = blqidC * colSz + localid;  // colid of the current thread
    size_t rowid = blqidR * rowSz;            // first row of the current thread

    value_type val;
#if BLAS_EXPERIMENTAL
    // Local computations
    while (rowid < dimR) {
      auto val = functor_traits<iniAddOp1_struct, value_type, Device>::eval(
          r2.eval(0));
      for (size_t j = colid; j < dimC; j += colSz) {
        val += r1.eval(rowid, j) * r2.eval(j);
      }
      scratch[localid] = val;
      // This barrier is mandatory to be sure the data is on the shared memory
      ndItem.barrier(cl::sycl::access::fence_space::local_space);
      // Reduction inside the block
      for (size_t offset = nThr >> 1; offset > 0; offset >>= 1) {
        if ((rowid < dimR) && (colid < offset)) {
          scratch[localid] += scratch[localid + offset];
        }
        // This barrier is mandatory to be sure the data are on the shared
        // memory
        ndItem.barrier(cl::sycl::access::fence_space::local_space);
      }
      // The result is stored in lhs
      if ((rowid < dimR) && (colid == 0)) {
        l.eval(rowid, blqidC) = scl * scratch[localid] + r3.eval(rowid);
      }
      rowid += rowSz;
    }
#endif  // BLAS_EXPERIMENTAL
    return val;
  }
#endif  // BLAS_EXPERIMENTAL

#if BLAS_EXPERIMENTAL
  value_type eval(cl::sycl::nd_item<1> ndItem) {
    return eval(ndItem.get_global(0));
  }
#endif  // BLAS_EXPERIMENTAL
};

/*! ModifRank1.
 * @brief RANK 1 UPDATE
 */
template <class RHS1, class RHS2, class RHS3, typename Device>
struct Evaluator<ModifRank1Expr<RHS1, RHS2, RHS3>, Device> {
  using Expression = ModifRank1Expr<RHS1, RHS2, RHS3>;
  using value_type = typename Expression::value_type;
  using cont_type = typename Evaluator<RHS1, Device>::cont_type;
  /* static constexpr bool supported = RHS1::supported && RHS2::supported &&
   * RHS3::supported; */

  Evaluator<RHS1, Device> r1;
  Evaluator<RHS2, Device> r2;
  Evaluator<RHS3, Device> r3;

  Evaluator(Expression &expr)
      : r1(Evaluator<RHS1, Device>(expr.r1)),
        r2(Evaluator<RHS2, Device>(expr.r2)),
        r3(Evaluator<RHS3, Device>(expr.r3)) {}

  size_t getSize() const { return r1.getSizeR(); }
  cont_type data() { return r1.data(); }

  bool eval_subexpr_if_needed(cont_type *cont, Device &dev) {
    r1.eval_subexpr_if_needed(NULL, dev);
    r2.eval_subexpr_if_needed(NULL, dev);
    r3.eval_subexpr_if_needed(NULL, dev);
    return true;
  }

  value_type eval(size_t i) {
    auto size = (r1.getAccess()) ? r1.getSizeC() : r1.getSizeR();
    auto row = (r1.getAccess()) ? (i / size) : (i % size);
    auto col = (r1.getAccess()) ? (i % size) : (i / size);

    auto val = r2.eval(row) * r3.eval(col);

    return val;
  }

  value_type eval(cl::sycl::nd_item<1> ndItem) {
    return eval(ndItem.get_global(0));
  }
};

}  // namespace blas

#endif  // BLAS2_TREES_HPP
