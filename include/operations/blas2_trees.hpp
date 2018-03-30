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
 *  @filename blas2_trees.hpp
 *
 **************************************************************************/

#ifndef BLAS2_TREES_HPP
#define BLAS2_TREES_HPP

#include <stdexcept>
#include <vector>

#include <operations/blas2_trees.hpp>
#include <operations/blas_operators.hpp>
#include <views/view_sycl.hpp>

namespace blas {

/*! PrdRowMatVct.
 * @brief CLASSICAL DOT PRODUCT GEMV
 * Each thread computes a dot product, If
 * the matrix is column-major the accesses are coalescent.
 */
template <class RHS1, class RHS2>
struct PrdRowMatVct {
  using IndexType = typename RHS2::IndexType;
  using value_type = typename RHS2::value_type;

  RHS1 r1;
  RHS2 r2;
  IndexType mult;

  PrdRowMatVct(RHS1 &_r1, RHS2 &_r2) : r1(_r1), r2(_r2){};

  value_type eval(IndexType i) {
    auto dim = r2.getSize();

    auto val = iniAddOp1_struct::eval(r2.eval(0));
    for (IndexType j = 0; j < dim; j++) {
      val += r1.eval(i, j) * r2.eval(j);
    }
    return val;
  }

  value_type eval(cl::sycl::nd_item<1> ndItem) {
    return eval(ndItem.get_global(0));
  }
  IndexType getSize() { return r1.getSizeR(); }
};

template <class RHS1, class RHS2>
PrdRowMatVct<RHS1, RHS2> make_prdRowMatVct(RHS1 &r1, RHS2 &r2) {
  return PrdRowMatVct<RHS1, RHS2>(r1, r2);
}

/** PrdRowMatVctMult
 * @brief MULTITHREAD DOT PRODUCT GEMV
 * P threads compute a dot product
 * If the matrix is column-major the accesses are coalescent.
 */
template <class LHS, class RHS1, class RHS2, class RHS3>
struct PrdRowMatVctMult {
  using value_type = typename RHS2::value_type;
  using IndexType = typename RHS2::IndexType;

  LHS l;
  value_type scl;

  RHS1 r1;
  RHS2 r2;
  RHS3 r3;
  IndexType nThr;

  PrdRowMatVctMult(LHS &_l, value_type _scl, RHS1 &_r1, RHS2 &_r2, RHS3 &_r3,
                   IndexType _nThr)
      : l(_l), scl(_scl), r1(_r1), r2(_r2), r3(_r3), nThr{_nThr} {};

  value_type eval(IndexType i) {
    auto dim = r2.getSize();

    auto val = iniAddOp1_struct::eval(r2.eval(0));
    for (IndexType j = 0; j < dim; j++) {
      val += r1.eval(i, j) * r2.eval(j);
    }
    l.eval(i) += scl * val;
    return val;
  }

  template <typename sharedT>
  value_type eval(sharedT scratch, cl::sycl::nd_item<1> ndItem) {
    IndexType localid = ndItem.get_local(0);
    IndexType localSz = ndItem.get_local_range(0);
    IndexType groupid = ndItem.get_group(0);

    IndexType dimR = r1.getSizeR();
    IndexType dimC = r1.getSizeC();

    IndexType rowSz = (localSz / nThr);  // number of rows per each workgroup
    IndexType rowid = groupid * rowSz + localid % rowSz;  // rowid of the thread

    IndexType colid = localid / rowSz;  // first column on which thread works

    // Local computations
    auto val = iniAddOp1_struct::eval(r2.eval(0));
    if (rowid < dimR) {
      for (IndexType j = colid; j < dimC; j += nThr) {
        val += r1.eval(rowid, j) * r2.eval(j);
      }
    }

    scratch[localid] = val;
    // This barrier is mandatory to be sure the data is on the shared memory
    ndItem.barrier(cl::sycl::access::fence_space::local_space);

    // Reduction inside the block
    for (IndexType offset = nThr >> 1; offset > 0; offset >>= 1) {
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

  IndexType getSize() { return r1.getSizeR(); }
};

template <class LHS, class RHS1, class RHS2, class RHS3, typename IndexType>
PrdRowMatVctMult<LHS, RHS1, RHS2, RHS3> make_prdRowMatVctMult(
    LHS &l, typename LHS::value_type scl, RHS1 &r1, RHS2 &r2, RHS3 &r3,
    IndexType nThr) {
  return PrdRowMatVctMult<LHS, RHS1, RHS2, RHS3>(l, scl, r1, r2, r3, nThr);
}

/*! PrdRowMatCvtMultShm.
 * @brief TWO KERNELS DOT PRODUCT GEMV
 * FIRST KERNEL: THE LOCAL COMPUTATIONS ARE MADE
 * The common data are copied to the scratch vector,
 * and later the computation begins.
 */
template <class LHS, class RHS1, class RHS2>
struct PrdRowMatVctMultShm {
  using IndexType = typename RHS2::IndexType;
  using value_type = typename RHS2::value_type;
  LHS l;
  RHS1 r1;
  RHS2 r2;
  IndexType nThr;

  PrdRowMatVctMultShm(LHS &_l, RHS1 &_r1, RHS2 &_r2, IndexType _nThr)
      : l(_l), r1(_r1), r2(_r2), nThr{_nThr} {};

  value_type eval(IndexType i) {
    auto dim = r2.getSize();

    auto val = iniAddOp1_struct::eval(r2.eval(0));
    for (IndexType j = 0; j < dim; j++) {
      val += r1.eval(i, j) * r2.eval(j);
    }
    l.eval(i) += val;
    return val;
  }

  template <typename sharedT>
  value_type eval(sharedT scratch, cl::sycl::nd_item<1> ndItem) {
    IndexType localid = ndItem.get_local(0);
    IndexType localSz = ndItem.get_local_range(0);
    IndexType groupid = ndItem.get_group(0);
    IndexType groupSz = ndItem.get_num_groups(0);

    IndexType dimR = r1.getSizeR();
    IndexType dimC = r1.getSizeC();

    IndexType blqSz =
        (groupSz + nThr - 1) / nThr;     // number of "real" workgroups
    IndexType blqidR = groupid % blqSz;  // 1st row id of the current workgroup
    IndexType blqidC = groupid / blqSz;  // col bloq id of the current workgroup

    IndexType rowSz =
        (dimR < localSz) ? dimR : localSz;  // number of rows per each workgroup
    IndexType colSz =
        (dimC + nThr - 1) / nThr;  // number of columns per each thread

    IndexType rowid = blqidR * rowSz + localid;  // rowid of the current thread
    IndexType colid = blqidC * colSz;  // first column of the current thread

    IndexType k;

    // Copying  to the scratch
    k = localid;
    for (IndexType j = colid + localid; j < colid + colSz; j += rowSz) {
      if ((rowid < dimR) && (j < dimC)) scratch[k] = r2.eval(j);
      k += rowSz;
    }

    // This barrier is mandatory to be sure the data are on the shared memory
    ndItem.barrier(cl::sycl::access::fence_space::local_space);

    // Local computation
    auto val = iniAddOp1_struct::eval(r2.eval(0));
    k = 0;
    for (IndexType j = colid; j < colid + colSz; j++) {
      if ((rowid < dimR) && (j < dimC)) val += r1.eval(rowid, j) * scratch[k++];
    }
    // The result is stored in lhs
    if (rowid < dimR) l.eval(rowid, blqidC) = val;

    return val;
  }

  IndexType getSize() { return r1.getSizeR(); }
};

template <class LHS, class RHS1, class RHS2, typename IndexType>
PrdRowMatVctMultShm<LHS, RHS1, RHS2> make_prdRowMatVctMultShm(LHS &l, RHS1 &r1,
                                                              RHS2 &r2,
                                                              IndexType nThr) {
  return PrdRowMatVctMultShm<LHS, RHS1, RHS2>(l, r1, r2, nThr);
}

/*! AddPrdRowMatVctMultShm.
 * @brief SECOND KERNEL: REDUCTION OF THE LOCAL COMPUTATIONS
 */
template <class LHS, class RHS1, class RHS2>
struct AddPrdRowMatVctMultShm {
  using IndexType = typename RHS2::IndexType;
  using value_type = typename RHS2::value_type;
  LHS l;
  value_type scl;
  RHS1 r1;
  RHS2 r2;

  AddPrdRowMatVctMultShm(LHS &_l, value_type _scl, RHS1 &_r1, RHS2 &_r2)
      : l(_l), scl(_scl), r1(_r1), r2(_r2){};

  value_type eval(IndexType i) {
    auto dimC = r1.getSizeC();

    auto val = iniAddOp1_struct::eval(r2.eval(0));
    for (IndexType j = 0; j < dimC; j++) {
      val += r1.eval(i, j);
    }
    l.eval(i) = scl * val + r2.eval(i);
    return val;
  }

  value_type eval(cl::sycl::nd_item<1> ndItem) {
    return eval(ndItem.get_global(0));
  }

  IndexType getSize() { return r1.getSizeR(); }
};

template <class LHS, class RHS1, class RHS2>
AddPrdRowMatVctMultShm<LHS, RHS1, RHS2> make_addPrdRowMatVctMultShm(
    LHS &l, typename RHS1::value_type &scl, RHS1 &r1, RHS2 &r2) {
  return AddPrdRowMatVctMultShm<LHS, RHS1, RHS2>(l, scl, r1, r2);
}

/*! RedRowMatVct.
 * @brief CLASSICAL AXPY GEMV
 */
// #define ORIGINAL_CODE 1
template <class RHS1, class RHS2>
struct RedRowMatVct {
  using IndexType = typename RHS2::IndexType;
  using value_type = typename RHS2::value_type;
  RHS1 r1;
  RHS2 r2;
  IndexType warpSize;

  RedRowMatVct(RHS1 &_r1, RHS2 &_r2, IndexType _warpSize)
      : r1(_r1), r2(_r2), warpSize(_warpSize){};

#if ORIGINAL_CODE
  value_type eval(IndexType i) {
    auto dim = r2.getSize();
    value_type v[warpSize];
    for (IndexType w = 0; w < warpSize; w++) {
      auto valWI = iniAddOp1_struct::eval(r2.eval(0));
      for (IndexType j = w; j < dim; j += warpSize) {
        valWI += r1.eval(i, j) * r2.eval(j);
      }
      v[w] = valWI;
    }
    auto valWG = iniAddOp1_struct::eval(r2.eval(0));
    for (IndexType w = 0; w < warpSize; w++) {
      valWG += v[w];
    }
    return valWG;
  }
#else
  value_type eval(IndexType i) {
    auto dim = r2.getSize();
    auto valWG = iniAddOp1_struct::eval(r2.eval(0));
    for (IndexType j = 0; j < dim; j++) {
      valWG += r1.eval(i, j) * r2.eval(j);
    }
    return valWG;
  }
#endif  // ORIGINAL_CODE

  value_type eval(cl::sycl::nd_item<1> ndItem) {
    return eval(ndItem.get_global(0));
  }

#if BLAS_EXPERIMENTAL
  template <typename sharedT>
  value_type eval(sharedT scratch, cl::sycl::nd_item<1> ndItem) {
    IndexType Pieces = 2;

    IndexType localid = ndItem.get_local(0);
    IndexType localSz = ndItem.get_local_range(0);
    IndexType groupid = ndItem.get_group(0);
    IndexType groupSz = ndItem.get_num_groups(0);
    IndexType globalid = ndItem.get_global(0);
    IndexType globalSz = ndItem.get_global_range(0);

    IndexType dimR = r1.getSizeR();
    IndexType dimC = r1.getSizeC();

    IndexType blqSz = groupSz;  // number of workgroups
    // row blq id of the current workgroup
    IndexType blqidR = (groupid + (Pieces * blqSz) - 1) / (Pieces * blqSz);
    IndexType blqidC =
        groupid % (Pieces * blqSz);  // 1st col id of the current workgroup

    // number of columns per each workgroup
    IndexType colSz = (dimC < (Pieces * localSz)) ? dimC : Pieces * localSz;
    // number of rows per each thread
    IndexType rowSz = (dimR + blqidR - 1) / blqidR;

    IndexType colid = blqidC * colSz + localid;  // colid of the current thread
    IndexType rowid = blqidR * rowSz;  // first row of the current thread

    value_type val;
#if BLAS_EXPERIMENTAL
    // Local computations
    while (rowid < dimR) {
      auto val = iniAddOp1_struct::eval(r2.eval(0));
      for (IndexType j = colid; j < dimC; j += colSz) {
        val += r1.eval(rowid, j) * r2.eval(j);
      }
      scratch[localid] = val;
      // This barrier is mandatory to be sure the data is on the shared memory
      ndItem.barrier(cl::sycl::access::fence_space::local_space);
      // Reduction inside the block
      for (IndexType offset = nThr >> 1; offset > 0; offset >>= 1) {
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
  IndexType getSize() { return r1.getSizeR(); }
};

template <class RHS1, class RHS2, typename IndexType>
RedRowMatVct<RHS1, RHS2> make_redRowMatVct(RHS1 &r1, RHS2 &r2,
                                           IndexType warpSize) {
  return RedRowMatVct<RHS1, RHS2>(r1, r2, warpSize);
}

/*! ModifRank1.
 * @brief RANK 1 UPDATE
 */
template <class RHS1, class RHS2, class RHS3>
struct ModifRank1 {
  using IndexType = typename RHS2::IndexType;
  using value_type = typename RHS2::value_type;
  RHS1 r1;
  RHS2 r2;
  RHS3 r3;

  ModifRank1(RHS1 &_r1, RHS2 &_r2, RHS3 &_r3) : r1(_r1), r2(_r2), r3(_r3){};

  value_type eval(IndexType i) {
    auto size = (r1.getAccess()) ? r1.getSizeC() : r1.getSizeR();
    auto row = (r1.getAccess()) ? (i / size) : (i % size);
    auto col = (r1.getAccess()) ? (i % size) : (i / size);

    auto val = r2.eval(row) * r3.eval(col);

    return val;
  }

  value_type eval(cl::sycl::nd_item<1> ndItem) {
    return eval(ndItem.get_global(0));
  }

  IndexType getSize() { return r1.getSize(); }
};

template <class RHS1, class RHS2, class RHS3>
ModifRank1<RHS1, RHS2, RHS3> make_modifRank1(RHS1 &r1, RHS2 &r2, RHS3 &r3) {
  return ModifRank1<RHS1, RHS2, RHS3>(r1, r2, r3);
}

}  // namespace blas

#endif  // BLAS2_TREES_HPP
