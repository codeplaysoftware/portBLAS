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
 *  @filename ger.hpp
 *
 **************************************************************************/

#ifndef GER_HPP
#define GER_HPP

#include "../blas_operators.hpp"
#include "operations/blas2_trees.h"
#include <stdexcept>
#include <vector>
#include <views/view_sycl.hpp>

namespace blas {

/**** GER BY ROWS M ROWS x N BLOCK USING PROPERLY THE SHARED MEMORY ****/
// template <typename lhs_t,  typename rhs_1_t, typename  rhs_2_t>
template <bool Single, bool Lower, bool Diag, bool Upper, typename lhs_t,
          typename rhs_1_t, typename rhs_2_t>
SYCL_BLAS_INLINE GerRow<Single, Lower, Diag, Upper, lhs_t, rhs_1_t,
                        rhs_2_t>::GerRow(lhs_t &_l, value_t _scl, rhs_1_t &_r1,
                                         rhs_2_t &_r2, index_t &_nWG_row,
                                         index_t &_nWG_col,
                                         index_t &_shrMemSize)
    : lhs_(_l),
      scalar_(_scl),
      rhs_1_(_r1),
      rhs_2_(_r2),
      nWG_row_(_nWG_row),
      nWG_col_(_nWG_col),
      local_memory_size_(_shrMemSize) {}

template <bool Single, bool Lower, bool Diag, bool Upper, typename lhs_t,
          typename rhs_1_t, typename rhs_2_t>
SYCL_BLAS_INLINE typename GerRow<Single, Lower, Diag, Upper, lhs_t, rhs_1_t,
                                 rhs_2_t>::index_t
GerRow<Single, Lower, Diag, Upper, lhs_t, rhs_1_t, rhs_2_t>::get_size() const {
  return rhs_1_.get_size();
}
template <bool Single, bool Lower, bool Diag, bool Upper, typename lhs_t,
          typename rhs_1_t, typename rhs_2_t>
SYCL_BLAS_INLINE bool
GerRow<Single, Lower, Diag, Upper, lhs_t, rhs_1_t, rhs_2_t>::valid_thread(
    cl::sycl::nd_item<1> ndItem) const {
  return true;
}

template <bool Single, bool Lower, bool Diag, bool Upper, typename lhs_t,
          typename rhs_1_t, typename rhs_2_t>
SYCL_BLAS_INLINE typename GerRow<Single, Lower, Diag, Upper, lhs_t, rhs_1_t,
                                 rhs_2_t>::value_t
GerRow<Single, Lower, Diag, Upper, lhs_t, rhs_1_t, rhs_2_t>::eval(
    typename GerRow<Single, Lower, Diag, Upper, lhs_t, rhs_1_t,
                    rhs_2_t>::index_t i) {
  auto size =
      (lhs_.is_row_access()) ? lhs_.get_size_col() : lhs_.get_size_row();
  auto row = (lhs_.is_row_access()) ? (i / size) : (i % size);
  auto col = (lhs_.is_row_access()) ? (i % size) : (i / size);

  auto val = scalar_ * rhs_1_.eval(row) * rhs_2_.eval(col);

  return lhs_.eval(i) += val;
}

template <bool Single, bool Lower, bool Diag, bool Upper, typename lhs_t,
          typename rhs_1_t, typename rhs_2_t>
SYCL_BLAS_INLINE typename GerRow<Single, Lower, Diag, Upper, lhs_t, rhs_1_t,
                                 rhs_2_t>::value_t
GerRow<Single, Lower, Diag, Upper, lhs_t, rhs_1_t, rhs_2_t>::eval(
    cl::sycl::nd_item<1> ndItem) {
  using index_t = typename GerRow<Single, Lower, Diag, Upper, lhs_t, rhs_1_t,
                                  rhs_2_t>::index_t;
  index_t localid = ndItem.get_local_id(0);
  index_t localSz = ndItem.get_local_range(0);
  index_t groupid = ndItem.get_group(0);

  index_t dimR = lhs_.get_size_row();
  index_t dimC = lhs_.get_size_col();

  index_t rowSz = (dimR + nWG_row_ - 1) / nWG_row_;

  index_t idWFR = (groupid % nWG_row_);
  index_t idWFC = (groupid / nWG_row_);
  index_t dimWFC =
      (dimC + (localSz * nWG_col_) - 1) / (localSz * nWG_col_) * localSz;

  index_t frs_row = idWFR * rowSz;
  index_t lst_row = std::min(dimR, frs_row + rowSz);

  index_t frs_col = idWFC * dimWFC + localid;
  index_t lst_col = std::min(dimC, frs_col + dimWFC);

  // PROBLEM IF ONLY SOME THREADS OF A WORKGROUP ARE CANCELED
  // TO SOLVE IT, USE GLOBAL VALUES OF frs_col AND lst_col
  if ((!Upper && (((idWFC * dimWFC) + ((!Diag) ? 1 : 0)) > (lst_row - 1))) ||
      (!Lower &&
       ((frs_row + ((!Diag) ? 1 : 0)) > ((idWFC * dimWFC + dimWFC) - 1)))) {
    ;
  } else if (Single) {
    for (index_t colid = frs_col; colid < lst_col; colid += localSz) {
      auto val = scalar_ * rhs_2_.eval(colid);
      for (index_t id_row = frs_row, row = 0; id_row < lst_row;
           id_row++, row++) {
        if (Lower && Upper && Diag) {
          lhs_.eval(id_row, colid) += rhs_1_.eval(id_row) * val;
        } else {
          if ((Lower && ((colid + ((!Diag) ? 1 : 0)) <= id_row)) ||
              (Upper && (colid >= (id_row + ((!Diag) ? 1 : 0))))) {
            lhs_.eval(id_row, colid) += rhs_1_.eval(id_row) * val;
          }
        }
      }
    }
  } else {
    for (index_t colid = frs_col; colid < lst_col; colid += localSz) {
      auto val1 = scalar_ * rhs_1_.eval(colid);
      auto val2 = scalar_ * rhs_2_.eval(colid);
      for (index_t id_row = frs_row, row = 0; id_row < lst_row;
           id_row++, row++) {
        if (Lower && Upper && Diag) {
          lhs_.eval(id_row, colid) +=
              rhs_1_.eval(id_row) * val2 + val1 * rhs_2_.eval(id_row);
        } else {
          if ((Lower && ((colid + ((!Diag) ? 1 : 0)) <= id_row)) ||
              (Upper && (colid >= (id_row + ((!Diag) ? 1 : 0))))) {
            lhs_.eval(id_row, colid) +=
                rhs_1_.eval(id_row) * val2 + rhs_2_.eval(id_row) * val1;
          }
        }
      }
    }
  }

  return lhs_.eval(frs_row, frs_col);
}

template <bool Single, bool Lower, bool Diag, bool Upper, typename lhs_t,
          typename rhs_1_t, typename rhs_2_t>
template <typename sharedT>
SYCL_BLAS_INLINE typename GerRow<Single, Lower, Diag, Upper, lhs_t, rhs_1_t,
                                 rhs_2_t>::value_t
GerRow<Single, Lower, Diag, Upper, lhs_t, rhs_1_t, rhs_2_t>::eval(
    sharedT shrMem, cl::sycl::nd_item<1> ndItem) {
  using index_t = typename GerRow<Single, Lower, Diag, Upper, lhs_t, rhs_1_t,
                                  rhs_2_t>::index_t;
  index_t localid = ndItem.get_local_id(0);
  index_t localSz = ndItem.get_local_range(0);
  index_t groupid = ndItem.get_group(0);

  index_t dimR = lhs_.get_size_row();
  index_t dimC = lhs_.get_size_col();

  index_t rowSz = (dimR + nWG_row_ - 1) / nWG_row_;
  index_t shrSz = local_memory_size_;

  index_t idWFR = (groupid % nWG_row_);
  index_t idWFC = (groupid / nWG_row_);
  index_t dimWFC =
      (dimC + (localSz * nWG_col_) - 1) / (localSz * nWG_col_) * localSz;

  index_t frs_row = idWFR * rowSz;
  index_t lst_row = std::min(dimR, frs_row + rowSz);

  index_t frs_col = idWFC * dimWFC + localid;
  index_t lst_col = std::min(dimC, frs_col + dimWFC);
  // PROBLEM IF ONLY SOME THREADS OF A WORKGROUP ARE CANCELED
  // TO SOLVE IT, USE GLOBAL VALUES OF frs_col AND lst_col
  if ((!Upper && (((idWFC * dimWFC) + ((!Diag) ? 1 : 0)) > (lst_row - 1))) ||
      (!Lower &&
       ((frs_row + ((!Diag) ? 1 : 0)) > ((idWFC * dimWFC + dimWFC) - 1)))) {
    ;
  } else if (Single) {
    for (index_t rowid = frs_row; rowid < lst_row; rowid += shrSz) {
      if (rowid > frs_row)
        // This barrier is mandatory to be sure the data is on the shared
        // memory
        ndItem.barrier(cl::sycl::access::fence_space::local_space);
      auto blqSz = std::min(shrSz, lst_row - rowid);
      for (index_t row = localid, id_row = rowid + localid; (row < blqSz);
           row += localSz, id_row += localSz) {
        shrMem[row] = scalar_ * rhs_1_.eval(id_row);
      }

      // This barrier is mandatory to be sure the data is on the shared memory
      ndItem.barrier(cl::sycl::access::fence_space::local_space);

      for (index_t colid = frs_col; (colid < lst_col); colid += localSz) {
        auto val = rhs_2_.eval(colid);
        for (index_t id_row = rowid, row = 0; row < blqSz; id_row++, row++) {
          if (Lower && Upper && Diag) {
            lhs_.eval(id_row, colid) += shrMem[row] * val;
          } else {
            if ((Lower && ((colid + ((!Diag) ? 1 : 0)) <= id_row)) ||
                (Upper && (colid >= (id_row + ((!Diag) ? 1 : 0))))) {
              lhs_.eval(id_row, colid) += shrMem[row] * val;
            }
          }
        }
      }
    }
  } else {
    auto shrSz1 = (shrSz >> 1);
    for (index_t rowid = frs_row; rowid < lst_row; rowid += shrSz) {
      if (rowid > frs_row)
        // This barrier is mandatory to be sure the data is on the shared
        // memory
        ndItem.barrier(cl::sycl::access::fence_space::local_space);
      auto blqSz = std::min(shrSz1, lst_row - rowid);
      for (index_t row = localid, id_row = rowid + localid; (row < blqSz);
           row += localSz, id_row += localSz) {
        shrMem[row] = scalar_ * rhs_1_.eval(id_row);
        shrMem[shrSz1 + row] = scalar_ * rhs_2_.eval(id_row);
      }

      // This barrier is mandatory to be sure the data is on the shared memory
      ndItem.barrier(cl::sycl::access::fence_space::local_space);

      for (index_t colid = frs_col; (colid < lst_col); colid += localSz) {
        auto val1 = rhs_1_.eval(colid);
        auto val2 = rhs_2_.eval(colid);
        for (index_t id_row = rowid, row = 0; row < blqSz; id_row++, row++) {
          if (Lower && Upper && Diag) {
            lhs_.eval(id_row, colid) +=
                shrMem[row] * val2 + shrMem[shrSz1 + row] * val1;
          } else {
            if ((Lower && ((colid + ((!Diag) ? 1 : 0)) <= id_row)) ||
                (Upper && (colid >= (id_row + ((!Diag) ? 1 : 0))))) {
              lhs_.eval(id_row, colid) +=
                  shrMem[row] * val2 + shrMem[shrSz1 + row] * val1;
            }
          }
        }
      }
    }
  }

  return shrMem[0];
}
template <bool Single, bool Lower, bool Diag, bool Upper, typename lhs_t,
          typename rhs_1_t, typename rhs_2_t>
SYCL_BLAS_INLINE void GerRow<Single, Lower, Diag, Upper, lhs_t, rhs_1_t,
                             rhs_2_t>::bind(cl::sycl::handler &h) {
  lhs_.bind(h);
  rhs_1_.bind(h);
  rhs_2_.bind(h);
}

/**** GER BY COLUMNS M ROWS x N BLOCK USING PROPERLY THE SHARED MEMORY ****/
// template <typename lhs_t,  typename rhs_1_t, typename  rhs_2_t>
template <bool Single, bool Lower, bool Diag, bool Upper, typename lhs_t,
          typename rhs_1_t, typename rhs_2_t>
SYCL_BLAS_INLINE GerCol<Single, Lower, Diag, Upper, lhs_t, rhs_1_t,
                        rhs_2_t>::GerCol(lhs_t &_l, value_t _scl, rhs_1_t &_r1,
                                         rhs_2_t &_r2, index_t &_nWG_row,
                                         index_t &_nWG_col,
                                         index_t &_shrMemSize)
    : lhs_(_l),
      scalar_(_scl),
      rhs_1_(_r1),
      rhs_2_(_r2),
      nWG_row_(_nWG_row),
      nWG_col_(_nWG_col),
      local_memory_size_(_shrMemSize) {}

template <bool Single, bool Lower, bool Diag, bool Upper, typename lhs_t,
          typename rhs_1_t, typename rhs_2_t>
SYCL_BLAS_INLINE typename GerCol<Single, Lower, Diag, Upper, lhs_t, rhs_1_t,
                                 rhs_2_t>::index_t
GerCol<Single, Lower, Diag, Upper, lhs_t, rhs_1_t, rhs_2_t>::get_size() const {
  return rhs_1_.get_size();
}
template <bool Single, bool Lower, bool Diag, bool Upper, typename lhs_t,
          typename rhs_1_t, typename rhs_2_t>
SYCL_BLAS_INLINE bool
GerCol<Single, Lower, Diag, Upper, lhs_t, rhs_1_t, rhs_2_t>::valid_thread(
    cl::sycl::nd_item<1> ndItem) const {
  return true;
}
template <bool Single, bool Lower, bool Diag, bool Upper, typename lhs_t,
          typename rhs_1_t, typename rhs_2_t>
SYCL_BLAS_INLINE typename GerCol<Single, Lower, Diag, Upper, lhs_t, rhs_1_t,
                                 rhs_2_t>::value_t
GerCol<Single, Lower, Diag, Upper, lhs_t, rhs_1_t, rhs_2_t>::eval(
    typename GerCol<Single, Lower, Diag, Upper, lhs_t, rhs_1_t,
                    rhs_2_t>::index_t i) {
  auto size =
      (lhs_.is_row_access()) ? lhs_.get_size_col() : lhs_.get_size_row();
  auto row = (lhs_.is_row_access()) ? (i / size) : (i % size);
  auto col = (lhs_.is_row_access()) ? (i % size) : (i / size);

  auto val = scalar_ * rhs_1_.eval(row) * rhs_2_.eval(col);

  return lhs_.eval(i) += val;
}

template <bool Single, bool Lower, bool Diag, bool Upper, typename lhs_t,
          typename rhs_1_t, typename rhs_2_t>
SYCL_BLAS_INLINE typename GerCol<Single, Lower, Diag, Upper, lhs_t, rhs_1_t,
                                 rhs_2_t>::value_t
GerCol<Single, Lower, Diag, Upper, lhs_t, rhs_1_t, rhs_2_t>::eval(
    cl::sycl::nd_item<1> ndItem) {
  using index_t = typename GerCol<Single, Lower, Diag, Upper, lhs_t, rhs_1_t,
                                  rhs_2_t>::index_t;
  index_t localid = ndItem.get_local_id(0);
  index_t localSz = ndItem.get_local_range(0);
  index_t groupid = ndItem.get_group(0);

  index_t dimR = lhs_.get_size_row();
  index_t dimC = lhs_.get_size_col();

  index_t colSz = (dimR < localSz) ? localSz : (dimC + nWG_col_ - 1) / nWG_col_;

  index_t idWFR = groupid % nWG_row_;  // row bloq id of the current workgroup
  index_t idWFC = groupid / nWG_row_;  // col blq id of the current workgroup
  index_t dimWFR =
      (dimR + (localSz * nWG_row_) - 1) / (localSz * nWG_row_) * localSz;

  index_t frs_row = idWFR * dimWFR + localid;
  index_t lst_row = std::min(dimR, frs_row + dimWFR);

  index_t frs_col = idWFC * colSz;
  index_t lst_col = std::min(dimC, frs_col + colSz);
  if ((!Upper &&
       ((frs_col + ((!Diag) ? 1 : 0)) > ((idWFR * dimWFR + dimWFR) - 1))) ||
      (!Lower && ((idWFR * dimWFR + ((!Diag) ? 1 : 0)) > (lst_col - 1)))) {
    ;
  } else if (Single) {
    for (index_t id_row = frs_row; id_row < lst_row; id_row += localSz) {
      auto val = scalar_ * rhs_1_.eval(id_row);
      for (index_t id_col =
               ((Lower) ? frs_col
                        : std::max(id_row + ((!Diag) ? 1 : 0), frs_col));
           id_col <
           ((Upper) ? lst_col : std::min(id_row + ((!Diag) ? 0 : 1), lst_col));
           id_col++) {
        lhs_.eval(id_row, id_col) += val * rhs_2_.eval(id_col);
      }
    }
  } else {
    for (index_t id_row = frs_row; id_row < lst_row; id_row += localSz) {
      auto val1 = scalar_ * rhs_1_.eval(id_row);
      auto val2 = scalar_ * rhs_2_.eval(id_row);
      for (index_t id_col =
               ((Lower) ? frs_col
                        : std::max(id_row + ((!Diag) ? 1 : 0), frs_col));
           id_col <
           ((Upper) ? lst_col : std::min(id_row + ((!Diag) ? 0 : 1), lst_col));
           id_col++) {
        lhs_.eval(id_row, id_col) +=
            val1 * rhs_2_.eval(id_col) + val2 * rhs_1_.eval(id_col);
      }
    }
  }

  return lhs_.eval(frs_row, frs_col);
}
template <bool Single, bool Lower, bool Diag, bool Upper, typename lhs_t,
          typename rhs_1_t, typename rhs_2_t>
template <typename sharedT>
SYCL_BLAS_INLINE typename GerCol<Single, Lower, Diag, Upper, lhs_t, rhs_1_t,
                                 rhs_2_t>::value_t
GerCol<Single, Lower, Diag, Upper, lhs_t, rhs_1_t, rhs_2_t>::eval(
    sharedT shrMem, cl::sycl::nd_item<1> ndItem) {
  using index_t = typename GerCol<Single, Lower, Diag, Upper, lhs_t, rhs_1_t,
                                  rhs_2_t>::index_t;
  index_t localid = ndItem.get_local_id(0);
  index_t localSz = ndItem.get_local_range(0);
  index_t groupid = ndItem.get_group(0);

  index_t dimR = lhs_.get_size_row();
  index_t dimC = lhs_.get_size_col();

  index_t colSz = (dimR < localSz) ? localSz : (dimC + nWG_col_ - 1) / nWG_col_;

  index_t idWFR = groupid % nWG_row_;  // row bloq id of the current workgroup
  index_t dimWFR =
      (dimR + (localSz * nWG_row_) - 1) / (localSz * nWG_row_) * localSz;

  index_t frs_row = idWFR * dimWFR + localid;
  index_t lst_row = std::min(dimR, frs_row + dimWFR);

  index_t frs_col = (groupid / nWG_row_) * colSz;
  index_t lst_col = std::min(dimC, frs_col + colSz);
  // PROBLEM IF ONLY SOME THREADS OF A WORKGROUP ARE CANCELED
  // TO SOLVE IT, USE GLOBAL VALUES OF frs_row AND lst_row
  if ((!Upper &&
       ((frs_col + ((!Diag) ? 1 : 0)) > ((idWFR * dimWFR + dimWFR) - 1))) ||
      (!Lower && ((idWFR * dimWFR + ((!Diag) ? 1 : 0)) > (lst_col - 1)))) {
    ;
  } else if (Single) {
    // The computation are made in blocks of local_memory_size_ elements
    for (index_t colid = frs_col; colid < lst_col;
         colid += local_memory_size_) {
      if (colid > frs_col) {
        // This barrier is mandatory to be sure the data is on the shared
        // memory
        ndItem.barrier(cl::sycl::access::fence_space::local_space);
      }
      auto blqSz = std::min(local_memory_size_, lst_col - colid);

      for (index_t col = localid; (col < blqSz); col += localSz) {
        shrMem[col] = scalar_ * rhs_2_.eval(colid + col);
      }

      // This barrier is mandatory to be sure the data is on the shared memory
      ndItem.barrier(cl::sycl::access::fence_space::local_space);

      for (index_t id_row = frs_row; id_row < lst_row; id_row += localSz) {
        auto val = rhs_1_.eval(id_row);
        for (index_t id_col = colid, col = 0; col < blqSz; id_col++, col++) {
          if (Lower && Upper && Diag) {
            lhs_.eval(id_row, id_col) += val * shrMem[col];
          } else {
            if ((Lower && ((id_col + ((!Diag) ? 1 : 0)) <= id_row)) ||
                (Upper && (id_col >= (id_row + ((!Diag) ? 1 : 0))))) {
              lhs_.eval(id_row, id_col) += val * shrMem[col];
            }
          }
        }
      }
    }
  } else {
    auto shrSz1 = (local_memory_size_ >> 1);
    // The computation are made in blocks of local_memory_size_/shrSz1 elements
    for (index_t colid = frs_col; colid < lst_col; colid += shrSz1) {
      if (colid > frs_col) {
        // This barrier is mandatory to be sure the data is on the shared
        // memory
        ndItem.barrier(cl::sycl::access::fence_space::local_space);
      }
      auto blqSz = std::min(shrSz1, lst_col - colid);

      for (index_t col = localid; (col < blqSz); col += localSz) {
        shrMem[col] = scalar_ * rhs_1_.eval(colid + col);
        shrMem[shrSz1 + col] = scalar_ * rhs_2_.eval(colid + col);
      }

      // This barrier is mandatory to be sure the data is on the shared memory
      ndItem.barrier(cl::sycl::access::fence_space::local_space);

      for (index_t id_row = frs_row; id_row < lst_row; id_row += localSz) {
        auto val1 = rhs_1_.eval(id_row);
        auto val2 = rhs_2_.eval(id_row);
        for (index_t id_col = colid, col = 0; col < blqSz; id_col++, col++) {
          if (Lower && Upper && Diag) {
            lhs_.eval(id_row, id_col) +=
                val1 * shrMem[shrSz1 + col] + val2 * shrMem[col];
          } else {
            if ((Lower && ((id_col + ((!Diag) ? 1 : 0)) <= id_row)) ||
                (Upper && (id_col >= (id_row + ((!Diag) ? 1 : 0))))) {
              lhs_.eval(id_row, id_col) +=
                  val1 * shrMem[shrSz1 + col] + val2 * shrMem[col];
            }
          }
        }
      }
    }
  }

  return shrMem[0];
}
template <bool Single, bool Lower, bool Diag, bool Upper, typename lhs_t,
          typename rhs_1_t, typename rhs_2_t>
SYCL_BLAS_INLINE void GerCol<Single, Lower, Diag, Upper, lhs_t, rhs_1_t,
                             rhs_2_t>::bind(cl::sycl::handler &h) {
  lhs_.bind(h);
  rhs_1_.bind(h);
  rhs_2_.bind(h);
}

}  // namespace blas

#endif
