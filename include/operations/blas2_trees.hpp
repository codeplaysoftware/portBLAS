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

/**** ADD A SET OF COLUMNS, 1 ROW PER THREAD ****/
template <class RHS>
struct AddSetColumns {
  using value_type = typename RHS::value_type;
  using IndexType = typename RHS::IndexType;

  RHS r;

  AddSetColumns(RHS &_r) : r(_r){};

  inline IndexType getSize() const { return r.getSizeR(); }

  inline bool valid_thread(cl::sycl::nd_item<1> ndItem) const {
    return ((ndItem.get_global_id(0) < getSize()));
  }

  value_type eval(IndexType i) {
    auto dimR = r.getSizeR();
    auto dimC = r.getSizeC();

    auto val = iniAddOp1_struct::eval(r.eval(0));
    if (i < dimR) {
      for (IndexType j = 0; j < dimC; j++) {
        val += r.eval(i, j);
      }
    }
    return val;
  }

  value_type eval(cl::sycl::nd_item<1> ndItem) {
    return eval(ndItem.get_global_id(0));
  }

  void bind(cl::sycl::handler &h) { r.bind(h); }
};

template <class RHS>
AddSetColumns<RHS> make_addSetColumns(RHS &r) {
  return AddSetColumns<RHS>(r);
}

/**** GEMV BY ROWS M ROWS x N BLOCK ****/

template <unsigned int interLoop, bool Lower, bool Diag, bool Upper, bool Unit,
          class LHS, class RHS1, class RHS2>
struct Gemv_Row {
  using value_type = typename RHS2::value_type;
  using IndexType = typename RHS2::IndexType;

  LHS l;
  RHS1 r1;
  RHS2 r2;
  IndexType nWG_row;
  IndexType nWG_col;
  IndexType shrMemSize;

  Gemv_Row(LHS &_l, RHS1 &_r1, RHS2 &_r2, IndexType &_nWG_row,
           IndexType &_nWG_col, IndexType &_shrMemSize)
      : l(_l),
        r1(_r1),
        r2(_r2),
        nWG_row(_nWG_row),
        nWG_col(_nWG_col),
        shrMemSize(_shrMemSize){};

  inline IndexType getSize() const { return r1.getSize(); }

  inline bool valid_thread(cl::sycl::nd_item<1> ndItem) const { return true; }

  // TODO (@JOSE) If this function is extra and it is not required please remove
  // it.
  value_type eval(IndexType i) {  // NOT VERIFIED
    auto dim = r2.getSize();

    auto val = iniAddOp1_struct::eval(r2.eval(0));
    for (IndexType j = 0; j < dim; j++) {
      auto prod = prdOp2_struct::eval(r1.eval(i, j), r2.eval(j));
      val = addOp2_struct::eval(val, prod);
    }
    return l.eval(i) = val;
  }

  value_type eval(cl::sycl::nd_item<1> ndItem) {
    IndexType localid = ndItem.get_local_id(0);
    IndexType localSz = ndItem.get_local_range(0);
    IndexType groupid = ndItem.get_group(0);

    IndexType dimR = r1.getSizeR();
    IndexType dimC = r1.getSizeC();

    IndexType rowSz = (dimR + nWG_row - 1) / nWG_row;

    IndexType idWFR =
        groupid / nWG_col;  // row bloq id of the current workgroup
    IndexType idWFC = groupid % nWG_col;  // col blq id of the current workgroup

    IndexType dimWFC =
        ((dimC + (localSz * nWG_col) - 1) / (localSz * nWG_col)) * localSz;

    IndexType frs_row = idWFR * rowSz;
    IndexType lst_row = std::min(dimR, frs_row + rowSz);

    IndexType frs_col = idWFC * dimWFC + interLoop * localid;
    IndexType lst_col = std::min(dimC, frs_col + dimWFC);

    IndexType id_col_thr = idWFC * localSz + localid;

    value_type val = addOp2_struct::init(r2);
    // PROBLEM IF ONLY SOME THREADS OF A WORKGROUP_OF ARE CANCELED
    // TO SOLVE IT, USE GLOBAL VALUES OF frs_col AND lst_col
    if ((!Upper && (((idWFC * dimWFC) + ((!Diag) ? 1 : 0)) > (lst_row - 1))) ||
        (!Lower &&
         ((frs_row + ((!Diag) ? 1 : 0)) > ((idWFC * dimWFC + dimWFC) - 1)))) {
      for (IndexType rowid = frs_row; rowid < lst_row; rowid += localSz) {
        l.eval(rowid, id_col_thr) = val;
      }
    } else {
#ifdef GROUP_OF_ROWS
      for (IndexType id_row = frs_row; (id_row < lst_row); id_row++) {
        l.eval(id_row, id_col_thr) = val;
      }
#endif
      if (interLoop == 1) {
#ifdef GROUP_OF_ROWS
        for (IndexType id_col = frs_col; id_col < lst_col; id_col += localSz) {
          auto elm = r2.eval(id_col);
          for (IndexType row = 0, id_row = frs_row; (id_row < lst_row);
               row++, id_row++) {
            if (Lower && Upper && Diag && !Unit) {
              auto prod = prdOp2_struct::eval(r1.eval(id_row, id_col), elm);
              l.eval(id_row, id_col_thr) =
                  addOp2_struct::eval(l.eval(id_row, id_col_thr), prod);
            } else {
              if ((Lower && ((id_col + ((!Diag || Unit) ? 1 : 0)) <= id_row)) ||
                  (Upper && (id_col >= (id_row + ((!Diag || Unit) ? 1 : 0))))) {
                auto prod = prdOp2_struct::eval(r1.eval(id_row, id_col), elm);
                l.eval(id_row, id_col_thr) =
                    addOp2_struct::eval(l.eval(id_row, id_col_thr), prod);
              }
              if (Diag && Unit && (id_row == id_col)) {
                l.eval(id_row, id_col_thr) = addOp2_struct::eval(
                    l.eval(id_row, id_col_thr), r1.eval(id_row, id_col));
              }
            }
          }
        }
#else
        if (id_col_thr < dimC) {
          for (IndexType row = 0, id_row = frs_row; (id_row < lst_row);
               row++, id_row++) {
            val = addOp2_struct::init(r2);
            for (IndexType id_col = frs_col; id_col < lst_col;
                 id_col += localSz) {
              if (Lower && Upper && Diag && !Unit) {
                auto prod = prdOp2_struct::eval(r1.eval(id_row, id_col),
                                                r2.eval(id_col));
                val = addOp2_struct::eval(val, prod);
              } else {
                if ((Lower &&
                     ((id_col + ((!Diag || Unit) ? 1 : 0)) <= id_row)) ||
                    (Upper &&
                     (id_col >= (id_row + ((!Diag || Unit) ? 1 : 0))))) {
                  auto prod = prdOp2_struct::eval(r1.eval(id_row, id_col),
                                                  r2.eval(id_col));
                  val = addOp2_struct::eval(val, prod);
                }
                if (Diag && Unit && (id_row == id_col)) {
                  val = addOp2_struct::eval(val, r1.eval(id_row, id_col));
                }
              }
            }
            l.eval(id_row, id_col_thr) = val;
          }
        }
#endif
      } else {
        for (IndexType row = 0, id_row = frs_row; (id_row < lst_row);
             row++, id_row++) {
          val = addOp2_struct::init(r2);
          for (IndexType id_col = frs_col; id_col < lst_col;
               id_col += localSz * interLoop) {
            auto lst_k_int = std::min(id_col + interLoop, lst_col);
            for (IndexType k_int =
                     ((Lower)
                          ? id_col
                          : std::max(row + ((!Diag || Unit) ? 1 : 0), id_col));
                 k_int < ((Upper) ? lst_k_int
                                  : std::min(row + ((!Diag || Unit) ? 0 : 1),
                                             lst_k_int));
                 k_int++) {
              auto prod =
                  prdOp2_struct::eval(r1.eval(id_row, k_int), r2.eval(k_int));
              val = addOp2_struct::eval(val, prod);
            }
          }
          l.eval(id_row, id_col_thr) = val;
        }
      }
    }
    return val;
  }

  template <typename sharedT>
  value_type eval(sharedT shrMem, cl::sycl::nd_item<1> ndItem) {
    IndexType localid = ndItem.get_local_id(0);
    IndexType localSz = ndItem.get_local_range(0);
    IndexType groupid = ndItem.get_group(0);

    IndexType dimR = r1.getSizeR();
    IndexType dimC = r1.getSizeC();

    IndexType rowSz = (dimR + nWG_row - 1) / nWG_row;
    IndexType shrSz = shrMemSize / localSz;

    IndexType idWFR =
        groupid / nWG_col;  // row bloq id of the current workgroup
    IndexType idWFC = groupid % nWG_col;  // col blq id of the current workgroup
    IndexType dimWFC =
        ((dimC + (localSz * nWG_col) - 1) / (localSz * nWG_col)) * localSz;

    IndexType frs_row = idWFR * rowSz;
    IndexType lst_row = std::min(dimR, frs_row + rowSz);

    IndexType frs_col = idWFC * dimWFC + interLoop * localid;
    IndexType lst_col = std::min(dimC, frs_col + dimWFC);

    // PROBLEM IF ONLY SOME THREADS OF A WORKGROUP ARE CANCELED
    // TO SOLVE IT, USE GLOBAL VALUES OF frs_col AND lst_col
    if ((!Upper && (((idWFC * dimWFC) + ((!Diag) ? 1 : 0)) > (lst_row - 1))) ||
        (!Lower &&
         ((frs_row + ((!Diag) ? 1 : 0)) > ((idWFC * dimWFC + dimWFC) - 1)))) {
      if (localid == 0) {
        value_type val = iniAddOp1_struct::eval(r2.eval(0));
        for (IndexType rowid = frs_row; rowid < lst_row; rowid++) {
          l.eval(rowid, idWFC) = val;
        }
      }
    } else {
      for (IndexType rowid = frs_row; rowid < lst_row; rowid += shrSz) {
        value_type val = addOp2_struct::init(r2);
        auto blqSz = std::min(shrSz, lst_row - rowid);
#ifdef GROUP_OF_ROWS
        for (IndexType row = 0, id_row = rowid; row < blqSz; row++, id_row++) {
          shrMem[row * localSz + localid] = val;
        }
#endif
        if (interLoop == 1) {
#ifdef GROUP_OF_ROWS
          for (IndexType id_col = frs_col; id_col < lst_col;
               id_col += localSz) {
            auto elm = r2.eval(id_col);
            for (IndexType row = 0, id_row = rowid; (row < blqSz);
                 row++, id_row++) {
              if (Lower && Upper && Diag && !Unit) {
                auto prod = prdOp2_struct::eval(r1.eval(id_row, id_col), elm);
                shrMem[row * localSz + localid] =
                    addOp2_struct::eval(shrMem[row * localSz + localid], prod);
              } else {
                if ((Lower &&
                     ((id_col + ((!Diag || Unit) ? 1 : 0)) <= id_row)) ||
                    (Upper &&
                     (id_col >= (id_row + ((!Diag || Unit) ? 1 : 0))))) {
                  auto prod = prdOp2_struct::eval(r1.eval(id_row, id_col), elm);
                  shrMem[row * localSz + localid] = addOp2_struct::eval(
                      shrMem[row * localSz + localid], prod);
                }
                if (Diag && Unit && (id_row == id_col)) {
                  shrMem[row * localSz + localid] = addOp2_struct::eval(
                      shrMem[row * localSz + localid], r1.eval(id_row, id_col));
                }
              }
            }
          }
#else
          for (IndexType row = 0, id_row = rowid; row < blqSz;
               row++, id_row++) {
            val = (Diag && Unit &&
                   ((id_row >= frs_col) && (id_row < lst_col) &&
                    (((id_row - frs_col) % localSz) == 0)))
                      ? r1.eval(id_row, id_row)
                      : addOp2_struct::init(r2);
            for (IndexType id_col = frs_col; id_col < lst_col;
                 id_col += localSz) {
              if (Lower && Upper && Diag && !Unit) {
                auto prod = prdOp2_struct::eval(r1.eval(id_row, id_col),
                                                r2.eval(id_col));
                val = addOp2_struct::eval(val, prod);
              } else {
                if ((Lower &&
                     ((id_col + ((!Diag || Unit) ? 1 : 0)) <= id_row)) ||
                    (Upper &&
                     (id_col >= (id_row + ((!Diag || Unit) ? 1 : 0))))) {
                  auto prod = prdOp2_struct::eval(r1.eval(id_row, id_col),
                                                  r2.eval(id_col));
                  val = addOp2_struct::eval(val, prod);
                }
              }
            }
            shrMem[row * localSz + localid] = val;
          }
#endif
        } else {
          for (IndexType row = 0, id_row = rowid; row < blqSz;
               row++, id_row++) {
            val = addOp2_struct::init(r2);
            for (IndexType id_col = frs_col; id_col < lst_col;
                 id_col += localSz * interLoop) {
              for (IndexType k_int = id_col;
                   k_int < std::min(id_col + interLoop, lst_col); k_int++) {
                if (Lower && Upper && Diag && !Unit) {
                  auto prod = prdOp2_struct::eval(r1.eval(id_row, k_int),
                                                  r2.eval(k_int));
                  val = addOp2_struct::eval(val, prod);
                } else {
                  if ((Lower &&
                       ((id_col + ((!Diag || Unit) ? 1 : 0)) <= id_row)) ||
                      (Upper &&
                       (id_col >= (id_row + ((!Diag || Unit) ? 1 : 0))))) {
                    auto prod = prdOp2_struct::eval(r1.eval(id_row, k_int),
                                                    r2.eval(k_int));
                    val = addOp2_struct::eval(val, prod);
                  }
                  if (Diag && Unit && (id_row == id_col)) {
                    val = addOp2_struct::eval(val, r1.eval(id_row, k_int));
                  }
                }
              }
            }
            shrMem[row * localSz + localid] = val;
          }
        }

        // This barrier is mandatory to be sure the data is on the shared memory
        ndItem.barrier(cl::sycl::access::fence_space::local_space);
        // Reduction inside the block
        for (IndexType offset = localSz >> 1; offset > 0; offset >>= 1) {
          if (localid < offset) {
            for (IndexType row = 0, id_row = rowid; row < blqSz;
                 row++, id_row++) {
              shrMem[row * localSz + localid] =
                  addOp2_struct::eval(shrMem[row * localSz + localid],
                                      shrMem[row * localSz + localid + offset]);
            }
          }
          // This barrier is mandatory to be sure the data are on the shared
          // memory
          ndItem.barrier(cl::sycl::access::fence_space::local_space);
        }
        if (localid == 0) {
          for (IndexType row = 0, id_row = rowid; row < blqSz;
               row++, id_row++) {
            l.eval(id_row, idWFC) = shrMem[row * localSz];
          }
        }
      }
    }

    return addOp2_struct::init(r2);
  }

  void bind(cl::sycl::handler &h) {
    l.bind(h);
    r1.bind(h);
    r2.bind(h);
  }
};

template <unsigned int interLoop = 1, bool Lower = true, bool Diag = true,
          bool Upper = true, bool Unit = false, typename LHS, typename RHS1,
          typename RHS2>
Gemv_Row<interLoop, Lower, Diag, Upper, Unit, LHS, RHS1, RHS2> make_Gemv_Row(
    LHS &l, RHS1 &r1, RHS2 &r2, typename RHS2::IndexType nWG_row,
    typename RHS2::IndexType nWG_col, typename RHS2::IndexType shrMemSize) {
  return Gemv_Row<interLoop, Lower, Diag, Upper, Unit, LHS, RHS1, RHS2>(
      l, r1, r2, nWG_row, nWG_col, shrMemSize);
}

/**** GEMV BY COLUMNS 1 ROW x M BLOCKS USING PROPERLY THE SHARED MEMORY ****/

// template <class LHS, class RHS1, class RHS2>
template <bool Lower, bool Diag, bool Upper, bool Unit, class LHS, class RHS1,
          class RHS2>
struct Gemv_Col {
  using value_type = typename RHS2::value_type;
  using IndexType = typename RHS2::IndexType;
  LHS l;
  RHS1 r1;
  RHS2 r2;
  IndexType nWG_row;
  IndexType nWG_col;
  IndexType shrMemSize;

  Gemv_Col(LHS &_l, RHS1 &_r1, RHS2 &_r2, IndexType &_nWG_row,
           IndexType &_nWG_col, IndexType &_shrMemSize)
      : l(_l),
        r1(_r1),
        r2(_r2),
        nWG_row(_nWG_row),
        nWG_col(_nWG_col),
        shrMemSize(_shrMemSize){};

  inline IndexType getSize() const { return r1.getSizeR(); }

  inline bool valid_thread(cl::sycl::nd_item<1> ndItem) const { return true; }

  value_type eval(IndexType i) {
    auto dim = r2.getSize();

    auto val = iniAddOp1_struct::eval(r2.eval(0));
    for (IndexType j = 0; j < dim; j++) {
      auto prod = prdOp2_struct::eval(r1.eval(i, j), r2.eval(j));
      val = addOp2_struct::eval(val, prod);
    }
    return l.eval(i) = val;
  }

  value_type eval(cl::sycl::nd_item<1> ndItem) {
    IndexType localid = ndItem.get_local_id(0);
    IndexType localSz = ndItem.get_local_range(0);
    IndexType groupid = ndItem.get_group(0);

    IndexType dimR = r1.getSizeR();
    IndexType dimC = r1.getSizeC();
    IndexType colSz = (dimC + nWG_col - 1) / nWG_col;

    IndexType idWFR = (groupid % nWG_row);
    IndexType idWFC = (groupid / nWG_row);
    IndexType dimWFR =
        (dimR + (localSz * nWG_row) - 1) / (localSz * nWG_row) * localSz;

    IndexType frs_row = idWFR * dimWFR + localid;
    IndexType lst_row = std::min(dimR, frs_row + dimWFR);

    IndexType frs_col = idWFC * colSz;
    IndexType lst_col = std::min(dimC, frs_col + colSz);
    // PROBLEM IF ONLY SOME THREADS OF A WORKGROUP ARE CANCELED
    // TO SOLVE IT, USE GLOBAL VALUES OF frs_row AND lst_row
    if ((!Upper &&
         ((frs_col + ((!Diag) ? 1 : 0)) > ((idWFR * dimWFR + dimWFR) - 1))) ||
        (!Lower && ((idWFR * dimWFR + ((!Diag) ? 1 : 0)) > (lst_col - 1)))) {
      auto val = iniAddOp1_struct::eval(r2.eval(0));
      for (IndexType rowid = frs_row; rowid < lst_row; rowid += localSz) {
        l.eval(rowid, idWFC) = val;
      }
    } else {
      // The product is computed
      for (IndexType rowid = frs_row; rowid < lst_row; rowid += localSz) {
        // The initial value of val is different for the first iteration
        auto val = (Diag && Unit && ((rowid >= frs_col) && (rowid < lst_col)))
                       ? r1.eval(rowid, rowid)
                       : iniAddOp1_struct::eval(r2.eval(0));
        for (IndexType id_col =
                 ((Lower)
                      ? frs_col
                      : std::max(rowid + ((!Diag || Unit) ? 1 : 0), frs_col));
             id_col <
             ((Upper) ? lst_col
                      : std::min(rowid + ((!Diag || Unit) ? 0 : 1), lst_col));
             id_col++) {
          auto prod =
              prdOp2_struct::eval(r1.eval(rowid, id_col), r2.eval(id_col));
          val = addOp2_struct::eval(val, prod);
        }
        // The result is stored in the correct component
        l.eval(rowid, idWFC) = val;
      }
    }

    return l.eval(frs_row, idWFC);
  }

  template <typename sharedT>
  value_type eval(sharedT shrMem, cl::sycl::nd_item<1> ndItem) {
    IndexType localid = ndItem.get_local_id(0);
    IndexType localSz = ndItem.get_local_range(0);
    IndexType groupid = ndItem.get_group(0);

    IndexType dimR = r1.getSizeR();
    IndexType dimC = r1.getSizeC();

    IndexType colSz = (dimC + nWG_col - 1) / nWG_col;
    IndexType idWFR = (groupid % nWG_row);
    IndexType idWFC = (groupid / nWG_row);
    IndexType dimWFR =
        (dimR + (localSz * nWG_row) - 1) / (localSz * nWG_row) * localSz;

    IndexType frs_row = idWFR * dimWFR + localid;
    IndexType lst_row = std::min(dimR, frs_row + dimWFR);

    IndexType frs_col = idWFC * colSz;
    IndexType lst_col = std::min(dimC, frs_col + colSz);

    // PROBLEM IF ONLY SOME THREADS OF A WORKGROUP ARE CANCELED
    // TO SOLVE IT, USE GLOBAL VALUES OF frs_row AND lst_row
    if ((!Upper &&
         ((frs_col + ((!Diag) ? 1 : 0)) > ((idWFR * dimWFR + dimWFR) - 1))) ||
        (!Lower && ((idWFR * dimWFR + ((!Diag) ? 1 : 0)) > (lst_col - 1)))) {
      auto val = iniAddOp1_struct::eval(r2.eval(0));
      for (IndexType rowid = frs_row; rowid < lst_row; rowid += localSz) {
        l.eval(rowid, idWFC) = val;
      }
    } else {
      // The computation are made in blocks of shrMemSize elements
      for (IndexType colid = frs_col; colid < lst_col; colid += shrMemSize) {
        auto blqSz = std::min(shrMemSize, lst_col - colid);
        // Copy a block of elements of vector r2 to the shared memory,
        // executing the expresion tree if it is needed
        for (IndexType col = localid; (col < blqSz); col += localSz) {
          shrMem[col] = r2.eval(colid + col);
        }
        // This barrier is mandatory to be sure the data is on the shared memory
        ndItem.barrier(cl::sycl::access::fence_space::local_space);

        // The product is computed
        for (IndexType rowid = frs_row; rowid < lst_row; rowid += localSz) {
          // The initial value of val is different for the first iteration
          auto val =
              ((colid == frs_col) ? iniAddOp1_struct::eval(r2.eval(0))
                                  : l.eval(rowid, idWFC)) +
              ((Diag && Unit && ((rowid >= colid) && (rowid < colid + blqSz)))
                   ? r1.eval(rowid, rowid)
                   : iniAddOp1_struct::eval(r2.eval(0)));
          for (IndexType id_col = colid, col = 0; col < blqSz;
               id_col++, col++) {
            if (Lower && Upper && Diag && !Unit) {
              auto prod =
                  prdOp2_struct::eval(r1.eval(rowid, id_col), shrMem[col]);
              val = addOp2_struct::eval(val, prod);
            } else {
              if ((Lower && ((id_col + ((!Diag || Unit) ? 1 : 0)) <= rowid)) ||
                  (Upper && (id_col >= (rowid + ((!Diag || Unit) ? 1 : 0))))) {
                auto prod =
                    prdOp2_struct::eval(r1.eval(rowid, id_col), shrMem[col]);
                val = addOp2_struct::eval(val, prod);
              }
            }
          }
          // The result is stored in the correct component
          l.eval(rowid, idWFC) = val;
        }
        // This barrier is mandatory to be sure the data is on the shared
        // memory
        ndItem.barrier(cl::sycl::access::fence_space::local_space);
      }
    }
    return l.eval(frs_row, idWFC);
  }

  void bind(cl::sycl::handler &h) {
    l.bind(h);
    r1.bind(h);
    r2.bind(h);
  }
};

// template <class LHS, class RHS1, class RHS2>
template <bool Lower = true, bool Diag = true, bool Upper = true,
          bool Unit = false, class LHS, class RHS1, class RHS2>
Gemv_Col<Lower, Diag, Upper, Unit, LHS, RHS1, RHS2> make_Gemv_Col(
    LHS &l, RHS1 &r1, RHS2 &r2, typename RHS2::IndexType nWG_row,
    typename RHS2::IndexType nWG_col, typename RHS2::IndexType shrMemSize) {
  return Gemv_Col<Lower, Diag, Upper, Unit, LHS, RHS1, RHS2>(
      l, r1, r2, nWG_row, nWG_col, shrMemSize);
}

/**** GER BY ROWS M ROWS x N BLOCK USING PROPERLY THE SHARED MEMORY ****/
// template <class LHS, class RHS1, class RHS2>
template <bool Single, bool Lower, bool Diag, bool Upper, class LHS, class RHS1,
          class RHS2>
struct Ger_Row {
  using value_type = typename RHS2::value_type;
  using IndexType = typename RHS2::IndexType;
  LHS l;
  RHS1 r1;
  RHS2 r2;
  IndexType nWG_row;
  IndexType nWG_col;
  IndexType shrMemSize;

  value_type scl;

  Ger_Row(LHS &_l, value_type _scl, RHS1 &_r1, RHS2 &_r2, IndexType &_nWG_row,
          IndexType &_nWG_col, IndexType &_shrMemSize)
      : l(_l),
        scl(_scl),
        r1(_r1),
        r2(_r2),
        nWG_row(_nWG_row),
        nWG_col(_nWG_col),
        shrMemSize(_shrMemSize){};

  inline IndexType getSize() const { return r1.getSize(); }

  inline bool valid_thread(cl::sycl::nd_item<1> ndItem) const { return true; }

  value_type eval(IndexType i) {
    auto size = (l.is_row_access()) ? l.getSizeC() : l.getSizeR();
    auto row = (l.is_row_access()) ? (i / size) : (i % size);
    auto col = (l.is_row_access()) ? (i % size) : (i / size);

    auto val = scl * r1.eval(row) * r2.eval(col);

    return l.eval(i) += val;
  }

  value_type eval(cl::sycl::nd_item<1> ndItem) {
    IndexType localid = ndItem.get_local_id(0);
    IndexType localSz = ndItem.get_local_range(0);
    IndexType groupid = ndItem.get_group(0);

    IndexType dimR = l.getSizeR();
    IndexType dimC = l.getSizeC();

    IndexType rowSz = (dimR + nWG_row - 1) / nWG_row;

    IndexType idWFR = (groupid % nWG_row);
    IndexType idWFC = (groupid / nWG_row);
    IndexType dimWFC =
        (dimC + (localSz * nWG_col) - 1) / (localSz * nWG_col) * localSz;

    IndexType frs_row = idWFR * rowSz;
    IndexType lst_row = std::min(dimR, frs_row + rowSz);

    IndexType frs_col = idWFC * dimWFC + localid;
    IndexType lst_col = std::min(dimC, frs_col + dimWFC);

    // PROBLEM IF ONLY SOME THREADS OF A WORKGROUP ARE CANCELED
    // TO SOLVE IT, USE GLOBAL VALUES OF frs_col AND lst_col
    if ((!Upper && (((idWFC * dimWFC) + ((!Diag) ? 1 : 0)) > (lst_row - 1))) ||
        (!Lower &&
         ((frs_row + ((!Diag) ? 1 : 0)) > ((idWFC * dimWFC + dimWFC) - 1)))) {
      ;
    } else if (Single) {
      for (IndexType colid = frs_col; colid < lst_col; colid += localSz) {
        auto val = scl * r2.eval(colid);
        for (IndexType id_row = frs_row, row = 0; id_row < lst_row;
             id_row++, row++) {
          if (Lower && Upper && Diag) {
            l.eval(id_row, colid) += r1.eval(id_row) * val;
          } else {
            if ((Lower && ((colid + ((!Diag) ? 1 : 0)) <= id_row)) ||
                (Upper && (colid >= (id_row + ((!Diag) ? 1 : 0))))) {
              l.eval(id_row, colid) += r1.eval(id_row) * val;
            }
          }
        }
      }
    } else {
      for (IndexType colid = frs_col; colid < lst_col; colid += localSz) {
        auto val1 = scl * r1.eval(colid);
        auto val2 = scl * r2.eval(colid);
        for (IndexType id_row = frs_row, row = 0; id_row < lst_row;
             id_row++, row++) {
          if (Lower && Upper && Diag) {
            l.eval(id_row, colid) +=
                r1.eval(id_row) * val2 + val1 * r2.eval(id_row);
          } else {
            if ((Lower && ((colid + ((!Diag) ? 1 : 0)) <= id_row)) ||
                (Upper && (colid >= (id_row + ((!Diag) ? 1 : 0))))) {
              l.eval(id_row, colid) +=
                  r1.eval(id_row) * val2 + r2.eval(id_row) * val1;
            }
          }
        }
      }
    }

    return l.eval(frs_row, frs_col);
  }

  template <typename sharedT>
  value_type eval(sharedT shrMem, cl::sycl::nd_item<1> ndItem) {
    IndexType localid = ndItem.get_local_id(0);
    IndexType localSz = ndItem.get_local_range(0);
    IndexType groupid = ndItem.get_group(0);

    IndexType dimR = l.getSizeR();
    IndexType dimC = l.getSizeC();

    IndexType rowSz = (dimR + nWG_row - 1) / nWG_row;
    IndexType shrSz = shrMemSize;

    IndexType idWFR = (groupid % nWG_row);
    IndexType idWFC = (groupid / nWG_row);
    IndexType dimWFC =
        (dimC + (localSz * nWG_col) - 1) / (localSz * nWG_col) * localSz;

    IndexType frs_row = idWFR * rowSz;
    IndexType lst_row = std::min(dimR, frs_row + rowSz);

    IndexType frs_col = idWFC * dimWFC + localid;
    IndexType lst_col = std::min(dimC, frs_col + dimWFC);
    // PROBLEM IF ONLY SOME THREADS OF A WORKGROUP ARE CANCELED
    // TO SOLVE IT, USE GLOBAL VALUES OF frs_col AND lst_col
    if ((!Upper && (((idWFC * dimWFC) + ((!Diag) ? 1 : 0)) > (lst_row - 1))) ||
        (!Lower &&
         ((frs_row + ((!Diag) ? 1 : 0)) > ((idWFC * dimWFC + dimWFC) - 1)))) {
      ;
    } else if (Single) {
      for (IndexType rowid = frs_row; rowid < lst_row; rowid += shrSz) {
        auto blqSz = std::min(shrSz, lst_row - rowid);
        for (IndexType row = localid, id_row = rowid + localid; (row < blqSz);
             row += localSz, id_row += localSz) {
          shrMem[row] = scl * r1.eval(id_row);
        }

        // This barrier is mandatory to be sure the data is on the shared memory
        ndItem.barrier(cl::sycl::access::fence_space::local_space);

        for (IndexType colid = frs_col; (colid < lst_col); colid += localSz) {
          auto val = r2.eval(colid);
          for (IndexType id_row = rowid, row = 0; row < blqSz;
               id_row++, row++) {
            if (Lower && Upper && Diag) {
              l.eval(id_row, colid) += shrMem[row] * val;
            } else {
              if ((Lower && ((colid + ((!Diag) ? 1 : 0)) <= id_row)) ||
                  (Upper && (colid >= (id_row + ((!Diag) ? 1 : 0))))) {
                l.eval(id_row, colid) += shrMem[row] * val;
              }
            }
          }
        }
        // This barrier is mandatory to be sure the data is on the shared
        // memory
        ndItem.barrier(cl::sycl::access::fence_space::local_space);
      }
    } else {
      auto shrSz1 = (shrSz / 2);
      for (IndexType rowid = frs_row; rowid < lst_row; rowid += shrSz) {
        auto blqSz = std::min(shrSz1, lst_row - rowid);
        for (IndexType row = localid, id_row = rowid + localid; (row < blqSz);
             row += localSz, id_row += localSz) {
          shrMem[row] = scl * r1.eval(id_row);
          shrMem[shrSz1 + row] = scl * r2.eval(id_row);
        }

        // This barrier is mandatory to be sure the data is on the shared memory
        ndItem.barrier(cl::sycl::access::fence_space::local_space);

        for (IndexType colid = frs_col; (colid < lst_col); colid += localSz) {
          auto val1 = r1.eval(colid);
          auto val2 = r2.eval(colid);
          for (IndexType id_row = rowid, row = 0; row < blqSz;
               id_row++, row++) {
            if (Lower && Upper && Diag) {
              l.eval(id_row, colid) +=
                  shrMem[row] * val2 + shrMem[shrSz1 + row] * val1;
            } else {
              if ((Lower && ((colid + ((!Diag) ? 1 : 0)) <= id_row)) ||
                  (Upper && (colid >= (id_row + ((!Diag) ? 1 : 0))))) {
                l.eval(id_row, colid) +=
                    shrMem[row] * val2 + shrMem[shrSz1 + row] * val1;
              }
            }
          }
        }
        // This barrier is mandatory to be sure the data is on the shared
        // memory
        ndItem.barrier(cl::sycl::access::fence_space::local_space);
      }
    }

    return shrMem[0];
  }

  void bind(cl::sycl::handler &h) {
    l.bind(h);
    r1.bind(h);
    r2.bind(h);
  }
};

template <bool Single = true, bool Lower = true, bool Diag = true,
          bool Upper = true, class LHS, class RHS1, class RHS2>
Ger_Row<Single, Lower, Diag, Upper, LHS, RHS1, RHS2> make_Ger_Row(
    LHS &l, typename LHS::value_type scl, RHS1 &r1, RHS2 &r2,
    typename RHS2::IndexType nWG_row, typename RHS2::IndexType nWG_col,
    typename RHS2::IndexType shrMemSize) {
  return Ger_Row<Single, Lower, Diag, Upper, LHS, RHS1, RHS2>(
      l, scl, r1, r2, nWG_row, nWG_col, shrMemSize);
}

/**** GER BY COLUMNS M ROWS x N BLOCK USING PROPERLY THE SHARED MEMORY ****/
// template <class LHS, class RHS1, class RHS2>
template <bool Single, bool Lower, bool Diag, bool Upper, class LHS, class RHS1,
          class RHS2>
struct Ger_Col {
  using value_type = typename RHS2::value_type;
  using IndexType = typename RHS2::IndexType;

  LHS l;
  RHS1 r1;
  RHS2 r2;
  IndexType nWG_row;
  IndexType nWG_col;
  IndexType shrMemSize;

  value_type scl;

  Ger_Col(LHS &_l, value_type _scl, RHS1 &_r1, RHS2 &_r2, IndexType &_nWG_row,
          IndexType &_nWG_col, IndexType &_shrMemSize)
      : l(_l),
        scl(_scl),
        r1(_r1),
        r2(_r2),
        nWG_row(_nWG_row),
        nWG_col(_nWG_col),
        shrMemSize(_shrMemSize){};

  inline IndexType getSize() const { return r1.getSize(); }

  inline bool valid_thread(cl::sycl::nd_item<1> ndItem) const { return true; }

  value_type eval(IndexType i) {
    auto size = (l.is_row_access()) ? l.getSizeC() : l.getSizeR();
    auto row = (l.is_row_access()) ? (i / size) : (i % size);
    auto col = (l.is_row_access()) ? (i % size) : (i / size);

    auto val = scl * r1.eval(row) * r2.eval(col);

    return l.eval(i) += val;
  }

  value_type eval(cl::sycl::nd_item<1> ndItem) {
    IndexType localid = ndItem.get_local_id(0);
    IndexType localSz = ndItem.get_local_range(0);
    IndexType groupid = ndItem.get_group(0);

    IndexType dimR = l.getSizeR();
    IndexType dimC = l.getSizeC();

    IndexType colSz =
        (dimR < localSz) ? localSz : (dimC + nWG_col - 1) / nWG_col;

    IndexType idWFR =
        groupid % nWG_row;  // row bloq id of the current workgroup
    IndexType idWFC = groupid / nWG_row;  // col blq id of the current workgroup
    IndexType dimWFR =
        (dimR + (localSz * nWG_row) - 1) / (localSz * nWG_row) * localSz;

    IndexType frs_row = idWFR * dimWFR + localid;
    IndexType lst_row = std::min(dimR, frs_row + dimWFR);

    IndexType frs_col = idWFC * colSz;
    IndexType lst_col = std::min(dimC, frs_col + colSz);
    if ((!Upper &&
         ((frs_col + ((!Diag) ? 1 : 0)) > ((idWFR * dimWFR + dimWFR) - 1))) ||
        (!Lower && ((idWFR * dimWFR + ((!Diag) ? 1 : 0)) > (lst_col - 1)))) {
      ;
    } else if (Single) {
      for (IndexType id_row = frs_row; id_row < lst_row; id_row += localSz) {
        auto val = scl * r1.eval(id_row);
        for (IndexType id_col =
                 ((Lower) ? frs_col
                          : std::max(id_row + ((!Diag) ? 1 : 0), frs_col));
             id_col < ((Upper) ? lst_col
                               : std::min(id_row + ((!Diag) ? 0 : 1), lst_col));
             id_col++) {
          l.eval(id_row, id_col) += val * r2.eval(id_col);
        }
      }
    } else {
      for (IndexType id_row = frs_row; id_row < lst_row; id_row += localSz) {
        auto val1 = scl * r1.eval(id_row);
        auto val2 = scl * r2.eval(id_row);
        for (IndexType id_col =
                 ((Lower) ? frs_col
                          : std::max(id_row + ((!Diag) ? 1 : 0), frs_col));
             id_col < ((Upper) ? lst_col
                               : std::min(id_row + ((!Diag) ? 0 : 1), lst_col));
             id_col++) {
          l.eval(id_row, id_col) +=
              val1 * r2.eval(id_col) + val2 * r1.eval(id_col);
        }
      }
    }

    return l.eval(frs_row, frs_col);
  }

  template <typename sharedT>
  value_type eval(sharedT shrMem, cl::sycl::nd_item<1> ndItem) {
    IndexType localid = ndItem.get_local_id(0);
    IndexType localSz = ndItem.get_local_range(0);
    IndexType groupid = ndItem.get_group(0);

    IndexType dimR = l.getSizeR();
    IndexType dimC = l.getSizeC();

    IndexType colSz =
        (dimR < localSz) ? localSz : (dimC + nWG_col - 1) / nWG_col;

    IndexType idWFR =
        groupid % nWG_row;  // row bloq id of the current workgroup
    IndexType dimWFR =
        (dimR + (localSz * nWG_row) - 1) / (localSz * nWG_row) * localSz;

    IndexType frs_row = idWFR * dimWFR + localid;
    IndexType lst_row = std::min(dimR, frs_row + dimWFR);

    IndexType frs_col = (groupid / nWG_row) * colSz;
    IndexType lst_col = std::min(dimC, frs_col + colSz);
    // PROBLEM IF ONLY SOME THREADS OF A WORKGROUP ARE CANCELED
    // TO SOLVE IT, USE GLOBAL VALUES OF frs_row AND lst_row
    if ((!Upper &&
         ((frs_col + ((!Diag) ? 1 : 0)) > ((idWFR * dimWFR + dimWFR) - 1))) ||
        (!Lower && ((idWFR * dimWFR + ((!Diag) ? 1 : 0)) > (lst_col - 1)))) {
      ;
    } else if (Single) {
      // The computation are made in blocks of shrMemSize elements
      for (IndexType colid = frs_col; colid < lst_col; colid += shrMemSize) {
        auto blqSz = std::min(shrMemSize, lst_col - colid);

        for (IndexType col = localid; (col < blqSz); col += localSz) {
          shrMem[col] = scl * r2.eval(colid + col);
        }

        // This barrier is mandatory to be sure the data is on the shared memory
        ndItem.barrier(cl::sycl::access::fence_space::local_space);

        for (IndexType id_row = frs_row; id_row < lst_row; id_row += localSz) {
          auto val = r1.eval(id_row);
          for (IndexType id_col = colid, col = 0; col < blqSz;
               id_col++, col++) {
            if (Lower && Upper && Diag) {
              l.eval(id_row, id_col) += val * shrMem[col];
            } else {
              if ((Lower && ((id_col + ((!Diag) ? 1 : 0)) <= id_row)) ||
                  (Upper && (id_col >= (id_row + ((!Diag) ? 1 : 0))))) {
                l.eval(id_row, id_col) += val * shrMem[col];
              }
            }
          }
        }
        // This barrier is mandatory to be sure the data is on the shared
        // memory
        ndItem.barrier(cl::sycl::access::fence_space::local_space);
      }
    } else {
      auto shrSz1 = (shrMemSize >> 2);
      // The computation are made in blocks of shrMemSize/shrSz1 elements
      for (IndexType colid = frs_col; colid < lst_col; colid += shrSz1) {
        auto blqSz = std::min(shrSz1, lst_col - colid);

        for (IndexType col = localid; (col < blqSz); col += localSz) {
          shrMem[col] = scl * r1.eval(colid + col);
          shrMem[shrSz1 + col] = scl * r2.eval(colid + col);
        }

        // This barrier is mandatory to be sure the data is on the shared memory
        ndItem.barrier(cl::sycl::access::fence_space::local_space);

        for (IndexType id_row = frs_row; id_row < lst_row; id_row += localSz) {
          auto val1 = r1.eval(id_row);
          auto val2 = r2.eval(id_row);
          for (IndexType id_col = colid, col = 0; col < blqSz;
               id_col++, col++) {
            if (Lower && Upper && Diag) {
              l.eval(id_row, id_col) +=
                  val1 * shrMem[shrSz1 + col] + val2 * shrMem[col];
            } else {
              if ((Lower && ((id_col + ((!Diag) ? 1 : 0)) <= id_row)) ||
                  (Upper && (id_col >= (id_row + ((!Diag) ? 1 : 0))))) {
                l.eval(id_row, id_col) +=
                    val1 * shrMem[shrSz1 + col] + val2 * shrMem[col];
              }
            }
          }
        }
        // This barrier is mandatory to be sure the data is on the shared
        // memory
        ndItem.barrier(cl::sycl::access::fence_space::local_space);
      }
    }

    return shrMem[0];
  }

  void bind(cl::sycl::handler &h) {
    l.bind(h);
    r1.bind(h);
    r2.bind(h);
  }
};

// template <class LHS, class RHS1, class RHS2>
template <bool Single = true, bool Lower = true, bool Diag = true,
          bool Upper = true, class LHS, class RHS1, class RHS2>
Ger_Col<Single, Lower, Diag, Upper, LHS, RHS1, RHS2> make_Ger_Col(
    LHS &l, typename LHS::value_type scl, RHS1 &r1, RHS2 &r2,
    typename RHS2::IndexType nWG_row, typename RHS2::IndexType nWG_col,
    typename RHS2::IndexType shrMemSize) {
  return Ger_Col<Single, Lower, Diag, Upper, LHS, RHS1, RHS2>(
      l, scl, r1, r2, nWG_row, nWG_col, shrMemSize);
}
}  // namespace blas

#endif  // BLAS2_TREES_HPP