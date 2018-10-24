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
 *  @filename gemv.hpp
 *
 **************************************************************************/

#ifndef GEMV_HPP
#define GEMV_HPP

#include <stdexcept>
#include <vector>
#include <operations/blas_operators.hpp>
#include <views/view_sycl.hpp>

namespace blas {

/**
 * @struct AddSetColumns
 * @brief Tree node representing a column sum (reduction?) - i.e. summing a row, with one row per thread
 */
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

/**
 * @fn make_addSetColumns
 * @brief Constructs an AddSetColumns structure. 
 */
template <class RHS>
AddSetColumns<RHS> make_addSetColumns(RHS &r) {
  return AddSetColumns<RHS>(r);
}

/**** GEMV BY ROWS M ROWS x N BLOCK ****/
/**
 * @struct Gemv_Row
 * @brief Tree node representing a row-based/row-parallel generalised matrix vector multiplication. 
 */
template <unsigned int interLoop, bool Lower, bool Diag, bool Upper, bool Unit,
          class LHS, class mA_T, class vX_T>
struct Gemv_Row {
  using value_type = typename vX_T::value_type;
  using IndexType = typename vX_T::IndexType;

  LHS l;
  mA_T r1;
  vX_T r2;
  IndexType nWG_row;
  IndexType nWG_col;
  IndexType shrMemSize;

  Gemv_Row(LHS &_l, mA_T &_r1, vX_T &_r2, IndexType &_nWG_row,
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
    
    // Get the number of rows of the matrix
    IndexType dimR = r1.getSizeR();
    // 
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
      if (interLoop == 1) {
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
      } else {
        // There's an implied question mark after each of these comments. 
        // They are just attempts to understand the code! 
        // Iterate over rows of the matrix
        for (IndexType row = 0, id_row = frs_row; (id_row < lst_row);
             row++, id_row++) {
          // initialise an add node, with r2, the vector
          // we need to initialise it, as otherwise the type will change during execution! 
          val = addOp2_struct::init(r2);
          // Iterate across blocks of columns, in chunks of localSz * interLoop
          for (IndexType id_col = frs_col; id_col < lst_col;
               id_col += localSz * interLoop) {
            // If the row length isn't a multiple of localSz * interLoop
            // we need to go for fewer columns. Pick the min. 
            auto lst_k_int = std::min(id_col + interLoop, lst_col);
            // Handle lower diagonal etc
            for (IndexType k_int =
                     ((Lower)
                          ? id_col
                          : std::max(row + ((!Diag || Unit) ? 1 : 0), id_col));
                 k_int < ((Upper) ? lst_k_int
                                  : std::min(row + ((!Diag || Unit) ? 0 : 1),
                                             lst_k_int));
                 k_int++) {
              // calculate the product between the row and the vector. 
              auto prod =
                  prdOp2_struct::eval(r1.eval(id_row, k_int), r2.eval(k_int));
              // add that to val?
              // Reassignment! 
              val = addOp2_struct::eval(val, prod);
            }
          }
          // what does this do?
          l.eval(id_row, id_col_thr) = val;
        }
      }
    }
    return val;
  }













  /* 
    Evaluate using shared memory.
  */
  template <typename sharedT>
  value_type eval(sharedT shrMem, cl::sycl::nd_item<1> ndItem) {
    IndexType localid = ndItem.get_local_id(0);
    IndexType localSz = ndItem.get_local_range(0);
    IndexType groupid = ndItem.get_group(0);

    // Get the dimensions of the row and column
    IndexType dimR = r1.getSizeR();
    IndexType dimC = r1.getSizeC();

    // 
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
        if (interLoop == 1) {
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

/*!
 @brief Generator/factory for row-based GEMV trees. 
 
 make_Gemv_Row(
    LHS &l,
    mA_T &r1,
    vX_T &r2,
    typename vX_T::IndexType nWG_row,
    typename vX_T::IndexType nWG_col,
    typename vX_T::IndexType shrMemSize
 )
 */
template <unsigned int interLoop = 1, bool Lower = true, bool Diag = true,
          bool Upper = true, bool Unit = false, typename LHS, typename mA_T,
          typename vX_T>
Gemv_Row<interLoop, Lower, Diag, Upper, Unit, LHS, mA_T, vX_T> make_Gemv_Row(
    LHS &l, mA_T &r1, vX_T &r2, typename vX_T::IndexType nWG_row,
    typename vX_T::IndexType nWG_col, typename vX_T::IndexType shrMemSize) {
  return Gemv_Row<interLoop, Lower, Diag, Upper, Unit, LHS, mA_T, vX_T>(
      l, r1, r2, nWG_row, nWG_col, shrMemSize);
}










































/**** GEMV BY COLUMNS 1 ROW x M BLOCKS USING PROPERLY THE SHARED MEMORY ****/

/**
 * @struct Gemv_Col
 * @brief Tree node representing a Gemv, with parallel expressed across columns * 
 */
template <bool Lower, bool Diag, bool Upper, bool Unit, class LHS, class mA_T,
          class vX_T>
struct Gemv_Col {
  using value_type = typename vX_T::value_type;
  using IndexType = typename vX_T::IndexType;
  LHS l;
  mA_T r1;
  vX_T r2;
  IndexType nWG_row;
  IndexType nWG_col;
  IndexType shrMemSize;

  Gemv_Col(LHS &_l, mA_T &_r1, vX_T &_r2, IndexType &_nWG_row,
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
        if (colid > frs_col) {
          // This barrier is mandatory to be sure the data is on the shared
          // memory
          ndItem.barrier(cl::sycl::access::fence_space::local_space);
        }
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

// template <class LHS, class mA_T, class vX_T>
template <bool Lower = true, bool Diag = true, bool Upper = true,
          bool Unit = false, class LHS, class mA_T, class vX_T>
Gemv_Col<Lower, Diag, Upper, Unit, LHS, mA_T, vX_T> make_Gemv_Col(
    LHS &l, mA_T &r1, vX_T &r2, typename vX_T::IndexType nWG_row,
    typename vX_T::IndexType nWG_col, typename vX_T::IndexType shrMemSize) {
  return Gemv_Col<Lower, Diag, Upper, Unit, LHS, mA_T, vX_T>(
      l, r1, r2, nWG_row, nWG_col, shrMemSize);
}

} // namespace blas
#endif