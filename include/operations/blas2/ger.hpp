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

namespace blas {

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
        if (rowid > frs_row)
          // This barrier is mandatory to be sure the data is on the shared
          // memory
          ndItem.barrier(cl::sycl::access::fence_space::local_space);
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
      }
    } else {
      auto shrSz1 = (shrSz >> 1);
      for (IndexType rowid = frs_row; rowid < lst_row; rowid += shrSz) {
        if (rowid > frs_row)
          // This barrier is mandatory to be sure the data is on the shared
          // memory
          ndItem.barrier(cl::sycl::access::fence_space::local_space);
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
        if (colid > frs_col) {
          // This barrier is mandatory to be sure the data is on the shared
          // memory
          ndItem.barrier(cl::sycl::access::fence_space::local_space);
        }
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
      }
    } else {
      auto shrSz1 = (shrMemSize >> 1);
      // The computation are made in blocks of shrMemSize/shrSz1 elements
      for (IndexType colid = frs_col; colid < lst_col; colid += shrSz1) {
        if (colid > frs_col) {
          // This barrier is mandatory to be sure the data is on the shared
          // memory
          ndItem.barrier(cl::sycl::access::fence_space::local_space);
        }
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

} // namespace blas

#endif 