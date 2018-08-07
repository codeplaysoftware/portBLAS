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

  IndexType getSize() { return r.getSizeR(); }

  bool valid_thread (cl::sycl::nd_item<1> ndItem) {
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

  void bind(cl::sycl::handler &h) {
    r.bind(h); 
  }

};

template <class RHS> AddSetColumns<RHS> make_addSetColumns(RHS &r) {
  return AddSetColumns<RHS>(r);
}

/**** GEMV BY ROWS M ROWS x N BLOCK ****/

// #define GROUP_OF_ROWS 1 // Not useful for GEMV by rows

template <unsigned int interLoop, bool Lower, bool Diag, bool Upper, bool Unit,
          class LHS, class RHS1, class RHS2>
struct Gemv_Row {
  
  using value_type = typename RHS2::value_type;
  using IndexType = typename RHS2::IndexType;
  
  LHS  l;
  RHS1 r1;
  RHS2 r2;
  IndexType nWG_row;
  IndexType nWG_col;
  IndexType shrMemSize;


  Gemv_Row(LHS &_l,RHS1 &_r1, RHS2 &_r2, IndexType &_nWG_row, IndexType &_nWG_col, IndexType &_shrMemSize)
    : l(_l), r1(_r1), r2(_r2), nWG_row(_nWG_row), nWG_col(_nWG_col), shrMemSize(_shrMemSize) {};

  IndexType getSize() { return r1.getSize(); }

  bool valid_thread (cl::sycl::nd_item<1> ndItem) { return true; }

  value_type eval(IndexType i) { // NOT VERIFIED
    auto dim = r2.getSize();

    auto val = iniAddOp1_struct::eval(r2.eval(0));
    for (IndexType j = 0; j < dim; j++) {
      auto prod = prdOp2_struct::eval(r1.eval(i,j),r2.eval(j));
      val = addOp2_struct::eval(val, prod);
    }
    return l.eval(i) = val;
  }

  value_type eval(cl::sycl::nd_item<1> ndItem) {
    IndexType localid = ndItem.get_local_id(0);
    IndexType localSz = ndItem.get_local_range(0);
    IndexType groupid = ndItem.get_group(0);
    IndexType groupSz = ndItem.get_group_range(0);
    IndexType glbalid = ndItem.get_global_id(0);
    IndexType glbalSz = ndItem.get_global_range(0);

    IndexType dimR = r1.getSizeR();
    IndexType dimC = r1.getSizeC();

    IndexType rowSz = (dimR + nWG_row - 1) / nWG_row;
    IndexType colSz = (dimC<localSz)? localSz: (dimC + nWG_col - 1) / nWG_col;
    IndexType shrSz = shrMemSize / localSz;

    IndexType idWFR = groupid / nWG_col;  // row bloq id of the current workgroup
    IndexType idWFC = groupid % nWG_col;  // col blq id of the current workgroup

    IndexType dimWFC = ((dimC + (localSz*nWG_col) - 1) /
                             (localSz*nWG_col)) * localSz;

    IndexType frs_row = idWFR*rowSz;
    IndexType lst_row = std::min(dimR,frs_row+rowSz);

    IndexType frs_col = idWFC * dimWFC + interLoop*localid;
    IndexType lst_col = std::min(dimC,frs_col+dimWFC);

    IndexType id_col_thr = idWFC * localSz + localid;


    value_type val = addOp2_struct::init(r2);
    // PROBLEM IF ONLY SOME THREADS OF A WORKGROUP_OF ARE CANCELED
    // TO SOLVE IT, USE GLOBAL VALUES OF frs_col AND lst_col
    if ((!Upper && (((idWFC*dimWFC)+((!Diag)?1:0))>(lst_row-1))) ||
        (!Lower && ((frs_row+((!Diag)?1:0))>((idWFC*dimWFC+dimWFC)-1)))) {
      for (IndexType rowid = frs_row; rowid < lst_row; rowid += localSz) {
        l.eval(rowid,id_col_thr) = val;
      }
    } else {

  #ifdef GROUP_OF_ROWS
      for (IndexType id_row=frs_row;(id_row<lst_row); id_row++) {
        l.eval(id_row,id_col_thr) = val;
      }
  #endif
      if (interLoop == 1) {
  #ifdef GROUP_OF_ROWS
        for (IndexType id_col = frs_col; id_col < lst_col; id_col += localSz) {
          auto elm = r2.eval(id_col);
          for (IndexType row=0, id_row=frs_row; (id_row<lst_row); row++, id_row++) {
            if (Lower && Upper && Diag && !Unit) {
              auto prod = prdOp2_struct::eval(r1.eval(id_row,id_col),elm);
              l.eval(id_row,id_col_thr) =
                      addOp2_struct::eval(l.eval(id_row,id_col_thr), prod);
            } else {
              if ((Lower && ((id_col+((!Diag||Unit)?1:0)) <= id_row)) ||
                (Upper && (id_col >= (id_row+((!Diag||Unit)?1:0))))) {
                auto prod = prdOp2_struct::eval(r1.eval(id_row,id_col),elm);
                l.eval(id_row,id_col_thr) =
                        addOp2_struct::eval(l.eval(id_row,id_col_thr), prod);
              }
              if (Diag && Unit && (id_row == id_col)) {
                l.eval(id_row,id_col_thr) =
                        addOp2_struct::eval(l.eval(id_row,id_col_thr),
                                            r1.eval(id_row,id_col));
              }
            }
          }
        }
  #else 
        if (id_col_thr < dimC) {
          for (IndexType row=0, id_row=frs_row; (id_row<lst_row); row++, id_row++) {
            val = addOp2_struct::init(r2);
            for (IndexType id_col = frs_col; id_col < lst_col; id_col += localSz) {
              if (Lower && Upper && Diag && !Unit) {
                auto prod = prdOp2_struct::eval(r1.eval(id_row,id_col),r2.eval(id_col));
                val = addOp2_struct::eval(val, prod);
              } else {
                if ((Lower && ((id_col+((!Diag||Unit)?1:0)) <= id_row)) ||
                    (Upper && (id_col >= (id_row+((!Diag||Unit)?1:0))))) {
                  auto prod = prdOp2_struct::eval(r1.eval(id_row,id_col),r2.eval(id_col));
                  val = addOp2_struct::eval(val, prod);
                }
                if (Diag && Unit && (id_row == id_col)) {
                  val = addOp2_struct::eval(val, r1.eval(id_row,id_col));
                }
              }
            }
            l.eval(id_row,id_col_thr) = val;
          }
        }
  #endif
      } else {
        for (IndexType row=0, id_row=frs_row; (id_row<lst_row); row++, id_row++) {
          val = addOp2_struct::init(r2);
          for (IndexType id_col = frs_col; id_col < lst_col; id_col += localSz*interLoop) {
            auto lst_k_int = std::min(id_col+interLoop,lst_col);
            for (IndexType k_int=((Lower)?id_col:std::max(row+((!Diag||Unit)?1:0),id_col));
                        k_int<((Upper)?lst_k_int:std::min(row+((!Diag||Unit)?0:1),lst_k_int)); k_int++) {
              auto prod = prdOp2_struct::eval(r1.eval(id_row,k_int),r2.eval(k_int));
              val = addOp2_struct::eval(val, prod);
            }
          }
          l.eval(id_row,id_col_thr) = val;
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
    IndexType groupSz = ndItem.get_group_range(0);
    IndexType glbalid = ndItem.get_global_id(0);
    IndexType glbalSz = ndItem.get_global_range(0);

    IndexType dimR = r1.getSizeR();
    IndexType dimC = r1.getSizeC();

    IndexType rowSz = (dimR + nWG_row - 1) / nWG_row;
    IndexType colSz = (dimC<localSz)? localSz: (dimC + nWG_col - 1) / nWG_col;
    IndexType shrSz = shrMemSize / localSz;

    IndexType idWFR = groupid / nWG_col;  // row bloq id of the current workgroup
    IndexType idWFC = groupid % nWG_col;  // col blq id of the current workgroup
    IndexType dimWFC = ((dimC + (localSz*nWG_col) - 1) /
                             (localSz*nWG_col)) * localSz;

    IndexType frs_row = idWFR*rowSz;
    IndexType lst_row = std::min(dimR,frs_row+rowSz);

    IndexType frs_col = idWFC * dimWFC + interLoop*localid;
    IndexType lst_col = std::min(dimC,frs_col+dimWFC);

    // PROBLEM IF ONLY SOME THREADS OF A WORKGROUP ARE CANCELED
    // TO SOLVE IT, USE GLOBAL VALUES OF frs_col AND lst_col
    if ((!Upper && (((idWFC*dimWFC)+((!Diag)?1:0))>(lst_row-1))) ||
        (!Lower && ((frs_row+((!Diag)?1:0))>((idWFC*dimWFC+dimWFC)-1)))) {
      if (localid == 0) {
        value_type val = iniAddOp1_struct::eval(r2.eval(0));
        for (IndexType rowid = frs_row; rowid < lst_row; rowid ++) {
          l.eval(rowid,idWFC) = val;
        }
      }
    } else {

      for (IndexType rowid=frs_row; rowid<lst_row; rowid+=shrSz) {
        value_type val = addOp2_struct::init(r2);
        auto blqSz = std::min(shrSz,lst_row-rowid);
  #ifdef GROUP_OF_ROWS
        for (IndexType row=0, id_row=rowid; row<blqSz; row++, id_row++) {
          shrMem[row*localSz+localid] = val;
        }
  #endif
        if (interLoop == 1) {
  #ifdef GROUP_OF_ROWS
          for (IndexType id_col = frs_col; id_col < lst_col; id_col += localSz) {
            auto elm = r2.eval(id_col);
            for (IndexType row=0, id_row=rowid; (row<blqSz); row++, id_row++) {
              if (Lower && Upper && Diag && !Unit) {
                auto prod = prdOp2_struct::eval(r1.eval(id_row,id_col),elm);
                shrMem[row*localSz+localid] =
                        addOp2_struct::eval(shrMem[row*localSz+localid], prod);
              } else {
                if ((Lower && ((id_col+((!Diag||Unit)?1:0)) <= id_row)) ||
                    (Upper && (id_col >= (id_row+((!Diag||Unit)?1:0))))) {
                  auto prod = prdOp2_struct::eval(r1.eval(id_row,id_col),elm);
                  shrMem[row*localSz+localid] =
                          addOp2_struct::eval(shrMem[row*localSz+localid], prod);
                }
                if (Diag && Unit && (id_row == id_col)) {
                  shrMem[row*localSz+localid] =
                          addOp2_struct::eval(shrMem[row*localSz+localid],
                                               r1.eval(id_row,id_col));
                }
              }
            }
          }
  #else
          for (IndexType row=0, id_row=rowid; row<blqSz; row++, id_row++) {
            val = (Diag && Unit && ((id_row >= frs_col) && (id_row < lst_col) &&
			             (((id_row-frs_col)%localSz) == 0)))?
                    	r1.eval(id_row,id_row): addOp2_struct::init(r2);
            for (IndexType id_col = frs_col; id_col < lst_col; id_col += localSz) {
              if (Lower && Upper && Diag && !Unit) {
                auto prod = prdOp2_struct::eval(r1.eval(id_row,id_col),r2.eval(id_col));
                val = addOp2_struct::eval(val, prod);
              } else {
                if ((Lower && ((id_col+((!Diag||Unit)?1:0)) <= id_row)) ||
                    (Upper && (id_col >= (id_row+((!Diag||Unit)?1:0))))) {
                  auto prod = prdOp2_struct::eval(r1.eval(id_row,id_col),r2.eval(id_col));
                  val = addOp2_struct::eval(val, prod);
                }
              }
            }
            shrMem[row*localSz+localid] = val;
          }
  #endif
        } else {
          for (IndexType row=0, id_row=rowid; row<blqSz; row++, id_row++) {
            val = addOp2_struct::init(r2);
            for (IndexType id_col = frs_col; id_col < lst_col; id_col += localSz*interLoop) {
              for (IndexType k_int=id_col; k_int<std::min(id_col+interLoop,lst_col);k_int++) {
                if (Lower && Upper && Diag && !Unit) {
                  auto prod = prdOp2_struct::eval(r1.eval(id_row,k_int),r2.eval(k_int));
                  val = addOp2_struct::eval(val, prod);
                } else {
                  if ((Lower && ((id_col+((!Diag||Unit)?1:0)) <= id_row)) ||
                      (Upper && (id_col >= (id_row+((!Diag||Unit)?1:0))))) {
                    auto prod = prdOp2_struct::eval(r1.eval(id_row,k_int),r2.eval(k_int));
                    val = addOp2_struct::eval(val, prod);
                  }
                  if (Diag && Unit && (id_row == id_col)) {
                    val = addOp2_struct::eval(val, r1.eval(id_row,k_int));
                  }
                }
              }
            }
            shrMem[row*localSz+localid] = val;
          }
        }

        // This barrier is mandatory to be sure the data is on the shared memory
        ndItem.barrier(cl::sycl::access::fence_space::local_space);
        // Reduction inside the block
        for (IndexType offset = localSz >> 1; offset > 0; offset >>= 1) {
          if (localid < offset) {
            for (IndexType row=0, id_row=rowid; row<blqSz; row++, id_row++) {
              shrMem[row*localSz+localid] =
                  addOp2_struct::eval(shrMem[row*localSz+localid],
                                      shrMem[row*localSz+localid+offset]);
            }
          }
          // This barrier is mandatory to be sure the data are on the shared memory
          ndItem.barrier(cl::sycl::access::fence_space::local_space);
        }
        if (localid == 0) {
          for (IndexType row=0, id_row=rowid; row<blqSz; row++, id_row++) {
            l.eval(id_row,idWFC) = shrMem[row*localSz];
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

template <unsigned int interLoop=1,
          bool Lower=true, bool Diag=true, bool Upper=true, bool Unit=false,
          typename LHS, typename RHS1, typename RHS2>
Gemv_Row<interLoop, Lower, Diag, Upper, Unit, LHS, RHS1, RHS2>
    make_Gemv_Row(LHS &l, RHS1 &r1, RHS2 &r2, typename RHS2::IndexType nWG_row, typename RHS2::IndexType nWG_col, typename RHS2::IndexType shrMemSize) {
  return Gemv_Row<interLoop, Lower, Diag, Upper, Unit, LHS, RHS1, RHS2>(l, r1, r2, nWG_row, nWG_col, shrMemSize);
}

/**** GEMV BY COLUMNS 1 ROW x M BLOCKS USING PROPERLY THE SHARED MEMORY ****/

//template <class LHS, class RHS1, class RHS2>
template <bool Lower, bool Diag, bool Upper, bool Unit,
          class LHS, class RHS1, class RHS2>
struct Gemv_Col {

  using value_type = typename RHS2::value_type;
  using IndexType = typename RHS2::IndexType;
  LHS l;
  RHS1 r1;
  RHS2 r2;
  IndexType nWG_row;
  IndexType nWG_col;
  IndexType shrMemSize;

  Gemv_Col(LHS &_l, RHS1 &_r1, RHS2 &_r2, IndexType &_nWG_row, IndexType &_nWG_col, IndexType &_shrMemSize)
      : l(_l), r1(_r1), r2(_r2), nWG_row(_nWG_row), nWG_col(_nWG_col), shrMemSize(_shrMemSize) {};

  IndexType getSize() { return r1.getSizeR(); }

  bool valid_thread (cl::sycl::nd_item<1> ndItem) { return true; }

  value_type eval(IndexType i) {
    auto dim = r2.getSize();

    auto val = iniAddOp1_struct::eval(r2.eval(0));
    for (IndexType j = 0; j < dim; j++) {
      auto prod = prdOp2_struct::eval(r1.eval(i,j),r2.eval(j));
      val = addOp2_struct::eval(val, prod);
    }
    return l.eval(i) = val;
  }

  value_type eval(cl::sycl::nd_item<1> ndItem) {
    IndexType localid = ndItem.get_local_id(0);
    IndexType localSz = ndItem.get_local_range(0);
    IndexType groupid = ndItem.get_group(0);
    IndexType groupSz = ndItem.get_group_range(0);
    IndexType glbalid = ndItem.get_global_id(0);

    IndexType dimR = r1.getSizeR();
    IndexType dimC = r1.getSizeC();
    IndexType colSz = (dimC + nWG_col - 1) / nWG_col;

    IndexType idWFR = (groupid % nWG_row);
    IndexType idWFC = (groupid / nWG_row);
    IndexType dimWFR = (dimR + (localSz*nWG_row) - 1) / (localSz*nWG_row) * localSz;

    IndexType frs_row = idWFR * dimWFR + localid;
    IndexType lst_row = std::min(dimR,frs_row + dimWFR);

    IndexType frs_col = idWFC*colSz;
    IndexType lst_col = std::min(dimC,frs_col+colSz);
    // PROBLEM IF ONLY SOME THREADS OF A WORKGROUP ARE CANCELED
    // TO SOLVE IT, USE GLOBAL VALUES OF frs_row AND lst_row
    if ((!Upper && ((frs_col+((!Diag)?1:0))>((idWFR*dimWFR+dimWFR)-1))) ||
        (!Lower && ((idWFR*dimWFR+((!Diag)?1:0))>(lst_col-1)))) {
      auto val = iniAddOp1_struct::eval(r2.eval(0));
      for (IndexType rowid = frs_row; rowid < lst_row; rowid += localSz) {
        l.eval(rowid,idWFC) = val;
      }
    } else {
      // The product is computed
      for (IndexType rowid = frs_row; rowid < lst_row; rowid += localSz) {
        // The initial value of val is different for the first iteration
        auto val = (Diag && Unit && ((rowid >= frs_col) && (rowid < lst_col)))?
                    r1.eval(rowid,rowid): iniAddOp1_struct::eval(r2.eval(0));
        for (IndexType id_col=((Lower)?frs_col:std::max(rowid+((!Diag||Unit)?1:0),frs_col));
                    id_col<((Upper)?lst_col:std::min(rowid+((!Diag||Unit)?0:1),lst_col)); id_col++) {
          auto prod = prdOp2_struct::eval(r1.eval(rowid,id_col), r2.eval(id_col));
          val = addOp2_struct::eval(val, prod);
        }
        // The result is stored in the correct component
        l.eval(rowid,idWFC) = val;
      }
    }

    return l.eval(frs_row,idWFC);
  }

  template <typename sharedT>
  value_type eval(sharedT shrMem, cl::sycl::nd_item<1> ndItem) {
    IndexType localid = ndItem.get_local_id(0);
    IndexType localSz = ndItem.get_local_range(0);
    IndexType groupid = ndItem.get_group(0);
    IndexType groupSz = ndItem.get_group_range(0);
    IndexType glbalid = ndItem.get_global_id(0);

    IndexType dimR = r1.getSizeR();
    IndexType dimC = r1.getSizeC();

    IndexType rowSz = (dimR < localSz)? dimR: localSz;
    IndexType colSz = (dimC + nWG_col - 1) / nWG_col;
    IndexType idWFR = (groupid % nWG_row);
    IndexType idWFC = (groupid / nWG_row);
    IndexType dimWFR = (dimR + (localSz*nWG_row) - 1) / (localSz*nWG_row) * localSz;

    IndexType frs_row = idWFR * dimWFR + localid;
    IndexType lst_row = std::min(dimR,frs_row + dimWFR);

    IndexType frs_col = idWFC*colSz;
    IndexType lst_col = std::min(dimC,frs_col+colSz);

    // PROBLEM IF ONLY SOME THREADS OF A WORKGROUP ARE CANCELED
    // TO SOLVE IT, USE GLOBAL VALUES OF frs_row AND lst_row
    if ((!Upper && ((frs_col+((!Diag)?1:0))>((idWFR*dimWFR+dimWFR)-1))) ||
        (!Lower && ((idWFR*dimWFR+((!Diag)?1:0))>(lst_col-1)))) {
      auto val = iniAddOp1_struct::eval(r2.eval(0));
      for (IndexType rowid = frs_row; rowid < lst_row; rowid += localSz) {
        l.eval(rowid,idWFC) = val;
      }
    } else {
      // The computation are made in blocks of shrMemSize elements
      for (IndexType colid=frs_col; colid<lst_col; colid+=shrMemSize) {
        if (colid > frs_col)
          // This barrier is mandatory to be sure the data is on the shared memory
          ndItem.barrier(cl::sycl::access::fence_space::local_space);

        auto blqSz = std::min(shrMemSize,lst_col-colid);
        // Copy a block of elements of vector r2 to the shared memory,
        // executing the expresion tree if it is needed
        for (IndexType col=localid; (col<blqSz); col+=localSz) {
          shrMem[col] = r2.eval(colid+col);
        }
        // This barrier is mandatory to be sure the data is on the shared memory
        ndItem.barrier(cl::sycl::access::fence_space::local_space);

        // The product is computed
        for (IndexType rowid = frs_row; rowid < lst_row; rowid += localSz) {
          // The initial value of val is different for the first iteration
          auto val = ((colid == frs_col)?
                      iniAddOp1_struct::eval(r2.eval(0)):
                      l.eval(rowid,idWFC))+
                      ((Diag && Unit && ((rowid >= colid) && (rowid < colid+blqSz)))?
                    	r1.eval(rowid,rowid): iniAddOp1_struct::eval(r2.eval(0)));
          for (IndexType id_col=colid, col=0; col<blqSz; id_col++, col++) {
            if (Lower && Upper && Diag && !Unit) {
              auto prod = prdOp2_struct::eval(r1.eval(rowid,id_col), shrMem[col]);
              val = addOp2_struct::eval(val, prod);
            } else {

              if ((Lower && ((id_col+((!Diag||Unit)?1:0)) <= rowid)) ||
                  (Upper && (id_col >= (rowid+((!Diag||Unit)?1:0))))) {
                auto prod = prdOp2_struct::eval(r1.eval(rowid,id_col), shrMem[col]);
                val = addOp2_struct::eval(val, prod);
              }
            }
          }
          // The result is stored in the correct component
          l.eval(rowid,idWFC) = val;
        }
      }
    }
    return l.eval(frs_row,idWFC);
  }
  
  void bind(cl::sycl::handler &h) {
    l.bind(h);  
    r1.bind(h);
    r2.bind(h);
  }
};

//template <class LHS, class RHS1, class RHS2>
template <bool Lower=true, bool Diag=true, bool Upper=true, bool Unit=false,
          class LHS, class RHS1, class RHS2>
Gemv_Col<Lower, Diag, Upper, Unit, LHS, RHS1, RHS2> make_Gemv_Col(
    LHS &l, RHS1 &r1, RHS2 &r2, typename RHS2::IndexType nWG_row, typename RHS2::IndexType nWG_col, typename RHS2::IndexType shrMemSize) {
  return Gemv_Col<Lower, Diag, Upper, Unit, LHS, RHS1, RHS2>(l, r1, r2, nWG_row, nWG_col, shrMemSize);
}

/**** GER BY ROWS M ROWS x N BLOCK USING PROPERLY THE SHARED MEMORY ****/
//template <class LHS, class RHS1, class RHS2>
template <bool Single, bool Lower, bool Diag, bool Upper,
          class LHS, class RHS1, class RHS2>
struct Ger_Row {
  
  using value_type = typename RHS2::value_type;
  using IndexType = typename RHS2::IndexType;
  LHS  l;
  RHS1 r1;
  RHS2 r2;
  IndexType nWG_row;
  IndexType nWG_col;
  IndexType shrMemSize;

  value_type scl;

  Ger_Row(LHS &_l, value_type _scl, RHS1 &_r1, RHS2 &_r2, IndexType &_nWG_row, IndexType &_nWG_col, IndexType &_shrMemSize)
    : l(_l), scl(_scl), r1(_r1), r2(_r2), nWG_row(_nWG_row), nWG_col(_nWG_col), shrMemSize(_shrMemSize) { };

  IndexType getSize() { return r1.getSize(); }

  bool valid_thread (cl::sycl::nd_item<1> ndItem) { return true; }

  value_type eval(IndexType i) {
    auto size = (l.getAccess()) ? l.getSizeC() : l.getSizeR();
    auto row = (l.getAccess()) ? (i / size) : (i % size);
    auto col = (l.getAccess()) ? (i % size) : (i / size);

    auto val = scl * r1.eval(row) * r2.eval(col);

    return l.eval(i) += val;
  }

  value_type eval(cl::sycl::nd_item<1> ndItem) {
    IndexType localid = ndItem.get_local_id(0);
    IndexType localSz = ndItem.get_local_range(0);
    IndexType groupid = ndItem.get_group(0);
    IndexType groupSz = ndItem.get_group_range(0);
    IndexType glbalid = ndItem.get_global_id(0);

    IndexType dimR = l.getSizeR();
    IndexType dimC = l.getSizeC();

    IndexType rowSz = (dimR + nWG_row - 1) / nWG_row;
    IndexType colSz = (dimR<localSz)? localSz: (dimC + nWG_col - 1) / nWG_col;
    IndexType shrSz = shrMemSize;

    IndexType idWFR = (groupid % nWG_row);
    IndexType idWFC = (groupid / nWG_row);
    IndexType dimWFC = (dimC + (localSz*nWG_col) - 1) / (localSz*nWG_col) * localSz;

    IndexType frs_row = idWFR*rowSz;
    IndexType lst_row = std::min(dimR,frs_row+rowSz);

    IndexType frs_col = idWFC * dimWFC + localid;
    IndexType lst_col = std::min(dimC,frs_col+dimWFC);

    // PROBLEM IF ONLY SOME THREADS OF A WORKGROUP ARE CANCELED
    // TO SOLVE IT, USE GLOBAL VALUES OF frs_col AND lst_col
    if ((!Upper && (((idWFC*dimWFC)+((!Diag)?1:0))>(lst_row-1))) ||
        (!Lower && ((frs_row+((!Diag)?1:0))>((idWFC*dimWFC+dimWFC)-1)))) {
      ;
    } else if (Single) {
      for (IndexType colid = frs_col; colid < lst_col; colid += localSz) {
        auto val = scl * r2.eval(colid);
        for (IndexType id_row=frs_row, row=0; id_row<lst_row; id_row++, row++) {
          if (Lower && Upper && Diag) {
            l.eval(id_row,colid) += r1.eval(id_row) * val;
          } else {
            if ((Lower && ((colid+((!Diag)?1:0)) <= id_row)) ||
              (Upper && (colid >= (id_row+((!Diag)?1:0))))) {
                l.eval(id_row,colid) += r1.eval(id_row) * val;
            }
          }
        }
      }
    } else {
      for (IndexType colid = frs_col; colid < lst_col; colid += localSz) {
        auto val1 = scl * r1.eval(colid);
        auto val2 = scl * r2.eval(colid);
        for (IndexType id_row=frs_row, row=0; id_row<lst_row; id_row++, row++) {
          if (Lower && Upper && Diag) {
            l.eval(id_row,colid) += r1.eval(id_row) * val2 + val1 * r2.eval(id_row);
          } else {
            if ((Lower && ((colid+((!Diag)?1:0)) <= id_row)) ||
              (Upper && (colid >= (id_row+((!Diag)?1:0))))) {
                l.eval(id_row,colid) += r1.eval(id_row) * val2 +
                                        r2.eval(id_row) * val1;
            }
          }
        }
      }
    }

    return l.eval(frs_row,frs_col);
  }

  template <typename sharedT>
  value_type eval(sharedT shrMem, cl::sycl::nd_item<1> ndItem) {
    IndexType localid = ndItem.get_local_id(0);
    IndexType localSz = ndItem.get_local_range(0);
    IndexType groupid = ndItem.get_group(0);
    IndexType groupSz = ndItem.get_group_range(0);
    IndexType glbalid = ndItem.get_global_id(0);

    IndexType dimR = l.getSizeR();
    IndexType dimC = l.getSizeC();

    IndexType rowSz = (dimR + nWG_row - 1) / nWG_row;
    IndexType colSz = (dimR<localSz)? localSz: (dimC + nWG_col - 1) / nWG_col;
    IndexType shrSz = shrMemSize;

    IndexType idWFR = (groupid % nWG_row);
    IndexType idWFC = (groupid / nWG_row);
    IndexType dimWFC = (dimC + (localSz*nWG_col) - 1) / (localSz*nWG_col) * localSz;

    IndexType frs_row = idWFR*rowSz;
    IndexType lst_row = std::min(dimR,frs_row+rowSz);

    IndexType frs_col = idWFC * dimWFC + localid;
    IndexType lst_col = std::min(dimC,frs_col+dimWFC);
    // PROBLEM IF ONLY SOME THREADS OF A WORKGROUP ARE CANCELED
    // TO SOLVE IT, USE GLOBAL VALUES OF frs_col AND lst_col
    if ((!Upper && (((idWFC*dimWFC)+((!Diag)?1:0))>(lst_row-1))) ||
        (!Lower && ((frs_row+((!Diag)?1:0))>((idWFC*dimWFC+dimWFC)-1)))) {
      ;
    } else if (Single) {
      for (IndexType rowid=frs_row; rowid<lst_row; rowid+=shrSz) {
        if (rowid > frs_row)
          // This barrier is mandatory to be sure the data is on the shared memory
          ndItem.barrier(cl::sycl::access::fence_space::local_space);

        auto blqSz = std::min(shrSz,lst_row-rowid);
        for (IndexType row=localid, id_row=rowid+localid; (row<blqSz); row+=localSz, id_row+=localSz) {
          shrMem[row] = scl * r1.eval(id_row);
        }

        // This barrier is mandatory to be sure the data is on the shared memory
        ndItem.barrier(cl::sycl::access::fence_space::local_space);

        for (IndexType colid = frs_col; (colid<lst_col); colid += localSz) {
          auto val = r2.eval(colid);
          for (IndexType id_row=rowid, row=0; row<blqSz; id_row++, row++) {
            if (Lower && Upper && Diag) {
              l.eval(id_row,colid) += shrMem[row] * val;
            } else {
              if ((Lower && ((colid+((!Diag)?1:0)) <= id_row)) ||
                  (Upper && (colid >= (id_row+((!Diag)?1:0))))) {
                l.eval(id_row,colid) += shrMem[row] * val;
              }
            }
          }
        }
      }
    } else {
      auto shrSz1 = (shrSz / 2);
      for (IndexType rowid=frs_row; rowid<lst_row; rowid+=shrSz) {
        if (rowid > frs_row)
          // This barrier is mandatory to be sure the data is on the shared memory
          ndItem.barrier(cl::sycl::access::fence_space::local_space);

        auto blqSz = std::min(shrSz1,lst_row-rowid);
        for (IndexType row=localid, id_row=rowid+localid; (row<blqSz); row+=localSz, id_row+=localSz) {
          shrMem[       row] = scl * r1.eval(id_row);
          shrMem[shrSz1+row] = scl * r2.eval(id_row);
        }

        // This barrier is mandatory to be sure the data is on the shared memory
        ndItem.barrier(cl::sycl::access::fence_space::local_space);

        for (IndexType colid = frs_col; (colid<lst_col); colid += localSz) {
          auto val1 = r1.eval(colid);
          auto val2 = r2.eval(colid);
          for (IndexType id_row=rowid, row=0; row<blqSz; id_row++, row++) {
            if (Lower && Upper && Diag) {
              l.eval(id_row,colid) += shrMem[       row] * val2 +
                                      shrMem[shrSz1+row] * val1;
            } else {
              if ((Lower && ((colid+((!Diag)?1:0)) <= id_row)) ||
                  (Upper && (colid >= (id_row+((!Diag)?1:0))))) {
                l.eval(id_row,colid) += shrMem[       row] * val2 +
                                        shrMem[shrSz1+row] * val1;
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

template <bool Single=true, bool Lower=true, bool Diag=true, bool Upper=true,
          class LHS, class RHS1, class RHS2>
Ger_Row<Single, Lower, Diag, Upper, LHS, RHS1, RHS2> make_Ger_Row(
    LHS &l, typename LHS::value_type scl, RHS1 &r1, RHS2 &r2, typename RHS2::IndexType nWG_row, typename RHS2::IndexType nWG_col, typename RHS2::IndexType shrMemSize) {
  return Ger_Row<Single, Lower, Diag, Upper, LHS, RHS1, RHS2>(l, scl, r1, r2, nWG_row, nWG_col, shrMemSize);
}

/**** GER BY COLUMNS M ROWS x N BLOCK USING PROPERLY THE SHARED MEMORY ****/
//template <class LHS, class RHS1, class RHS2>
template <bool Single, bool Lower, bool Diag, bool Upper,
          class LHS, class RHS1, class RHS2>
struct Ger_Col {
  using value_type = typename RHS2::value_type;
  using IndexType = typename RHS2::IndexType;

  LHS  l;
  RHS1 r1;
  RHS2 r2;
  IndexType nWG_row;
  IndexType nWG_col;
  IndexType shrMemSize;

  value_type scl;

  Ger_Col(LHS &_l, value_type _scl, RHS1 &_r1, RHS2 &_r2, IndexType &_nWG_row, IndexType &_nWG_col, IndexType &_shrMemSize)
    : l(_l), scl(_scl), r1(_r1), r2(_r2), nWG_row(_nWG_row), nWG_col(_nWG_col), shrMemSize(_shrMemSize) { };

  IndexType getSize() { return r1.getSize(); }

  bool valid_thread (cl::sycl::nd_item<1> ndItem) { return true; }

  value_type eval(IndexType i) {
    auto size = (l.getAccess()) ? l.getSizeC() : l.getSizeR();
    auto row = (l.getAccess()) ? (i / size) : (i % size);
    auto col = (l.getAccess()) ? (i % size) : (i / size);

    auto val = scl * r1.eval(row) * r2.eval(col);

    return l.eval(i) += val;
  }

  value_type eval(cl::sycl::nd_item<1> ndItem) {
    IndexType localid = ndItem.get_local_id(0);
    IndexType localSz = ndItem.get_local_range(0);
    IndexType groupid = ndItem.get_group(0);
    IndexType groupSz = ndItem.get_group_range(0);
    IndexType glbalid = ndItem.get_global_id(0);

    IndexType dimR = l.getSizeR();
    IndexType dimC = l.getSizeC();

    IndexType rowSz = (dimR + nWG_row - 1) / nWG_row;
    IndexType colSz = (dimR<localSz)? localSz: (dimC + nWG_col - 1) / nWG_col;
    IndexType shrSz = shrMemSize;

    IndexType idWFR  = groupid % nWG_row;  // row bloq id of the current workgroup
    IndexType idWFC  = groupid / nWG_row;  // col blq id of the current workgroup
    IndexType dimWFR = (dimR + (localSz*nWG_row) - 1) / (localSz*nWG_row) * localSz;

    IndexType frs_row = idWFR * dimWFR + localid;
    IndexType lst_row = std::min(dimR,frs_row + dimWFR);

    IndexType frs_col = idWFC*colSz;
    IndexType lst_col = std::min(dimC,frs_col+colSz);
    if ((!Upper && ((frs_col+((!Diag)?1:0))>((idWFR*dimWFR+dimWFR)-1))) ||
        (!Lower && ((idWFR*dimWFR+((!Diag)?1:0))>(lst_col-1)))) {
      ;
    } else if (Single) {
      for (IndexType id_row = frs_row; id_row < lst_row; id_row += localSz) {
        auto val = scl * r1.eval(id_row);
        for (IndexType id_col=((Lower)?frs_col:std::max(id_row+((!Diag)?1:0),frs_col));
                    id_col<((Upper)?lst_col:std::min(id_row+((!Diag)?0:1),lst_col)); id_col++) {
          l.eval(id_row,id_col) += val * r2.eval(id_col);
        }
      }
    } else {
      for (IndexType id_row = frs_row; id_row < lst_row; id_row += localSz) {
        auto val1 = scl * r1.eval(id_row);
        auto val2 = scl * r2.eval(id_row);
        for (IndexType id_col=((Lower)?frs_col:std::max(id_row+((!Diag)?1:0),frs_col));
                    id_col<((Upper)?lst_col:std::min(id_row+((!Diag)?0:1),lst_col)); id_col++) {
          l.eval(id_row,id_col) += val1 * r2.eval(id_col) +
                                   val2 * r1.eval(id_col);
        }
      }
    }

    return l.eval(frs_row,frs_col);
  }

  template <typename sharedT>
  value_type eval(sharedT shrMem, cl::sycl::nd_item<1> ndItem) {
    IndexType localid = ndItem.get_local_id(0);
    IndexType localSz = ndItem.get_local_range(0);
    IndexType groupid = ndItem.get_group(0);
    IndexType groupSz = ndItem.get_group_range(0);
    IndexType glbalid = ndItem.get_global_id(0);

    IndexType dimR = l.getSizeR();
    IndexType dimC = l.getSizeC();

    IndexType rowSz = (dimR + nWG_row - 1) / nWG_row;
    IndexType colSz = (dimR<localSz)? localSz: (dimC + nWG_col - 1) / nWG_col;
    IndexType shrSz = shrMemSize;

    IndexType idWFR  = groupid % nWG_row;  // row bloq id of the current workgroup
    IndexType idWFC  = groupid / nWG_row;  // col blq id of the current workgroup
    IndexType dimWFR = (dimR + (localSz*nWG_row) - 1) / (localSz*nWG_row) * localSz;

    IndexType frs_row = idWFR * dimWFR + localid;
    IndexType lst_row = std::min(dimR,frs_row + dimWFR);

    IndexType frs_col = idWFC*colSz;
    IndexType lst_col = std::min(dimC,frs_col+colSz);
    // PROBLEM IF ONLY SOME THREADS OF A WORKGROUP ARE CANCELED
    // TO SOLVE IT, USE GLOBAL VALUES OF frs_row AND lst_row
    if ((!Upper && ((frs_col+((!Diag)?1:0))>((idWFR*dimWFR+dimWFR)-1))) ||
        (!Lower && ((idWFR*dimWFR+((!Diag)?1:0))>(lst_col-1)))) {
      ;
    } else if (Single) {
      // The computation are made in blocks of shrMemSize elements
      for (IndexType colid=frs_col; colid<lst_col; colid+=shrMemSize) {
        if (colid > frs_col)
          // This barrier is mandatory to be sure the data is on the shared memory
          ndItem.barrier(cl::sycl::access::fence_space::local_space);

        auto blqSz = std::min(shrMemSize,lst_col-colid);

        for (IndexType col=localid; (col<blqSz); col+=localSz) {
          shrMem[col] = scl * r2.eval(colid+col);
        }

        // This barrier is mandatory to be sure the data is on the shared memory
        ndItem.barrier(cl::sycl::access::fence_space::local_space);

        for (IndexType id_row = frs_row; id_row < lst_row; id_row += localSz) {
          auto val = r1.eval(id_row);
          for (IndexType id_col=colid, col=0; col<blqSz; id_col++, col++) {
            if (Lower && Upper && Diag) {
              l.eval(id_row,id_col) += val * shrMem[col];
            } else {
              if ((Lower && ((id_col+((!Diag)?1:0)) <= id_row)) ||
                  (Upper && (id_col >= (id_row+((!Diag)?1:0))))) {
                l.eval(id_row,id_col) += val * shrMem[col];
              }
            }
          }
        }
      }
    } else {
      auto shrSz1 = (shrMemSize / 2);
      // The computation are made in blocks of shrMemSize/shrSz1 elements
      for (IndexType colid=frs_col; colid<lst_col; colid+=shrSz1) {
        if (colid > frs_col)
          // This barrier is mandatory to be sure the data is on the shared memory
          ndItem.barrier(cl::sycl::access::fence_space::local_space);

        auto blqSz = std::min(shrSz1,lst_col-colid);

        for (IndexType col=localid; (col<blqSz); col+=localSz) {
          shrMem[       col] = scl * r1.eval(colid+col);
          shrMem[shrSz1+col] = scl * r2.eval(colid+col);
        }

        // This barrier is mandatory to be sure the data is on the shared memory
        ndItem.barrier(cl::sycl::access::fence_space::local_space);

        for (IndexType id_row = frs_row; id_row < lst_row; id_row += localSz) {
          auto val1 = r1.eval(id_row);
          auto val2 = r2.eval(id_row);
          for (IndexType id_col=colid, col=0; col<blqSz; id_col++, col++) {
            if (Lower && Upper && Diag) {
              l.eval(id_row,id_col) += val1 * shrMem[shrSz1+col] +
                                       val2 * shrMem[       col];
            } else {
              if ((Lower && ((id_col+((!Diag)?1:0)) <= id_row)) ||
                  (Upper && (id_col >= (id_row+((!Diag)?1:0))))) {
                l.eval(id_row,id_col) += val1 * shrMem[shrSz1+col] +
                                         val2 * shrMem[       col];
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

//template <class LHS, class RHS1, class RHS2>
template <bool Single=true, bool Lower=true, bool Diag=true, bool Upper=true,
          class LHS, class RHS1, class RHS2>
Ger_Col<Single, Lower, Diag, Upper, LHS, RHS1, RHS2> make_Ger_Col(
    LHS &l, typename LHS::value_type scl, RHS1 &r1, RHS2 &r2, typename RHS2::IndexType nWG_row, typename RHS2::IndexType nWG_col, typename RHS2::IndexType shrMemSize) {
  return Ger_Col<Single, Lower, Diag, Upper, LHS, RHS1, RHS2>(l, scl, r1, r2, nWG_row, nWG_col, shrMemSize);
}

/**************************************************/
/*************** PREVIOUS VERSIONS ****************/
/**************************************************/

/**** GEMV BY ROWS 1 ROW x 1 BLOCK ****/
template <unsigned int interLoop, class LHS, class RHS1, class RHS2>
struct GemvR_1Row_1WG {
  LHS  l;
  RHS1 r1;
  RHS2 r2;

  using value_type = typename RHS2::value_type;
  using IndexType = typename RHS2::IndexType;

  GemvR_1Row_1WG(LHS &_l,RHS1 &_r1, RHS2 &_r2)
    : l(_l), r1(_r1), r2(_r2) {};

  IndexType getSize() { return r1.getSizeR(); }

  bool valid_thread (cl::sycl::nd_item<1> ndItem) { return true; }

  value_type eval(IndexType i) { // NOT VERIFIED
    auto dim = r2.getSize();

    auto val = iniAddOp1_struct::eval(r2.eval(0));
    for (IndexType j = 0; j < dim; j++) {
      val += r1.eval(i, j) * r2.eval(j);
    }
    return l.eval(i) = val;
  }

  value_type eval(cl::sycl::nd_item<1> ndItem) {
    return eval(ndItem.get_global_id(0));
  }

  template <typename sharedT>
  value_type eval(sharedT shrMem, cl::sycl::nd_item<1> ndItem) {
    IndexType localid = ndItem.get_local_id(0);
    IndexType localSz = ndItem.get_local_range(0);
    IndexType groupid = ndItem.get_group(0);
    IndexType glbalSz = ndItem.get_global_range(0);

    IndexType vecS = r2.getSize();

    value_type val = addOp2_struct::init(r2);

    if (interLoop == 1) {
      IndexType frs_thrd = localid;
      for (IndexType k = frs_thrd; k < vecS; k += localSz) {
        val += r1.eval(groupid,k) * r2.eval(k);
      }
    } else { // NOT VERIFIED
      IndexType frs_thrd = interLoop * (groupid * localSz + localid);
      for (IndexType k = frs_thrd; k < vecS; k += interLoop * glbalSz) {
        for (IndexType k_int=k; k_int<std::min(k+interLoop,vecS);k_int++) {
          val = addOp2_struct::eval(val, r1.eval(groupid,k_int));
        }
      }
    }

    shrMem[localid] = val;
    // This barrier is mandatory to be sure the data is on the shared memory
    ndItem.barrier(cl::sycl::access::fence_space::local_space);

    // Reduction inside the block
    for (IndexType offset = localSz >> 1; offset > 0; offset >>= 1) {
      if (localid < offset) {
        shrMem[localid] += shrMem[localid + offset];
      }
      // This barrier is mandatory to be sure the data are on the shared memory
      ndItem.barrier(cl::sycl::access::fence_space::local_space);
    }
    if (localid == 0) {
      l.eval(groupid) = shrMem[localid];
    }

    return l.eval(groupid);
  }
  
  void bind(cl::sycl::handler &h) {
    l.bind(h);  
    r1.bind(h);
    r2.bind(h);
  }
};

template <unsigned int interLoop=1, typename LHS, typename RHS1, typename RHS2>
GemvR_1Row_1WG<interLoop, LHS, RHS1, RHS2> make_GemvR_1Row_1WG(LHS &l, RHS1 &r1,
                                                              RHS2 &r2) {
  return GemvR_1Row_1WG<interLoop, LHS, RHS1, RHS2>(l, r1, r2);
}

/**** GEMV BY ROWS 1 ROW x 1 BLOCK, WITHOUT LOCAL ADDITION ****/
template <unsigned int interLoop, class LHS, class RHS1, class RHS2>
struct GemvR_1Row_1WG_NoRed {
  LHS  l;
  RHS1 r1;
  RHS2 r2;

  using value_type = typename RHS2::value_type;
    using IndexType = typename RHS2::IndexType;

  GemvR_1Row_1WG_NoRed(LHS &_l,RHS1 &_r1, RHS2 &_r2)
    : l(_l), r1(_r1), r2(_r2) {};

  IndexType getSize() { return l.getSize(); }

  bool valid_thread (cl::sycl::nd_item<1> ndItem) { return true; }

  value_type eval(IndexType i) { // NOT VERIFIED
    auto dim = r2.getSize();

    auto val = iniAddOp1_struct::eval(r2.eval(0));
    for (IndexType j = 0; j < dim; j++) {
      val += r1.eval(i, j) * r2.eval(j);
    }
    return l.eval(i) = val;
  }

  value_type eval(cl::sycl::nd_item<1> ndItem) {
    IndexType localid = ndItem.get_local_id(0);
    IndexType localSz = ndItem.get_local_range(0);
    IndexType groupid = ndItem.get_group(0);
    IndexType glbalSz = ndItem.get_group_range(0) * localSz;

    IndexType vecS = r2.getSize();

    value_type val = addOp2_struct::init(r2);

    if (interLoop == 1) {
      IndexType frs_thrd = localid;
      for (IndexType k = frs_thrd; k < vecS; k += localSz) {
        auto prod = prdOp2_struct::eval(r1.eval(groupid,k),r2.eval(k));
        val = addOp2_struct::eval(val, prod);
      }
    } else { // NOT VERIFIED
      IndexType frs_thrd = interLoop * (groupid * localSz + localid);
      for (IndexType k = frs_thrd; k < vecS; k += interLoop * glbalSz) {
        for (IndexType k_int=k; k_int<std::min(k+interLoop,vecS);k_int++) {
          val = addOp2_struct::eval(val, r1.eval(groupid,k_int));
        }
      }
    }

    return l.eval(groupid,localid) = val;
  }
  void bind(cl::sycl::handler &h) {
    l.bind(h);  
    r1.bind(h);
    r2.bind(h);
  }
};

template <unsigned int interLoop=1, typename LHS, typename RHS1, typename RHS2>
GemvR_1Row_1WG_NoRed<interLoop, LHS, RHS1, RHS2>
            make_GemvR_1Row_1WG_NoRed(LHS &l, RHS1 &r1, RHS2 &r2) {
  return GemvR_1Row_1WG_NoRed<interLoop, LHS, RHS1, RHS2>(l, r1, r2);
}

/**** GEMV BY ROWS 1 ROW x N BLOCK ****/
template <unsigned int interLoop, class LHS, class RHS1, class RHS2>
struct GemvR_1Row_NWG {

  using value_type = typename RHS2::value_type;
  using IndexType = typename RHS2::IndexType;

  LHS  l;
  RHS1 r1;
  RHS2 r2;
  IndexType nWG_col;

  GemvR_1Row_NWG(LHS &_l,RHS1 &_r1, RHS2 &_r2, IndexType &_nWG_col)
    : l(_l), r1(_r1), r2(_r2), nWG_col(_nWG_col){};

  IndexType getSize() { return r1.getSizeR(); }

  bool valid_thread (cl::sycl::nd_item<1> ndItem) { return true; }

  value_type eval(IndexType i) { // NOT VERIFIED
    auto dim = r2.getSize();

    auto val = iniAddOp1_struct::eval(r2.eval(0));
    for (IndexType j = 0; j < dim; j++) {
      val += r1.eval(i, j) * r2.eval(j);
    }
    return l.eval(i) = val;
  }

  value_type eval(cl::sycl::nd_item<1> ndItem) {
    return eval(ndItem.get_global_id(0));
  }

  template <typename sharedT>
  value_type eval(sharedT shrMem, cl::sycl::nd_item<1> ndItem) {
    IndexType localid = ndItem.get_local_id(0);
    IndexType localSz = ndItem.get_local_range(0);
    IndexType groupid = ndItem.get_group(0);
    IndexType groupSz = ndItem.get_group_range(0);
    IndexType glbalSz = ndItem.get_global_range(0);

    IndexType dimR  = r1.getSizeR();
    IndexType dimC  = r1.getSizeC();
    IndexType blqSz = (groupSz + nWG_col - 1) / nWG_col;  // number of "real" workgroups

    IndexType blqidR = groupid / nWG_col;  // row bloq id of the current workgroup
    IndexType blqidC = groupid % nWG_col;  // col blq id of the current workgroup

    IndexType vecS = r2.getSize();

    value_type val = addOp2_struct::init(r2);

    if (interLoop == 1) {
      IndexType frs_thrd = blqidC * localSz + localid;
      for (IndexType k = frs_thrd; k < vecS; k += localSz*nWG_col) {
        auto prod = prdOp2_struct::eval(r1.eval(blqidR,k),r2.eval(k));
        val = addOp2_struct::eval(val, prod);
      }
    } else { // NOT VERIFIED
      IndexType frs_thrd = interLoop * (groupid * localSz + localid);
      for (IndexType k = frs_thrd; k < vecS; k += interLoop * glbalSz) {
        for (IndexType k_int=k; k_int<std::min(k+interLoop,vecS);k_int++) {
          val = addOp2_struct::eval(val, r1.eval(groupid,k_int));
        }
      }
    }

    shrMem[localid] = val;
    // This barrier is mandatory to be sure the data is on the shared memory
    ndItem.barrier(cl::sycl::access::fence_space::local_space);

    // Reduction inside the block
    for (IndexType offset = localSz >> 1; offset > 0; offset >>= 1) {
      if (localid < offset) {
        shrMem[localid] =
            addOp2_struct::eval(shrMem[localid], shrMem[localid + offset]);
      }
      // This barrier is mandatory to be sure the data are on the shared memory
      ndItem.barrier(cl::sycl::access::fence_space::local_space);
    }
    if (localid == 0) {
      l.eval(blqidR,blqidC) = shrMem[localid];
    }

    return l.eval(blqidR,blqidC);
  }
  void bind(cl::sycl::handler &h) {
    l.bind(h);  
    r1.bind(h);
    r2.bind(h);
  }
};

template <unsigned int interLoop=1, typename LHS, typename RHS1, typename RHS2>
GemvR_1Row_NWG<interLoop, LHS, RHS1, RHS2>
    make_GemvR_1Row_NWG(LHS &l, RHS1 &r1, RHS2 &r2, typename RHS2::IndexType nWG_col) {
  return GemvR_1Row_NWG<interLoop, LHS, RHS1, RHS2>(l, r1, r2, nWG_col);
}

/**** GEMV BY ROWS M ROWS x N BLOCK ****/
#define GROUP_ROWS 1 // Not useful for GEMV by rows
//#define SHARED_ACCESS 1
template <unsigned int interLoop, class LHS, class RHS1, class RHS2>
struct GemvR_MRow_NWG {

  using value_type = typename RHS2::value_type;
  using IndexType = typename RHS2::IndexType;

  LHS  l;
  RHS1 r1;
  RHS2 r2;
  IndexType n_rows;
  IndexType nWG_col;


  GemvR_MRow_NWG(LHS &_l,RHS1 &_r1, RHS2 &_r2, IndexType &_n_rows, IndexType &_nWG_col)
    : l(_l), r1(_r1), r2(_r2), n_rows(_n_rows), nWG_col(_nWG_col){};

  IndexType getSize() { return r1.getSizeR(); }

  bool valid_thread (cl::sycl::nd_item<1> ndItem) { return true; }

  value_type eval(IndexType i) { // NOT VERIFIED
    auto dim = r2.getSize();

    auto val = iniAddOp1_struct::eval(r2.eval(0));
    for (IndexType j = 0; j < dim; j++) {
      val += r1.eval(i, j) * r2.eval(j);
    }
    return l.eval(i) = val;
  }

  value_type eval(cl::sycl::nd_item<1> ndItem) {
    return eval(ndItem.get_global_id(0));
  }

  template <typename sharedT>
  value_type eval(sharedT shrMem, cl::sycl::nd_item<1> ndItem) {
    IndexType localid = ndItem.get_local_id(0);
    IndexType localSz = ndItem.get_local_range(0);
    IndexType groupid = ndItem.get_group(0);
    IndexType groupSz = ndItem.get_group_range(0);
    IndexType glbalid = ndItem.get_global_id(0);
    IndexType glbalSz = ndItem.get_global_range(0);

    IndexType dimR = r1.getSizeR();
    IndexType dimC = r1.getSizeC();
    IndexType blqSz = (groupSz + nWG_col - 1) / nWG_col;  // number of "real" workgroups

    IndexType blqidR = groupid / nWG_col;  // row bloq id of the current workgroup
    IndexType blqidC = groupid % nWG_col;  // col blq id of the current workgroup

    IndexType vecS = r2.getSize();

    IndexType frs_row = blqidR*n_rows;

    value_type val = addOp2_struct::init(r2);
#ifdef GROUP_ROWS
    IndexType num_rows = 0;
    for (IndexType row=0, id_row=frs_row;
          (row<n_rows) && (id_row<dimR); row++, id_row++, num_rows++) {
  #ifdef SHARED_ACCESS
      shrMem[row*localSz+localid] = val;
  #else
      shrMem[row+n_rows*localid] = val;
  #endif
    }
#endif
    if (interLoop == 1) {
      IndexType frs_thrd = blqidC * localSz + localid;

#ifdef GROUP_ROWS
      for (IndexType k = frs_thrd; k < vecS; k += localSz*nWG_col) {
        auto elm = r2.eval(k);
        for (IndexType row=0, id_row=frs_row;
              (row<n_rows); row++, id_row++) {
          auto prod = prdOp2_struct::eval(r1.eval(id_row,k),elm);
  #ifdef SHARED_ACCESS
          shrMem[row*localSz+localid] =
                  addOp2_struct::eval(shrMem[row*localSz+localid], prod);
  #else
          shrMem[row+n_rows*localid] =
                  addOp2_struct::eval(shrMem[row+n_rows*localid], prod);
  #endif
        }
      }
#else
      IndexType id_row = frs_row;
      for (IndexType row=0; (row<n_rows); row++) {
        val = addOp2_struct::init(r2);
        for (IndexType k = frs_thrd; k < vecS; k += localSz*nWG_col) {
          auto prod = prdOp2_struct::eval(r1.eval(id_row,k),r2.eval(k));
          val = addOp2_struct::eval(val, prod);
        }
  #ifdef SHARED_ACCESS
        shrMem[row*localSz+localid] = val;
  #else
        shrMem[row+n_rows*localid] = val;
  #endif
        id_row++;
      }
#endif
    } else { // NOT VERIFIED
      IndexType frs_thrd = interLoop * (groupid * localSz + localid);
      for (IndexType k = frs_thrd; k < vecS; k += interLoop * glbalSz) {
        for (IndexType k_int=k; k_int<std::min(k+interLoop,vecS);k_int++) {
          val = addOp2_struct::eval(val, r1.eval(groupid,k_int));
        }
      }
    }

    // This barrier is mandatory to be sure the data is on the shared memory
    ndItem.barrier(cl::sycl::access::fence_space::local_space);
    // Reduction inside the block
    for (IndexType offset = localSz >> 1; offset > 0; offset >>= 1) {
      if (localid < offset) {
        for (IndexType row=0; row<n_rows; row++) {
#ifdef SHARED_ACCESS
          shrMem[row*localSz+localid] =
              addOp2_struct::eval(shrMem[row*localSz+localid],
                                  shrMem[row*localSz+localid+offset]);
#else
          shrMem[row+n_rows*localid] =
              addOp2_struct::eval(shrMem[row+n_rows*localid],
                                  shrMem[row+n_rows*(localid+offset)]);
#endif
        }
      }
      // This barrier is mandatory to be sure the data are on the shared memory
      ndItem.barrier(cl::sycl::access::fence_space::local_space);
    }
    if (localid == 0) {
      IndexType id_row=frs_row;
      for (IndexType row=0; row<n_rows; row++) {
#ifdef SHARED_ACCESS
        l.eval(id_row,blqidC) = shrMem[row*localSz];
#else
        l.eval(id_row,blqidC) = shrMem[row];
#endif
        id_row++;
      }
    }

    return l.eval(blqidR*n_rows,blqidC);
  }

  void bind(cl::sycl::handler &h) {
    l.bind(h);  
    r1.bind(h);
    r2.bind(h);
  }
};

template <unsigned int interLoop=1, typename LHS, typename RHS1, typename RHS2>
GemvR_MRow_NWG<interLoop, LHS, RHS1, RHS2>
    make_GemvR_MRow_NWG(LHS &l, RHS1 &r1, RHS2 &r2, typename RHS2::IndexType n_rows, typename RHS2::IndexType nWG_col) {
  return GemvR_MRow_NWG<interLoop, LHS, RHS1, RHS2>(l, r1, r2, n_rows, nWG_col);
}

/**** GEMV BY COLUMNS 1 ROW x 1 THREAD ****/
template <class RHS1, class RHS2>
struct GemvC_1Row_1Thread {
  using value_type = typename RHS2::value_type;
  using IndexType = typename RHS2::IndexType;

  RHS1 r1;
  RHS2 r2;


  GemvC_1Row_1Thread(RHS1 &_r1, RHS2 &_r2) : r1(_r1), r2(_r2){};

  IndexType getSize() { return r1.getSizeR(); }

  bool valid_thread (cl::sycl::nd_item<1> ndItem) { return true; }

  value_type eval(IndexType i) {
    auto dim = r2.getSize();

    auto val = iniAddOp1_struct::eval(r2.eval(0));
    for (IndexType j = 0; j < dim; j++) {
      auto prod = prdOp2_struct::eval(r1.eval(i,j),r2.eval(j));
      val = addOp2_struct::eval(val, prod);
    }
    return val;
  }

  value_type eval(cl::sycl::nd_item<1> ndItem) {
    return eval(ndItem.get_global_id(0));
  }
  void bind(cl::sycl::handler &h) {
    r1.bind(h);
    r2.bind(h);
  }
};

template <class RHS1, class RHS2>
GemvC_1Row_1Thread<RHS1, RHS2> make_GemvC_1Row_1Thread(RHS1 &r1, RHS2 &r2) {
  return GemvC_1Row_1Thread<RHS1, RHS2>(r1, r2);
}

/**** GEMV BY COLUMNS 1 ROW x 1 THREAD USING SHARED MEMORY ****/
template <class LHS, class RHS1, class RHS2, class RHS3>
struct GemvC_1Row_1Thread_ShMem {
  
  using value_type = typename RHS2::value_type;
  using IndexType = typename RHS2::IndexType;
  
  LHS l;
  value_type scl;
  RHS1 r1;
  RHS2 r2;
  RHS3 r3;

  GemvC_1Row_1Thread_ShMem(LHS &_l, value_type _scl, RHS1 &_r1, RHS2 &_r2, RHS3 &_r3)
      : l(_l), scl(_scl), r1(_r1), r2(_r2), r3(_r3) {};

  IndexType getSize() { return r1.getSizeR(); }

  bool valid_thread (cl::sycl::nd_item<1> ndItem) { return true; }

  value_type eval(IndexType i) {
    auto dim = r2.getSize();

    auto val = iniAddOp1_struct::eval(r2.eval(0));
    for (IndexType j = 0; j < dim; j++) {
      auto prod = prdOp2_struct::eval(r1.eval(i,j),r2.eval(j));
      val = addOp2_struct::eval(val, prod);
    }
    auto prod = prdOp2_struct::eval(scl, val);
    return l.eval(i) = addOp2_struct::eval(prod, r3.eval(i));
  }

  value_type eval(cl::sycl::nd_item<1> ndItem) {
    return eval(ndItem.get_global_id(0));
  }

  template <typename sharedT>
  value_type eval(sharedT scratch, cl::sycl::nd_item<1> ndItem) {
    IndexType localid = ndItem.get_local_id(0);
    IndexType localSz = ndItem.get_local_range(0);
    IndexType groupid = ndItem.get_group(0);
    IndexType groupSz = ndItem.get_group_range(0);
    IndexType glbalid = ndItem.get_global_id(0);

    IndexType dimR = r1.getSizeR();
    IndexType dimC = r1.getSizeC();

    auto val = iniAddOp1_struct::eval(r2.eval(0));
    for (IndexType k=0; k<dimC; k+=localSz) {
      ndItem.barrier(cl::sycl::access::fence_space::local_space);
      scratch[localid] = r2.eval(k+localid);
      ndItem.barrier(cl::sycl::access::fence_space::local_space);
      for (IndexType j=0; j<std::min(dimC-k,localSz); j++) {
        auto prod = prdOp2_struct::eval(r1.eval(glbalid,k+j),scratch[j]);
        val = addOp2_struct::eval(val, prod);
      }
    }
    auto prod = prdOp2_struct::eval(scl, val);
    return l.eval(glbalid) = addOp2_struct::eval(prod, r3.eval(glbalid));
  }
    void bind(cl::sycl::handler &h) {
    l.bind(h);  
    r1.bind(h);
    r2.bind(h);
    r3.bind(h);
  }
};

template <class LHS, class RHS1, class RHS2, class RHS3>
GemvC_1Row_1Thread_ShMem<LHS, RHS1, RHS2, RHS3> make_GemvC_1Row_1Thread_ShMem(
    LHS &l, typename LHS::value_type scl, RHS1 &r1, RHS2 &r2, RHS3 &r3) {
  return GemvC_1Row_1Thread_ShMem<LHS, RHS1, RHS2, RHS3>(l, scl, r1, r2, r3);
}

/**** GEMV BY COLUMNS 1 ROW x 1 THREAD USING SHARED MEMORY MINIMIZING SYNCHRONIZATION ****/
/**** This option uses too much memory, failing when the local memory is completed ****/
template <class LHS, class RHS1, class RHS2, class RHS3>
struct GemvC_1Row_1Thread_ShMem_Full {
  using value_type = typename RHS2::value_type;
  using IndexType = typename RHS2::IndexType;
  
  LHS l;
  value_type scl;
  RHS1 r1;
  RHS2 r2;
  RHS3 r3;

  GemvC_1Row_1Thread_ShMem_Full(LHS &_l, value_type _scl, RHS1 &_r1, RHS2 &_r2, RHS3 &_r3)
      : l(_l), scl(_scl), r1(_r1), r2(_r2), r3(_r3) {};

  IndexType getSize() { return r1.getSizeR(); }

  bool valid_thread (cl::sycl::nd_item<1> ndItem) { return true; }

  value_type eval(IndexType i) {
    auto dim = r2.getSize();

    auto val = iniAddOp1_struct::eval(r2.eval(0));
    for (IndexType j = 0; j < dim; j++) {
      auto prod = prdOp2_struct::eval(r1.eval(i,j),r2.eval(j));
      val = addOp2_struct::eval(val, prod);
    }
    auto prod = prdOp2_struct::eval(scl, val);
    return l.eval(i) = addOp2_struct::eval(prod, r3.eval(i));
  }

  value_type eval(cl::sycl::nd_item<1> ndItem) {
    return eval(ndItem.get_global_id(0));
  }

  template <typename sharedT>
  value_type eval(sharedT scratch, cl::sycl::nd_item<1> ndItem) {
    IndexType localid = ndItem.get_local_id(0);
    IndexType localSz = ndItem.get_local_range(0);
    IndexType groupid = ndItem.get_group(0);
    IndexType groupSz = ndItem.get_group_range(0);
    IndexType glbalid = ndItem.get_global_id(0);

    IndexType dimR = r1.getSizeR();
    IndexType dimC = r1.getSizeC();

    for (IndexType k=0; k<dimC; k+=localSz) {
      if ((k+localid) < dimC) scratch[k+localid] = r2.eval(k+localid);
    }

    ndItem.barrier(cl::sycl::access::fence_space::local_space);

    auto val = iniAddOp1_struct::eval(r2.eval(0));
    if (glbalid < dimR) {
      for (IndexType k=0; k<dimC; k++) {
        auto prod = prdOp2_struct::eval(r1.eval(glbalid,k),scratch[k]);
        val = addOp2_struct::eval(val, prod);
      }
      auto prod = prdOp2_struct::eval(scl, val);
      l.eval(glbalid) = val = addOp2_struct::eval(prod, r3.eval(glbalid));
    }
    return val;
  }
    void bind(cl::sycl::handler &h) {
    l.bind(h);  
    r1.bind(h);
    r2.bind(h);
    r3.bind(h);
  }
};

template <class LHS, class RHS1, class RHS2, class RHS3>
GemvC_1Row_1Thread_ShMem_Full<LHS, RHS1, RHS2, RHS3> make_GemvC_1Row_1Thread_ShMem_Full(
    LHS &l, typename LHS::value_type scl, RHS1 &r1, RHS2 &r2, RHS3 &r3) {
  return GemvC_1Row_1Thread_ShMem_Full<LHS, RHS1, RHS2, RHS3>(l, scl, r1, r2, r3);
}

/**** GEMV BY COLUMNS 1 ROW x M THREADS ****/
template <class LHS, class RHS1, class RHS2, class RHS3>
struct GemvC_1Row_MThreads {

  using value_type = typename RHS2::value_type;
  using IndexType = typename RHS2::IndexType;

  LHS l;
  value_type scl;
  RHS1 r1;
  RHS2 r2;
  RHS3 r3;
  IndexType nThr;

  GemvC_1Row_MThreads(LHS &_l, value_type _scl, RHS1 &_r1,
                            RHS2 &_r2, RHS3 &_r3, IndexType &_nThr)
      : l(_l), scl(_scl), r1(_r1), r2(_r2), r3(_r3), nThr(_nThr) {};

  IndexType getSize() { return r1.getSizeR(); }

  bool valid_thread (cl::sycl::nd_item<1> ndItem) { return true; }

  value_type eval(IndexType i) {
    auto dim = r2.getSize();

    auto val = iniAddOp1_struct::eval(r2.eval(0));
    for (IndexType j = 0; j < dim; j++) {
      auto prod = prdOp2_struct::eval(r1.eval(i,j),r2.eval(j));
      val = addOp2_struct::eval(val, prod);
    }
    return val;
  }

  template <typename sharedT>
  value_type eval(sharedT scratch, cl::sycl::nd_item<1> ndItem) {
    IndexType localid = ndItem.get_local_id(0);
    IndexType localSz = ndItem.get_local_range(0);
    IndexType groupid = ndItem.get_group(0);
    IndexType groupSz = ndItem.get_group_range(0);
    IndexType glbalid = ndItem.get_global_id(0);

    IndexType dimR = r1.getSizeR();
    IndexType dimC = r1.getSizeC();

    IndexType rowSz = (localSz + nThr - 1) / nThr;
    IndexType colSz = (dimC    + nThr - 1) / nThr;

    IndexType idWFR = (localid % rowSz);
    IndexType idWFC = (localid / rowSz);

    IndexType rowid = (groupid * rowSz) + idWFR;
    IndexType colid = colSz * idWFC;
    auto val = iniAddOp1_struct::eval(r2.eval(0));
    for (IndexType k=colid; k<std::min(dimC,colid+colSz); k++) {
      auto prod = prdOp2_struct::eval(r1.eval(rowid,k),r2.eval(k));
      val = addOp2_struct::eval(val, prod);
    }

    scratch[localid] = val;
    // This barrier is mandatory to be sure the data is on the shared memory
    ndItem.barrier(cl::sycl::access::fence_space::local_space);

    // Reduction inside the block
    for (IndexType offset = nThr >> 1; offset > 0; offset >>= 1) {
      if ((rowid < dimR) && (idWFC < offset)) {
        scratch[localid] += scratch[localid + offset * rowSz];
      }
      // This barrier is mandatory to be sure the data are on the shared memory
      ndItem.barrier(cl::sycl::access::fence_space::local_space);
    }
    // The result is stored in lhs
    if ((rowid < dimR) && (idWFC == 0)) {
      auto prod = prdOp2_struct::eval(scl, scratch[localid]);
      l.eval(rowid) = addOp2_struct::eval(prod, r3.eval(rowid));
    }

    return val;
  }
    void bind(cl::sycl::handler &h) {
    l.bind(h);  
    r1.bind(h);
    r2.bind(h);
    r3.bind(h);
  }
};

template <class LHS, class RHS1, class RHS2, class RHS3>
GemvC_1Row_MThreads<LHS, RHS1, RHS2, RHS3> make_GemvC_1Row_MThreads(
    LHS &l, typename LHS::value_type scl, RHS1 &r1, RHS2 &r2, RHS3 &r3, typename RHS2::IndexTypeRHS2::IndexType nThr) {
  return GemvC_1Row_MThreads<LHS, RHS1, RHS2, RHS3>(l, scl, r1, r2, r3, nThr);
}

/**** GEMV BY COLUMNS 1 ROW x M THREADS USING SHARED MEMORY****/
template <class LHS, class RHS1, class RHS2, class RHS3>
struct GemvC_1Row_MThreads_ShMem {
  
  using value_type = typename RHS2::value_type;
  using IndexType = typename RHS2::IndexType;
  
  LHS l;
  value_type scl;
  RHS1 r1;
  RHS2 r2;
  RHS3 r3;
  IndexType nThr;

  GemvC_1Row_MThreads_ShMem(LHS &_l, value_type _scl, RHS1 &_r1,
                            RHS2 &_r2, RHS3 &_r3, IndexType &_nThr)
      : l(_l), scl(_scl), r1(_r1), r2(_r2), r3(_r3), nThr(_nThr) {};

  IndexType getSize() { return r1.getSizeR(); }

  bool valid_thread (cl::sycl::nd_item<1> ndItem) { return true; }

  value_type eval(IndexType i) {
    auto dim = r2.getSize();

    auto val = iniAddOp1_struct::eval(r2.eval(0));
    for (IndexType j = 0; j < dim; j++) {
      auto prod = prdOp2_struct::eval(r1.eval(i,j),r2.eval(j));
      val = addOp2_struct::eval(val, prod);
    }
    return val;
  }

  template <typename sharedT>
  value_type eval(sharedT scratch, cl::sycl::nd_item<1> ndItem) {
    IndexType localid = ndItem.get_local_id(0);
    IndexType localSz = ndItem.get_local_range(0);
    IndexType groupid = ndItem.get_group(0);
    IndexType groupSz = ndItem.get_group_range(0);
    IndexType glbalid = ndItem.get_global_id(0);

    IndexType dimR = r1.getSizeR();
    IndexType dimC = r1.getSizeC();

    IndexType rowSz = (localSz + nThr - 1) / nThr;
    IndexType colSz = (dimC    + nThr - 1) / nThr;

    IndexType idWFR = (localid % rowSz);
    IndexType idWFC = (localid / rowSz);

    IndexType rowid = (groupid * rowSz) + idWFR;
    IndexType colid = colSz * idWFC;

    auto val = iniAddOp1_struct::eval(r2.eval(0));
    for (IndexType k=colid; k<std::min(colid+colSz,dimC); k+=rowSz) {
      ndItem.barrier(cl::sycl::access::fence_space::local_space);
      scratch[localid] = ((k+idWFR)<dimC)?r2.eval(k+idWFR):0.0;
      ndItem.barrier(cl::sycl::access::fence_space::local_space);
      if (rowid < dimR) {
        for (IndexType j=k; j<std::min(k+rowSz,std::min(colid+colSz,dimC)); j++) {
          auto prod = prdOp2_struct::eval(r1.eval(rowid,j),scratch[idWFC*rowSz+j-k]);
          val = addOp2_struct::eval(val, prod);
        }
      }
    }

    ndItem.barrier(cl::sycl::access::fence_space::local_space);
    scratch[localid] = val;
    // This barrier is mandatory to be sure the data is on the shared memory
    ndItem.barrier(cl::sycl::access::fence_space::local_space);

    // Reduction inside the block
    for (IndexType offset = nThr >> 1; offset > 0; offset >>= 1) {
      if ((rowid < dimR) && (idWFC < offset)) {
        scratch[localid] = addOp2_struct::eval(scratch[localid],
                                            scratch[localid + offset * rowSz]);
      }
      // This barrier is mandatory to be sure the data are on the shared memory
      ndItem.barrier(cl::sycl::access::fence_space::local_space);
    }
    // The result is stored in lhs
    if ((rowid < dimR) && (idWFC == 0)) {
      auto prod = prdOp2_struct::eval(scl, scratch[localid]);
      l.eval(rowid) = addOp2_struct::eval(prod, r3.eval(rowid));
    }

    return val;
  }
    void bind(cl::sycl::handler &h) {
    l.bind(h);  
    r1.bind(h);
    r2.bind(h);
    r3.bind(h);
  }
};

template <class LHS, class RHS1, class RHS2, class RHS3>
GemvC_1Row_MThreads_ShMem<LHS, RHS1, RHS2, RHS3> make_GemvC_1Row_MThreads_ShMem(
    LHS &l, typename LHS::value_type scl, RHS1 &r1, RHS2 &r2, RHS3 &r3, typename RHS2::IndexType nThr) {
  return GemvC_1Row_MThreads_ShMem<LHS, RHS1, RHS2, RHS3>(l, scl, r1, r2, r3, nThr);
}

/**** GEMV BY COLUMNS 1 ROW x M THREADS USING SHARED MEMORY, WITHOUT LOCAL ADDITION ****/
template <class LHS, class RHS1, class RHS2>
struct GemvC_1Row_MThreads_ShMem_NoRed {
 
  using value_type = typename RHS2::value_type;
  using IndexType = typename RHS2::IndexType;

  LHS l;
  RHS1 r1;
  RHS2 r2;
  IndexType nThr;

  GemvC_1Row_MThreads_ShMem_NoRed(LHS &_l, RHS1 &_r1, RHS2 &_r2, IndexType &_nThr)
      : l(_l), r1(_r1), r2(_r2), nThr(_nThr) {};

  IndexType getSize() { return r1.getSizeR(); }

  bool valid_thread (cl::sycl::nd_item<1> ndItem) { return true; }

  value_type eval(IndexType i) {
    auto dim = r2.getSize();

    auto val = iniAddOp1_struct::eval(r2.eval(0));
    for (IndexType j = 0; j < dim; j++) {
      auto prod = prdOp2_struct::eval(r1.eval(i,j),r2.eval(j));
      val = addOp2_struct::eval(val, prod);
    }
    return val;
  }

  template <typename sharedT>
  value_type eval(sharedT scratch, cl::sycl::nd_item<1> ndItem) {
    IndexType localid = ndItem.get_local_id(0);
    IndexType localSz = ndItem.get_local_range(0);
    IndexType groupid = ndItem.get_group(0);
    IndexType groupSz = ndItem.get_group_range(0);
    IndexType glbalid = ndItem.get_global_id(0);

    IndexType dimR = r1.getSizeR();
    IndexType dimC = r1.getSizeC();

    IndexType rowSz = (localSz + nThr - 1) / nThr;
    IndexType colSz = (dimC    + nThr - 1) / nThr;

    IndexType idWFR = (localid % rowSz);
    IndexType idWFC = (localid / rowSz);

    IndexType rowid = (groupid * rowSz) + idWFR;
    IndexType colid = colSz * idWFC;

    auto val = iniAddOp1_struct::eval(r2.eval(0));
    for (IndexType k=colid; k<std::min(dimC,colid+colSz); k+=rowSz) {
      ndItem.barrier(cl::sycl::access::fence_space::local_space);
      scratch[localid] = ((k+idWFR)<dimC)?r2.eval(k+idWFR):0.0;
      ndItem.barrier(cl::sycl::access::fence_space::local_space);
      for (IndexType j=k; j<std::min(k+rowSz,std::min(colid+colSz,dimC)); j++) {
        auto prod = prdOp2_struct::eval(r1.eval(rowid,j),scratch[idWFC*rowSz+j-k]);
        val = addOp2_struct::eval(val, prod);
      }
    }

    if (rowid < dimR) l.eval(rowid,idWFC) = val;

    return val;
  }
    void bind(cl::sycl::handler &h) {
    l.bind(h);
    r1.bind(h);
    r2.bind(h);
  }
};

template <class LHS, class RHS1, class RHS2>
GemvC_1Row_MThreads_ShMem_NoRed<LHS, RHS1, RHS2> make_GemvC_1Row_MThreads_ShMem_NoRed(
    LHS &l, RHS1 &r1, RHS2 &r2, typename RHS2::IndexType nThr) {
  return GemvC_1Row_MThreads_ShMem_NoRed<LHS, RHS1, RHS2>(l, r1, r2, nThr);
}

/**** GEMV BY COLUMNS 1 ROW x M BLOCKS ****/
template <class LHS, class RHS1, class RHS2>
struct GemvC_1Row_MBlocks {

  using value_type = typename RHS2::value_type;
  using IndexType = typename RHS2::IndexType;

  LHS l;
  RHS1 r1;
  RHS2 r2;
  IndexType nBlq;

  GemvC_1Row_MBlocks(LHS &_l, RHS1 &_r1, RHS2 &_r2, IndexType &_nBlq)
      : l(_l), r1(_r1), r2(_r2), nBlq(_nBlq) {};

  IndexType getSize() { return r1.getSizeR(); }

  bool valid_thread (cl::sycl::nd_item<1> ndItem) { return true; }

  value_type eval(IndexType i) {
    auto dim = r2.getSize();

    auto val = iniAddOp1_struct::eval(r2.eval(0));
    for (IndexType j = 0; j < dim; j++) {
      auto prod = prdOp2_struct::eval(r1.eval(i,j),r2.eval(j));
      val = addOp2_struct::eval(val, prod);
    }
    return val;
  }

  value_type eval(cl::sycl::nd_item<1> ndItem) {
    IndexType localid = ndItem.get_local_id(0);
    IndexType localSz = ndItem.get_local_range(0);
    IndexType groupid = ndItem.get_group(0);
    IndexType groupSz = ndItem.get_group_range(0);
    IndexType glbalid = ndItem.get_global_id(0);

    IndexType dimR = r1.getSizeR();
    IndexType dimC = r1.getSizeC();

    IndexType rowSz = localSz;
    IndexType colSz = (dimC    + nBlq - 1) / nBlq;

    IndexType dimWF = (groupSz + nBlq - 1) / nBlq;
    IndexType idWFR = (groupid / nBlq);
    IndexType idWFC = (groupid % nBlq);

    IndexType rowid = (idWFR * rowSz) + localid;
    IndexType colid = colSz * idWFC;
    auto val = iniAddOp1_struct::eval(r2.eval(0));
    if (rowid < dimR) {
      for (IndexType k=colid; k<std::min(dimC,colid+colSz); k++) {
        auto prod = prdOp2_struct::eval(r1.eval(rowid,k),r2.eval(k));
        val = addOp2_struct::eval(val, prod);
      }
      l.eval(rowid,idWFC) = val;
    }
    return val;
  }
    void bind(cl::sycl::handler &h) {
    l.bind(h);
    r1.bind(h);
    r2.bind(h);
  }
};

template <class LHS, class RHS1, class RHS2>
GemvC_1Row_MBlocks<LHS, RHS1, RHS2> make_GemvC_1Row_MBlocks(
    LHS &l, RHS1 &r1, RHS2 &r2, typename RHS2::IndexType nBlq) {
  return GemvC_1Row_MBlocks<LHS, RHS1, RHS2>(l, r1, r2, nBlq);
}

/**** GEMV BY COLUMNS 1 ROW x M BLOCKS USING SHARED MEMORY ****/
template <class LHS, class RHS1, class RHS2>
struct GemvC_1Row_MBlocks_ShMem {
 
  using value_type = typename RHS2::value_type;
  using IndexType = typename RHS2::IndexType;

  LHS l;
  RHS1 r1;
  RHS2 r2;
  IndexType nBlq;

  GemvC_1Row_MBlocks_ShMem(LHS &_l, RHS1 &_r1, RHS2 &_r2, IndexType &_nBlq)
      : l(_l), r1(_r1), r2(_r2), nBlq(_nBlq) {};

  IndexType getSize() { return r1.getSizeR(); }

  bool valid_thread (cl::sycl::nd_item<1> ndItem) { return true; }

  value_type eval(IndexType i) {
    auto dim = r2.getSize();

    auto val = iniAddOp1_struct::eval(r2.eval(0));
    for (IndexType j = 0; j < dim; j++) {
      auto prod = prdOp2_struct::eval(r1.eval(i,j),r2.eval(j));
      val = addOp2_struct::eval(val, prod);
    }
    return val;
  }

  template <typename sharedT>
  value_type eval(sharedT scratch, cl::sycl::nd_item<1> ndItem) {
    IndexType localid = ndItem.get_local_id(0);
    IndexType localSz = ndItem.get_local_range(0);
    IndexType groupid = ndItem.get_group(0);
    IndexType groupSz = ndItem.get_group_range(0);
    IndexType glbalid = ndItem.get_global_id(0);

    IndexType dimR = r1.getSizeR();
    IndexType dimC = r1.getSizeC();

    IndexType rowSz = localSz;
    IndexType colSz = (dimC    + nBlq - 1) / nBlq;

    IndexType dimWF = (groupSz + nBlq - 1) / nBlq;
    IndexType idWFR = (groupid / nBlq);
    IndexType idWFC = (groupid % nBlq);

    IndexType rowid = (idWFR * rowSz) + localid;
    IndexType colid = colSz * idWFC;
    auto val = iniAddOp1_struct::eval(r2.eval(0));
    for (IndexType k=colid; k<std::min(colid+colSz,dimC); k+=rowSz) {
      ndItem.barrier(cl::sycl::access::fence_space::local_space);
      scratch[localid] = r2.eval(k+localid);
      ndItem.barrier(cl::sycl::access::fence_space::local_space);
      if (rowid < dimR) {
        for (IndexType j=k; j<std::min(k+rowSz,std::min(colid+colSz,dimC)); j++) {
          auto prod = prdOp2_struct::eval(r1.eval(rowid,j),scratch[j-k]);
          val = addOp2_struct::eval(val, prod);
        }
      }
    }

    if (rowid < dimR) l.eval(rowid,idWFC) = val;
    return val;
  }
    void bind(cl::sycl::handler &h) {
    l.bind(h);
    r1.bind(h);
    r2.bind(h);
  }
};

template <class LHS, class RHS1, class RHS2>
GemvC_1Row_MBlocks_ShMem<LHS, RHS1, RHS2> make_GemvC_1Row_MBlocks_ShMem(
    LHS &l, RHS1 &r1, RHS2 &r2, typename RHS2::IndexType nBlq) {
  return GemvC_1Row_MBlocks_ShMem<LHS, RHS1, RHS2>(l, r1, r2, nBlq);
}

/**** GEMV BY COLUMNS 1 ROW x M BLOCKS USING SHARED MEMORY MINIMIZING SYNCHRONIZATION ****/
template <class LHS, class RHS1, class RHS2>
struct GemvC_1Row_MBlocks_ShMem_Full {
  
  using value_type = typename RHS2::value_type;
  using IndexType = typename RHS2::IndexType;

  LHS l;
  RHS1 r1;
  RHS2 r2;
  IndexType nBlq;

  GemvC_1Row_MBlocks_ShMem_Full(LHS &_l, RHS1 &_r1, RHS2 &_r2, IndexType &_nBlq)
      : l(_l), r1(_r1), r2(_r2), nBlq(_nBlq) {};

  IndexType getSize() { return r1.getSizeR(); }

  bool valid_thread (cl::sycl::nd_item<1> ndItem) { return true; }

  value_type eval(IndexType i) {
    auto dim = r2.getSize();

    auto val = iniAddOp1_struct::eval(r2.eval(0));
    for (IndexType j = 0; j < dim; j++) {
      auto prod = prdOp2_struct::eval(r1.eval(i,j),r2.eval(j));
      val = addOp2_struct::eval(val, prod);
    }
    return val;
  }

  template <typename sharedT>
  value_type eval(sharedT scratch, cl::sycl::nd_item<1> ndItem) {
    IndexType localid = ndItem.get_local_id(0);
    IndexType localSz = ndItem.get_local_range(0);
    IndexType groupid = ndItem.get_group(0);
    IndexType groupSz = ndItem.get_group_range(0);

    IndexType dimR = r1.getSizeR();
    IndexType dimC = r1.getSizeC();

    IndexType rowSz = (dimR < localSz)? dimR: localSz;
    IndexType colSz = (dimC    + nBlq - 1) / nBlq;

    IndexType dimWF = (groupSz + nBlq - 1) / nBlq;
    IndexType idWFR = (groupid % dimWF);
    IndexType idWFC = (groupid / dimWF);

    IndexType rowid = (idWFR * rowSz) + localid;
    IndexType colid = colSz * idWFC;

    IndexType j;
    j = localid;
    for (IndexType k=colid+localid; k<std::min(colid+colSz,dimC); k+=rowSz) {
      scratch[j] = r2.eval(k);
      j+=rowSz;
    }

    ndItem.barrier(cl::sycl::access::fence_space::local_space);

    auto val = iniAddOp1_struct::eval(r2.eval(0));
    if (rowid < dimR) {
      j = 0;
      for (IndexType k=colid; k<std::min(colid+colSz,dimC); k++) {
        val += r1.eval(rowid,k) * scratch[j++];
      }
      l.eval(rowid,idWFC) = val;
    }
    return val;
  }
    void bind(cl::sycl::handler &h) {
    l.bind(h);
    r1.bind(h);
    r2.bind(h);
  }
};

template <class LHS, class RHS1, class RHS2>
GemvC_1Row_MBlocks_ShMem_Full<LHS, RHS1, RHS2> make_GemvC_1Row_MBlocks_ShMem_Full(
    LHS &l, RHS1 &r1, RHS2 &r2, typename RHS2::IndexType nBlq) {
  return GemvC_1Row_MBlocks_ShMem_Full<LHS, RHS1, RHS2>(l, r1, r2, nBlq);
}

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


  bool valid_thread (cl::sycl::nd_item<1> ndItem) { return true; }

  value_type eval(IndexType i) {
    auto dim = r2.getSize();

    auto val = iniAddOp1_struct::eval(r2.eval(0));
    for (IndexType j = 0; j < dim; j++) {
      val += r1.eval(i, j) * r2.eval(j);
    }
    return val;
  }

  value_type eval(cl::sycl::nd_item<1> ndItem) {
    return eval(ndItem.get_global_id(0));
  }
  IndexType getSize() { return r1.getSizeR(); }

  void bind(cl::sycl::handler &h) {
    r1.bind(h);
    r2.bind(h);
  }
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



  bool valid_thread (cl::sycl::nd_item<1> ndItem) { return true; }

  value_type eval(IndexType i) {
    auto dim = r2.getSize();

    auto val = iniAddOp1_struct::eval(r2.eval(0));
    for (IndexType j = 0; j < dim; j++) {
      val += r1.eval(i, j) * r2.eval(j);
    }
    l.eval(i) = scl * val + r3.eval(i);
    return val;
  }

  template <typename sharedT>
  value_type eval(sharedT scratch, cl::sycl::nd_item<1> ndItem) {
    IndexType localid = ndItem.get_local_id(0);
    IndexType localSz = ndItem.get_local_range(0);
    IndexType groupid = ndItem.get_group(0);
    IndexType groupSz = ndItem.get_group_range(0);

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

  void bind(cl::sycl::handler &h) {
    l.bind(h);
    r1.bind(h);
    r2.bind(h);
    r3.bind(h);
  }
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
    IndexType localid = ndItem.get_local_id(0);
    IndexType localSz = ndItem.get_local_range(0);
    IndexType groupid = ndItem.get_group(0);
    IndexType groupSz = ndItem.get_group_range(0);

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
    for (IndexType j = colid + localid; j < std::min(colid+colSz,dimC); j += rowSz) {
      scratch[k] = r2.eval(j);
      k += rowSz;
    }

    // This barrier is mandatory to be sure the data are on the shared memory
    ndItem.barrier(cl::sycl::access::fence_space::local_space);

    // Local computation
    auto val = iniAddOp1_struct::eval(r2.eval(0));
    if (rowid < dimR) {
      k = 0;
      for (IndexType j = colid; j < std::min(colid+colSz,dimC); j++) {
        val += r1.eval(rowid, j) * scratch[k++];
      }
      // The result is stored in lhs
      l.eval(rowid, blqidC) = val;
    }

    return val;
  }

  IndexType getSize() { return r1.getSizeR(); }
  void bind(cl::sycl::handler &h) {
    l.bind(h);
    r1.bind(h);
    r2.bind(h);
  }
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

bool valid_thread (cl::sycl::nd_item<1> ndItem) { return true; }

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
    return eval(ndItem.get_global_id(0));
  }

  IndexType getSize() { return r1.getSizeR(); }

  void bind(cl::sycl::handler &h) {
    l.bind(h);
    r1.bind(h);
    r2.bind(h);
  }
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

 bool valid_thread (cl::sycl::nd_item<1> ndItem) { return true; }

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
    return eval(ndItem.get_global_id(0));
  }

#if BLAS_EXPERIMENTAL
  template <typename sharedT>
  value_type eval(sharedT scratch, cl::sycl::nd_item<1> ndItem) {
    IndexType Pieces = 2;

    IndexType localid = ndItem.get_local_id(0);
    IndexType localSz = ndItem.get_local_range(0);
    IndexType groupid = ndItem.get_group(0);
    IndexType groupSz = ndItem.get_group_range(0);
    IndexType globalid = ndItem.get_global_id(0);
    IndexType globalSz = ndItem.get_global_id_range(0);

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
    return eval(ndItem.get_global_id(0));
  }
#endif  // BLAS_EXPERIMENTAL
  IndexType getSize() { return r1.getSizeR(); }

  void bind(cl::sycl::handler &h) {
    r1.bind(h);
    r2.bind(h);
  }
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

  bool valid_thread (cl::sycl::nd_item<1> ndItem) { return true; }

  value_type eval(IndexType i) {
    auto size = (r1.getAccess()) ? r1.getSizeC() : r1.getSizeR();
    auto row = (r1.getAccess()) ? (i / size) : (i % size);
    auto col = (r1.getAccess()) ? (i % size) : (i / size);

    auto val = r2.eval(row) * r3.eval(col);

    return val;
  }

  value_type eval(cl::sycl::nd_item<1> ndItem) {
    return eval(ndItem.get_global_id(0));
  }

  IndexType getSize() { return r1.getSize(); }

  void bind(cl::sycl::handler &h) {
    r1.bind(h);
    r2.bind(h);
    r3.bind(h);
  }
};

template <class RHS1, class RHS2, class RHS3>
ModifRank1<RHS1, RHS2, RHS3> make_modifRank1(RHS1 &r1, RHS2 &r2, RHS3 &r3) {
  return ModifRank1<RHS1, RHS2, RHS3>(r1, r2, r3);
}

template <class LHS, class RHS1, class RHS2>
struct Ger_1Row_1WG {
  
  using value_type = typename RHS2::value_type;
  using IndexType = typename RHS2::IndexType;
  
  LHS l;
  RHS1 r1;
  RHS2 r2;
  value_type scl;

  Ger_1Row_1WG(LHS &_l, value_type _scl, RHS1 &_r1, RHS2 &_r2)
    : l(_l), scl(_scl), r1(_r1), r2(_r2) { };

  IndexType getSize() { return r1.getSize(); }

  bool valid_thread (cl::sycl::nd_item<1> ndItem) { return true; }

  value_type eval(IndexType i) {
    auto size = (l.getAccess()) ? l.getSizeC() : l.getSizeR();
    auto row = (l.getAccess()) ? (i / size) : (i % size);
    auto col = (l.getAccess()) ? (i % size) : (i / size);

    auto val = scl * r1.eval(row) * r2.eval(col);

    return l.eval(i) += val;
  }

  value_type eval(cl::sycl::nd_item<1> ndItem) {
    IndexType localid = ndItem.get_local_id(0);
    IndexType localSz = ndItem.get_local_range(0);
    IndexType groupid = ndItem.get_group(0);
    IndexType groupSz = ndItem.get_group_range(0);
    IndexType glbalid = ndItem.get_global_id(0);

    IndexType dimR = l.getSizeR();
    IndexType dimC = l.getSizeC();

    value_type val = scl * r1.eval(groupid);

    IndexType frs_thrd = localid;
    for (IndexType k = frs_thrd; k < dimC; k += localSz) {
      l.eval(groupid,k) += val * r2.eval(k);
    }

    return val;
  }
  void bind(cl::sycl::handler &h) {
    l.bind(h);
    r1.bind(h);
    r2.bind(h);
  }
};

template <class LHS, class RHS1, class RHS2>
Ger_1Row_1WG<LHS, RHS1, RHS2> make_Ger_1Row_1WG(
    LHS &l, typename LHS::value_type scl, RHS1 &r1, RHS2 &r2) {
  return Ger_1Row_1WG<LHS, RHS1, RHS2>(l, scl, r1, r2);
}

template <class LHS, class RHS1, class RHS2>
struct Ger_MRow_NWG {
  
  using value_type = typename RHS2::value_type;
  using IndexType = typename RHS2::IndexType;
  
  LHS l;
  RHS1 r1;
  RHS2 r2;
  IndexType n_rows;
  IndexType nWG_col;

  value_type scl;

  Ger_MRow_NWG(LHS &_l, value_type _scl, RHS1 &_r1, RHS2 &_r2, IndexType &_n_rows, IndexType &_nWG_col)
    : l(_l), scl(_scl), r1(_r1), r2(_r2), n_rows(_n_rows), nWG_col(_nWG_col) { };

  IndexType getSize() { return r1.getSize(); }

  bool valid_thread (cl::sycl::nd_item<1> ndItem) { return true; }

  value_type eval(IndexType i) {
    auto size = (l.getAccess()) ? l.getSizeC() : l.getSizeR();
    auto row = (l.getAccess()) ? (i / size) : (i % size);
    auto col = (l.getAccess()) ? (i % size) : (i / size);

    auto val = scl * r1.eval(row) * r2.eval(col);

    return l.eval(i) += val;
  }

  template <typename sharedT>
  value_type eval(sharedT shrMem, cl::sycl::nd_item<1> ndItem) {
    IndexType localid = ndItem.get_local_id(0);
    IndexType localSz = ndItem.get_local_range(0);
    IndexType groupid = ndItem.get_group(0);
    IndexType groupSz = ndItem.get_group_range(0);
    IndexType glbalid = ndItem.get_global_id(0);

    IndexType dimR = l.getSizeR();
    IndexType dimC = l.getSizeC();

    IndexType nWG_row = (groupSz + nWG_col - 1) / nWG_col;  // number of "row" workgroups
    IndexType blqidR  = groupid % nWG_row;  // row bloq id of the current workgroup
    IndexType blqidC  = groupid / nWG_row;  // col blq id of the current workgroup

    IndexType dimWFC = (dimC + (localSz*nWG_col) - 1) / (localSz*nWG_col) * localSz;

    IndexType frs_row = blqidR*n_rows;

    for (IndexType row=localid; (row<n_rows); row+=localSz) {
      shrMem[row] = scl * r1.eval(frs_row+row);
    }

    // This barrier is mandatory to be sure the data is on the shared memory
    ndItem.barrier(cl::sycl::access::fence_space::local_space);

    IndexType frs_thrd = blqidC * dimWFC + localid;
    IndexType lst_thrd = std::min(dimC,frs_thrd + dimWFC);
    for (IndexType k = frs_thrd; k < lst_thrd; k += localSz) {
      auto val = r2.eval(k);
      for (IndexType id_row=frs_row, row=0; id_row<std::min(dimR,frs_row+n_rows); id_row++, row++) {
          l.eval(id_row,k) += shrMem[row] * val;

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

template <class LHS, class RHS1, class RHS2>
Ger_MRow_NWG<LHS, RHS1, RHS2> make_Ger_MRow_NWG(
    LHS &l, typename LHS::value_type scl, RHS1 &r1, RHS2 &r2, typename RHS2::IndexType n_rows, typename RHS2::IndexType nWG_col) {
  return Ger_MRow_NWG<LHS, RHS1, RHS2>(l, scl, r1, r2, n_rows, nWG_col);
}

template <class LHS, class RHS1, class RHS2>
struct Ger_1Row_1Thread {
  
  using value_type = typename RHS2::value_type;
  using IndexType = typename RHS2::IndexType;
  
  LHS l;
  RHS1 r1;
  RHS2 r2;
  value_type scl;

  Ger_1Row_1Thread(LHS &_l, value_type _scl, RHS1 &_r1, RHS2 &_r2)
    : l(_l), scl(_scl), r1(_r1), r2(_r2) { };

  IndexType getSize() { return r1.getSize(); }

  bool valid_thread (cl::sycl::nd_item<1> ndItem) { return true; }

  value_type eval(IndexType i) {
    auto size = (l.getAccess()) ? l.getSizeC() : l.getSizeR();
    auto row = (l.getAccess()) ? (i / size) : (i % size);
    auto col = (l.getAccess()) ? (i % size) : (i / size);

    auto val = scl * r1.eval(row) * r2.eval(col);

    return l.eval(i) += val;
  }

  value_type eval(cl::sycl::nd_item<1> ndItem) {
    IndexType localid = ndItem.get_local_id(0);
    IndexType localSz = ndItem.get_local_range(0);
    IndexType groupid = ndItem.get_group(0);
    IndexType groupSz = ndItem.get_group_range(0);
    IndexType glbalid = ndItem.get_global_id(0);

    IndexType dimR = l.getSizeR();
    IndexType dimC = l.getSizeC();

    IndexType row_id = groupid*localSz+localid;

    value_type val = scl * r1.eval(row_id);

    if (row_id < dimR) {
      for (IndexType k = 0; k < dimC; k ++) {
        l.eval(row_id,k) += val * r2.eval(k);
      }
    }

    return val;
  }

  void bind(cl::sycl::handler &h) {
    l.bind(h);
    r1.bind(h);
    r2.bind(h);
  }
};

template <class LHS, class RHS1, class RHS2>
Ger_1Row_1Thread<LHS, RHS1, RHS2> make_Ger_1Row_1Thread(
    LHS &l, typename LHS::value_type scl, RHS1 &r1, RHS2 &r2) {
  return Ger_1Row_1Thread<LHS, RHS1, RHS2>(l, scl, r1, r2);
}

template <class LHS, class RHS1, class RHS2>
struct Ger_1Row_NWG_ShMem {
  
  using value_type = typename RHS2::value_type;
  using IndexType = typename RHS2::IndexType;
  
  LHS l;
  RHS1 r1;
  RHS2 r2;
  IndexType n_cols;
  IndexType nWG_row;


  value_type scl;

  Ger_1Row_NWG_ShMem(LHS &_l, value_type _scl, RHS1 &_r1, RHS2 &_r2, IndexType &_n_cols, IndexType &_nWG_row)
    : l(_l), scl(_scl), r1(_r1), r2(_r2), n_cols(_n_cols), nWG_row(_nWG_row) { };

  IndexType getSize() { return r1.getSize(); }

  bool valid_thread (cl::sycl::nd_item<1> ndItem) { return true; }

  value_type eval(IndexType i) {
    auto size = (l.getAccess()) ? l.getSizeC() : l.getSizeR();
    auto row = (l.getAccess()) ? (i / size) : (i % size);
    auto col = (l.getAccess()) ? (i % size) : (i / size);

    auto val = scl * r1.eval(row) * r2.eval(col);

    return l.eval(i) += val;
  }

  template <typename sharedT>
  value_type eval(sharedT shrMem, cl::sycl::nd_item<1> ndItem) {
    IndexType localid = ndItem.get_local_id(0);
    IndexType localSz = ndItem.get_local_range(0);
    IndexType groupid = ndItem.get_group(0);
    IndexType groupSz = ndItem.get_group_range(0);
    IndexType glbalid = ndItem.get_global_id(0);

    IndexType dimR = l.getSizeR();
    IndexType dimC = l.getSizeC();

    IndexType nWG_col = (groupSz + nWG_row - 1) / nWG_row;  // number of "col" workgroups
    IndexType blqidR  = groupid % nWG_row;  // row bloq id of the current workgroup
    IndexType blqidC  = groupid / nWG_row;  // col blq id of the current workgroup

    IndexType dimWFR = (dimR + (localSz*nWG_row) - 1) / (localSz*nWG_row) * localSz;

    IndexType frs_col = blqidC*n_cols;

    for (IndexType col=localid; (col<n_cols); col+=localSz) {
      shrMem[col] = scl * r2.eval(frs_col+col);
    }

    // This barrier is mandatory to be sure the data is on the shared memory
    ndItem.barrier(cl::sycl::access::fence_space::local_space);

    IndexType frs_row = blqidR * dimWFR + localid;
    IndexType lst_row = std::min(dimR,frs_row + dimWFR);
    for (IndexType k = frs_row; k < lst_row; k += localSz) {
      auto val = r1.eval(k);
      for (IndexType id_col=frs_col, col=0; id_col<std::min(dimC,frs_col+n_cols); id_col++, col++) {
        l.eval(k,id_col) += val * shrMem[col];
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

template <class LHS, class RHS1, class RHS2>
Ger_1Row_NWG_ShMem<LHS, RHS1, RHS2> make_Ger_1Row_NWG_ShMem(
    LHS &l, typename LHS::value_type scl, RHS1 &r1, RHS2 &r2, typename RHS2::IndexType n_rows, typename RHS2::IndexType nWG_col) {
  return Ger_1Row_NWG_ShMem<LHS, RHS1, RHS2>(l, scl, r1, r2, n_rows, nWG_col);
}

}  // namespace blas

#endif  // BLAS2_TREES_HPP