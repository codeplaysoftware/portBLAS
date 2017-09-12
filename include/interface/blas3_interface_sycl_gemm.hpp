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
 *  @filename blas3_interface_sycl.hpp
 *
 **************************************************************************/

#ifndef BLAS3_INTERFACE_SYCL_GEMM_HPP
#define BLAS3_INTERFACE_SYCL_GEMM_HPP


#include <algorithm>
#include <cctype>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>


#include <executors/executor_sycl.hpp>
#include <operations/blas3_trees_gemm.hpp>


using namespace cl::sycl;


namespace blas {


template <typename T>
struct Wrap {};


/*!
 * @brief Launch a kernel which calls a GemmFactory instance given by template
 *        parameter Gemm.
 */
template <typename Gemm, typename ExecutorType, typename T>
void _gemm_tr(Executor<ExecutorType> ex, int _M, int _N, int _K, T _alpha,
              cl::sycl::buffer<T, 1> _A, int _lda,
              cl::sycl::buffer<T, 1> _B, int _ldb, T _beta,
              cl::sycl::buffer<T, 1> _C, int _ldc) {
  ex.sycl_queue().submit([&](handler &h) {
    auto accA = _A.template get_access<access::mode::read>(h);
    auto accB = _B.template get_access<access::mode::read>(h);
    auto accC = _C.template get_access<access::mode::read_write>(h);
    accessor<T, 1, access::mode::read_write, access::target::local> scratch(
        range<1>(Gemm::scratch_size), h);
    h.parallel_for<Wrap<Gemm>>(Gemm::get_nd_range(_M, _N), [=](nd_item<1> id) {
      Gemm::run(id, id.get_group(0), id.get_local(0), _M, _N, _K, T(_alpha),
                accA.get_pointer(), _lda, accB.get_pointer(), _ldb, T(_beta),
                accC.get_pointer(), _ldc, scratch.get_pointer());
    });
  });
  ex.sycl_queue().wait();
}


/*!
 * @brief Select the correct transpose version of GemmFactory, depending on the
 *        runtime values of transpose.
 */
template <bool DoubleBuffer, bool ConflictA, bool ConflictB, size_t ClSize,
          typename TileT, typename ExecutorType, typename T>
void _select_gemm(
    Executor<ExecutorType> ex, bool _TransA, bool _TransB, int _M,
               int _N, int _K, T _alpha, cl::sycl::buffer<T, 1> _A, int _lda,
               cl::sycl::buffer<T, 1> _B, int _ldb, T _beta,
               cl::sycl::buffer<T, 1> _C, int _ldc) {
  #define ENABLE_GEMM_TRANSPOSE(_trans_a, _trans_b) \
  if (_TransA == _trans_a && _TransB == _trans_b) { \
    _gemm_tr<\
      GemmFactory<DoubleBuffer, ConflictA, ConflictB, ClSize, TileT, \
                  _trans_a, _trans_b, T>>( \
        ex, _M, _N, _K, _alpha, _A, _lda, _B, _ldb, _beta, _C, _ldc); \
    return; \
  }

  const bool NoTrans = false;
  const bool Trans   =  true;

  ENABLE_GEMM_TRANSPOSE(NoTrans, NoTrans);
  ENABLE_GEMM_TRANSPOSE(  Trans, NoTrans);
  ENABLE_GEMM_TRANSPOSE(NoTrans,   Trans);
  ENABLE_GEMM_TRANSPOSE(  Trans,   Trans);

  #undef ENABLE_GEMM_TRANSPOSE
}


/*!
 * @brief This is a top-level wrapper for GemmFactory, which provides a
 *        "standard" BLAS gemm interface.
 *
 * See netlib.org/blas for details.
 */
template <typename ExecutorType, typename T>
void _gemm(Executor<ExecutorType> ex, char _TransA, char _TransB, int _M,
           int _N, int _K, T _alpha, cl::sycl::buffer<T, 1> _A, int _lda,
           cl::sycl::buffer<T, 1> _B, int _ldb, T _beta,
           cl::sycl::buffer<T, 1> _C, int _ldc) {
  _TransA = tolower(_TransA);
  _TransB = tolower(_TransB);
  if(_TransA != 'n' && _TransA != 't' && _TransA != 'c') {
    throw std::invalid_argument("invalid _TransA");
  } else if(_TransB != 'n' && _TransB != 't' && _TransB != 'c') {
    throw std::invalid_argument("invalid _TransB");
  }

  bool _TrA = _TransA != 'n';
  bool _TrB = _TransB != 'n';

  #define BIND_DATA_SIZE(_m, _n, _k) \
    if (_M == (_m) && _N == (_n) && _K == (_k))

  #define BIND_DEFAULT

  #define TO_TPARAMS(_db, _tir, _tic, _twr, _twc) {\
    _select_gemm<_db, false, false, 64, Tile<_tir, _tic, _twr, _twc>>( \
        ex, _TrA, _TrB, _M, _N, _K, _alpha, _A, _lda, _B, _ldb, \
        _beta, _C, _ldc); \
    return; \
  }

  if (ex.get_device_type() == Executor<SYCL>::device_type::INTELGPU) {
    BIND_DATA_SIZE(1024, 4096, 1024) TO_TPARAMS(false, 4, 4, 16, 16);
    BIND_DATA_SIZE(  10, 1024, 1024) TO_TPARAMS(false, 2, 2,  8,  8);
    BIND_DEFAULT                     TO_TPARAMS(false, 8, 8,  8,  8);
  } else {
    BIND_DATA_SIZE(  10, 1024, 1024) TO_TPARAMS( true, 1, 1, 16, 16);
    BIND_DEFAULT                     TO_TPARAMS(false, 8, 8, 16, 16);
  }

  #undef BIND_DATA_SIZE
  #undef BIND_DEFAULT
  #undef TO_TPARAMS
}


}  // namespace blas


#endif  // BLAS3_INTERFACE_SYCL_HPP

