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
 *  @filename blas3_interface.hpp
 *
 **************************************************************************/

#ifndef BLAS3_INTERFACE_HPP
#define BLAS3_INTERFACE_HPP

#include <algorithm>
#include <cctype>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>

#include <executors/executor_sycl.hpp>
#include <interface/blas_interface_sycl.hpp>
#include <operations/blas3_trees.hpp>

namespace blas {

#ifndef WG_SIZE
#define WG_SIZE 128
#endif
/*!
 * @brief Select the correct transpose version of GemmFactory, depending on the
 *        runtime values of transpose.
 */
template <int WgSize, bool DoubleBuffer, bool ConflictA, bool ConflictB,
          int ClSize, typename TileT, typename Executor, typename ContainerT0,
          typename ContainerT1, typename ContainerT2, typename T,
          typename IndexType>
typename Executor::Return_Type _select_gemm(
    Executor& ex, bool _TransA, bool _TransB, IndexType _M, IndexType _N,
    IndexType _K, T _alpha, ContainerT0 _A, IndexType _lda, ContainerT1 _B,
    IndexType _ldb, T _beta, ContainerT2 _C, IndexType _ldc) {
  typename Executor::Return_Type ret;

  auto buffer_a = make_matrix_view(ex, _A, _M, _K, _lda, Access::ColMajor());
  auto buffer_b = make_matrix_view(ex, _B, _K, _N, _ldb, Access::ColMajor());
  auto buffer_c = make_matrix_view(ex, _C, _M, _N, _ldc, Access::ColMajor());
#ifndef NAIVE_GEMM
#define ENABLE_GEMM_TRANSPOSE(_trans_a, _trans_b)                              \
  if (_TransA == _trans_a && _TransB == _trans_b) {                            \
    if (ex.has_local_memory()) {                                               \
      auto gemm = make_gemm<DoubleBuffer, ConflictA, ConflictB, ClSize, TileT, \
                            _trans_a, _trans_b>(buffer_a, buffer_b, buffer_c,  \
                                                T(_alpha), T(_beta));          \
      ret = ex.gemm_executor(gemm);                                            \
    } else {                                                                   \
      auto gemm = make_gemm_no_local_mem<ClSize, TileT, _trans_a, _trans_b>(   \
          buffer_a, buffer_b, buffer_c, T(_alpha), T(_beta));                  \
      ret = ex.gemm_executor(gemm);                                            \
    }                                                                          \
    return ret;                                                                \
  }
#else
#define ENABLE_GEMM_TRANSPOSE(_trans_a, _trans_b)                \
  if (_TransA == _trans_a && _TransB == _trans_b) {              \
    auto gemm = make_gemm_reference<WgSize, _trans_a, _trans_b>( \
        buffer_a, buffer_b, buffer_c, T(_alpha), T(_beta));      \
    ret = ex.gemm_executor(gemm);                                \
    return ret;                                                  \
  }
#endif
  const bool NoTrans = false;
  const bool Trans = true;

  ENABLE_GEMM_TRANSPOSE(NoTrans, NoTrans);
  ENABLE_GEMM_TRANSPOSE(Trans, NoTrans);
  ENABLE_GEMM_TRANSPOSE(NoTrans, Trans);
  ENABLE_GEMM_TRANSPOSE(Trans, Trans);

#undef ENABLE_GEMM_TRANSPOSE
  return ret;
}  // namespace blas

/*!
 * @brief This is a top-level wrapper for GemmFactory, which provides a
 *        "standard" BLAS gemm interface.
 *
 * See netlib.org/blas for details.
 */
template <typename Executor, typename ContainerT0, typename ContainerT1,
          typename ContainerT2, typename T, typename IndexType>
cl::sycl::event _gemm(Executor& ex, char _TransA, char _TransB, IndexType _M,
                      IndexType _N, IndexType _K, T _alpha, ContainerT0 _A,
                      IndexType _lda, ContainerT1 _B, IndexType _ldb, T _beta,
                      ContainerT2 _C, IndexType _ldc) {
  _TransA = tolower(_TransA);
  _TransB = tolower(_TransB);

  if (_TransA != 'n' && _TransA != 't' && _TransA != 'c') {
    throw std::invalid_argument("invalid _TransA");
  } else if (_TransB != 'n' && _TransB != 't' && _TransB != 'c') {
    throw std::invalid_argument("invalid _TransB");
  }

  bool _TrA = _TransA != 'n';
  bool _TrB = _TransB != 'n';
#define BIND_DATA_SIZE(_m, _n, _k) if (_M == (_m) && _N == (_n) && _K == (_k))

#define BIND_DEFAULT
  /*
   * @tparam _tir  the number of rows processed by each work item
   * @tparam _tic  the number of columns processed by each work item
   * @tparam _twr  the number of item-level tiles within each column of
   *                 block-level tile
   * @tparam _twc  the number of item-level tiles within each row of
   *                 block-level tile
   * @tparam _wg the total number of work-groupsize for the naive algorithm. It
   * is only used for the naive algorithm.
   * @tparam _clsize  the size of the cache line of the architecture in bytes
   *                 (If the value passed in is smaller than the actual cache
   *                 line size, some values fetched will be wasted, which can
   *                 significantly reduce performance. It can be set to a
   *                 multiple of the physical cache line size. In this case, it
   *                 will significantly increase scratchpad memory usage, but
   *                 will result in fewer local barriers.)
   * Note:
   * _tir * _twr must be equal to _tic * _twc.
   * This is ensured iff: (item_rows | wg_cols)  and  (item_cols | wg_rows)
   * _clsize cannot be bigger than _twr * _twc * sizeof(T)
   */
#define TO_TPARAMS(_wg, _db, _clsize, _tir, _tic, _twr, _twc)              \
  {                                                                        \
    return _select_gemm<_wg, _db, false, false, _clsize,                   \
                        Tile<_tir, _tic, _twr, _twc>>(                     \
        ex, _TrA, _TrB, _M, _N, _K, _alpha, _A, _lda, _B, _ldb, _beta, _C, \
        _ldc);                                                             \
  }
#ifndef NAIVE_GEMM
#if defined(DYNAMIC)
  if (ex.get_device_type() ==
      Executor::Queue_Interface_Type::device_type::SYCL_INTEL_GPU) {
    BIND_DATA_SIZE(1024, 4096, 1024) TO_TPARAMS(128, false, 64, 4, 4, 16, 16);
    BIND_DATA_SIZE(10, 1024, 1024) TO_TPARAMS(128, false, 64, 2, 2, 8, 8);
    BIND_DEFAULT TO_TPARAMS(128, false, 64, 8, 8, 8, 8);
  } else if ((ex.get_device_type() == Executor::Queue_Interface_Type::
                                          device_type::SYCL_RCAR_CVENGINE) ||
             (ex.get_device_type() == Executor::Queue_Interface_Type::
                                          device_type::SYCL_RCAR_HOST_CPU)) {
    if (_M < 512 && _N < 512) {
      BIND_DEFAULT TO_TPARAMS(32, false, 128, 4, 8, 8, 4);
    } else {
      BIND_DEFAULT TO_TPARAMS(32, false, 128, 8, 4, 4, 8);
    }
  } else {
    BIND_DATA_SIZE(10, 1024, 1024) TO_TPARAMS(128, true, 64, 1, 1, 16, 16);
    BIND_DEFAULT TO_TPARAMS(128, false, 64, 8, 8, 16, 16);
  }
#elif defined(INTEL_GPU)
  BIND_DATA_SIZE(1024, 4096, 1024) TO_TPARAMS(128, false, 64, 4, 4, 16, 16);
  BIND_DATA_SIZE(10, 1024, 1024) TO_TPARAMS(128, false, 64, 2, 2, 8, 8);
  BIND_DEFAULT TO_TPARAMS(128, false, 64, 8, 8, 16, 16);
#elif defined(RCAR)
  if (_M < 512 && _N < 512) {
    BIND_DEFAULT TO_TPARAMS(32, false, 128, 4, 8, 8, 4);
  } else {
    BIND_DEFAULT TO_TPARAMS(32, false, 128, 8, 4, 4, 8);
  }
#else  // any other specified devices
  BIND_DATA_SIZE(10, 1024, 1024) TO_TPARAMS(128, true, 64, 1, 1, 16, 16);
  BIND_DEFAULT TO_TPARAMS(128, false, 64, 8, 8, 16, 16);
#endif
#else
  BIND_DEFAULT TO_TPARAMS(WG_SIZE, false, 64, 8, 8, 8, 8);
#endif

#undef BIND_DATA_SIZE
#undef BIND_DEFAULT
#undef TO_TPARAMS
}

}  // namespace blas

#endif  // BLAS3_INTERFACE_HPP
