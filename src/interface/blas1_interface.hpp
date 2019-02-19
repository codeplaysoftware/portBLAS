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
 *  @filename blas1_interface.hpp
 *
 **************************************************************************/

#ifndef SYCL_BLAS_BLAS1_INTERFACE_HPP
#define SYCL_BLAS_BLAS1_INTERFACE_HPP

#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>

#include "blas_meta.h"
#include "container/sycl_iterator.h"
#include "executors/executor.h"
#include "interface/blas1_interface.h"
#include "operations/blas1_trees.h"
#include "operations/blas_constants.h"
#include "operations/blas_operators.hpp"

namespace blas {
namespace internal {
/**
 * \brief AXPY constant times a vector plus a vector.
 *
 * Implements AXPY \f$y = ax + y\f$
 *
 * @param Executor<ExecutorType> ex
 * @param _vx  VectorView
 * @param _incx Increment in X axis
 * @param _vy  VectorView
 * @param _incy Increment in Y axis
 */
template <typename Executor, typename ContainerT0, typename ContainerT1,
          typename T, typename IndexType, typename IncrementType>
typename Executor::Policy::event_type _axpy(Executor &ex, IndexType _N,
                                            T _alpha, ContainerT0 _vx,
                                            IncrementType _incx,
                                            ContainerT1 _vy,
                                            IncrementType _incy) {
  auto vx = make_vector_view(ex, _vx, _incx, _N);
  auto vy = make_vector_view(ex, _vy, _incy, _N);

  auto scalOp = make_op<ScalarOp, prdOp2_struct>(_alpha, vx);
  auto addOp = make_op<BinaryOp, addOp2_struct>(vy, scalOp);
  auto assignOp = make_op<Assign>(vy, addOp);
  auto ret = ex.execute(assignOp);
  return ret;
}

/**
 * \brief COPY copies a vector, x, to a vector, y.
 *
 * @param Executor<ExecutorType> ex
 * @param _vx  VectorView
 * @param _incx Increment in X axis
 * @param _vy  VectorView
 * @param _incy Increment in Y axis
 */
template <typename Executor, typename IndexType, typename ContainerT0,
          typename ContainerT1, typename IncrementType>
typename Executor::Policy::event_type _copy(Executor &ex, IndexType _N,
                                            ContainerT0 _vx,
                                            IncrementType _incx,
                                            ContainerT1 _vy,
                                            IncrementType _incy) {
  auto vx = make_vector_view(ex, _vx, _incx, _N);
  auto vy = make_vector_view(ex, _vy, _incy, _N);
  auto assignOp2 = make_op<Assign>(vy, vx);
  auto ret = ex.execute(assignOp2);
  return ret;
}

/**
 * \brief Compute the inner product of two vectors with extended precision
    accumulation.
 * @param Executor<ExecutorType> ex
 * @param _vx  VectorView
 * @param _incx Increment in X axis
 * @param _vx  VectorView
 * @param _incy Increment in Y axis
 */
template <typename Executor, typename ContainerT0, typename ContainerT1,
          typename ContainerT2, typename IndexType, typename IncrementType>
typename Executor::Policy::event_type _dot(Executor &ex, IndexType _N,
                                           ContainerT0 _vx, IncrementType _incx,
                                           ContainerT1 _vy, IncrementType _incy,
                                           ContainerT2 _rs) {
  auto vx = make_vector_view(ex, _vx, _incx, _N);
  auto vy = make_vector_view(ex, _vy, _incy, _N);
  auto rs = make_vector_view(ex, _rs, static_cast<IncrementType>(1),
                             static_cast<IndexType>(1));
  auto prdOp = make_op<BinaryOp, prdOp2_struct>(vx, vy);

  auto localSize = ex.get_policy_handler().get_work_group_size();
  auto nWG = 2 * localSize;

  auto assignOp = make_AssignReduction<addOp2_struct>(rs, prdOp, localSize,
                                                      localSize * nWG);
  auto ret = ex.execute(assignOp);
  return ret;
}

/**
 * \brief ASUM Takes the sum of the absolute values
 * @param Executor<ExecutorType> ex
 * @param _vx  VectorView
 * @param _incx Increment in X axis
 */
template <typename Executor, typename ContainerT0, typename ContainerT1,
          typename IndexType, typename IncrementType>
typename Executor::Policy::event_type _asum(Executor &ex, IndexType _N,
                                            ContainerT0 _vx,
                                            IncrementType _incx,
                                            ContainerT1 _rs) {
  auto vx = make_vector_view(ex, _vx, _incx, _N);
  auto rs = make_vector_view(ex, _rs, static_cast<IncrementType>(1),
                             static_cast<IndexType>(1));

  const auto localSize = ex.get_policy_handler().get_work_group_size();
  const auto nWG = 2 * localSize;
  auto assignOp = make_AssignReduction<addAbsOp2_struct>(rs, vx, localSize,
                                                         localSize * nWG);
  auto ret = ex.execute(assignOp);
  return ret;
}

/**
 * \brief IAMAX finds the index of the first element having maximum
 * @param _vx  VectorView
 * @param _incx Increment in X axis
 */
template <typename Executor, typename ContainerT, typename ContainerI,
          typename IndexType, typename IncrementType>
typename Executor::Policy::event_type _iamax(Executor &ex, IndexType _N,
                                             ContainerT _vx,
                                             IncrementType _incx,
                                             ContainerI _rs) {
  auto vx = make_vector_view(ex, _vx, _incx, _N);
  auto rs = make_vector_view(ex, _rs, static_cast<IncrementType>(1),
                             static_cast<IndexType>(1));
  const auto localSize = ex.get_policy_handler().get_work_group_size();
  const auto nWG = 2 * localSize;
  auto tupOp = make_tuple_op(vx);
  auto assignOp = make_AssignReduction<maxIndOp2_struct>(rs, tupOp, localSize,
                                                         localSize * nWG);
  auto ret = ex.execute(assignOp);
  return ret;
}

/**
 * \brief IAMIN finds the index of the first element having minimum
 * @param _vx  VectorView
 * @param _incx Increment in X axis
 */
template <typename Executor, typename ContainerT, typename ContainerI,
          typename IndexType, typename IncrementType>
typename Executor::Policy::event_type _iamin(Executor &ex, IndexType _N,
                                             ContainerT _vx,
                                             IncrementType _incx,
                                             ContainerI _rs) {
  auto vx = make_vector_view(ex, _vx, _incx, _N);
  auto rs = make_vector_view(ex, _rs, static_cast<IncrementType>(1),
                             static_cast<IndexType>(1));

  const auto localSize = ex.get_policy_handler().get_work_group_size();
  const auto nWG = 2 * localSize;
  auto tupOp = make_tuple_op(vx);
  auto assignOp = make_AssignReduction<minIndOp2_struct>(rs, tupOp, localSize,
                                                         localSize * nWG);
  auto ret = ex.execute(assignOp);
  return ret;
}

/**
 * \brief SWAP interchanges two vectors
 *
 * @param Executor ex
 * @param _vx  VectorView
 * @param _incx Increment in X axis
 * @param _vy  VectorView
 * @param _incy Increment in Y axis
 */
template <typename Executor, typename ContainerT0, typename ContainerT1,
          typename IndexType, typename IncrementType>
typename Executor::Policy::event_type _swap(Executor &ex, IndexType _N,
                                            ContainerT0 _vx,
                                            IncrementType _incx,
                                            ContainerT1 _vy,
                                            IncrementType _incy) {
  auto vx = make_vector_view(ex, _vx, _incx, _N);
  auto vy = make_vector_view(ex, _vy, _incy, _N);
  auto swapOp = make_op<DoubleAssign>(vy, vx, vx, vy);
  auto ret = ex.execute(swapOp);

  return ret;
}

/**
 * \brief SCALAR  operation on a vector
 * @param Executor ex
 * @param _vx  VectorView
 * @param _incx Increment in X axis
 */
template <typename Executor, typename T, typename ContainerT0,
          typename IndexType, typename IncrementType>
typename Executor::Policy::event_type _scal(Executor &ex, IndexType _N,
                                            T _alpha, ContainerT0 _vx,
                                            IncrementType _incx) {
  auto vx = make_vector_view(ex, _vx, _incx, _N);
  auto scalOp = make_op<ScalarOp, prdOp2_struct>(_alpha, vx);
  auto assignOp = make_op<Assign>(vx, scalOp);
  auto ret = ex.execute(assignOp);
  return ret;
}

/**
 * \brief NRM2 Returns the euclidian norm of a vector
 * @param Executor<ExecutorType> ex
 * @param _vx  VectorView
 * @param _incx Increment in X axis
 */
template <typename Executor, typename ContainerT0, typename ContainerT1,
          typename IndexType, typename IncrementType>
typename Executor::Policy::event_type _nrm2(Executor &ex, IndexType _N,
                                            ContainerT0 _vx,
                                            IncrementType _incx,
                                            ContainerT1 _rs) {
  auto vx = make_vector_view(ex, _vx, _incx, _N);
  auto rs = make_vector_view(ex, _rs, static_cast<IncrementType>(1),
                             static_cast<IndexType>(1));
  auto prdOp = make_op<UnaryOp, prdOp1_struct>(vx);

  const auto localSize = ex.get_policy_handler().get_work_group_size();
  const auto nWG = 2 * localSize;
  auto assignOp = make_AssignReduction<addOp2_struct>(rs, prdOp, localSize,
                                                      localSize * nWG);
  ex.execute(assignOp);
  auto sqrtOp = make_op<UnaryOp, sqtOp1_struct>(rs);
  auto assignOpFinal = make_op<Assign>(rs, sqrtOp);
  auto ret = ex.execute(assignOpFinal);
  return ret;
}

/**
 * .
 * @brief _rot constructor given plane rotation
 *  *
 * @param Executor<ExecutorType> ex
 * @param _vx  VectorView
 * @param _incx Increment in X axis
 * @param _vx  VectorView
 * @param _incy Increment in Y axis
 * @param _sin  sine
 * @param _cos cosine
 * @param _N data size
 *
 */
template <typename Executor, typename ContainerT0, typename ContainerT1,
          typename T, typename IndexType, typename IncrementType>
typename Executor::Policy::event_type _rot(Executor &ex, IndexType _N,
                                           ContainerT0 _vx, IncrementType _incx,
                                           ContainerT1 _vy, IncrementType _incy,
                                           T _cos, T _sin) {
  auto vx = make_vector_view(ex, _vx, _incx, _N);
  auto vy = make_vector_view(ex, _vy, _incy, _N);
  auto scalOp1 = make_op<ScalarOp, prdOp2_struct>(_cos, vx);
  auto scalOp2 = make_op<ScalarOp, prdOp2_struct>(_sin, vy);
  auto scalOp3 = make_op<ScalarOp, prdOp2_struct>(-_sin, vx);
  auto scalOp4 = make_op<ScalarOp, prdOp2_struct>(_cos, vy);
  auto addOp12 = make_op<BinaryOp, addOp2_struct>(scalOp1, scalOp2);
  auto addOp34 = make_op<BinaryOp, addOp2_struct>(scalOp3, scalOp4);
  auto DoubleAssignView = make_op<DoubleAssign>(vx, vy, addOp12, addOp34);
  auto ret = ex.execute(DoubleAssignView);
  return ret;
}

/**
 * \brief Compute the inner product of two vectors with extended
    precision accumulation and result.
 *
 * @param Executor<ExecutorType> ex
 * @param _vx  VectorView
 * @param _incx Increment in X axis
 * @param _vx  VectorView
 * @param _incy Increment in Y axis
 */
template <typename Executor, typename ContainerT0, typename ContainerT1,
          typename IndexType, typename IncrementType>
typename scalar_type<ContainerT0>::type _dot(Executor &ex, IndexType _N,
                                             ContainerT0 _vx,
                                             IncrementType _incx,
                                             ContainerT1 _vy,
                                             IncrementType _incy) {
  using T = typename scalar_type<ContainerT0>::type;
  auto res = std::vector<T>(1);
  auto gpu_res = make_sycl_iterator_buffer<T>(static_cast<IndexType>(1));
  blas::internal::_dot(ex, _N, _vx, _incx, _vy, _incy, gpu_res);
  ex.get_policy_handler().copy_to_host(gpu_res, res.data(), 1);
  return res[0];
}

/**
 * \brief ICAMAX finds the index of the first element having maximum
 * @param _vx  VectorView
 * @param _incx Increment in X axis
 */
template <typename Executor, typename ContainerT, typename IndexType,
          typename IncrementType>
IndexType _iamax(Executor &ex, IndexType _N, ContainerT _vx,
                 IncrementType _incx) {
  using T = typename scalar_type<ContainerT>::type;
  using IndValTuple = IndexValueTuple<T, IndexType>;
  std::vector<IndValTuple> rsT(1, IndValTuple(IndexType(-1), T(-1)));
  auto gpu_res =
      make_sycl_iterator_buffer<IndValTuple>(static_cast<IndexType>(1));
  blas::internal::_iamax(ex, _N, _vx, _incx, gpu_res);
  ex.get_policy_handler().copy_to_host(gpu_res, rsT.data(), 1);
  return rsT[0].get_index();
}

/**
 * \brief ICAMIN finds the index of the first element having minimum
 * @param _vx  VectorView
 * @param _incx Increment in X axis
 */
template <typename Executor, typename ContainerT, typename IndexType,
          typename IncrementType>
IndexType _iamin(Executor &ex, IndexType _N, ContainerT _vx,
                 IncrementType _incx) {
  using T = typename scalar_type<ContainerT>::type;
  using IndValTuple = IndexValueTuple<T, IndexType>;
  std::vector<IndValTuple> rsT(1, IndValTuple(IndexType(-1), T(-1)));
  auto gpu_res =
      make_sycl_iterator_buffer<IndValTuple>(static_cast<IndexType>(1));
  blas::internal::_iamin(ex, _N, _vx, _incx, gpu_res);
  ex.get_policy_handler().copy_to_host(gpu_res, rsT.data(), 1);
  return rsT[0].get_index();
}

/**
 * \brief ASUM Takes the sum of the absolute values
 *
 * @param Executor<ExecutorType> ex
 * @param _vx  VectorView
 * @param _incx Increment in X axis
 */
template <typename Executor, typename ContainerT, typename IndexType,
          typename IncrementType>
typename scalar_type<ContainerT>::type _asum(Executor &ex, IndexType _N,
                                             ContainerT _vx,
                                             IncrementType _incx) {
  using T = typename scalar_type<ContainerT>::type;
  auto res = std::vector<T>(1, T(0));
  auto gpu_res = make_sycl_iterator_buffer<T>(static_cast<IndexType>(1));
  blas::internal::_asum(ex, _N, _vx, _incx, gpu_res);
  ex.get_policy_handler().copy_to_host(gpu_res, res.data(), 1);
  return res[0];
}

/**
 * \brief NRM2 Returns the euclidian norm of a vector
 *
 * @param Executor<ExecutorType> ex
 * @param _vx  VectorView
 * @param _incx Increment in X axis
 */
template <typename Executor, typename ContainerT, typename IndexType,
          typename IncrementType>
typename scalar_type<ContainerT>::type _nrm2(Executor &ex, IndexType _N,
                                             ContainerT _vx,
                                             IncrementType _incx) {
  using T = typename scalar_type<ContainerT>::type;
  auto res = std::vector<T>(1, T(0));
  auto gpu_res = make_sycl_iterator_buffer<T>(static_cast<IndexType>(1));
  blas::internal::_nrm2(ex, _N, _vx, _incx, gpu_res);
  ex.get_policy_handler().copy_to_host(gpu_res, res.data(), 1);
  return res[0];
}

}  // namespace internal
}  // namespace blas

#endif  // BLAS1_INTERFACE_HPP
