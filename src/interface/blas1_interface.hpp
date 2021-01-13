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
 * @param executor_t<ExecutorType> ex
 * @param _vx  BufferIterator
 * @param _incx Increment in X axis
 * @param _vy  BufferIterator
 * @param _incy Increment in Y axis
 */
template <typename executor_t, typename container_0_t, typename container_1_t,
          typename element_t, typename index_t, typename increment_t>
typename executor_t::policy_t::event_t _axpy(
    executor_t &ex, index_t _N, element_t _alpha, container_0_t _vx,
    increment_t _incx, container_1_t _vy, increment_t _incy) {
  auto vx = make_vector_view(ex, _vx, _incx, _N);
  auto vy = make_vector_view(ex, _vy, _incy, _N);

  auto scalOp = make_op<ScalarOp, ProductOperator>(_alpha, vx);
  auto addOp = make_op<BinaryOp, AddOperator>(vy, scalOp);
  auto assignOp = make_op<Assign>(vy, addOp);
  auto ret = ex.execute(assignOp);
  return ret;
}

/**
 * \brief COPY copies a vector, x, to a vector, y.
 *
 * @param executor_t<ExecutorType> ex
 * @param _vx  BufferIterator
 * @param _incx Increment in X axis
 * @param _vy  BufferIterator
 * @param _incy Increment in Y axis
 */
template <typename executor_t, typename index_t, typename container_0_t,
          typename container_1_t, typename increment_t>
typename executor_t::policy_t::event_t _copy(executor_t &ex, index_t _N,
                                             container_0_t _vx,
                                             increment_t _incx,
                                             container_1_t _vy,
                                             increment_t _incy) {
  auto vx = make_vector_view(ex, _vx, _incx, _N);
  auto vy = make_vector_view(ex, _vy, _incy, _N);
  auto assignOp2 = make_op<Assign>(vy, vx);
  auto ret = ex.execute(assignOp2);
  return ret;
}

/**
 * \brief Compute the inner product of two vectors with extended precision
    accumulation.
 * @param executor_t<ExecutorType> ex
 * @param _vx  BufferIterator
 * @param _incx Increment in X axis
 * @param _vx  BufferIterator
 * @param _incy Increment in Y axis
 */
template <typename executor_t, typename container_0_t, typename container_1_t,
          typename container_2_t, typename index_t, typename increment_t>
typename executor_t::policy_t::event_t _dot(
    executor_t &ex, index_t _N, container_0_t _vx, increment_t _incx,
    container_1_t _vy, increment_t _incy, container_2_t _rs) {
  auto vx = make_vector_view(ex, _vx, _incx, _N);
  auto vy = make_vector_view(ex, _vy, _incy, _N);
  auto rs = make_vector_view(ex, _rs, static_cast<increment_t>(1),
                             static_cast<index_t>(1));
  auto prdOp = make_op<BinaryOp, ProductOperator>(vx, vy);

  auto localSize = ex.get_policy_handler().get_work_group_size();
  auto nWG = 2 * localSize;

  auto assignOp =
      make_AssignReduction<AddOperator>(rs, prdOp, localSize, localSize * nWG);
  auto ret = ex.execute(assignOp);
  return ret;
}

/**
 * \brief ASUM Takes the sum of the absolute values
 * @param executor_t<ExecutorType> ex
 * @param _vx  BufferIterator
 * @param _incx Increment in X axis
 */
template <typename executor_t, typename container_0_t, typename container_1_t,
          typename index_t, typename increment_t>
typename executor_t::policy_t::event_t _asum(executor_t &ex, index_t _N,
                                             container_0_t _vx,
                                             increment_t _incx,
                                             container_1_t _rs) {
  auto vx = make_vector_view(ex, _vx, _incx, _N);
  auto rs = make_vector_view(ex, _rs, static_cast<increment_t>(1),
                             static_cast<index_t>(1));

  const auto localSize = ex.get_policy_handler().get_work_group_size();
  const auto nWG = 2 * localSize;
  auto assignOp = make_AssignReduction<AbsoluteAddOperator>(rs, vx, localSize,
                                                            localSize * nWG);
  auto ret = ex.execute(assignOp);
  return ret;
}

/**
 * \brief IAMAX finds the index of the first element having maximum
 * @param _vx  BufferIterator
 * @param _incx Increment in X axis
 */
template <typename executor_t, typename container_t, typename ContainerI,
          typename index_t, typename increment_t>
typename executor_t::policy_t::event_t _iamax(executor_t &ex, index_t _N,
                                              container_t _vx,
                                              increment_t _incx,
                                              ContainerI _rs) {
  auto vx = make_vector_view(ex, _vx, _incx, _N);
  auto rs = make_vector_view(ex, _rs, static_cast<increment_t>(1),
                             static_cast<index_t>(1));
  const auto localSize = ex.get_policy_handler().get_work_group_size();
  const auto nWG = 2 * localSize;
  auto tupOp = make_tuple_op(vx);
  auto assignOp =
      make_AssignReduction<IMaxOperator>(rs, tupOp, localSize, localSize * nWG);
  auto ret = ex.execute(assignOp);
  return ret;
}

/**
 * \brief IAMIN finds the index of the first element having minimum
 * @param _vx  BufferIterator
 * @param _incx Increment in X axis
 */
template <typename executor_t, typename container_t, typename ContainerI,
          typename index_t, typename increment_t>
typename executor_t::policy_t::event_t _iamin(executor_t &ex, index_t _N,
                                              container_t _vx,
                                              increment_t _incx,
                                              ContainerI _rs) {
  auto vx = make_vector_view(ex, _vx, _incx, _N);
  auto rs = make_vector_view(ex, _rs, static_cast<increment_t>(1),
                             static_cast<index_t>(1));

  const auto localSize = ex.get_policy_handler().get_work_group_size();
  const auto nWG = 2 * localSize;
  auto tupOp = make_tuple_op(vx);
  auto assignOp =
      make_AssignReduction<IMinOperator>(rs, tupOp, localSize, localSize * nWG);
  auto ret = ex.execute(assignOp);
  return ret;
}

/**
 * \brief SWAP interchanges two vectors
 *
 * @param executor_t ex
 * @param _vx  BufferIterator
 * @param _incx Increment in X axis
 * @param _vy  BufferIterator
 * @param _incy Increment in Y axis
 */
template <typename executor_t, typename container_0_t, typename container_1_t,
          typename index_t, typename increment_t>
typename executor_t::policy_t::event_t _swap(executor_t &ex, index_t _N,
                                             container_0_t _vx,
                                             increment_t _incx,
                                             container_1_t _vy,
                                             increment_t _incy) {
  auto vx = make_vector_view(ex, _vx, _incx, _N);
  auto vy = make_vector_view(ex, _vy, _incy, _N);
  auto swapOp = make_op<DoubleAssign>(vy, vx, vx, vy);
  auto ret = ex.execute(swapOp);

  return ret;
}

/**
 * \brief SCALAR  operation on a vector
 * @param executor_t ex
 * @param _vx  BufferIterator
 * @param _incx Increment in X axis
 */
template <typename executor_t, typename element_t, typename container_0_t,
          typename index_t, typename increment_t>
typename executor_t::policy_t::event_t _scal(executor_t &ex, index_t _N,
                                             element_t _alpha,
                                             container_0_t _vx,
                                             increment_t _incx) {
  auto vx = make_vector_view(ex, _vx, _incx, _N);
  if (_alpha == element_t{0}) {
    auto zeroOp = make_op<UnaryOp, AdditionIdentity>(vx);
    auto assignOp = make_op<Assign>(vx, zeroOp);
    auto ret = ex.execute(assignOp);
    return ret;
  } else {
    auto scalOp = make_op<ScalarOp, ProductOperator>(_alpha, vx);
    auto assignOp = make_op<Assign>(vx, scalOp);
    auto ret = ex.execute(assignOp);
    return ret;
  }
}

/**
 * \brief NRM2 Returns the euclidian norm of a vector
 * @param executor_t<ExecutorType> ex
 * @param _vx  BufferIterator
 * @param _incx Increment in X axis
 */
template <typename executor_t, typename container_0_t, typename container_1_t,
          typename index_t, typename increment_t>
typename executor_t::policy_t::event_t _nrm2(executor_t &ex, index_t _N,
                                             container_0_t _vx,
                                             increment_t _incx,
                                             container_1_t _rs) {
  auto vx = make_vector_view(ex, _vx, _incx, _N);
  auto rs = make_vector_view(ex, _rs, static_cast<increment_t>(1),
                             static_cast<index_t>(1));
  auto prdOp = make_op<UnaryOp, SquareOperator>(vx);

  const auto localSize = ex.get_policy_handler().get_work_group_size();
  const auto nWG = 2 * localSize;
  auto assignOp =
      make_AssignReduction<AddOperator>(rs, prdOp, localSize, localSize * nWG);
  auto ret0 = ex.execute(assignOp);
  auto sqrtOp = make_op<UnaryOp, SqrtOperator>(rs);
  auto assignOpFinal = make_op<Assign>(rs, sqrtOp);
  auto ret1 = ex.execute(assignOpFinal);
  return blas::concatenate_vectors(ret0, ret1);
}

/**
 * .
 * @brief _rot constructor given plane rotation
 *  *
 * @param executor_t<ExecutorType> ex
 * @param _vx  BufferIterator
 * @param _incx Increment in X axis
 * @param _vx  BufferIterator
 * @param _incy Increment in Y axis
 * @param _sin  sine
 * @param _cos cosine
 * @param _N data size
 *
 */
template <typename executor_t, typename container_0_t, typename container_1_t,
          typename element_t, typename index_t, typename increment_t>
typename executor_t::policy_t::event_t _rot(
    executor_t &ex, index_t _N, container_0_t _vx, increment_t _incx,
    container_1_t _vy, increment_t _incy, element_t _cos, element_t _sin) {
  auto vx = make_vector_view(ex, _vx, _incx, _N);
  auto vy = make_vector_view(ex, _vy, _incy, _N);
  auto scalOp1 = make_op<ScalarOp, ProductOperator>(_cos, vx);
  auto scalOp2 = make_op<ScalarOp, ProductOperator>(_sin, vy);
  auto scalOp3 = make_op<ScalarOp, ProductOperator>(element_t{-_sin}, vx);
  auto scalOp4 = make_op<ScalarOp, ProductOperator>(_cos, vy);
  auto addOp12 = make_op<BinaryOp, AddOperator>(scalOp1, scalOp2);
  auto addOp34 = make_op<BinaryOp, AddOperator>(scalOp3, scalOp4);
  auto DoubleAssignView = make_op<DoubleAssign>(vx, vy, addOp12, addOp34);
  auto ret = ex.execute(DoubleAssignView);
  return ret;
}

/**
 * \brief Compute the inner product of two vectors with extended
    precision accumulation and result.
 *
 * @param executor_t<ExecutorType> ex
 * @param _vx  BufferIterator
 * @param _incx Increment in X axis
 * @param _vx  BufferIterator
 * @param _incy Increment in Y axis
 */
template <typename executor_t, typename container_0_t, typename container_1_t,
          typename index_t, typename increment_t>
typename ValueType<container_0_t>::type _dot(executor_t &ex, index_t _N,
                                             container_0_t _vx,
                                             increment_t _incx,
                                             container_1_t _vy,
                                             increment_t _incy) {
  using element_t = typename ValueType<container_0_t>::type;
  auto res = std::vector<element_t>(1);
  auto gpu_res = make_sycl_iterator_buffer<element_t>(static_cast<index_t>(1));
  blas::internal::_dot(ex, _N, _vx, _incx, _vy, _incy, gpu_res);
  ex.get_policy_handler().copy_to_host(gpu_res, res.data(), 1);
  return res[0];
}

/**
 * \brief ICAMAX finds the index of the first element having maximum
 * @param _vx  BufferIterator
 * @param _incx Increment in X axis
 */
template <typename executor_t, typename container_t, typename index_t,
          typename increment_t>
index_t _iamax(executor_t &ex, index_t _N, container_t _vx, increment_t _incx) {
  using element_t = typename ValueType<container_t>::type;
  using IndValTuple = IndexValueTuple<index_t, element_t>;
  std::vector<IndValTuple> rsT(1, IndValTuple(index_t(-1), element_t(-1)));
  auto gpu_res =
      make_sycl_iterator_buffer<IndValTuple>(static_cast<index_t>(1));
  blas::internal::_iamax(ex, _N, _vx, _incx, gpu_res);
  ex.get_policy_handler().copy_to_host(gpu_res, rsT.data(), 1);
  return rsT[0].get_index();
}

/**
 * \brief ICAMIN finds the index of the first element having minimum
 * @param _vx  BufferIterator
 * @param _incx Increment in X axis
 */
template <typename executor_t, typename container_t, typename index_t,
          typename increment_t>
index_t _iamin(executor_t &ex, index_t _N, container_t _vx, increment_t _incx) {
  using element_t = typename ValueType<container_t>::type;
  using IndValTuple = IndexValueTuple<index_t, element_t>;
  std::vector<IndValTuple> rsT(1, IndValTuple(index_t(-1), element_t(-1)));
  auto gpu_res =
      make_sycl_iterator_buffer<IndValTuple>(static_cast<index_t>(1));
  blas::internal::_iamin(ex, _N, _vx, _incx, gpu_res);
  ex.get_policy_handler().copy_to_host(gpu_res, rsT.data(), 1);
  return rsT[0].get_index();
}

/**
 * \brief ASUM Takes the sum of the absolute values
 *
 * @param executor_t<ExecutorType> ex
 * @param _vx  BufferIterator
 * @param _incx Increment in X axis
 */
template <typename executor_t, typename container_t, typename index_t,
          typename increment_t>
typename ValueType<container_t>::type _asum(executor_t &ex, index_t _N,
                                            container_t _vx,
                                            increment_t _incx) {
  using element_t = typename ValueType<container_t>::type;
  auto res = std::vector<element_t>(1, element_t(0));
  auto gpu_res = make_sycl_iterator_buffer<element_t>(static_cast<index_t>(1));
  blas::internal::_asum(ex, _N, _vx, _incx, gpu_res);
  ex.get_policy_handler().copy_to_host(gpu_res, res.data(), 1);
  return res[0];
}

/**
 * \brief NRM2 Returns the euclidian norm of a vector
 *
 * @param executor_t<ExecutorType> ex
 * @param _vx  BufferIterator
 * @param _incx Increment in X axis
 */
template <typename executor_t, typename container_t, typename index_t,
          typename increment_t>
typename ValueType<container_t>::type _nrm2(executor_t &ex, index_t _N,
                                            container_t _vx,
                                            increment_t _incx) {
  using element_t = typename ValueType<container_t>::type;
  auto res = std::vector<element_t>(1, element_t(0));
  auto gpu_res = make_sycl_iterator_buffer<element_t>(static_cast<index_t>(1));
  blas::internal::_nrm2(ex, _N, _vx, _incx, gpu_res);
  ex.get_policy_handler().copy_to_host(gpu_res, res.data(), 1);
  return res[0];
}

}  // namespace internal
}  // namespace blas

#endif  // BLAS1_INTERFACE_HPP
