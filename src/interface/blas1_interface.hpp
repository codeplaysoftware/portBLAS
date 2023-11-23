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
 *  portBLAS: BLAS implementation using SYCL
 *
 *  @filename blas1_interface.hpp
 *
 **************************************************************************/

#ifndef PORTBLAS_BLAS1_INTERFACE_HPP
#define PORTBLAS_BLAS1_INTERFACE_HPP

#include <cmath>
#include <stdexcept>
#include <vector>

#include "blas_meta.h"
#include "container/sycl_iterator.h"
#include "interface/blas1/backend/backend.hpp"
#include "interface/blas1_interface.h"
#include "operations/blas1_trees.h"
#include "operations/blas_constants.h"
#include "operations/blas_operators.hpp"
#include "sb_handle/portblas_handle.h"
#include "views/view.h"

namespace blas {
namespace internal {
/**
 * \brief AXPY constant times a vector plus a vector.
 *
 * Implements AXPY \f$y = ax + y\f$
 *
 * @param sb_handle_t sb_handle
 * @param _vx  BufferIterator or USM pointer
 * @param _incx Increment in X axis
 * @param _vy  BufferIterator or USM pointer
 * @param _incy Increment in Y axis
 * @param _dependencies Vector of events
 */
template <typename sb_handle_t, typename container_0_t, typename container_1_t,
          typename element_t, typename index_t, typename increment_t>
typename sb_handle_t::event_t _axpy(
    sb_handle_t &sb_handle, index_t _N, element_t _alpha, container_0_t _vx,
    increment_t _incx, container_1_t _vy, increment_t _incy,
    const typename sb_handle_t::event_t &_dependencies) {
  typename VectorViewType<container_0_t, index_t, increment_t>::type vx =
      make_vector_view(_vx, _incx, _N);
  auto vy = make_vector_view(_vy, _incy, _N);

  auto scalOp = make_op<ScalarOp, ProductOperator>(_alpha, vx);
  auto addOp = make_op<BinaryOp, AddOperator>(vy, scalOp);
  auto assignOp = make_op<Assign>(vy, addOp);
  auto ret = sb_handle.execute(assignOp, _dependencies);
  return ret;
}

/**
 * \brief COPY copies a vector, x, to a vector, y.
 *
 * @param sb_handle_t sb_handle
 * @param _vx  BufferIterator or USM pointer
 * @param _incx Increment in X axis
 * @param _vy  BufferIterator or USM pointer
 * @param _incy Increment in Y axis
 * @param _dependencies Vector of events
 */
template <typename sb_handle_t, typename index_t, typename container_0_t,
          typename container_1_t, typename increment_t>
typename sb_handle_t::event_t _copy(
    sb_handle_t &sb_handle, index_t _N, container_0_t _vx, increment_t _incx,
    container_1_t _vy, increment_t _incy,
    const typename sb_handle_t::event_t &_dependencies) {
  typename VectorViewType<container_0_t, index_t, increment_t>::type vx =
      make_vector_view(_vx, _incx, _N);
  auto vy = make_vector_view(_vy, _incy, _N);
  auto assignOp2 = make_op<Assign>(vy, vx);
  auto ret = sb_handle.execute(assignOp2, _dependencies);
  return ret;
}

/**
 * \brief Computes the inner product of two vectors with double precision
 * accumulation (Asynchronous version that returns an event)
 * @tparam sb_handle_t SB_Handle type
 * @tparam container_0_t Buffer Iterator or USM pointer
 * @tparam container_1_t Buffer Iterator or USM pointer
 * @tparam container_2_t Buffer Iterator or USM pointer
 * @tparam index_t Index type
 * @tparam increment_t Increment type
 * @param sb_handle SB_Handle
 * @param _N Input buffer sizes.
 * @param _vx Memory object holding input vector x
 * @param _incx Stride of vector x (i.e. measured in elements of _vx)
 * @param _vy Memory object holding input vector y
 * @param _incy Stride of vector y (i.e. measured in elements of _vy)
 * @param _rs Output memory object
 * @param _dependencies Vector of events
 * @return Vector of events to wait for.
 */
template <typename sb_handle_t, typename container_0_t, typename container_1_t,
          typename container_2_t, typename index_t, typename increment_t>
typename sb_handle_t::event_t _dot(
    sb_handle_t &sb_handle, index_t _N, container_0_t _vx, increment_t _incx,
    container_1_t _vy, increment_t _incy, container_2_t _rs,
    const typename sb_handle_t::event_t &_dependencies) {
  return blas::dot::backend::_dot(sb_handle, _N, _vx, _incx, _vy, _incy, _rs,
                                  _dependencies);
}

/**
 * \brief Computes the inner product of two vectors with double precision
 * accumulation and adds a scalar to the result (Asynchronous version that
 * returns an event)
 * @tparam sb_handle_t SB_Handle type
 * @tparam container_0_t Buffer Iterator or USM pointer
 * @tparam container_1_t Buffer Iterator or USM pointer
 * @tparam container_2_t Buffer Iterator or USM pointer
 * @tparam index_t Index type
 * @tparam increment_t Increment type
 * @param sb_handle SB_Handle
 * @param _N Input buffer sizes. If size 0, the result will be sb.
 * @param sb Scalar to add to the results of the inner product.
 * @param _vx Memory object holding input vector x
 * @param _incx Stride of vector x (i.e. measured in elements of _vx)
 * @param _vy Memory object holding input vector y
 * @param _incy Stride of vector y (i.e. measured in elements of _vy)
 * @param _rs Output memory object
 * @param _dependencies Vector of events
 * @return Vector of events to wait for.
 */
template <typename sb_handle_t, typename container_0_t, typename container_1_t,
          typename container_2_t, typename index_t, typename increment_t>
typename sb_handle_t::event_t _sdsdot(
    sb_handle_t &sb_handle, index_t _N, float sb, container_0_t _vx,
    increment_t _incx, container_1_t _vy, increment_t _incy, container_2_t _rs,
    const typename sb_handle_t::event_t &_dependencies) {
  if (!_N) {
    using element_t = typename ValueType<container_2_t>::type;
    sb_handle.wait(_dependencies);
    auto ret = blas::helper::copy_to_device(
        sb_handle.get_queue(), reinterpret_cast<element_t *>(&sb), _rs, 1);
    sb_handle.wait(ret);
    return {ret};
  } else {
    auto rs = make_vector_view(_rs, static_cast<increment_t>(1),
                               static_cast<index_t>(1));
    auto dotOp = blas::dot::backend::_dot(sb_handle, _N, _vx, _incx, _vy, _incy,
                                          _rs, _dependencies);
    auto addOp = make_op<ScalarOp, AddOperator>(sb, rs);
    auto assignOp2 = make_op<Assign>(rs, addOp);
    auto ret = sb_handle.execute(assignOp2, dotOp);
    return blas::concatenate_vectors(dotOp, ret);
  }
}

/**
 * \brief ASUM Takes the sum of the absolute values
 * @param sb_handle_t sb_handle
 * @param _vx  BufferIterator or USM pointer
 * @param _incx Increment in X axis
 * @param _rs BufferIterator or USM pointer
 * @param _dependencies Vector of events
 */
template <typename sb_handle_t, typename container_0_t, typename container_1_t,
          typename index_t, typename increment_t>
typename sb_handle_t::event_t _asum(
    sb_handle_t &sb_handle, index_t _N, container_0_t _vx, increment_t _incx,
    container_1_t _rs, const typename sb_handle_t::event_t &_dependencies) {
  // keep compatibility with older sycl versions
#if SYCL_LANGUAGE_VERSION < 202000
  typename VectorViewType<container_0_t, index_t, increment_t>::type vx =
      make_vector_view(_vx, _incx, _N);
  auto rs = make_vector_view(_rs, static_cast<increment_t>(1),
                             static_cast<index_t>(1));

  const auto localSize = sb_handle.get_work_group_size();
  const auto nWG = 2 * localSize;
  auto assignOp = make_assign_reduction<AbsoluteAddOperator>(rs, vx, localSize,
                                                             localSize * nWG);
  auto ret = sb_handle.execute(assignOp, _dependencies);
  return ret;
#else
  return blas::asum::backend::_asum(sb_handle, _N, _vx, _incx, _rs,
                                    _dependencies);
#endif
}

#if SYCL_LANGUAGE_VERSION >= 202000
/*! _asum_impl.
 * @brief Internal implementation of the Absolute sum operator.
 *
 * This function contains the code that sets up and executes the kernels
 * required to perform the asum operation.
 *
 * This function is called by blas::internal::backend::asum which, dependent on
 * the platform being compiled for and other parameters, provides different
 * template parameters to ensure the most optimal kernel is constructed.
 *
 * @tparam localSize  specifies the number of threads per work group used by
 *                    the kernel
 * @tparam localMemSize specifies the size of local shared memory to use, which
 *                      is device and implementation dependent. If 0 the
 *                      implementation use a kernel implementation which doesn't
 *                      require local memory.
 */
template <int localSize, int localMemSize, typename sb_handle_t,
          typename container_0_t, typename container_1_t, typename index_t,
          typename increment_t>
typename sb_handle_t::event_t _asum_impl(
    sb_handle_t &sb_handle, index_t _N, container_0_t _vx, increment_t _incx,
    container_1_t _rs, const index_t number_WG,
    const typename sb_handle_t::event_t &_dependencies) {
  typename VectorViewType<container_0_t, index_t, increment_t>::type vx =
      make_vector_view(_vx, _incx, _N);
  auto rs = make_vector_view(_rs, static_cast<increment_t>(1),
                             static_cast<index_t>(1));
  typename sb_handle_t::event_t ret;
  auto asumOp = make_wg_atomic_reduction<AbsoluteAddOperator>(rs, vx);
  if constexpr (localMemSize != 0) {
    ret = sb_handle.execute(asumOp, static_cast<index_t>(localSize),
                            static_cast<index_t>(number_WG * localSize),
                            static_cast<index_t>(localMemSize), _dependencies);
  } else {
    ret = sb_handle.execute(asumOp, static_cast<index_t>(localSize),
                            static_cast<index_t>(number_WG * localSize),
                            _dependencies);
  }
  return ret;
}
#endif

/**
 * \brief IAMAX finds the index of the first element having maximum
 * @param _vx  BufferIterator or USM pointer
 * @param _incx Increment in X axis
 * @param _rs BufferIterator or USM pointer
 * @param _dependencies Vector of events
 */
template <typename sb_handle_t, typename container_t, typename ContainerI,
          typename index_t, typename increment_t>
typename sb_handle_t::event_t _iamax(
    sb_handle_t &sb_handle, index_t _N, container_t _vx, increment_t _incx,
    ContainerI _rs, const typename sb_handle_t::event_t &_dependencies) {
  typename VectorViewType<container_t, index_t, increment_t>::type vx =
      make_vector_view(_vx, _incx, _N);
  auto rs = make_vector_view(_rs, static_cast<increment_t>(1),
                             static_cast<index_t>(1));
  const auto localSize = sb_handle.get_work_group_size();
  const auto nWG = 2 * localSize;
  auto tupOp = make_tuple_op(vx);
  auto assignOp = make_assign_reduction<IMaxOperator>(rs, tupOp, localSize,
                                                      localSize * nWG);
  auto ret = sb_handle.execute(assignOp, _dependencies);
  return ret;
}

/**
 * \brief IAMIN finds the index of the first element having minimum
 * @param _vx  BufferIterator or USM pointer
 * @param _incx Increment in X axis
 * @param _rs BufferIterator or USM pointer
 * @param _dependencies Vector of events
 */
template <typename sb_handle_t, typename container_t, typename ContainerI,
          typename index_t, typename increment_t>
typename sb_handle_t::event_t _iamin(
    sb_handle_t &sb_handle, index_t _N, container_t _vx, increment_t _incx,
    ContainerI _rs, const typename sb_handle_t::event_t &_dependencies) {
  typename VectorViewType<container_t, index_t, increment_t>::type vx =
      make_vector_view(_vx, _incx, _N);
  auto rs = make_vector_view(_rs, static_cast<increment_t>(1),
                             static_cast<index_t>(1));

  const auto localSize = sb_handle.get_work_group_size();
  const auto nWG = 2 * localSize;
  auto tupOp = make_tuple_op(vx);
  auto assignOp = make_assign_reduction<IMinOperator>(rs, tupOp, localSize,
                                                      localSize * nWG);
  auto ret = sb_handle.execute(assignOp, _dependencies);
  return ret;
}

/**
 * \brief SWAP interchanges two vectors
 *
 * @param sb_handle_t sb_handle
 * @param _vx  BufferIterator or USM pointer
 * @param _incx Increment in X axis
 * @param _vy  BufferIterator or USM pointer
 * @param _incy Increment in Y axis
 * @param _dependencies Vector of events
 */
template <typename sb_handle_t, typename container_0_t, typename container_1_t,
          typename index_t, typename increment_t>
typename sb_handle_t::event_t _swap(
    sb_handle_t &sb_handle, index_t _N, container_0_t _vx, increment_t _incx,
    container_1_t _vy, increment_t _incy,
    const typename sb_handle_t::event_t &_dependencies) {
  auto vx = make_vector_view(_vx, _incx, _N);
  auto vy = make_vector_view(_vy, _incy, _N);
  auto swapOp = make_op<DoubleAssign>(vy, vx, vx, vy);
  auto ret = sb_handle.execute(swapOp, _dependencies);

  return ret;
}

/**
 * \brief SCALAR operation on a vector
 * @param sb_handle_t sb_handle
 * @param _vx  BufferIterator or USM pointer
 * @param _incx Increment in X axis
 * @param _dependencies Vector of events
 */
template <typename sb_handle_t, typename element_t, typename container_0_t,
          typename index_t, typename increment_t>
typename sb_handle_t::event_t _scal(
    sb_handle_t &sb_handle, index_t _N, element_t _alpha, container_0_t _vx,
    increment_t _incx, const typename sb_handle_t::event_t &_dependencies) {
  auto vx = make_vector_view(_vx, _incx, _N);
  if (_alpha == element_t{0}) {
    auto zeroOp = make_op<UnaryOp, AdditionIdentity>(vx);
    auto assignOp = make_op<Assign>(vx, zeroOp);
    auto ret = sb_handle.execute(assignOp, _dependencies);
    return ret;
  } else {
    auto scalOp = make_op<ScalarOp, ProductOperator>(_alpha, vx);
    auto assignOp = make_op<Assign>(vx, scalOp);
    auto ret = sb_handle.execute(assignOp, _dependencies);
    return ret;
  }
}

/**
 * \brief NRM2 Returns the euclidian norm of a vector
 * @param sb_handle_t sb_handle
 * @param _vx  BufferIterator or USM pointer
 * @param _incx Increment in X axis
 * @param _rs BufferIterator or USM pointer
 * @param _dependencies Vector of events
 */
template <typename sb_handle_t, typename container_0_t, typename container_1_t,
          typename index_t, typename increment_t>
typename sb_handle_t::event_t _nrm2(
    sb_handle_t &sb_handle, index_t _N, container_0_t _vx, increment_t _incx,
    container_1_t _rs, const typename sb_handle_t::event_t &_dependencies) {
  return blas::nrm2::backend::_nrm2(sb_handle, _N, _vx, _incx, _rs,
                                    _dependencies);
}

/*! _nrm2_impl.
 * @brief Internal implementation of the nrm2 operator.
 *
 * This function contains the code that sets up and executes the kernels
 * required to perform the nrm2 operation.
 *
 * This function is called by blas::internal::backend::nrm2 which, dependent on
 * the platform being compiled for and other parameters, provides different
 * template parameters to ensure the most optimal kernel is constructed.
 *
 * @tparam localSize  specifies the number of threads per work group used by
 *                    the kernel
 * @tparam localMemSize specifies the size of local shared memory to use, which
 *                      is device and implementation dependent. If 0 the
 *                      implementation use a kernel implementation which doesn't
 *                      require local memory.
 */
template <int localSize, int localMemSize, typename sb_handle_t,
          typename container_0_t, typename container_1_t, typename index_t,
          typename increment_t>
typename sb_handle_t::event_t _nrm2_impl(
    sb_handle_t &sb_handle, index_t _N, container_0_t _vx, increment_t _incx,
    container_1_t _rs, const index_t number_WG,
    const typename sb_handle_t::event_t &_dependencies) {
  typename VectorViewType<container_0_t, index_t, increment_t>::type vx =
      make_vector_view(_vx, _incx, _N);
  auto rs = make_vector_view(_rs, static_cast<increment_t>(1),
                             static_cast<index_t>(1));
  auto prdOp = make_op<UnaryOp, SquareOperator>(vx);

  auto assignOp = make_wg_atomic_reduction<AddOperator>(rs, prdOp);
  typename sb_handle_t::event_t ret0;
  if constexpr (localMemSize != 0) {
    ret0 = sb_handle.execute(assignOp, static_cast<index_t>(localSize),
                             static_cast<index_t>(number_WG * localSize),
                             static_cast<index_t>(localMemSize), _dependencies);
  } else {
    ret0 = sb_handle.execute(assignOp, static_cast<index_t>(localSize),
                             static_cast<index_t>(number_WG * localSize),
                             _dependencies);
  }
  auto sqrtOp = make_op<UnaryOp, SqrtOperator>(rs);
  auto assignOpFinal = make_op<Assign>(rs, sqrtOp);
  auto ret1 = sb_handle.execute(assignOpFinal, ret0);
  return blas::concatenate_vectors(ret0, ret1);
}

/**
 * @brief _dot_impl Internal implementation of the dot operator.
 *
 * This function contains the code that sets up and executes the kernels
 * required to perform the dot operation (also used in sdsdot).
 *
 * This function is called by blas::dot::backend::_dot which, depending on
 * the TUNING_TARGET and other RT parameters (size for instance), selects
 * different template parameters / configuration to ensure the adequate kernel
 * is called.
 *
 * @tparam localSize  specifies the number of threads per work group used by
 *                    the kernel
 * @tparam localMemSize specifies the size of local shared memory to use, which
 *                      is device and implementation dependent. If 0 the
 *                      implementation use a kernel implementation which doesn't
 *                      require local memory.
 */
template <int localSize, int localMemSize, typename sb_handle_t,
          typename container_0_t, typename container_1_t,
          typename container_2_t, typename index_t, typename increment_t>
typename sb_handle_t::event_t _dot_impl(
    sb_handle_t &sb_handle, index_t _N, container_0_t _vx, increment_t _incx,
    container_1_t _vy, increment_t _incy, container_2_t _rs,
    const index_t _number_wg,
    const typename sb_handle_t::event_t &_dependencies) {
  typename sb_handle_t::event_t ret_event;
  // Skip if N==0, _rs is not overwritten
  if (!_N) return {_dependencies};

  auto vx = make_vector_view(_vx, _incx, _N);
  auto vy = make_vector_view(_vy, _incy, _N);
  auto rs = make_vector_view(_rs, static_cast<increment_t>(1),
                             static_cast<index_t>(1));

  auto prdOp = make_op<BinaryOpConst, ProductOperator>(vx, vy);
  auto assignOp = make_wg_atomic_reduction<AddOperator>(rs, prdOp);

  if constexpr (localMemSize) {
    ret_event =
        sb_handle.execute(assignOp, static_cast<index_t>(localSize),
                          static_cast<index_t>(_number_wg * localSize),
                          static_cast<index_t>(localMemSize), _dependencies);
  } else {
    ret_event = sb_handle.execute(assignOp, static_cast<index_t>(localSize),
                                  static_cast<index_t>(_number_wg * localSize),
                                  _dependencies);
  }
  return ret_event;
}

/**
 * .
 * @brief _rot constructor given plane rotation
 *  *
 * @param sb_handle_t sb_handle
 * @param _vx  BufferIterator or USM pointer
 * @param _incx Increment in X axis
 * @param _vx  BufferIterator or USM pointer
 * @param _incy Increment in Y axis
 * @param _sin  sine
 * @param _cos cosine
 * @param _N data size
 * @param _dependencies Vector of events
 */
template <typename sb_handle_t, typename container_0_t, typename container_1_t,
          typename element_t, typename index_t, typename increment_t>
typename sb_handle_t::event_t _rot(
    sb_handle_t &sb_handle, index_t _N, container_0_t _vx, increment_t _incx,
    container_1_t _vy, increment_t _incy, element_t _cos, element_t _sin,
    const typename sb_handle_t::event_t &_dependencies) {
  auto vx = make_vector_view(_vx, _incx, _N);
  auto vy = make_vector_view(_vy, _incy, _N);
  auto scalOp1 = make_op<ScalarOp, ProductOperator>(_cos, vx);
  auto scalOp2 = make_op<ScalarOp, ProductOperator>(_sin, vy);
  auto scalOp3 = make_op<ScalarOp, ProductOperator>(element_t{-_sin}, vx);
  auto scalOp4 = make_op<ScalarOp, ProductOperator>(_cos, vy);
  auto addOp12 = make_op<BinaryOp, AddOperator>(scalOp1, scalOp2);
  auto addOp34 = make_op<BinaryOp, AddOperator>(scalOp3, scalOp4);
  auto DoubleAssignView = make_op<DoubleAssign>(vx, vy, addOp12, addOp34);
  auto ret = sb_handle.execute(DoubleAssignView, _dependencies);
  return ret;
}

/**
 * @brief Performs a modified Givens rotation of points.
 * Given two vectors x and y and a modified Givens transformation matrix, each
 * element of x and y is replaced as follows:
 *
 * [xi] = [h11 h12] * [xi]
 * [yi]   [h21 h22]   [yi]
 *
 * where h11, h12, h21 and h22 represent the modified Givens transformation
 * matrix.
 *
 * The value of the flag parameter can be used to modify the matrix as follows:
 *
 * -1.0: [h11 h12]     0.0: [1.0 h12]     1.0: [h11 1.0]     -2.0 = [1.0 0.0]
 *       [h21 h22]          [h21 1.0]          [-1.0 h22]           [0.0 1.0]
 *
 * @tparam sb_handle_t SB_Handle type
 * @tparam container_0_t Buffer Iterator or USM pointer
 * @tparam container_1_t Buffer Iterator or USM pointer
 * @tparam container_2_t Buffer Iterator or USM pointer
 * @tparam index_t Index type
 * @tparam increment_t Increment type
 * @param sb_handle SB_Handle
 * @param _N Input buffer sizes (for vx and vy).
 * @param[in, out] _vx Memory object holding input vector x
 * @param _incx Stride of vector x (i.e. measured in elements of _vx)
 * @param[in, out] _vy Memory object holding input vector y
 * @param _incy Stride of vector y (i.e. measured in elements of _vy)
 * @param[in] _param Buffer with the following layout: [flag, h11, h21, h12,
 * h22].
 * @param _dependencies Vector of events
 * @return Vector of events to wait for.
 */
template <typename sb_handle_t, typename container_0_t, typename container_1_t,
          typename container_2_t, typename index_t, typename increment_t>
typename sb_handle_t::event_t _rotm(
    sb_handle_t &sb_handle, index_t _N, container_0_t _vx, increment_t _incx,
    container_1_t _vy, increment_t _incy, container_2_t _param,
    const typename sb_handle_t::event_t &_dependencies) {
  auto vx = make_vector_view(_vx, _incx, _N);
  auto vy = make_vector_view(_vy, _incy, _N);

  constexpr size_t param_size = 5;
  using element_t = typename ValueType<container_0_t>::type;
  std::array<element_t, param_size> param_host;

  /* This implementation can be further optimized for small input vectors by
   * creating a custom kernel that modifies param instead of copying it back to
   * the host */
  auto copy_event = blas::helper::copy_to_host(sb_handle.get_queue(), _param,
                                               param_host.data(), param_size);
  sb_handle.wait(copy_event);

  const element_t flag = param_host[0];
  element_t h11 = param_host[1];
  element_t h21 = param_host[2];
  element_t h12 = param_host[3];
  element_t h22 = param_host[4];

  using m_two = constant<element_t, const_val::m_two>;
  using m_one = constant<element_t, const_val::m_one>;
  using zero = constant<element_t, const_val::zero>;
  using one = constant<element_t, const_val::one>;

  if (flag == zero::value()) {
    h11 = one::value();
    h22 = one::value();
  } else if (flag == one::value()) {
    h12 = one::value();
    h21 = m_one::value();
  } else if (flag == m_two::value()) {
    h11 = one::value();
    h12 = zero::value();
    h21 = zero::value();
    h22 = one::value();
  }

  auto h11TimesVx = make_op<ScalarOp, ProductOperator>(h11, vx);
  auto h12TimesVy = make_op<ScalarOp, ProductOperator>(h12, vy);
  auto h21TimesVx = make_op<ScalarOp, ProductOperator>(h21, vx);
  auto h22TimesVy = make_op<ScalarOp, ProductOperator>(h22, vy);
  auto vxResult = make_op<BinaryOp, AddOperator>(h11TimesVx, h12TimesVy);
  auto vyResult = make_op<BinaryOp, AddOperator>(h21TimesVx, h22TimesVy);
  auto DoubleAssignView = make_op<DoubleAssign>(vx, vy, vxResult, vyResult);
  auto ret = sb_handle.execute(DoubleAssignView, _dependencies);

  return ret;
}

/**
 * Given the Cartesian coordinates (x1, y1) of a point, the rotmg routines
 * compute the components of a modified Givens transformation matrix H that
 * zeros the y-component of the resulting point:
 *
 *                      [xi] = H * [xi * sqrt(d1) ]
 *                      [0 ]       [yi * sqrt(d2) ]
 *
 * Depending on the flag parameter, the components of H are set as follows:
 *
 * -1.0: [h11 h12]     0.0: [1.0 h12]     1.0: [h11 1.0]     -2.0 = [1.0 0.0]
 *       [h21 h22]          [h21 1.0]          [-1.0 h22]           [0.0 1.0]
 *
 * Rotmg may apply scaling operations to d1, d2 and x1 to avoid overflows.
 *
 * @tparam sb_handle_t SB_Handle type
 * @tparam container_0_t Buffer Iterator or USM pointer
 * @tparam container_1_t Buffer Iterator or USM pointer
 * @tparam container_2_t Buffer Iterator or USM pointer
 * @tparam container_3_t Buffer Iterator or USM pointer
 * @tparam container_4_t Buffer Iterator or USM pointer
 * @param sb_handle SB_Handle
 * @param _d1[in,out] On entry, memory object holding the scaling factor for the
 * x-coordinate. On exit, the re-scaled _d1.
 * @param _d2[in,out] On entry, memory object holding the scaling factor for the
 * y-coordinate. On exit, the re-scaled _d2.
 * @param _x1[in,out] On entry, memory object holding the x-coordinate. On exit,
 * the re-scaled _x1
 * @param _y1[in] Memory object holding the y-coordinate of the point.
 * @param _param[out] Buffer with the following layout: [flag, h11, h21, h12,
 * h22].
 * @param _dependencies Vector of events
 * @return Vector of events to wait for.
 */
template <typename sb_handle_t, typename container_0_t, typename container_1_t,
          typename container_2_t, typename container_3_t,
          typename container_4_t>
typename sb_handle_t::event_t _rotmg(
    sb_handle_t &sb_handle, container_0_t _d1, container_1_t _d2,
    container_2_t _x1, container_3_t _y1, container_4_t _param,
    const typename sb_handle_t::event_t &_dependencies) {
  constexpr int inc = 1;
  constexpr int vector_size = 1;
  constexpr int param_size = 5;

  auto d1_view = make_vector_view(_d1, inc, vector_size);
  auto d2_view = make_vector_view(_d2, inc, vector_size);
  auto x1_view = make_vector_view(_x1, inc, vector_size);
  auto y1_view = make_vector_view(_y1, inc, vector_size);
  auto param_view = make_vector_view(_param, inc, param_size);

  auto operation =
      Rotmg<decltype(d1_view)>(d1_view, d2_view, x1_view, y1_view, param_view);
  auto ret = sb_handle.execute(operation, _dependencies);

  return ret;
}

/**
 * \brief Given the Cartesian coordinates (a, b) of a point, the rotg routines
 * return the parameters c, s, r, and z associated with the Givens rotation.
 * @tparam sb_handle_t SB_Handle type
 * @tparam container_0_t Buffer Iterator or USM pointer
 * @tparam container_1_t Buffer Iterator or USM pointer
 * @tparam container_2_t Buffer Iterator or USM pointer
 * @tparam container_3_t Buffer Iterator or USM pointer
 * @param sb_handle SB_Handle
 * @param a[in, out] On entry, memory object holding the x-coordinate of the
 * point. On exit, the scalar z.
 * @param b[in, out] On entry, memory object holding the y-coordinate of the
 * point. On exit, the scalar r.
 * @param c[out] Memory object holding the parameter c.
 * @param s[out] Memory object holding the parameter s.
 * @param _dependencies Vector of events
 * @return Vector of events to wait for.
 */
template <
    typename sb_handle_t, typename container_0_t, typename container_1_t,
    typename container_2_t, typename container_3_t,
    typename std::enable_if<!is_sycl_scalar<container_0_t>::value, bool>::type>
typename sb_handle_t::event_t _rotg(
    sb_handle_t &sb_handle, container_0_t a, container_1_t b, container_2_t c,
    container_3_t s, const typename sb_handle_t::event_t &_dependencies) {
  auto a_view = make_vector_view(a, 1, 1);
  auto b_view = make_vector_view(b, 1, 1);
  auto c_view = make_vector_view(c, 1, 1);
  auto s_view = make_vector_view(s, 1, 1);

  auto operation = Rotg<decltype(a_view)>(a_view, b_view, c_view, s_view);
  auto ret = sb_handle.execute(operation, _dependencies);

  return ret;
}

/**
 * \brief Synchronous version of rotg.
 * Given the Cartesian coordinates (a, b) of a point, the rotg routines
 * return the parameters c, s, r, and z associated with the Givens rotation.
 * @tparam sb_handle_t SB_Handle type
 * @tparam scalar_t Scalar type
 * @param sb_handle SB_Handle
 * @param a[in, out] On entry, x-coordinate of the point. On exit, the scalar z.
 * @param b[in, out] On entry, y-coordinate of the point. On exit, the scalar r.
 * @param c[out] Scalar representing the output c.
 * @param s[out] Scalar representing the output s.
 * @param _dependencies Vector of events
 */
template <typename sb_handle_t, typename scalar_t,
          typename std::enable_if<is_sycl_scalar<scalar_t>::value, bool>::type>
void _rotg(sb_handle_t &sb_handle, scalar_t &a, scalar_t &b, scalar_t &c,
           scalar_t &s, const typename sb_handle_t::event_t &_dependencies) {
  auto device_a =
      blas::helper::allocate<blas::helper::AllocType::buffer, scalar_t>(
          1, sb_handle.get_queue());
  auto device_b =
      blas::helper::allocate<blas::helper::AllocType::buffer, scalar_t>(
          1, sb_handle.get_queue());
  auto device_c =
      blas::helper::allocate<blas::helper::AllocType::buffer, scalar_t>(
          1, sb_handle.get_queue());
  auto device_s =
      blas::helper::allocate<blas::helper::AllocType::buffer, scalar_t>(
          1, sb_handle.get_queue());
  auto copy_a =
      blas::helper::copy_to_device(sb_handle.get_queue(), &a, device_a, 1);
  auto copy_b =
      blas::helper::copy_to_device(sb_handle.get_queue(), &b, device_b, 1);
  auto copy_c =
      blas::helper::copy_to_device(sb_handle.get_queue(), &c, device_c, 1);
  auto copy_s =
      blas::helper::copy_to_device(sb_handle.get_queue(), &s, device_s, 1);

  typename sb_handle_t::event_t ret = concatenate_vectors(
      _dependencies,
      typename sb_handle_t::event_t{copy_a, copy_b, copy_c, copy_s});

  auto event = blas::internal::_rotg(sb_handle, device_a, device_b, device_c,
                                     device_s, ret);

  auto event1 =
      blas::helper::copy_to_host(sb_handle.get_queue(), device_c, &c, 1);
  auto event2 =
      blas::helper::copy_to_host(sb_handle.get_queue(), device_s, &s, 1);
  auto event3 =
      blas::helper::copy_to_host(sb_handle.get_queue(), device_a, &a, 1);
  auto event4 =
      blas::helper::copy_to_host(sb_handle.get_queue(), device_b, &b, 1);

  sb_handle.wait({event1, event2, event3, event4});
}

/**
 * \brief Computes the inner product of two vectors with double precision
 * accumulation (synchronous version that returns the result directly)
 * @tparam sb_handle_t SB_Handle type
 * @tparam container_0_t Buffer Iterator or USM pointer
 * @tparam container_1_t Buffer Iterator or USM pointer
 * @tparam index_t Index type
 * @tparam increment_t Increment type
 * @param sb_handle SB_Handle
 * @param _N Input buffer sizes.
 * @param _vx Memory object holding input vector x
 * @param _incx Stride of vector x (i.e. measured in elements of _vx)
 * @param _vy Memory object holding input vector y
 * @param _incy Stride of vector y (i.e. measured in elements of _vy)
 * @param _rs Output memory object
 * @param _dependencies Vector of events
 * @return Vector of events to wait for.
 */
template <typename sb_handle_t, typename container_0_t, typename container_1_t,
          typename index_t, typename increment_t>
typename ValueType<container_0_t>::type _dot(
    sb_handle_t &sb_handle, index_t _N, container_0_t _vx, increment_t _incx,
    container_1_t _vy, increment_t _incy,
    const typename sb_handle_t::event_t &_dependencies) {
  constexpr bool is_usm = std::is_pointer<container_0_t>::value;
  using element_t = typename ValueType<container_0_t>::type;
  element_t res{0};
  auto gpu_res = helper::allocate < is_usm ? helper::AllocType::usm
                                           : helper::AllocType::buffer,
       element_t > (static_cast<index_t>(1), sb_handle.get_queue());
  auto copyTodD =
      blas::helper::copy_to_device(sb_handle.get_queue(), &res, gpu_res, 1);
  typename sb_handle_t::event_t all_deps = concatenate_vectors(
      _dependencies, typename sb_handle_t::event_t{copyTodD});

  auto dotOp =
      internal::_dot(sb_handle, _N, _vx, _incx, _vy, _incy, gpu_res, all_deps);

  sb_handle.wait(dotOp);
  auto copyToH = helper::copy_to_host(sb_handle.get_queue(), gpu_res, &res, 1);
  sb_handle.wait(copyToH);

  helper::deallocate<is_usm ? helper::AllocType::usm
                            : helper::AllocType::buffer>(gpu_res,
                                                         sb_handle.get_queue());
  return res;
}

/**
 * \brief Computes the inner product of two vectors with double precision
 * accumulation and adds a scalar to the result (synchronous version that
 * returns the result directly)
 * @tparam sb_handle_t SB_Handle type
 * @tparam container_0_t Buffer Iterator or USM pointer
 * @tparam container_1_t Buffer Iterator or USM pointer
 * @tparam index_t Index type
 * @tparam increment_t Increment type
 * @param sb_handle SB_Handle
 * @param _N Input buffer sizes. If size 0, the result will be sb.
 * @param sb Scalar to add to the results of the inner product.
 * @param _vx Memory object holding input vector x
 * @param _incx Stride of vector x (i.e. measured in elements of _vx)
 * @param _vy Memory object holding input vector y
 * @param _incy Stride of vector y (i.e. measured in elements of _vy)
 * @param _rs Output memory object
 * @param _dependencies Vector of events
 * @return Vector of events to wait for.
 */
template <typename sb_handle_t, typename container_0_t, typename container_1_t,
          typename index_t, typename increment_t>
typename ValueType<container_0_t>::type _sdsdot(
    sb_handle_t &sb_handle, index_t _N, float sb, container_0_t _vx,
    increment_t _incx, container_1_t _vy, increment_t _incy,
    const typename sb_handle_t::event_t &_dependencies) {
  constexpr bool is_usm = std::is_pointer<container_0_t>::value;
  using element_t = typename ValueType<container_0_t>::type;
  element_t res{0};
  auto gpu_res = blas::helper::allocate < is_usm ? helper::AllocType::usm
                                                 : helper::AllocType::buffer,
       element_t > (static_cast<index_t>(1), sb_handle.get_queue());
  auto copyTodD =
      blas::helper::copy_to_device(sb_handle.get_queue(), &res, gpu_res, 1);
  typename sb_handle_t::event_t all_deps = concatenate_vectors(
      _dependencies, typename sb_handle_t::event_t{copyTodD});

  auto sdsdot_event = blas::internal::_sdsdot(sb_handle, _N, sb, _vx, _incx,
                                              _vy, _incy, gpu_res, all_deps);
  sb_handle.wait(sdsdot_event);
  auto copyToH =
      blas::helper::copy_to_host(sb_handle.get_queue(), gpu_res, &res, 1);
  sb_handle.wait(copyToH);

  blas::helper::deallocate<is_usm ? helper::AllocType::usm
                                  : helper::AllocType::buffer>(
      gpu_res, sb_handle.get_queue());
  return res;
}

/**
 * \brief ICAMAX finds the index of the first element having maximum
 * @param _vx  BufferIterator or USM pointer
 * @param _incx Increment in X axis
 * @param _dependencies Vector of events
 */
template <typename sb_handle_t, typename container_t, typename index_t,
          typename increment_t>
index_t _iamax(sb_handle_t &sb_handle, index_t _N, container_t _vx,
               increment_t _incx,
               const typename sb_handle_t::event_t &_dependencies) {
  constexpr bool is_usm = std::is_pointer<container_t>::value;
  using element_t = typename ValueType<container_t>::type;
  using IndValTuple = IndexValueTuple<index_t, element_t>;
  std::vector<IndValTuple> rsT(1, IndValTuple(index_t(-1), element_t(-1)));
  auto gpu_res = blas::helper::allocate < is_usm ? helper::AllocType::usm
                                                 : helper::AllocType::buffer,
       IndValTuple > (static_cast<index_t>(1), sb_handle.get_queue());
  auto iamax_event =
      blas::internal::_iamax(sb_handle, _N, _vx, _incx, gpu_res, _dependencies);
  sb_handle.wait(iamax_event);
  auto event = blas::helper::copy_to_host<IndValTuple>(sb_handle.get_queue(),
                                                       gpu_res, rsT.data(), 1);
  sb_handle.wait(event);
  blas::helper::deallocate<is_usm ? helper::AllocType::usm
                                  : helper::AllocType::buffer>(
      gpu_res, sb_handle.get_queue());
  return rsT[0].get_index();
}

/**
 * \brief ICAMIN finds the index of the first element having minimum
 * @param _vx  BufferIterator or USM pointer
 * @param _incx Increment in X axis
 * @param _dependencies Vector of events
 */
template <typename sb_handle_t, typename container_t, typename index_t,
          typename increment_t>
index_t _iamin(sb_handle_t &sb_handle, index_t _N, container_t _vx,
               increment_t _incx,
               const typename sb_handle_t::event_t &_dependencies) {
  constexpr bool is_usm = std::is_pointer<container_t>::value;
  using element_t = typename ValueType<container_t>::type;
  using IndValTuple = IndexValueTuple<index_t, element_t>;
  std::vector<IndValTuple> rsT(1, IndValTuple(index_t(-1), element_t(-1)));
  auto gpu_res = blas::helper::allocate < is_usm ? helper::AllocType::usm
                                                 : helper::AllocType::buffer,
       IndValTuple > (static_cast<index_t>(1), sb_handle.get_queue());
  auto iamin_event =
      blas::internal::_iamin(sb_handle, _N, _vx, _incx, gpu_res, _dependencies);
  sb_handle.wait(iamin_event);
  auto event =
      blas::helper::copy_to_host(sb_handle.get_queue(), gpu_res, rsT.data(), 1);
  sb_handle.wait(event);
  blas::helper::deallocate<is_usm ? helper::AllocType::usm
                                  : helper::AllocType::buffer>(
      gpu_res, sb_handle.get_queue());
  return rsT[0].get_index();
}

/**
 * \brief ASUM Takes the sum of the absolute values
 *
 * @param sb_handle_t sb_handle
 * @param _vx  BufferIterator or USM pointer
 * @param _incx Increment in X axis
 * @param _dependencies Vector of events
 */
template <typename sb_handle_t, typename container_t, typename index_t,
          typename increment_t>
typename ValueType<container_t>::type _asum(
    sb_handle_t &sb_handle, index_t _N, container_t _vx, increment_t _incx,
    const typename sb_handle_t::event_t &_dependencies) {
  constexpr bool is_usm = std::is_pointer<container_t>::value;
  using element_t = typename ValueType<container_t>::type;
  auto res = std::vector<element_t>(1, element_t(0));
  auto gpu_res = blas::helper::allocate < is_usm ? helper::AllocType::usm
                                                 : helper::AllocType::buffer,
       element_t > (static_cast<index_t>(1), sb_handle.get_queue());
  const typename sb_handle_t::event_t init_res_event = {
      blas::helper::copy_to_device(sb_handle.get_queue(), res.data(), gpu_res,
                                   1)};
  auto local_deps = concatenate_vectors(_dependencies, init_res_event);
  auto asum_event =
      blas::internal::_asum(sb_handle, _N, _vx, _incx, gpu_res, local_deps);
  sb_handle.wait(asum_event);
  auto event =
      blas::helper::copy_to_host(sb_handle.get_queue(), gpu_res, res.data(), 1);
  sb_handle.wait(event);
  blas::helper::deallocate<is_usm ? helper::AllocType::usm
                                  : helper::AllocType::buffer>(
      gpu_res, sb_handle.get_queue());
  return res[0];
}

/**
 * \brief NRM2 Returns the euclidian norm of a vector
 *
 * @param sb_handle_t sb_handle
 * @param _vx  BufferIterator or USM pointer
 * @param _incx Increment in X axis
 * @param _dependencies Vector of events
 */
template <typename sb_handle_t, typename container_t, typename index_t,
          typename increment_t>
typename ValueType<container_t>::type _nrm2(
    sb_handle_t &sb_handle, index_t _N, container_t _vx, increment_t _incx,
    const typename sb_handle_t::event_t &_dependencies) {
  constexpr bool is_usm = std::is_pointer<container_t>::value;
  using element_t = typename ValueType<container_t>::type;
  auto res = std::vector<element_t>(1, element_t(0));
  auto gpu_res = blas::helper::allocate < is_usm ? helper::AllocType::usm
                                                 : helper::AllocType::buffer,
       element_t > (static_cast<index_t>(1), sb_handle.get_queue());
  typename sb_handle_t::event_t copy_init_val = {blas::helper::copy_to_device(
      sb_handle.get_queue(), res.data(), gpu_res, 1)};
  const auto local_deps = concatenate_vectors(_dependencies, copy_init_val);
  auto nrm2_event =
      blas::internal::_nrm2(sb_handle, _N, _vx, _incx, gpu_res, local_deps);
  sb_handle.wait(nrm2_event);
  auto event =
      blas::helper::copy_to_host(sb_handle.get_queue(), gpu_res, res.data(), 1);
  sb_handle.wait(event);
  blas::helper::deallocate<is_usm ? helper::AllocType::usm
                                  : helper::AllocType::buffer>(
      gpu_res, sb_handle.get_queue());
  return res[0];
}

}  // namespace internal
}  // namespace blas

#endif  // BLAS1_INTERFACE_HPP
