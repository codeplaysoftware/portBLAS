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

#ifndef SYCL_BLAS_BLAS2_TREES_HPP
#define SYCL_BLAS_BLAS2_TREES_HPP

#include "blas2/gemv.hpp"
#include "blas2/ger.hpp"

namespace blas {

/*! Scalar2DOp.
 * @brief Implements an scalar operation.
 * (e.g alpha OP A, with alpha scalar and A matrix)
 */
template <typename operator_t, typename scalar_t, typename rhs_t>
Scalar2DOp<operator_t, scalar_t, rhs_t>::Scalar2DOp(scalar_t _scl, rhs_t &_r)
    : scalar_(_scl), rhs_(_r) {}

template <typename operator_t, typename scalar_t, typename rhs_t>
SYCL_BLAS_INLINE typename Scalar2DOp<operator_t, scalar_t, rhs_t>::index_t
Scalar2DOp<operator_t, scalar_t, rhs_t>::get_size() const {
  return rhs_.get_size();
}
template <typename operator_t, typename scalar_t, typename rhs_t>
SYCL_BLAS_INLINE typename Scalar2DOp<operator_t, scalar_t, rhs_t>::index_t
Scalar2DOp<operator_t, scalar_t, rhs_t>::getSizeL() const {
  return rhs_.getSizeL();
}
template <typename operator_t, typename scalar_t, typename rhs_t>
SYCL_BLAS_INLINE typename Scalar2DOp<operator_t, scalar_t, rhs_t>::index_t
Scalar2DOp<operator_t, scalar_t, rhs_t>::get_size_row() const {
  return rhs_.get_size_row();
}
template <typename operator_t, typename scalar_t, typename rhs_t>
SYCL_BLAS_INLINE typename Scalar2DOp<operator_t, scalar_t, rhs_t>::index_t
Scalar2DOp<operator_t, scalar_t, rhs_t>::get_size_col() const {
  return rhs_.get_size_col();
}
template <typename operator_t, typename scalar_t, typename rhs_t>
SYCL_BLAS_INLINE bool Scalar2DOp<operator_t, scalar_t, rhs_t>::valid_thread(
    cl::sycl::nd_item<1> ndItem) const {
  return ((ndItem.get_global_id(0) <
           Scalar2DOp<operator_t, scalar_t, rhs_t>::get_size()));
}
template <typename operator_t, typename scalar_t, typename rhs_t>
SYCL_BLAS_INLINE typename Scalar2DOp<operator_t, scalar_t, rhs_t>::value_t
Scalar2DOp<operator_t, scalar_t, rhs_t>::eval(
    typename Scalar2DOp<operator_t, scalar_t, rhs_t>::index_t i) {
  return operator_t::eval(internal::get_scalar(scalar_), rhs_.eval(i));
}
template <typename operator_t, typename scalar_t, typename rhs_t>
SYCL_BLAS_INLINE typename Scalar2DOp<operator_t, scalar_t, rhs_t>::value_t
Scalar2DOp<operator_t, scalar_t, rhs_t>::eval(cl::sycl::nd_item<1> ndItem) {
  return Scalar2DOp<operator_t, scalar_t, rhs_t>::eval(ndItem.get_global_id(0));
}
template <typename operator_t, typename scalar_t, typename rhs_t>
SYCL_BLAS_INLINE void Scalar2DOp<operator_t, scalar_t, rhs_t>::bind(
    cl::sycl::handler &h) {
  rhs_.bind(h);
}

template <typename operator_t, typename scalar_t, typename rhs_t>
SYCL_BLAS_INLINE void
Scalar2DOp<operator_t, scalar_t, rhs_t>::adjust_access_displacement() {
  rhs_.adjust_access_displacement();
}

/*! Unary2DOp.
 * Implements a Unary Operation ( operator_t(A), e.g. A++), with A a matrix.
 */
template <typename operator_t, typename rhs_t>
Unary2DOp<operator_t, rhs_t>::Unary2DOp(rhs_t &_r) : rhs_(_r) {}

template <typename operator_t, typename rhs_t>
SYCL_BLAS_INLINE typename Unary2DOp<operator_t, rhs_t>::index_t
Unary2DOp<operator_t, rhs_t>::get_size() const {
  return rhs_.get_size();
}
template <typename operator_t, typename rhs_t>
SYCL_BLAS_INLINE typename Unary2DOp<operator_t, rhs_t>::index_t
Unary2DOp<operator_t, rhs_t>::getSizeL() const {
  return rhs_.getSizeL();
}
template <typename operator_t, typename rhs_t>
SYCL_BLAS_INLINE typename Unary2DOp<operator_t, rhs_t>::index_t
Unary2DOp<operator_t, rhs_t>::get_size_row() const {
  return rhs_.get_size_row();
}
template <typename operator_t, typename rhs_t>
SYCL_BLAS_INLINE typename Unary2DOp<operator_t, rhs_t>::index_t
Unary2DOp<operator_t, rhs_t>::get_size_col() const {
  return rhs_.get_size_col();
}
template <typename operator_t, typename rhs_t>
SYCL_BLAS_INLINE bool Unary2DOp<operator_t, rhs_t>::valid_thread(
    cl::sycl::nd_item<1> ndItem) const {
  return ((ndItem.get_global_id(0) < Unary2DOp<operator_t, rhs_t>::get_size()));
}
template <typename operator_t, typename rhs_t>
SYCL_BLAS_INLINE typename Unary2DOp<operator_t, rhs_t>::value_t
Unary2DOp<operator_t, rhs_t>::eval(
    typename Unary2DOp<operator_t, rhs_t>::index_t i) {
  return operator_t::eval(rhs_.eval(i));
}
template <typename operator_t, typename rhs_t>
SYCL_BLAS_INLINE typename Unary2DOp<operator_t, rhs_t>::value_t
Unary2DOp<operator_t, rhs_t>::eval(cl::sycl::nd_item<1> ndItem) {
  return Unary2DOp<operator_t, rhs_t>::eval(ndItem.get_global_id(0));
}
template <typename operator_t, typename rhs_t>
SYCL_BLAS_INLINE void Unary2DOp<operator_t, rhs_t>::bind(cl::sycl::handler &h) {
  rhs_.bind(h);
}
template <typename operator_t, typename rhs_t>
SYCL_BLAS_INLINE void
Unary2DOp<operator_t, rhs_t>::adjust_access_displacement() {
  rhs_.adjust_access_displacement();
}

/*! Binary2DOp.
 * @brief Implements a Binary Operation (A OP B) with A and B matrices.
 */
template <typename operator_t, typename lhs_t, typename rhs_t>
Binary2DOp<operator_t, lhs_t, rhs_t>::Binary2DOp(lhs_t &_l, rhs_t &_r)
    : lhs_(_l), rhs_(_r){};

template <typename operator_t, typename lhs_t, typename rhs_t>
SYCL_BLAS_INLINE typename Binary2DOp<operator_t, lhs_t, rhs_t>::index_t
Binary2DOp<operator_t, lhs_t, rhs_t>::get_size() const {
  return rhs_.get_size();
}
template <typename operator_t, typename lhs_t, typename rhs_t>
SYCL_BLAS_INLINE typename Binary2DOp<operator_t, lhs_t, rhs_t>::index_t
Binary2DOp<operator_t, lhs_t, rhs_t>::getSizeL() const {
  return rhs_.get_size_row();
}
template <typename operator_t, typename lhs_t, typename rhs_t>
SYCL_BLAS_INLINE typename Binary2DOp<operator_t, lhs_t, rhs_t>::index_t
Binary2DOp<operator_t, lhs_t, rhs_t>::get_size_row() const {
  return rhs_.get_size_row();
}
template <typename operator_t, typename lhs_t, typename rhs_t>
SYCL_BLAS_INLINE typename Binary2DOp<operator_t, lhs_t, rhs_t>::index_t
Binary2DOp<operator_t, lhs_t, rhs_t>::get_size_col() const {
  return rhs_.get_size_col();
}
template <typename operator_t, typename lhs_t, typename rhs_t>
SYCL_BLAS_INLINE bool Binary2DOp<operator_t, lhs_t, rhs_t>::valid_thread(
    cl::sycl::nd_item<1> ndItem) const {
  return ((ndItem.get_global_id(0) < get_size()));
}
template <typename operator_t, typename lhs_t, typename rhs_t>
SYCL_BLAS_INLINE typename Binary2DOp<operator_t, lhs_t, rhs_t>::value_t
Binary2DOp<operator_t, lhs_t, rhs_t>::eval(
    typename Binary2DOp<operator_t, lhs_t, rhs_t>::index_t i) {
  using index_t = Binary2DOp<operator_t, lhs_t, rhs_t>::index_t;
  const index_t nb_rows = get_size_row();
  const index_t col = i / nb_rows;
  const index_t row = i - nb_rows * col;
  return operator_t::eval(lhs_.eval(col * lhs_.getSizeL() + row),
                          rhs_.eval(col * rhs_.getSizeL() + row));
}
template <typename operator_t, typename lhs_t, typename rhs_t>
SYCL_BLAS_INLINE typename Binary2DOp<operator_t, lhs_t, rhs_t>::value_t
Binary2DOp<operator_t, lhs_t, rhs_t>::eval(cl::sycl::nd_item<1> ndItem) {
  return Binary2DOp<operator_t, lhs_t, rhs_t>::eval(ndItem.get_global_id(0));
}
template <typename operator_t, typename lhs_t, typename rhs_t>
SYCL_BLAS_INLINE void Binary2DOp<operator_t, lhs_t, rhs_t>::bind(
    cl::sycl::handler &h) {
  lhs_.bind(h);
  rhs_.bind(h);
}
template <typename operator_t, typename lhs_t, typename rhs_t>
SYCL_BLAS_INLINE void
Binary2DOp<operator_t, lhs_t, rhs_t>::adjust_access_displacement() {
  lhs_.adjust_access_displacement();
  rhs_.adjust_access_displacement();
}

/*! Assign2D.
 * @brief Assign the rhs to the lhs
 */
template <typename lhs_t, typename rhs_t>
Assign2D<lhs_t, rhs_t>::Assign2D(lhs_t &_l, rhs_t _r) : lhs_(_l), rhs_(_r){};

template <typename lhs_t, typename rhs_t>
SYCL_BLAS_INLINE typename Assign2D<lhs_t, rhs_t>::index_t
Assign2D<lhs_t, rhs_t>::get_size() const {
  return rhs_.get_size();
}
template <typename lhs_t, typename rhs_t>
SYCL_BLAS_INLINE typename Assign2D<lhs_t, rhs_t>::index_t
Assign2D<lhs_t, rhs_t>::getSizeL() const {
  return rhs_.get_size_row();
}
template <typename lhs_t, typename rhs_t>
SYCL_BLAS_INLINE typename Assign2D<lhs_t, rhs_t>::index_t
Assign2D<lhs_t, rhs_t>::get_size_row() const {
  return rhs_.get_size_row();
}
template <typename lhs_t, typename rhs_t>
SYCL_BLAS_INLINE typename Assign2D<lhs_t, rhs_t>::index_t
Assign2D<lhs_t, rhs_t>::get_size_col() const {
  return rhs_.get_size_col();
}
template <typename lhs_t, typename rhs_t>
SYCL_BLAS_INLINE bool Assign2D<lhs_t, rhs_t>::valid_thread(
    cl::sycl::nd_item<1> ndItem) const {
  return ((ndItem.get_global_id(0) < Assign2D<lhs_t, rhs_t>::get_size()));
}
template <typename lhs_t, typename rhs_t>
SYCL_BLAS_INLINE typename Assign2D<lhs_t, rhs_t>::value_t
Assign2D<lhs_t, rhs_t>::eval(typename Assign2D<lhs_t, rhs_t>::index_t i) {
  const index_t nb_rows = get_size_row();
  const index_t col = i / nb_rows;
  const index_t row = i - nb_rows * col;
  auto val = lhs_.eval(col * lhs_.getSizeL() + row) =
      rhs_.eval(col * rhs_.getSizeL() + row);
  return val;
}
template <typename lhs_t, typename rhs_t>
SYCL_BLAS_INLINE typename Assign2D<lhs_t, rhs_t>::value_t
Assign2D<lhs_t, rhs_t>::eval(cl::sycl::nd_item<1> ndItem) {
  return Assign2D<lhs_t, rhs_t>::eval(ndItem.get_global_id(0));
}
template <typename lhs_t, typename rhs_t>
SYCL_BLAS_INLINE void Assign2D<lhs_t, rhs_t>::bind(cl::sycl::handler &h) {
  lhs_.bind(h);
  rhs_.bind(h);
}
template <typename lhs_t, typename rhs_t>
SYCL_BLAS_INLINE void Assign2D<lhs_t, rhs_t>::adjust_access_displacement() {
  lhs_.adjust_access_displacement();
  rhs_.adjust_access_displacement();
}

}  // namespace blas

#endif  // BLAS2_TREES_HPP
