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
 *  @filename blas_constants.h
 *
 **************************************************************************/

#ifndef SYCL_BLAS_CONSTANTS_H
#define SYCL_BLAS_CONSTANTS_H

#include "blas_meta.h"

#include <CL/sycl.hpp>

#include <complex>
#include <limits>

namespace blas {

template <typename scalar_t, typename index_t>
struct IndexValueTuple;

// This template is specialised to help with the getting of the value of nested
// tuples
// (a, x).get_value() == x
// (a, (b, x)).get_value() == x
template <typename value_t>
struct GetTupleValue {
  using return_t = value_t;

  SYCL_BLAS_INLINE static return_t get(const value_t val) { return val; }
};
template <typename index_t, typename value_t>
struct GetTupleValue<IndexValueTuple<index_t, value_t>> {
  using return_t = value_t;

  SYCL_BLAS_INLINE static return_t get(
      const IndexValueTuple<index_t, value_t> val) {
    return val.get_value();
  }
};

/*!
@brief Container for a scalar value and an index.
*/
template <typename ix_t, typename val_t>
struct IndexValueTuple {
  using value_t = val_t;
  using index_t = ix_t;

  index_t ind;
  value_t val;

  // This operator is required due to a ComputeCPP bug
  // (If the RHS of this operator is static const, then llvm.memcpy is broken)
  constexpr IndexValueTuple(const IndexValueTuple<index_t, value_t> &other)
      : val(other.val), ind(other.ind) {}

  constexpr explicit IndexValueTuple(index_t _ind, value_t _val)
      : ind(_ind), val(_val){};
  SYCL_BLAS_INLINE index_t get_index() const { return ind; }
  SYCL_BLAS_INLINE typename GetTupleValue<value_t>::return_t get_value() const {
    return GetTupleValue<value_t>::get(val);
  }
  // This operator is required due to a ComputeCPP bug
  // (If the RHS of this operator is static const, then llvm.memcpy is broken)
  IndexValueTuple<index_t, value_t> &operator=(
      const IndexValueTuple<index_t, value_t> &other) {
    val = other.val;
    ind = other.ind;

    return *this;
  }
};

/*!
@brief Enum class used to indicate a constant value associated with a type.
*/
enum class const_val : int {
  zero = 0,
  one = 1,
  m_one = -1,
  two = 2,
  m_two = -2,
  max = 3,
  min = 4,
  abs_max = 5,
  abs_min = 6,
  collapse = 7,
};

/*!
@brief Template struct used to represent constants within a compile-time
expression tree, each instantiation will have a static constexpr member
variable of the type value_t initialized to the specified constant.
@tparam value_t Value type of the constant.
@tparam kIndicator Enumeration specifying the constant.
*/
template <typename value_t, const_val Indicator>
struct constant {
  constexpr static SYCL_BLAS_INLINE value_t value() {
    return static_cast<value_t>(Indicator);
  }
};

template <typename value_t>
struct constant<value_t, const_val::max> {
  constexpr static SYCL_BLAS_INLINE value_t value() {
    return std::numeric_limits<value_t>::max();
  }
};

template <typename value_t>
struct constant<value_t, const_val::min> {
  constexpr static SYCL_BLAS_INLINE value_t value() {
    return std::numeric_limits<value_t>::min();
  }
};

template <typename value_t>
struct constant<value_t, const_val::abs_max> {
  constexpr static SYCL_BLAS_INLINE value_t value() {
    return std::numeric_limits<value_t>::max();
  }
};

template <typename value_t>
struct constant<value_t, const_val::abs_min> {
  constexpr static SYCL_BLAS_INLINE value_t value() {
    return static_cast<value_t>(0);
  }
};

template <typename value_t, typename index_t>
struct constant<IndexValueTuple<index_t, value_t>, const_val::abs_max> {
  constexpr static SYCL_BLAS_INLINE IndexValueTuple<index_t, value_t> value() {
    return IndexValueTuple<index_t, value_t>(
        std::numeric_limits<index_t>::max(),
        std::numeric_limits<value_t>::max());
  }
};

template <typename value_t, typename index_t>
struct constant<IndexValueTuple<index_t, value_t>, const_val::abs_min> {
  constexpr static SYCL_BLAS_INLINE IndexValueTuple<index_t, value_t> value() {
    return IndexValueTuple<index_t, value_t>(
        std::numeric_limits<index_t>::max(), 0);
  }
};

template <typename value_t, typename index_t>
struct constant<IndexValueTuple<index_t, value_t>, const_val::max> {
  constexpr static SYCL_BLAS_INLINE IndexValueTuple<index_t, value_t> value() {
    return IndexValueTuple<index_t, value_t>(
        std::numeric_limits<index_t>::max(),
        std::numeric_limits<value_t>::max());
  }
};

template <typename value_t, typename index_t>
struct constant<IndexValueTuple<index_t, value_t>, const_val::min> {
  constexpr static SYCL_BLAS_INLINE IndexValueTuple<index_t, value_t> value() {
    return IndexValueTuple<index_t, value_t>(
        std::numeric_limits<index_t>::max(),
        std::numeric_limits<value_t>::min());
  }
};

template <typename value_t, typename index_t, const_val Indicator>
struct constant<IndexValueTuple<index_t, value_t>, Indicator> {
  constexpr static SYCL_BLAS_INLINE IndexValueTuple<index_t, value_t> value() {
    return IndexValueTuple<index_t, value_t>(
        std::numeric_limits<index_t>::max(),
        constant<value_t, Indicator>::value());
  }
};

template <typename value_t, typename index_t>
struct constant<IndexValueTuple<index_t, value_t>, const_val::collapse> {
  constexpr static SYCL_BLAS_INLINE IndexValueTuple<index_t, value_t> value() {
    return IndexValueTuple<index_t, value_t>(
        std::numeric_limits<index_t>::max(),
        std::numeric_limits<value_t>::max());
  }
};

template <typename value_t, const_val Indicator>
struct constant<std::complex<value_t>, Indicator> {
  constexpr static SYCL_BLAS_INLINE std::complex<value_t> value() {
    return std::complex<value_t>(constant<value_t, Indicator>::value(),
                                 constant<value_t, Indicator>::value());
  }
};

template <>
struct constant<cl::sycl::half, const_val::zero>
    : constant<float, const_val::zero> {};

template <>
struct constant<cl::sycl::half, const_val::one>
    : constant<float, const_val::one> {};

template <>
struct constant<cl::sycl::half, const_val::m_one>
    : constant<float, const_val::m_one> {};

template <>
struct constant<cl::sycl::half, const_val::two>
    : constant<float, const_val::two> {};

template <>
struct constant<cl::sycl::half, const_val::m_two>
    : constant<float, const_val::m_two> {};

template <>
struct constant<cl::sycl::half, const_val::max>
    : constant<float, const_val::max> {};

template <>
struct constant<cl::sycl::half, const_val::min>
    : constant<float, const_val::min> {};

template <>
struct constant<cl::sycl::half, const_val::abs_max>
    : constant<float, const_val::abs_max> {};

template <>
struct constant<cl::sycl::half, const_val::abs_min>
    : constant<float, const_val::abs_min> {};

template <>
struct constant<cl::sycl::half, const_val::collapse>
    : constant<float, const_val::collapse> {};

template <typename iv_type, const_val IndexIndicator, const_val ValueIndicator>
struct constant_pair {
  constexpr static SYCL_BLAS_INLINE iv_type value() {
    return iv_type(
        constant<typename iv_type::index_t, IndexIndicator>::value(),
        constant<typename iv_type::value_t, ValueIndicator>::value());
  }
};

}  // namespace blas

#endif  // BLAS_CONSTANTS_H
