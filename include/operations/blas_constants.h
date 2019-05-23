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

#include "../blas_meta.h"
#include <complex>
#include <limits>

namespace blas {

/*!
@brief Container for a scalar value and an index.
*/
template <typename scalar_t, typename index_t>
struct IndexValueTuple {
  using value_t = scalar_t;
  value_t val;
  index_t ind;

  constexpr explicit IndexValueTuple(index_t _ind, value_t _val)
      : val(_val), ind(_ind){};
  SYCL_BLAS_INLINE index_t get_index() const { return ind; }
  SYCL_BLAS_INLINE value_t get_value() const { return val; }
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
  imax = 5,
  imin = 6
};
/*!
@def define a specialization of the constant template value for each indicator.
@ref ConstValue.
@tparam primitive_t The value type to specialize for.
@tparam Indicator The constant to specialize for.
*/
template <typename primitive_t, const_val Indicator>
struct ConstValue {
  constexpr static SYCL_BLAS_INLINE primitive_t init() {
    return static_cast<primitive_t>(static_cast<int>(Indicator));
  }
};
template <typename primitive_t>
struct ConstValue<primitive_t, const_val::max> {
  constexpr static SYCL_BLAS_INLINE primitive_t init() {
    return std::numeric_limits<primitive_t>::min();
  }
};

template <typename primitive_t>
struct ConstValue<primitive_t, const_val::min> {
  constexpr static SYCL_BLAS_INLINE primitive_t init() {
    return std::numeric_limits<primitive_t>::max();
  }
};

/*!
@brief Template struct used to represent constants within a compile-time
expression tree, each instantiation will have a static constexpr member variable
of the type value_t initialized to the specified constant.
@tparam value_t Value type of the constant.
@tparam kIndicator Enumeration specifying the constant.
*/
template <typename value_t, const_val Indicator>
struct constant {
  constexpr static SYCL_BLAS_INLINE value_t value() {
    return ConstValue<value_t, Indicator>::init();
  }
};

template <typename value_t, typename index_t>
struct constant<IndexValueTuple<value_t, index_t>, const_val::imax> {
  constexpr static SYCL_BLAS_INLINE IndexValueTuple<value_t, index_t> value() {
    return IndexValueTuple<value_t, index_t>(
        std::numeric_limits<index_t>::max(),
        static_cast<value_t>(0));  // This is used for absolute max, -1 == 1
  }
};

template <typename value_t, typename index_t>
struct constant<IndexValueTuple<value_t, index_t>, const_val::imin> {
  constexpr static SYCL_BLAS_INLINE IndexValueTuple<value_t, index_t> value() {
    return IndexValueTuple<value_t, index_t>(
        std::numeric_limits<index_t>::max(),
        std::numeric_limits<value_t>::max());
  }
};
template <typename value_t, const_val Indicator>
struct constant<std::complex<value_t>, Indicator> {
  constexpr static SYCL_BLAS_INLINE std::complex<value_t> value() {
    return std::complex<value_t>(ConstValue<value_t, Indicator>::init(),
                                 ConstValue<value_t, Indicator>::init());
  }
};

}  // namespace blas

#endif  // BLAS_CONSTANTS_H
