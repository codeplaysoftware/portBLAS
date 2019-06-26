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

#include <complex>
#include <limits>
//#include <utility>
#include "blas_meta.h"
namespace blas {

/*!
@brief Container for a scalar value and an index.
*/
template <typename scalar_t, typename index_t>
struct IndexValueTuple {
  using value_t = scalar_t;
  using ind_t = index_t;
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
// enum class const_val : int {
//   zero = 0,
//   one = 1,
//   m_one = -1,
//   two = 2,
//   m_two = -2,
//   max = 3,
//   min = 4,
//   abs_max = 5,
//   abs_min = 6,
// };

namespace const_val {

struct zero {
  template <typename value_t>
  constexpr static SYCL_BLAS_INLINE value_t value() {
    return static_cast<value_t>(0);
  }
};

struct one {
  template <typename value_t>
  constexpr static SYCL_BLAS_INLINE value_t value() {
    return static_cast<value_t>(1);
  }
};

struct m_one {
  template <typename value_t>
  constexpr static SYCL_BLAS_INLINE value_t value() {
    return static_cast<value_t>(-1);
  }
};

struct two {
  template <typename value_t>
  constexpr static SYCL_BLAS_INLINE value_t value() {
    return static_cast<value_t>(2);
  }
};

struct m_two {
  template <typename value_t>
  constexpr static SYCL_BLAS_INLINE value_t value() {
    return static_cast<value_t>(-2);
  }
};

struct max {
  template <typename value_t>
  constexpr static SYCL_BLAS_INLINE value_t value() {
    return std::numeric_limits<value_t>::max();
  }
};

struct min {
  template <typename value_t>
  constexpr static SYCL_BLAS_INLINE value_t value() {
    return std::numeric_limits<value_t>::min();
  }
};

struct abs_max {
  template <typename value_t>
  constexpr static SYCL_BLAS_INLINE value_t value() {
    return std::numeric_limits<value_t>::max();
  }
};

struct abs_min {
  template <typename value_t>
  constexpr static SYCL_BLAS_INLINE value_t value() {
    return static_cast<value_t>(0);
  }
};

// template <typename value_const>
// struct complex {
//   template <typename elem_t>
//   constexpr static SYCL_BLAS_INLINE std::complex<value_t> value() {
//     return std::complex<value_t>(value_const::value<value_t>(),
//                                  value_const::value<value_t>());
//   }
// };

template <typename value_const, typename index_const>
struct index_val {
  template <typename tuple_t>
  constexpr static SYCL_BLAS_INLINE tuple_t value() {
    return tuple_t(index_const::template value<typename tuple_t::ind_t>(),
                   value_const::template value<typename tuple_t::value_t>());
  }
};

}  // namespace const_val

// template <typename elem_t, typename init_t>
// struct constant {
//   constexpr static SYCL_BLAS_INLINE elem_t value() {
//     return init_t::<elem_t> value();
//   };
// };

/*!
@brief Template struct used to represent constants within a compile-time
expression tree, each instantiation will have a static constexpr member
variable of the type value_t initialized to the specified constant.
@tparam value_t Value type of the constant.
@tparam kIndicator Enumeration specifying the constant.
*/
// template <typename value_t, const_val Indicator>
// struct constant {
//   constexpr static SYCL_BLAS_INLINE value_t value() {
//     return static_cast<value_t>(Indicator);
//   }
// };

// template <typename value_t>
// struct constant<value_t, const_val::max> {
//   constexpr static SYCL_BLAS_INLINE value_t value() {
//     return std::numeric_limits<value_t>::max();
//   }
// };

// template <typename value_t>
// struct constant<value_t, const_val::min> {
//   constexpr static SYCL_BLAS_INLINE value_t value() {
//     return std::numeric_limits<value_t>::min();
//   }
// };

// template <typename value_t>
// struct constant<value_t, const_val::abs_max> {
//   constexpr static SYCL_BLAS_INLINE value_t value() {
//     return std::numeric_limits<value_t>::max();
//   }
// };

// template <typename value_t>
// struct constant<value_t, const_val::abs_min> {
//   constexpr static SYCL_BLAS_INLINE value_t value() {
//     return static_cast<value_t>(0);
//   }
// };

// template <typename value_t, typename index_t>
// struct constant<IndexValueTuple<value_t, index_t>, const_val::abs_min> {
//   constexpr static SYCL_BLAS_INLINE IndexValueTuple<value_t, index_t> value()
//   {
//     return IndexValueTuple<value_t, index_t>(
//         constant<index_t, const_val::max>::value(),
//         constant<value_t, const_val::abs_min>::
//             value());  // This is used for absolute max, -1 == 1
//   }
// };

// template <typename value_t, typename index_t>
// struct constant<IndexValueTuple<value_t, index_t>, const_val::abs_max> {
//   constexpr static SYCL_BLAS_INLINE IndexValueTuple<value_t, index_t> value()
//   {
//     return IndexValueTuple<value_t, index_t>(
//         constant<index_t, const_val::max>::value(),
//         constant<value_t, const_val::abs_max>::value());
//     // This is used for absolute max, -1 == 1
//   }
// };

// template <typename value_t, const_val Indicator>
// struct constant<std::complex<value_t>, Indicator> {
//   constexpr static SYCL_BLAS_INLINE std::complex<value_t> value() {
//     return std::complex<value_t>(constant<value_t, Indicator>::value(),
//                                  constant<value_t, Indicator>::value());
//   }
// };

}  // namespace blas

#endif  // BLAS_CONSTANTS_H
