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
 *  @filename blas_operators.hpp
 *
 **************************************************************************/
// NO H for this one as this one is internal. but all the macro will be
// generated by cmake in cpp file
#ifndef PORTBLAS_OPERATORS_HPP
#define PORTBLAS_OPERATORS_HPP

#include <iostream>
#include <stdexcept>
#include <type_traits>
#include <vector>

#include <CL/sycl.hpp>

#include "operations/blas_constants.hpp"
#include "operations/blas_operators.h"

namespace blas {
struct Operators {};

/**
 * AbsoluteValue.
 *
 * SYCL 1.2 defines different functions for abs for floating point
 * and integer numbers, following the OpenCL convention.
 * To choose the appropriate one we use this template specialization
 * that is enabled for floating point to use fabs, and abs for everything
 * else.
 */
struct AbsoluteValue {
#ifdef BLAS_ENABLE_HALF
  template <typename value_t>
  using is_floating_point =
      std::integral_constant<bool,
                             std::is_floating_point<value_t>::value ||
                                 std::is_same<value_t, cl::sycl::half>::value>;
#else
  template <typename value_t>
  using is_floating_point = std::is_floating_point<value_t>;
#endif  // BLAS_ENABLE_HALF

  template <typename value_t>
  static PORTBLAS_INLINE value_t eval(
      const value_t &val,
      typename std::enable_if<!is_floating_point<value_t>::value>::type * = 0) {
    return cl::sycl::abs(val);
  }

  template <typename value_t>
  static PORTBLAS_INLINE value_t
  eval(const value_t &val,
       typename std::enable_if<is_floating_point<value_t>::value>::type * = 0) {
    return cl::sycl::fabs(val);
  }
};

/*!
Definitions of unary operators
*/
struct AdditionIdentity : public Operators {
  template <typename rhs_t>
  static PORTBLAS_INLINE rhs_t eval(const rhs_t) {
    return constant<rhs_t, const_val::zero>::value();
  }
};

struct ProductIdentity : public Operators {
  template <typename rhs_t>
  static PORTBLAS_INLINE rhs_t eval(const rhs_t r) {
    return (constant<rhs_t, const_val::one>::value());
  }
};

struct IdentityOperator : public Operators {
  template <typename rhs_t>
  static PORTBLAS_INLINE rhs_t eval(const rhs_t r) {
    return (r);
  }
};

struct SignOperator : public Operators {
  template <typename rhs_t>
  static PORTBLAS_INLINE rhs_t eval(const rhs_t r) {
    return cl::sycl::sign(r);
  }
};

struct NegationOperator : public Operators {
  template <typename rhs_t>
  static PORTBLAS_INLINE rhs_t eval(const rhs_t r) {
    return (-r);
  }
};

struct SqrtOperator : public Operators {
  template <typename rhs_t>
  static PORTBLAS_INLINE rhs_t eval(const rhs_t r) {
    return (cl::sycl::sqrt(r));
  }
};

struct HypotenuseOperator : public Operators {
  template <typename lhs_t, typename rhs_t>
  static PORTBLAS_INLINE rhs_t eval(const lhs_t l, const rhs_t r) {
    return (cl::sycl::hypot(l, r));
  }
};

struct DoubleOperator : public Operators {
  template <typename rhs_t>
  static PORTBLAS_INLINE rhs_t eval(const rhs_t r) {
    return (r + r);
  }
};

struct SquareOperator : public Operators {
  template <typename rhs_t>
  static PORTBLAS_INLINE rhs_t eval(const rhs_t r) {
    return (r * r);
  }
};

/*!
 Definitions of binary operators
*/

struct AddOperator : public Operators {
  template <typename lhs_t, typename rhs_t>
  static PORTBLAS_INLINE rhs_t eval(const lhs_t &l, const rhs_t &r) {
    return (l + r);
  }

  template <typename rhs_t>
  constexpr static PORTBLAS_INLINE typename rhs_t::value_t init() {
    return constant<typename rhs_t::value_t, const_val::zero>::value();
  }

  template <typename element_t, typename index_t>
  static PORTBLAS_INLINE element_t get_final_value(const element_t &l,
                                                   const index_t &) {
    return l;
  }
};

struct ProductOperator : public Operators {
  template <typename lhs_t, typename rhs_t>
  static PORTBLAS_INLINE rhs_t eval(const lhs_t &l, const rhs_t &r) {
    return (l * r);
  }

  template <typename rhs_t>
  constexpr static PORTBLAS_INLINE typename rhs_t::value_t init() {
    return constant<typename rhs_t::value_t, const_val::one>::value();
  }

  template <typename element_t, typename index_t>
  static PORTBLAS_INLINE element_t get_final_value(const element_t &l,
                                                   const index_t &) {
    return l;
  }
};

struct DivisionOperator : public Operators {
  template <typename lhs_t, typename rhs_t>
  static PORTBLAS_INLINE rhs_t eval(const lhs_t &l, const rhs_t &r) {
    return (l / r);
  }

  template <typename rhs_t>
  constexpr static PORTBLAS_INLINE typename rhs_t::value_t init() {
    return constant<typename rhs_t::value_t, const_val::one>::value();
  }

  template <typename element_t, typename index_t>
  static PORTBLAS_INLINE element_t get_final_value(const element_t &l,
                                                   const index_t &) {
    return l;
  }
};

struct MeanOperator : public Operators {
  template <typename element_t>
  static PORTBLAS_INLINE element_t eval(const element_t &accumulator,
                                        const element_t &val) {
    return accumulator + val;
  }

  template <typename rhs_t>
  constexpr static PORTBLAS_INLINE typename rhs_t::value_t init() {
    return constant<typename rhs_t::value_t, const_val::zero>::value();
  }

  template <typename element_t, typename index_t>
  static PORTBLAS_INLINE element_t get_final_value(const element_t &l,
                                                   const index_t &r) {
    return (l / static_cast<element_t>(r));
  }
};

struct MaxOperator : public Operators {
  template <typename lhs_t, typename rhs_t>
  static PORTBLAS_INLINE rhs_t eval(const lhs_t &l, const rhs_t &r) {
    return ((l > r) ? l : r);
  }

  template <typename rhs_t>
  constexpr static PORTBLAS_INLINE typename rhs_t::value_t init() {
    return constant<typename rhs_t::value_t, const_val::min>::value();
  }

  template <typename element_t, typename index_t>
  static PORTBLAS_INLINE element_t get_final_value(const element_t &l,
                                                   const index_t &) {
    return l;
  }
};

struct MinOperator : public Operators {
  template <typename lhs_t, typename rhs_t>
  static PORTBLAS_INLINE rhs_t eval(const lhs_t &l, const rhs_t &r) {
    return ((l < r) ? l : r);
  }

  template <typename rhs_t>
  constexpr static PORTBLAS_INLINE typename rhs_t::value_t init() {
    return constant<typename rhs_t::value_t, const_val::max>::value();
  }

  template <typename element_t, typename index_t>
  static PORTBLAS_INLINE element_t get_final_value(const element_t &l,
                                                   const index_t &) {
    return l;
  }
};

struct AbsoluteAddOperator : public Operators {
  template <typename lhs_t, typename rhs_t>
  static PORTBLAS_INLINE rhs_t eval(const lhs_t &l, const rhs_t &r) {
    return AbsoluteValue::eval(l) + AbsoluteValue::eval(r);
  }  // namespace blas

  template <typename rhs_t>
  constexpr static PORTBLAS_INLINE typename rhs_t::value_t init() {
    return constant<typename rhs_t::value_t, const_val::zero>::value();
  }

  template <typename element_t, typename index_t>
  static PORTBLAS_INLINE element_t get_final_value(const element_t &l,
                                                   const index_t &) {
    return l;
  }
};

struct IMaxOperator : public Operators {
  template <typename lhs_t, typename rhs_t>
  static PORTBLAS_INLINE rhs_t eval(const lhs_t &l, const rhs_t &r) {
    if (AbsoluteValue::eval(static_cast<lhs_t>(l).get_value()) <
            AbsoluteValue::eval(static_cast<rhs_t>(r).get_value()) ||
        (AbsoluteValue::eval(static_cast<lhs_t>(l).get_value()) ==
             AbsoluteValue::eval(static_cast<rhs_t>(r).get_value()) &&
         l.get_index() > r.get_index())) {
      return static_cast<rhs_t>(r);
    } else {
      return static_cast<lhs_t>(l);
    }
  }

  template <typename rhs_t>
  constexpr static PORTBLAS_INLINE typename rhs_t::value_t init() {
    return constant_pair<typename rhs_t::value_t, const_val::max,
                         const_val::zero>::value();
  }
};

struct IMinOperator : public Operators {
  template <typename lhs_t, typename rhs_t>
  static PORTBLAS_INLINE rhs_t eval(const lhs_t &l, const rhs_t &r) {
    if (AbsoluteValue::eval(static_cast<lhs_t>(l).get_value()) >
            AbsoluteValue::eval(static_cast<rhs_t>(r).get_value()) ||
        (AbsoluteValue::eval(static_cast<lhs_t>(l).get_value()) ==
             AbsoluteValue::eval(static_cast<rhs_t>(r).get_value()) &&
         l.get_index() > r.get_index())) {
      return static_cast<rhs_t>(r);
    } else {
      return static_cast<lhs_t>(l);
    }
  }

  template <typename rhs_t>
  constexpr static PORTBLAS_INLINE typename rhs_t::value_t init() {
    return constant_pair<typename rhs_t::value_t, const_val::max,
                         const_val::abs_max>::value();
  }
};

struct CollapseIndexTupleOperator : public Operators {
  template <typename lhs_t, typename rhs_t>
  static PORTBLAS_INLINE
      typename ResolveReturnType<CollapseIndexTupleOperator, rhs_t>::type
      eval(const lhs_t &l, const rhs_t &r) {
    return typename rhs_t::value_t(static_cast<rhs_t>(r).get_index() * l +
                                       static_cast<rhs_t>(r).val.get_index(),
                                   static_cast<rhs_t>(r).get_value());
  }

  template <typename rhs_t>
  constexpr static PORTBLAS_INLINE typename rhs_t::value_t init() {
    return constant_pair<typename rhs_t::value_t, const_val::zero,
                         constant_pair<typename rhs_t::value_t, const_val::zero,
                                       const_val::zero>::value()>::value();
  }
};
}  // namespace blas

#endif  // BLAS_OPERATORS_HPP
