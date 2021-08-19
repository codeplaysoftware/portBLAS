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
 *  @filename blas_operators.hpp
 *
 **************************************************************************/
// NO H for this one as this one is internal. but all the macro will be
// generated by cmake in cpp file
#ifndef SYCL_BLAS_OPERATORS_HPP
#define SYCL_BLAS_OPERATORS_HPP

#include <iostream>
#include <stdexcept>
#include <type_traits>
#include <vector>

#include <CL/sycl.hpp>

#include "operations/blas_constants.hpp"
#include "operations/blas_operators.h"

namespace blas {
struct Operators {};

/* StripASP.
 * When using ComputeCpp CE, the Device Compiler uses Address Spaces
 * to deal with the different global memories.
 * However, this causes problem with std type traits, which see the
 * types with address space qualifiers as different from the C++
 * standard types.
 *
 * This is StripASP function servers as a workaround that removes
 * the address space for various types.
 */
template <typename type_with_address_space_t>
struct StripASP {
  typedef type_with_address_space_t type;
};

#if defined(__SYCL_DEVICE_ONLY__) && defined(__COMPUTECPP__)
#define GENERATE_STRIP_ASP(entry_type, pointer_type)                   \
  template <>                                                          \
  struct StripASP<typename std::remove_pointer<                        \
      typename cl::sycl::pointer_type<entry_type>::pointer_t>::type> { \
    typedef entry_type type;                                           \
  };

#define GENERATE_STRIP_ASP_LOCATION(data_t) \
  GENERATE_STRIP_ASP(data_t, constant_ptr)  \
  GENERATE_STRIP_ASP(data_t, private_ptr)   \
  GENERATE_STRIP_ASP(data_t, local_ptr)     \
  GENERATE_STRIP_ASP(data_t, global_ptr)

#define GENERATE_STRIP_ASP_TUPLE(index_t, data_t, pointer_type)     \
  template <>                                                       \
  struct StripASP<                                                  \
      typename std::remove_pointer<typename cl::sycl::pointer_type< \
          IndexValueTuple<index_t, data_t>>::pointer_t>::type> {    \
    typedef IndexValueTuple<index_t, data_t> type;                  \
  };
#define GENERATE_STRIP_ASP_NEST_TUPLE(index_t, data_t, pointer_type)         \
  template <>                                                                \
  struct StripASP<typename std::remove_pointer<                              \
      typename cl::sycl::pointer_type<IndexValueTuple<                       \
          index_t, IndexValueTuple<index_t, data_t>>>::pointer_t>::type> {   \
    typedef IndexValueTuple<index_t, IndexValueTuple<index_t, data_t>> type; \
  };

#define INDEX_VALUE_STRIP_ASP_LOCATION(index_t, data_t)   \
  GENERATE_STRIP_ASP_TUPLE(index_t, data_t, constant_ptr) \
  GENERATE_STRIP_ASP_TUPLE(index_t, data_t, private_ptr)  \
  GENERATE_STRIP_ASP_TUPLE(index_t, data_t, local_ptr)    \
  GENERATE_STRIP_ASP_TUPLE(index_t, data_t, global_ptr)

#define NEST_INDEX_VALUE_STRIP_ASP_LOCATION(index_t, data_t)   \
  GENERATE_STRIP_ASP_NEST_TUPLE(index_t, data_t, constant_ptr) \
  GENERATE_STRIP_ASP_NEST_TUPLE(index_t, data_t, private_ptr)  \
  GENERATE_STRIP_ASP_NEST_TUPLE(index_t, data_t, local_ptr)    \
  GENERATE_STRIP_ASP_NEST_TUPLE(index_t, data_t, global_ptr)
#endif  // __SYCL_DEVICE_ONLY__  && __COMPUTECPP__

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
  template <typename value_t>
  using stripped_t = typename StripASP<value_t>::type;

#ifdef BLAS_DATA_TYPE_HALF
  template <typename value_t>
  using is_floating_point = std::integral_constant<
      bool, std::is_floating_point<stripped_t<value_t>>::value ||
                std::is_same<stripped_t<value_t>, cl::sycl::half>::value>;
#else
  template <typename value_t>
  using is_floating_point = std::is_floating_point<value_t>;
#endif  // BLAS_DATA_TYPE_HALF

  template <typename value_t>
  static SYCL_BLAS_INLINE value_t eval(
      const value_t &val,
      typename std::enable_if<!is_floating_point<value_t>::value>::type * = 0) {
    return cl::sycl::abs(val);
  }

  template <typename value_t>
  static SYCL_BLAS_INLINE value_t
  eval(const value_t &val,
       typename std::enable_if<is_floating_point<value_t>::value>::type * = 0) {
    return cl::sycl::fabs(val);
  }
};

#if defined(__SYCL_DEVICE_ONLY__) && defined(__COMPUTECPP__)
GENERATE_STRIP_ASP_LOCATION(double)
GENERATE_STRIP_ASP_LOCATION(float)
INDEX_VALUE_STRIP_ASP_LOCATION(int, float)
INDEX_VALUE_STRIP_ASP_LOCATION(long, float)
INDEX_VALUE_STRIP_ASP_LOCATION(long long, float)
INDEX_VALUE_STRIP_ASP_LOCATION(int, double)
INDEX_VALUE_STRIP_ASP_LOCATION(long, double)
INDEX_VALUE_STRIP_ASP_LOCATION(long long, double)
NEST_INDEX_VALUE_STRIP_ASP_LOCATION(int, float)
NEST_INDEX_VALUE_STRIP_ASP_LOCATION(long, float)
NEST_INDEX_VALUE_STRIP_ASP_LOCATION(long long, float)
NEST_INDEX_VALUE_STRIP_ASP_LOCATION(int, double)
NEST_INDEX_VALUE_STRIP_ASP_LOCATION(long, double)
NEST_INDEX_VALUE_STRIP_ASP_LOCATION(long long, double)
#endif

/*!
Definitions of unary operators
*/
struct AdditionIdentity : public Operators {
  template <typename rhs_t>
  static SYCL_BLAS_INLINE rhs_t eval(const rhs_t) {
    return constant<rhs_t, const_val::zero>::value();
  }
};

struct ProductIdentity : public Operators {
  template <typename rhs_t>
  static SYCL_BLAS_INLINE rhs_t eval(const rhs_t r) {
    return (constant<rhs_t, const_val::one>::value());
  }
};

struct IdentityOperator : public Operators {
  template <typename rhs_t>
  static SYCL_BLAS_INLINE rhs_t eval(const rhs_t r) {
    return (r);
  }
};

struct NegationOperator : public Operators {
  template <typename rhs_t>
  static SYCL_BLAS_INLINE rhs_t eval(const rhs_t r) {
    return (-r);
  }
};

struct SqrtOperator : public Operators {
  template <typename rhs_t>
  static SYCL_BLAS_INLINE rhs_t eval(const rhs_t r) {
    return (cl::sycl::sqrt(r));
  }
};

struct DoubleOperator : public Operators {
  template <typename rhs_t>
  static SYCL_BLAS_INLINE rhs_t eval(const rhs_t r) {
    return (r + r);
  }
};

struct SquareOperator : public Operators {
  template <typename rhs_t>
  static SYCL_BLAS_INLINE rhs_t eval(const rhs_t r) {
    return (r * r);
  }
};

/*!
 Definitions of binary operators
*/

struct AddOperator : public Operators {
  template <typename lhs_t, typename rhs_t>
  static SYCL_BLAS_INLINE typename StripASP<rhs_t>::type eval(const lhs_t &l,
                                                              const rhs_t &r) {
    return (l + r);
  }

  template <typename rhs_t>
  constexpr static SYCL_BLAS_INLINE typename rhs_t::value_t init() {
    return constant<typename rhs_t::value_t, const_val::zero>::value();
  }
};

struct ProductOperator : public Operators {
  template <typename lhs_t, typename rhs_t>
  static SYCL_BLAS_INLINE typename StripASP<rhs_t>::type eval(const lhs_t &l,
                                                              const rhs_t &r) {
    return (l * r);
  }

  template <typename rhs_t>
  constexpr static SYCL_BLAS_INLINE typename rhs_t::value_t init() {
    return constant<typename rhs_t::value_t, const_val::one>::value();
  }
};

struct DivisionOperator : public Operators {
  template <typename lhs_t, typename rhs_t>
  static SYCL_BLAS_INLINE typename StripASP<rhs_t>::type eval(const lhs_t &l,
                                                              const rhs_t &r) {
    return (l / r);
  }

  template <typename rhs_t>
  constexpr static SYCL_BLAS_INLINE typename rhs_t::value_t init() {
    return constant<typename rhs_t::value_t, const_val::one>::value();
  }
};

struct MaxOperator : public Operators {
  template <typename lhs_t, typename rhs_t>
  static SYCL_BLAS_INLINE typename StripASP<rhs_t>::type eval(const lhs_t &l,
                                                              const rhs_t &r) {
    return ((l > r) ? l : r);
  }

  template <typename rhs_t>
  constexpr static SYCL_BLAS_INLINE typename rhs_t::value_t init() {
    return constant<typename rhs_t::value_t, const_val::min>::value();
  }
};

struct MinOperator : public Operators {
  template <typename lhs_t, typename rhs_t>
  static SYCL_BLAS_INLINE typename StripASP<rhs_t>::type eval(const lhs_t &l,
                                                              const rhs_t &r) {
    return ((l < r) ? l : r);
  }

  template <typename rhs_t>
  constexpr static SYCL_BLAS_INLINE typename rhs_t::value_t init() {
    return constant<typename rhs_t::value_t, const_val::max>::value();
  }
};

struct AbsoluteAddOperator : public Operators {
  template <typename lhs_t, typename rhs_t>
  static SYCL_BLAS_INLINE typename StripASP<rhs_t>::type eval(const lhs_t &l,
                                                              const rhs_t &r) {
    return AbsoluteValue::eval(l) + AbsoluteValue::eval(r);
  }  // namespace blas

  template <typename rhs_t>
  constexpr static SYCL_BLAS_INLINE typename rhs_t::value_t init() {
    return constant<typename rhs_t::value_t, const_val::zero>::value();
  }
};

struct IMaxOperator : public Operators {
  template <typename lhs_t, typename rhs_t>
  static SYCL_BLAS_INLINE typename StripASP<rhs_t>::type eval(const lhs_t &l,
                                                              const rhs_t &r) {
    if (AbsoluteValue::eval(
            static_cast<typename StripASP<lhs_t>::type>(l).get_value()) <
            AbsoluteValue::eval(
                static_cast<typename StripASP<rhs_t>::type>(r).get_value()) ||
        (AbsoluteValue::eval(
             static_cast<typename StripASP<lhs_t>::type>(l).get_value()) ==
             AbsoluteValue::eval(
                 static_cast<typename StripASP<rhs_t>::type>(r).get_value()) &&
         l.get_index() > r.get_index())) {
      return static_cast<typename StripASP<rhs_t>::type>(r);
    } else {
      return static_cast<typename StripASP<lhs_t>::type>(l);
    }
  }

  template <typename rhs_t>
  constexpr static SYCL_BLAS_INLINE typename rhs_t::value_t init() {
    return constant_pair<typename rhs_t::value_t, const_val::max,
                         const_val::zero>::value();
  }
};

struct IMinOperator : public Operators {
  template <typename lhs_t, typename rhs_t>
  static SYCL_BLAS_INLINE typename StripASP<rhs_t>::type eval(const lhs_t &l,
                                                              const rhs_t &r) {
    if (AbsoluteValue::eval(
            static_cast<typename StripASP<lhs_t>::type>(l).get_value()) >
            AbsoluteValue::eval(
                static_cast<typename StripASP<rhs_t>::type>(r).get_value()) ||
        (AbsoluteValue::eval(
             static_cast<typename StripASP<lhs_t>::type>(l).get_value()) ==
             AbsoluteValue::eval(
                 static_cast<typename StripASP<rhs_t>::type>(r).get_value()) &&
         l.get_index() > r.get_index())) {
      return static_cast<typename StripASP<rhs_t>::type>(r);
    } else {
      return static_cast<typename StripASP<lhs_t>::type>(l);
    }
  }

  template <typename rhs_t>
  constexpr static SYCL_BLAS_INLINE typename rhs_t::value_t init() {
    return constant_pair<typename rhs_t::value_t, const_val::max,
                         const_val::abs_max>::value();
  }
};

struct CollapseIndexTupleOperator : public Operators {
  template <typename lhs_t, typename rhs_t>
  static SYCL_BLAS_INLINE
      typename ResolveReturnType<CollapseIndexTupleOperator,
                                 typename StripASP<rhs_t>::type>::type
      eval(const lhs_t &l, const rhs_t &r) {
    return typename StripASP<rhs_t>::type::value_t(
        static_cast<typename StripASP<rhs_t>::type>(r).get_index() * l +
            static_cast<typename StripASP<rhs_t>::type>(r).val.get_index(),
        static_cast<typename StripASP<rhs_t>::type>(r).get_value());
  }

  template <typename rhs_t>
  constexpr static SYCL_BLAS_INLINE typename rhs_t::value_t init() {
    return constant_pair<typename rhs_t::value_t, const_val::zero,
                         constant_pair<typename rhs_t::value_t, const_val::zero,
                                       const_val::zero>::value()>::value();
  }
};
}  // namespace blas

#endif  // BLAS_OPERATORS_HPP
