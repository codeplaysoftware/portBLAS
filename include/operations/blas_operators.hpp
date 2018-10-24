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

#ifndef BLAS_OPERATORS_HPP
#define BLAS_OPERATORS_HPP

#include <iostream>
#include <stdexcept>
#include <vector>

#include <CL/sycl.hpp>

#include <operations/blas_constants.hpp>

namespace blas {

/*!
@def Macro for defining a unary operator.
@param name Name of the operator.
@param expr Return expression of the eval function of the oeprator.
*/
#define SYCLBLAS_DEFINE_UNARY_OPERATOR(name, expr) \
  struct name {                                    \
    template <typename R>                          \
    static R eval(const R r) {                     \
      return expr;                                 \
    }                                              \
  };

/*!
@brief Macro for defining a binary operator.
@param name Name of the operator.
@param inital Initial value used in the init function of the operator.
@param expr Return expression of the eval function of the operator.
*/
#define SYCLBLAS_DEFINE_BINARY_OPERATOR(name, initial, expr)          \
  struct name {                                                       \
    template <typename L, typename R>                                 \
    static typename strip_asp<R>::type eval(const L &l, const R &r) { \
      return expr;                                                    \
    }                                                                 \
                                                                      \
    template <typename R>                                             \
    static typename R::value_type init(const R &r) {                  \
      return constant<typename R::value_type, initial>::value;        \
    }                                                                 \
  };

/* strip_asp.
 * When using ComputeCpp CE, the Device Compiler uses Address Spaces
 * to deal with the different global memories.
 * However, this causes problem with std type traits, which see the
 * types with address space qualifiers as different from the C++
 * standard types.
 *
 * This is strip_asp function servers as a workaround that removes
 * the address space for various types.
 */
template <typename TypeWithAddressSpace>
struct strip_asp {
  typedef TypeWithAddressSpace type;
};

#if defined(__SYCL_DEVICE_ONLY__) && defined(__COMPUTECPP__)
#define GENERATE_STRIP_ASP(entry_type, pointer_type)                   \
  template <>                                                          \
  struct strip_asp<typename std::remove_pointer<                       \
      typename cl::sycl::pointer_type<entry_type>::pointer_t>::type> { \
    typedef entry_type type;                                           \
  };

#define GENERATE_STRIP_ASP_LOCATION(data_type) \
  GENERATE_STRIP_ASP(data_type, constant_ptr)  \
  GENERATE_STRIP_ASP(data_type, private_ptr)   \
  GENERATE_STRIP_ASP(data_type, local_ptr)     \
  GENERATE_STRIP_ASP(data_type, global_ptr)

GENERATE_STRIP_ASP_LOCATION(double)
GENERATE_STRIP_ASP_LOCATION(float)
#undef GENERATE_STRIP_ASP_LOCATION
#undef GENERATE_STRIP_ASP

#define GENERATE_STRIP_ASP_TUPLE(data_type, value_type, pointer_type)  \
  template <>                                                          \
  struct strip_asp<                                                    \
      typename std::remove_pointer<typename cl::sycl::pointer_type<    \
          IndexValueTuple<data_type, value_type>>::pointer_t>::type> { \
    typedef IndexValueTuple<data_type, value_type> type;               \
  };

#define INDEX_VALUE_STRIP_ASP_LOCATION(data_type, index_type)   \
  GENERATE_STRIP_ASP_TUPLE(data_type, index_type, constant_ptr) \
  GENERATE_STRIP_ASP_TUPLE(data_type, index_type, private_ptr)  \
  GENERATE_STRIP_ASP_TUPLE(data_type, index_type, local_ptr)    \
  GENERATE_STRIP_ASP_TUPLE(data_type, index_type, global_ptr)

INDEX_VALUE_STRIP_ASP_LOCATION(float, int)
INDEX_VALUE_STRIP_ASP_LOCATION(float, long)
INDEX_VALUE_STRIP_ASP_LOCATION(float, unsigned int)
INDEX_VALUE_STRIP_ASP_LOCATION(float, unsigned long)
INDEX_VALUE_STRIP_ASP_LOCATION(double, int)
INDEX_VALUE_STRIP_ASP_LOCATION(double, long)
INDEX_VALUE_STRIP_ASP_LOCATION(double, unsigned int)
INDEX_VALUE_STRIP_ASP_LOCATION(double, unsigned long)
#undef INDEX_VALUE_STRIP_ASP_LOCATION
#undef GENERATE_STRIP_ASP_TUPLE
#endif  // __SYCL_DEVICE_ONLY__  && __COMPUTECPP__

/**
 * syclblas_abs.
 *
 * SYCL 1.2 defines different functions for abs for floating point
 * and integer numbers, following the OpenCL convention.
 * To choose the appropriate one we use this template specialization
 * that is enabled for floating point to use fabs, and abs for everything
 * else.
 */
struct syclblas_abs {
  template <typename Type>
  static Type eval(const Type &val,
                   typename std::enable_if<!std::is_floating_point<
                       typename strip_asp<Type>::type>::value>::type * = 0) {
    return cl::sycl::abs(val);
  }

  template <typename Type>
  static Type eval(const Type &val,
                   typename std::enable_if<std::is_floating_point<
                       typename strip_asp<Type>::type>::value>::type * = 0) {
    return cl::sycl::fabs(val);
  }
};

/*!
Definitions of unary, bianry and ternary operators using the above macros.
*/
SYCLBLAS_DEFINE_UNARY_OPERATOR(iniAddOp1_struct,
                               (constant<R, const_val::zero>::value))
SYCLBLAS_DEFINE_UNARY_OPERATOR(iniPrdOp1_struct,
                               (constant<R, const_val::one>::value))
SYCLBLAS_DEFINE_UNARY_OPERATOR(posOp1_struct, (r))
SYCLBLAS_DEFINE_UNARY_OPERATOR(negOp1_struct, (-r))
SYCLBLAS_DEFINE_UNARY_OPERATOR(sqtOp1_struct, (cl::sycl::sqrt(r)))
SYCLBLAS_DEFINE_UNARY_OPERATOR(tupOp1_struct, r)
SYCLBLAS_DEFINE_UNARY_OPERATOR(addOp1_struct, (r + r))
SYCLBLAS_DEFINE_UNARY_OPERATOR(prdOp1_struct, (r * r))
SYCLBLAS_DEFINE_BINARY_OPERATOR(addOp2_struct, const_val::zero, (l + r))
SYCLBLAS_DEFINE_BINARY_OPERATOR(prdOp2_struct, const_val::one, (l * r))
SYCLBLAS_DEFINE_BINARY_OPERATOR(divOp2_struct, const_val::one, (l / r))
SYCLBLAS_DEFINE_BINARY_OPERATOR(maxOp2_struct, const_val::min,
                                ((l > r) ? l : r))
SYCLBLAS_DEFINE_BINARY_OPERATOR(minOp2_struct, const_val::max,
                                ((l < r) ? l : r))
SYCLBLAS_DEFINE_BINARY_OPERATOR(addAbsOp2_struct, const_val::zero,
                                (syclblas_abs::eval(l) + syclblas_abs::eval(r)))
SYCLBLAS_DEFINE_BINARY_OPERATOR(
    maxIndOp2_struct, const_val::imin,
    (syclblas_abs::eval(
         static_cast<typename strip_asp<L>::type>(l).get_value()) <
         syclblas_abs::eval(
             static_cast<typename strip_asp<R>::type>(r).get_value()) ||
     (syclblas_abs::eval(
          static_cast<typename strip_asp<L>::type>(l).get_value()) ==
          syclblas_abs::eval(
              static_cast<typename strip_asp<R>::type>(r).get_value()) &&
      l.get_index() > r.get_index()))
        ? static_cast<typename strip_asp<R>::type>(r)
        : static_cast<typename strip_asp<L>::type>(l))
SYCLBLAS_DEFINE_BINARY_OPERATOR(
    minIndOp2_struct, const_val::imax,
    (syclblas_abs::eval(
         static_cast<typename strip_asp<L>::type>(l).get_value()) >
         syclblas_abs::eval(
             static_cast<typename strip_asp<R>::type>(r).get_value()) ||
     (syclblas_abs::eval(
          static_cast<typename strip_asp<L>::type>(l).get_value()) ==
          syclblas_abs::eval(
              static_cast<typename strip_asp<R>::type>(r).get_value()) &&
      l.get_index() > r.get_index()))
        ? static_cast<typename strip_asp<R>::type>(r)
        : static_cast<typename strip_asp<L>::type>(l))

/*!
Undefine SYCLBLAS_DEIFNE_*_OPERATOR macros.
*/
#undef SYCLBLAS_DEFINE_UNARY_OPERATOR
#undef SYCLBLAS_DEFINE_BINARY_OPERATOR

}  // namespace blas

#endif  // BLAS_OPERATORS_HPP
