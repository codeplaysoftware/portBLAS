/***************************************************************************
 *
 *  @license
 *  Copyright (C) 2016 Codeplay Software Limited
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
@param inital Initial value used in the init function of the oeprator.
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
#define GENERATE_STRIP_ASP(ENTRY_TYPE)                             \
  template <>                                                      \
  struct strip_asp<__attribute__((address_space(1))) ENTRY_TYPE> { \
    typedef ENTRY_TYPE type;                                       \
  };                                                               \
                                                                   \
  template <>                                                      \
  struct strip_asp<__attribute__((address_space(2))) ENTRY_TYPE> { \
    typedef ENTRY_TYPE type;                                       \
  };                                                               \
                                                                   \
  template <>                                                      \
  struct strip_asp<__attribute__((address_space(3))) ENTRY_TYPE> { \
    typedef ENTRY_TYPE type;                                       \
  };

GENERATE_STRIP_ASP(IndVal<double>)
GENERATE_STRIP_ASP(IndVal<float>)
GENERATE_STRIP_ASP(double)
GENERATE_STRIP_ASP(float)
#endif  // __SYCL_DEVICE_ONLY__  && __COMPUTECPP__

/**
 * syclblas_abs.
 *
 * SYCL 1.2 defines different functions for abs for floating point
 * and integer numbers, following the OpenCL convention.
 * To choose the appropriate one we use this template specialization
 * that is enabled for floating point to use fabs, and abs for everything else.
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
SYCLBLAS_DEFINE_UNARY_OPERATOR(
    sqtOp1_struct,
    (static_cast<double>(cl::sycl::sqrt(static_cast<double>(r)))))
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
    (syclblas_abs::eval(static_cast<typename strip_asp<L>::type>(l).getVal()) <
         syclblas_abs::eval(
             static_cast<typename strip_asp<R>::type>(r).getVal()) ||
     (syclblas_abs::eval(
          static_cast<typename strip_asp<L>::type>(l).getVal()) ==
          syclblas_abs::eval(
              static_cast<typename strip_asp<R>::type>(r).getVal()) &&
      l.getInd() > r.getInd()))
        ? static_cast<typename strip_asp<R>::type>(r)
        : static_cast<typename strip_asp<L>::type>(l))
SYCLBLAS_DEFINE_BINARY_OPERATOR(
    minIndOp2_struct, const_val::imax,
    (syclblas_abs::eval(static_cast<typename strip_asp<L>::type>(l).getVal()) >
         syclblas_abs::eval(
             static_cast<typename strip_asp<R>::type>(r).getVal()) ||
     (syclblas_abs::eval(
          static_cast<typename strip_asp<L>::type>(l).getVal()) ==
          syclblas_abs::eval(
              static_cast<typename strip_asp<R>::type>(r).getVal()) &&
      l.getInd() > r.getInd()))
        ? static_cast<typename strip_asp<R>::type>(r)
        : static_cast<typename strip_asp<L>::type>(l))

/*!
Undefine SYCLBLAS_DEIFNE_*_OPERATOR macros.
*/
#undef SYCLBLAS_DEFINE_UNARY_OPERATOR
#undef SYCLBLAS_DEFINE_BINARY_OPERATOR

}  // namespace blas

#endif  // BLAS_OPERATORS_HPP
