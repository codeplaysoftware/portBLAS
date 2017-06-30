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
 *  @filename blas_constants.hpp
 *
 **************************************************************************/

#ifndef BLAS_CONSTANTS_HPP
#define BLAS_CONSTANTS_HPP

#include <complex>
#include <limits>

/*!
@def Macro used to define a specialization of the constant template struct.
@ref constant.
@param type The value type to specialize for.
@param indicator The constant to specialize for.
@param val The value to asssign to the specialization.
*/
#define SYCLBLAS_DEFINE_CONSTANT(type, indicator, val) \
  template <>                                          \
  struct constant<type, indicator> {                   \
    static const type value;                           \
  };                                                   \
  const type constant<type, indicator>::value = val;  // temporary work around
                                                      // for duplicator issue
                                                      // will be fixed int the
                                                      // next release

namespace blas {

/*!
@brief Container for a scalar value and an index.
*/
template <typename ScalarT>
struct IndVal {
  using value_type = ScalarT;
  size_t ind;
  value_type val;

  constexpr explicit IndVal(size_t _ind, value_type _val)
      : ind(_ind), val(_val){};
  size_t getInd() const { return ind; }
  value_type getVal() const { return val; }
};

/*!
@brief Enum class used to indicate a constant value associated with a type.
*/
enum class const_val : int {
  zero = 0,
  one = 1,
  m_one = 2,
  two = 3,
  m_two = 4,
  max = 5,
  min = 6,
  imax = 7,
  imin = 8
};

/*!
@brief Template struct used to represent constants within a compile-time
expression tree, each instantiation will have a static constexpr member variable
of the type valueT initialized to the specified constant.
@tparam valueT Value type of the constant.
@tparam kIndicator Enumeration specifying the constant.
*/
template <typename valueT, const_val kIndicator>
struct constant;

/*!
Specializations of template struct constant.
*/
SYCLBLAS_DEFINE_CONSTANT(int, const_val::zero, 0)
SYCLBLAS_DEFINE_CONSTANT(int, const_val::one, 1)
SYCLBLAS_DEFINE_CONSTANT(int, const_val::m_one, -1)
SYCLBLAS_DEFINE_CONSTANT(int, const_val::two, 2)
SYCLBLAS_DEFINE_CONSTANT(int, const_val::m_two, -2)
SYCLBLAS_DEFINE_CONSTANT(int, const_val::max, (std::numeric_limits<int>::max()))
SYCLBLAS_DEFINE_CONSTANT(int, const_val::min, (std::numeric_limits<int>::min()))
SYCLBLAS_DEFINE_CONSTANT(float, const_val::zero, 0.0f)
SYCLBLAS_DEFINE_CONSTANT(float, const_val::one, 1.0f)
SYCLBLAS_DEFINE_CONSTANT(float, const_val::m_one, -1.0f)
SYCLBLAS_DEFINE_CONSTANT(float, const_val::two, 2.0f)
SYCLBLAS_DEFINE_CONSTANT(float, const_val::m_two, -2.0f)
SYCLBLAS_DEFINE_CONSTANT(float, const_val::max,
                         (std::numeric_limits<float>::max()))
SYCLBLAS_DEFINE_CONSTANT(float, const_val::min,
                         (std::numeric_limits<float>::min()))
SYCLBLAS_DEFINE_CONSTANT(double, const_val::zero, 0.0)
SYCLBLAS_DEFINE_CONSTANT(double, const_val::one, 1.0)
SYCLBLAS_DEFINE_CONSTANT(double, const_val::m_one, -1.0)
SYCLBLAS_DEFINE_CONSTANT(double, const_val::two, 2.0)
SYCLBLAS_DEFINE_CONSTANT(double, const_val::m_two, -2.0)
SYCLBLAS_DEFINE_CONSTANT(double, const_val::max,
                         (std::numeric_limits<double>::max()))
SYCLBLAS_DEFINE_CONSTANT(double, const_val::min,
                         (std::numeric_limits<double>::min()))
SYCLBLAS_DEFINE_CONSTANT(std::complex<float>, const_val::zero,
                         (std::complex<float>(0.0f, 0.0f)))
SYCLBLAS_DEFINE_CONSTANT(std::complex<float>, const_val::one,
                         (std::complex<float>(1.0f, 0.0f)))
SYCLBLAS_DEFINE_CONSTANT(std::complex<float>, const_val::m_one,
                         (std::complex<float>(-1.0f, 0.0f)))
SYCLBLAS_DEFINE_CONSTANT(std::complex<float>, const_val::two,
                         (std::complex<float>(2.0f, 0.0f)))
SYCLBLAS_DEFINE_CONSTANT(std::complex<float>, const_val::m_two,
                         (std::complex<float>(-2.0f, 0.0f)))
SYCLBLAS_DEFINE_CONSTANT(
    std::complex<float>, const_val::max,
    (std::complex<float>(std::numeric_limits<float>::max(),
                         std::numeric_limits<float>::max())))
SYCLBLAS_DEFINE_CONSTANT(
    std::complex<float>, const_val::min,
    (std::complex<float>(std::numeric_limits<float>::min(),
                         std::numeric_limits<float>::min())))
SYCLBLAS_DEFINE_CONSTANT(std::complex<double>, const_val::zero,
                         (std::complex<double>(0.0f, 0.0f)))
SYCLBLAS_DEFINE_CONSTANT(std::complex<double>, const_val::one,
                         (std::complex<double>(1.0f, 0.0f)))
SYCLBLAS_DEFINE_CONSTANT(std::complex<double>, const_val::m_one,
                         (std::complex<double>(-1.0f, 0.0f)))
SYCLBLAS_DEFINE_CONSTANT(std::complex<double>, const_val::two,
                         (std::complex<double>(2.0f, 0.0f)))
SYCLBLAS_DEFINE_CONSTANT(std::complex<double>, const_val::m_two,
                         (std::complex<double>(-2.0f, 0.0f)))
SYCLBLAS_DEFINE_CONSTANT(
    std::complex<double>, const_val::max,
    (std::complex<double>(std::numeric_limits<double>::max(),
                          std::numeric_limits<double>::max())))
SYCLBLAS_DEFINE_CONSTANT(
    std::complex<double>, const_val::min,
    (std::complex<double>(std::numeric_limits<double>::min(),
                          std::numeric_limits<double>::min())))
SYCLBLAS_DEFINE_CONSTANT(IndVal<double>, const_val::imax,
                         (IndVal<double>(std::numeric_limits<size_t>::max(),
                                         std::numeric_limits<double>::max())))
SYCLBLAS_DEFINE_CONSTANT(IndVal<double>, const_val::imin,
                         (IndVal<double>(std::numeric_limits<size_t>::max(),
                                         std::numeric_limits<double>::min())))
}  // namespace blas

/*!
Undefine SYCLBLAS_DEFINE_CONSTANT
*/
#undef SYCLBLASS_DEFINE_CONSTANT

#endif  // BLAS_CONSTANTS_HPP
