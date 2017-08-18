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
#define SYCLBLAS_DEFINE_BINARY_OPERATOR(name, initial, expr)   \
  struct name {                                                \
    template <typename L, typename R>                          \
    static R eval(const L l, const R r) {                      \
      return expr;                                             \
    }                                                          \
                                                               \
    template <typename R>                                      \
    static R init() {                                          \
      return constant<R, initial>::value;                      \
    }                                                          \
  };

/** wang random generator
 */
struct wang {
  static uint64_t eval(uint64_t key) {
    key = (~key) + (key << 21);  // key = (key << 21) - key - 1;
    key = key ^ (key >> 24);
    key = (key + (key << 3)) + (key << 8);  // key * 265
    key = key ^ (key >> 14);
    key = (key + (key << 2)) + (key << 4);  // key * 21
    key = key ^ (key >> 28);
    key = key + (key << 31);
    return key;
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
SYCLBLAS_DEFINE_UNARY_OPERATOR(sqtOp1_struct, std::sqrt(r))
SYCLBLAS_DEFINE_UNARY_OPERATOR(tupOp1_struct, r)
SYCLBLAS_DEFINE_UNARY_OPERATOR(addOp1_struct, (r + r))
SYCLBLAS_DEFINE_UNARY_OPERATOR(prdOp1_struct, (r * r))
SYCLBLAS_DEFINE_UNARY_OPERATOR(absOp1_struct, std::fabs(r))
SYCLBLAS_DEFINE_BINARY_OPERATOR(addOp2_struct, const_val::zero, (l + r))
SYCLBLAS_DEFINE_BINARY_OPERATOR(prdOp2_struct, const_val::one, (l * r))
SYCLBLAS_DEFINE_BINARY_OPERATOR(divOp2_struct, const_val::one, (l / r))
SYCLBLAS_DEFINE_BINARY_OPERATOR(maxOp2_struct, const_val::min,
                                ((l > r) ? l : r))
SYCLBLAS_DEFINE_BINARY_OPERATOR(minOp2_struct, const_val::max,
                                ((l < r) ? l : r))
SYCLBLAS_DEFINE_BINARY_OPERATOR(addAbsOp2_struct, const_val::zero,
                                (absOp1_struct::eval(l) +
                                 absOp1_struct::eval(r)))
SYCLBLAS_DEFINE_BINARY_OPERATOR(
    maxIndOp2_struct, const_val::imin,
    ((absOp1_struct::eval(l.getVal()) < absOp1_struct::eval(r.getVal())) ||
     (absOp1_struct::eval(l.getVal()) == absOp1_struct::eval(r.getVal()) &&
      l.getInd() > r.getInd()))
        ? r
        : l)
SYCLBLAS_DEFINE_BINARY_OPERATOR(
    minIndOp2_struct, const_val::imax,
    ((absOp1_struct::eval(l.getVal()) > absOp1_struct::eval(r.getVal())) ||
     (absOp1_struct::eval(l.getVal()) == absOp1_struct::eval(r.getVal()) &&
      l.getInd() > r.getInd()))
        ? r
        : l)

/*!
Undefine SYCLBLAS_DEIFNE_*_OPERATOR macros.
*/
#undef SYCLBLAS_DEFINE_UNARY_OPERATOR
#undef SYCLBLAS_DEFINE_BINARY_OPERATOR

}  // namespace blas

#endif  // BLAS_OPERATORS_HPP
