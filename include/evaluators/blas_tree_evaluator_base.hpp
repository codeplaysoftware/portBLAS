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
 *  @filename blas_tree_evaluator_base.hpp
 *
 **************************************************************************/

#ifndef BLAS_TREE_EVALUATOR_BASE_HPP
#define BLAS_TREE_EVALUATOR_BASE_HPP

#include <evaluators/blas_functor_traits.hpp>
#include <executors/blas_device.hpp>
#include <operations/blas_types.hpp>
#include <views/operview_base.hpp>

namespace blas {
namespace internal {

/*! DetectScalar.
 * @brief Class specialization used to detect scalar values in scalar expression
 * nodes.
 * When the value is not an integral basic type,
 * it is assumed to be a vector and the first value
 * is used.
 */
template <typename T, bool is_supported_scalar = blas_type_support<T>::value>
struct DetectScalar;
template <typename T>
struct DetectScalar<T, true> {
  static T get_scalar(T &scalar) { return scalar; }
};
template <typename T>
struct DetectScalar<T, false> {
  static typename T::value_type get_Scalar(T &opSCL) { return opSCL.eval(0); }
};

/*! get_scalar.
 * @brief Template autodecuction function for DetectScalar.
*/
template <typename T>
auto get_scalar(T &scl) -> decltype(DetectScalar<T>::get_scalar(scl)) {
  return DetectScalar<T>::get_scalar(scl);
}

}  // namespace internal

/*!
 * Evaluator.
 * @brief Evaluation class which specifies how an expression node is evaluated
 * on a specific device.
 * @tparam Expression Expression to be evaluated.
 * @tparam Device Device for which evaluation is specified.
 */
template <class Expression, typename Device>
struct Evaluator;

}  // namespace blas

#endif
