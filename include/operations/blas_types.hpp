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
 *  @filename blas1_tree_evaluator.hpp
 *
 **************************************************************************/

#ifndef BLAS_TYPES_HPP
#define BLAS_TYPES_HPP

#include <complex>
#include <type_traits>

/*!
 * @brief Scalar types support indicator
 * @tparam ScalarT Scalar type to be checked
 */
template <class ScalarT>
struct blas_type_support {
  static constexpr bool value =
      std::is_same<ScalarT, float>::value ||
      std::is_same<ScalarT, double>::value ||
      std::is_same<ScalarT, std::complex<float>>::value ||
      std::is_same<ScalarT, std::complex<double>>::value;
};

#endif /* end of include guard: BLAS_TYPES_HPP */
