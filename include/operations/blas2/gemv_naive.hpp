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
 *  @filename gemv_naive.hpp
 *
 **************************************************************************/

#ifndef GEMV_NAIVE_HPP
#define GEMV_NAIVE_HPP

#include <operations/blas_operators.hpp>
#include <stdexcept>
#include <vector>
#include <views/view_sycl.hpp>

namespace blas {
template <class Output_T, class MA_T, class VX_T>
struct NaiveGemv {
  using value_type = typename VX_T::value_type;
  using IndexType = typename VX_T::IndexType;

  Output_T l;
  MA_T r1;
  VX_T r2;
  IndexType nWG_row;
  IndexType nWG_col;
  IndexType shrMemSize;
};

}  // namespace blas

#endif  // GEMV_NAIVE_HPP