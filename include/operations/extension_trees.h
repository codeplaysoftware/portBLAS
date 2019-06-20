/***************************************************************************
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
 *  @filename extension_trees.h
 *
 **************************************************************************/

#ifndef SYCL_BLAS_EXTENSION_TREES_H
#define SYCL_BLAS_EXTENSION_TREES_H

#include <CL/sycl.hpp>

namespace blas {

  /*
   * @brief Determines which type of reduction to perform
   */
  enum class Reduction_t : int {
    full = 0,            // Not implemented yet
    partial_rows = 1,
    partial_columns = 2  // Not implemented yet
  };

/*!
 * TODO: more info here
 */
template <typename input_t, typename output_t, int ClSize,
          int WgSize, int WorkPerItem, typename element_t, int Reduction_type>
class Reduction {
public:
  using index_t = typename std::make_signed<typename input_t::index_t>::type;
  input_t in_;
  output_t out_;
  const index_t rows_;
  const index_t cols_;
  const index_t leading_dim_;
  Reduction(input_t in, output_t out, index_t num_rows, index_t num_cols);
};

/*!
 * TODO: more info here
 */
 template <typename input_t, typename output_t, int ClSize,
           int WgSize, int WorkPerItem, typename element_t, bool IsFinal>
class ReductionPartialRows;

// TODO: make_reduction function

}  // namespace blas

#endif  // SYCL_BLAS_EXTENSION_TREES_H
