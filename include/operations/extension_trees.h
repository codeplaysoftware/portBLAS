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

/*!
 * @brief Determines which type of reduction to perform
 */
enum class Reduction_t : int {
  full = 0,  // Not implemented yet
  partial_rows = 1,
  partial_columns = 2  // Not implemented yet
};

/*!
 * @brief Wrapper around the reduction.
 *
 * The executor will specialize the execution for every reduction type and use
 * the specific reduction classes
 */
template <typename operator_t, typename input_t, typename output_t, int ClSize,
          int WgSize, typename element_t, int Reduction_type>
class Reduction {
 public:
  using index_t = typename input_t::index_t;
  input_t in_;
  output_t out_;
  const index_t rows_;
  const index_t cols_;
  Reduction(input_t in, output_t out, index_t num_rows, index_t num_cols);
};

/*!
 * @brief Calculates the parameters of the row reduction step (used by the
 * executor and the kernel)
 */
template <typename index_t, typename element_t, int ClSize, int WgSize>
struct ReductionRows_Params {
  /* The number of elements per cache line size depends on the element type */
  static constexpr index_t cl_elems = ClSize / sizeof(element_t);

  /* Work group dimensions */
  static constexpr index_t work_group_rows = cl_elems;
  static constexpr index_t work_group_cols = WgSize / work_group_rows;

  /* Local memory dimensions */
  static constexpr index_t local_memory_size =
      work_group_rows * work_group_cols;
};

/*!
 * @brief This class holds the kernel for the partial reduction of the rows.
 *
 * The output buffer will contain the same number of rows as the input buffer
 * and a smaller number of columns. Eventually this will result in a single
 * column. The number of work groups can be chosen to control the number of
 * steps before the reduction of the rows is complete.
 */
template <typename operator_t, typename input_t, typename output_t, int ClSize,
          int WgSize, typename element_t>
class ReductionPartialRows;

}  // namespace blas

#endif  // SYCL_BLAS_EXTENSION_TREES_H
