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
 *  @filename quantize.h
 *
 **************************************************************************/

#ifndef SYCL_BLAS_QUANTIZE_H
#define SYCL_BLAS_QUANTIZE_H

#include "executors/executor.h"

namespace blas {
namespace internal {

/**
 * @brief Quantizes a buffer of one type to a buffer of another
 * @tparam input_t Input data type
 * @tparam output_t Output data type
 * @tparam executor_t Type of the executor
 * @param ex Executor where the operation will run
 * @param[in] input Buffer holding the input data
 * @param[out] output Buffer where ouput will be stored
 * @return Event associated with the operation
 * @note Internal function
 */
template <typename input_t, typename output_t, typename executor_t>
typename executor_t::policy_t::event_t _quantize(
    executor_t& ex, cl::sycl::buffer<input_t> input,
    cl::sycl::buffer<output_t> output);

}  // namespace internal

/**
 * @brief Quantizes a buffer of one type to a buffer of another
 * @tparam executor_t Type of the executor
 * @tparam container_input_t Container type of the input buffer
 * @tparam container_output_t Container type of the output buffer
 * @param ex Executor where the operation will run
 * @param[in] input Container holding the input data
 * @param[out] output Container where ouput will be stored
 * @return Event associated with the operation
 */
template <typename executor_t, typename container_input_t,
          typename container_output_t>
typename executor_t::policy_t::event_t _quantize(executor_t& ex,
                                                 container_input_t input,
                                                 container_output_t output) {
  auto input_buf = ex.get_policy_handler().get_buffer(input).get_buffer();
  auto output_buf = ex.get_policy_handler().get_buffer(output).get_buffer();
  using input_t = typename container_input_t::scalar_t;
  using output_t = typename container_output_t::scalar_t;
  return internal::_quantize<input_t, output_t>(ex, input_buf, output_buf);
}

}  // namespace blas

#endif  // SYCL_BLAS_QUANTIZE_H
