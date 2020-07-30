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
 *  @filename quantize.hpp
 *
 **************************************************************************/

#ifndef SYCL_BLAS_QUANTIZE_HPP
#define SYCL_BLAS_QUANTIZE_HPP

#include "executors/executor.h"
#include "policy/sycl_policy_handler.h"

namespace blas {
namespace internal {

/**
 * @brief Type of the accessor to use as input when performing quantization
 * @tparam input_t Input data type
 */
template <typename input_t>
using quantized_input_acc_t =
    cl::sycl::accessor<input_t, 1, cl::sycl::access::mode::read,
                       cl::sycl::access::target::global_buffer>;

/**
 * @brief Type of the accessor to use as output when performing quantization
 * @tparam output_t Output data type
 */
template <typename output_t>
using quantized_output_acc_t =
    cl::sycl::accessor<output_t, 1, cl::sycl::access::mode::write,
                       cl::sycl::access::target::global_buffer>;

/**
 * @brief Kernel that performs quantization.
 *        The generic form just performs a static_cast of each element.
 * @tparam input_t Input data type
 * @tparam output_t Output data type
 */
template <typename input_t, typename output_t>
struct QuantizeKernel {
  quantized_input_acc_t<input_t> input_;
  quantized_output_acc_t<output_t> output_;

  void operator()(cl::sycl::id<1> index) const {
    output_[index] = static_cast<output_t>(input_[index]);
  }
};

/**
 * @brief Struct that dispatches a kernel to perform quantization
 * @tparam input_t Input data type
 * @tparam output_t Output data type
 */
template <typename input_t, typename output_t>
struct Quantize {
  template <typename executor_t>
  static typename executor_t::policy_t::event_t run(
      executor_t& ex, cl::sycl::buffer<input_t>& input,
      cl::sycl::buffer<output_t>& output) {
    return {
        ex.get_policy_handler().get_queue().submit([&](cl::sycl::handler& cgh) {
          const auto kernel = QuantizeKernel<input_t, output_t>{
              quantized_input_acc_t<input_t>{input, cgh},
              quantized_output_acc_t<output_t>{output, cgh}};
          cgh.parallel_for(cl::sycl::range<1>{input.get_size()}, kernel);
        })};
  }
};

/**
 * @brief Specialization for quantizing to same type.
 *        In this case no algorithm is required,
 *        the data is just copied from one buffer to another.
 * @tparam scalar_t Input and output data type
 */
template <typename scalar_t>
struct Quantize<scalar_t, scalar_t> {
  template <typename executor_t>
  static typename executor_t::policy_t::event_t run(
      executor_t& ex, cl::sycl::buffer<scalar_t>& input,
      cl::sycl::buffer<scalar_t>& output) {
    return {
        ex.get_policy_handler().get_queue().submit([&](cl::sycl::handler& cgh) {
          auto input_acc = quantized_input_acc_t<scalar_t>{input, cgh};
          auto output_acc = quantized_output_acc_t<scalar_t>{output, cgh};
          cgh.copy(input_acc, output_acc);
        })};
  }
};

}  // namespace internal
}  // namespace blas

#endif  // SYCL_BLAS_QUANTIZE_HPP
