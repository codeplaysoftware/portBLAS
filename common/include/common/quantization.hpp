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
 *  @filename quantization.hpp
 *
 **************************************************************************/

#ifndef UTILS_QUANTIZATION_HPP
#define UTILS_QUANTIZATION_HPP

#include <sycl_blas.h>

#include <CL/sycl.hpp>

#include <type_traits>
#include <vector>

namespace utils {
namespace internal {

/**
 * @brief How data types should be stored before quantization.
 *        Most data quantizes to and from float.
 * @tparam scalar_t Data type
 */
template <typename scalar_t>
struct DataStorage {
  using type = float;
};

/**
 * @brief double can be stored as itself because it's supported natively
 */
template <>
struct DataStorage<double> {
  using type = double;
};

}  // namespace internal

/**
 * @brief How data types should be stored before quantization
 * @tparam scalar_t Data type
 */
template <typename scalar_t>
using data_storage_t = typename internal::DataStorage<scalar_t>::type;

////////////////////////////////////////////////////////////////////////////////
// Testing: make_quantized_buffer

namespace internal {

template <typename scalar_t>
using quantized_buffer_t =
    decltype(blas::make_sycl_iterator_buffer<scalar_t>(int{}));

/**
 * @brief Helper for constructing a quantized buffer
 *
 * 1. Constructs a buffer to hold the input data, which is float or double
 * 2. Copies input data to the buffer
 * 3. Constructs a buffer to hold data of scalar_t
 * 4. Performs quantization from input data to scalar_t
 */
template <typename scalar_t>
struct MakeQuantizedBuffer {
  using data_t = data_storage_t<scalar_t>;

  using return_t = quantized_buffer_t<scalar_t>;

  template <typename executor_t>
  static return_t run(executor_t& ex, std::vector<data_t>& input_vec) {
    auto data_gpu_x_v = blas::make_sycl_iterator_buffer<data_t>(
        static_cast<int>(input_vec.size()));
    ex.get_policy_handler().copy_to_device(input_vec.data(), data_gpu_x_v,
                                           input_vec.size());
    auto gpu_x_v = blas::make_sycl_iterator_buffer<scalar_t>(
        static_cast<int>(input_vec.size()));
    blas::_quantize(ex, data_gpu_x_v, gpu_x_v);
    return gpu_x_v;
  }

  template <typename executor_t>
  static return_t run(executor_t& ex, data_t& input_scalar) {
    auto data_gpu_x_v =
        blas::make_sycl_iterator_buffer<data_t>(static_cast<int>(1));
    ex.get_policy_handler().copy_to_device(&input_scalar, data_gpu_x_v, 1);
    auto gpu_x_v =
        blas::make_sycl_iterator_buffer<scalar_t>(static_cast<int>(1));
    blas::_quantize(ex, data_gpu_x_v, gpu_x_v);
    return gpu_x_v;
  }
};

/**
 * @brief Helper for constructing a quantized buffer
 *        where no quantization actually takes place,
 *        as is the case with float and double.
 *
 * 1. Constructs a buffer to hold data of scalar_t
 * 2. Copies input data to the buffer
 */
template <typename scalar_t>
struct MakeQuantizedBufferNoConversion {
  using data_t = scalar_t;

  using return_t = quantized_buffer_t<scalar_t>;

  template <typename executor_t>
  static return_t run(executor_t& ex, std::vector<data_t>& input_vec) {
    auto gpu_x_v = blas::make_sycl_iterator_buffer<scalar_t>(
        static_cast<int>(input_vec.size()));
    ex.get_policy_handler().copy_to_device(input_vec.data(), gpu_x_v,
                                           input_vec.size());
    return gpu_x_v;
  }

  template <typename executor_t>
  static return_t run(executor_t& ex, data_t& input_scalar) {
    auto gpu_x_v =
        blas::make_sycl_iterator_buffer<scalar_t>(static_cast<int>(1));
    ex.get_policy_handler().copy_to_device(&input_scalar, gpu_x_v, 1);
    return gpu_x_v;
  }
};

/**
 * @brief Buffer of float doesn't need a quantization step
 */
template <>
struct MakeQuantizedBuffer<float>
    : public MakeQuantizedBufferNoConversion<float> {};

/**
 * @brief Buffer of double doesn't need a quantization step
 */
template <>
struct MakeQuantizedBuffer<double>
    : public MakeQuantizedBufferNoConversion<double> {};

////////////////////////////////////////////////////////////////////////////////
// Testing: quantized_copy_to_host

/**
 * @brief Helper for copying data from device to host
 *        while also quantizing the data
 *
 * 1. Constructs a buffer to hold the output data, which is float or double
 * 2. Performs quantization from scalar_t to output data
 * 3. Copies output data to host
 */
template <typename scalar_t>
struct QuantizedCopyToHost {
  using data_t = data_storage_t<scalar_t>;

  template <typename executor_t>
  using return_t = typename executor_t::policy_t::event_t;

  template <typename executor_t>
  static return_t<executor_t> run(executor_t& ex,
                                  quantized_buffer_t<scalar_t>& device_buffer,
                                  std::vector<data_t>& output_vec) {
    auto data_gpu_x_v = blas::make_sycl_iterator_buffer<data_t>(
        static_cast<int>(output_vec.size()));
    blas::_quantize(ex, device_buffer, data_gpu_x_v);
    return ex.get_policy_handler().copy_to_host(data_gpu_x_v, output_vec.data(),
                                                output_vec.size());
  }

  template <typename executor_t>
  static return_t<executor_t> run(executor_t& ex,
                                  quantized_buffer_t<scalar_t>& device_buffer,
                                  data_t& output_scalar) {
    auto data_gpu_x_v =
        blas::make_sycl_iterator_buffer<data_t>(static_cast<int>(1));
    blas::_quantize(ex, device_buffer, data_gpu_x_v);
    return ex.get_policy_handler().copy_to_host(data_gpu_x_v, &output_scalar,
                                                1);
  }
};

/**
 * @brief Helper for copying data from device to host
 *        where no quantization actually takes place,
 *        as is the case with float and double.
 *
 * 1. Copies from device buffer to output data on host
 */
template <typename scalar_t>
struct QuantizedCopyToHostNoConversion {
  using data_t = scalar_t;

  template <typename executor_t>
  using return_t = typename executor_t::policy_t::event_t;

  template <typename executor_t>
  static return_t<executor_t> run(executor_t& ex,
                                  quantized_buffer_t<scalar_t>& device_buffer,
                                  std::vector<data_t>& output_vec) {
    return ex.get_policy_handler().copy_to_host(
        device_buffer, output_vec.data(), output_vec.size());
  }

  template <typename executor_t>
  static return_t<executor_t> run(executor_t& ex,
                                  quantized_buffer_t<scalar_t>& device_buffer,
                                  data_t& output_scalar) {
    return ex.get_policy_handler().copy_to_host(device_buffer, &output_scalar,
                                                1);
  }
};

/**
 * @brief Buffer of float doesn't need a quantization step
 */
template <>
struct QuantizedCopyToHost<float>
    : public QuantizedCopyToHostNoConversion<float> {};

/**
 * @brief Buffer of double doesn't need a quantization step
 */
template <>
struct QuantizedCopyToHost<double>
    : public QuantizedCopyToHostNoConversion<double> {};

}  // namespace internal

////////////////////////////////////////////////////////////////////////////////
// Exposed interface

/**
 * @brief Constructs a buffer containing data that was quantized
 *        from the provided input vector
 *
 * The vector will not be written back on buffer destruction.
 *
 * @return Buffer containing quantized data
 * @note scalar_t cannot be deduced, it has to be provided
 */
template <typename scalar_t, typename executor_t>
auto make_quantized_buffer(executor_t& ex,
                           std::vector<data_storage_t<scalar_t>>& input_vec)
    -> decltype(internal::MakeQuantizedBuffer<scalar_t>::run(ex, input_vec)) {
  return internal::MakeQuantizedBuffer<scalar_t>::run(ex, input_vec);
}

/**
 * @brief Constructs a buffer containing data that was quantized
 *        from the provided input scalar
 *
 * The scalar will not be written back on buffer destruction.
 *
 * @return Buffer containing quantized data
 * @note scalar_t cannot be deduced, it has to be provided
 */
template <typename scalar_t, typename executor_t>
auto make_quantized_buffer(executor_t& ex,
                           data_storage_t<scalar_t>& input_scalar)
    -> decltype(internal::MakeQuantizedBuffer<scalar_t>::run(ex,
                                                             input_scalar)) {
  return internal::MakeQuantizedBuffer<scalar_t>::run(ex, input_scalar);
}

/**
 * @brief Performs a copy from a buffer of scalar_t to output vector
 *        while also quantizing the data
 * @return Event associated with the operation
 * @note scalar_t cannot be deduced, it has to be provided
 */
template <typename scalar_t, typename executor_t>
auto quantized_copy_to_host(
    executor_t& ex, internal::quantized_buffer_t<scalar_t>& device_buffer,
    std::vector<data_storage_t<scalar_t>>& output_vec) ->
    typename executor_t::policy_t::event_t {
  return internal::QuantizedCopyToHost<scalar_t>::run(ex, device_buffer,
                                                      output_vec);
}

/**
 * @brief Performs a copy from a buffer of scalar_t to output scalar
 *        while also quantizing the data
 * @return Event associated with the operation
 * @note scalar_t cannot be deduced, it has to be provided
 */
template <typename scalar_t, typename executor_t>
auto quantized_copy_to_host(
    executor_t& ex, internal::quantized_buffer_t<scalar_t>& device_buffer,
    data_storage_t<scalar_t>& output_scalar) ->
    typename executor_t::policy_t::event_t {
  return internal::QuantizedCopyToHost<scalar_t>::run(ex, device_buffer,
                                                      output_scalar);
}

}  // namespace utils

#endif  // UTILS_QUANTIZATION_HPP
