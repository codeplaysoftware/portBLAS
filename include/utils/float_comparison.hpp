/*
 * Copyright 2019 Codeplay Software Ltd.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use these files except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/*
 * This file provides tools to compare floating-point numbers.
 * The function almost_equal returns a boolean indicating if two scalars can
 * be considered equal. The function compare_vectors checks if two vectors of
 * the same size are almost equal, prints a message on stderr if a value
 * mismatch is found, and returns a boolean as well.
 * Neither of the methods raises an exception nor calls exit(1)
 */

#ifndef UTILS_FLOAT_COMPARISON_H_
#define UTILS_FLOAT_COMPARISON_H_

#include <cmath>
#include <iostream>

#include <CL/sycl.hpp>

#ifdef BLAS_DATA_TYPE_HALF
inline std::ostream& operator<<(std::ostream& os, const cl::sycl::half& value) {
  os << static_cast<float>(value);
  return os;
}
#endif  // BLAS_DATA_TYPE_HALF

namespace utils {

template <typename scalar_t>
bool isnan(scalar_t value) noexcept {
  return std::isnan(value);
}

template <typename scalar_t>
bool isinf(scalar_t value) noexcept {
  return std::isinf(value);
}

template <typename scalar_t>
bool abs(scalar_t value) noexcept {
  return std::abs(value);
}

#ifdef BLAS_DATA_TYPE_HALF

template <>
inline bool isnan<cl::sycl::half>(cl::sycl::half value) noexcept {
  return std::isnan(static_cast<float>(value));
}

template <>
inline bool isinf<cl::sycl::half>(cl::sycl::half value) noexcept {
  return std::isinf(static_cast<float>(value));
}

template <>
inline bool abs<cl::sycl::half>(cl::sycl::half value) noexcept {
  return std::abs(static_cast<float>(value));
}

#endif  // BLAS_DATA_TYPE_HALF

/**
 * Indicates the tolerated margin for relative differences
 */
template <typename scalar_t>
inline scalar_t getRelativeErrorMargin() {
  /* Measured empirically with gemm. The dimensions of the matrices (even k)
   * don't seem to have an impact on the observed relative differences
   * In the cases where the relative error is relevant (non close to zero),
   * relative differences of up to 0.002 were observed for float
   */
  return static_cast<scalar_t>(0.005);
}

template <>
inline double getRelativeErrorMargin<double>() {
  /* Measured empirically with gemm. The dimensions of the matrices (even k)
   * don't seem to have an impact on the observed relative differences
   * In the cases where the relative error is relevant (non close to zero),
   * relative differences of up to 10^-12 were observed for double
   */
  return 0.0000000001;  // 10^-10
}

/**
 * Indicates the tolerated margin for absolute differences (used in case the
 * scalars are close to 0)
 */
template <typename scalar_t>
inline scalar_t getAbsoluteErrorMargin() {
  /* Measured empirically with gemm.
   * In the cases where the relative error is irrelevant (close to zero),
   * absolute differences of up to 0.0006 were observed for float
   */
  return 0.001;
}

template <>
inline double getAbsoluteErrorMargin<double>() {
  /* Measured empirically with gemm.
   * In the cases where the relative error is irrelevant (close to zero),
   * absolute differences of up to 10^-12 were observed for double
   */
  return 0.0000000001;  // 10^-10
}

/**
 * Compare two scalars and returns false if the difference is not acceptable.
 */
template <typename scalar_t>
inline bool almost_equal(scalar_t const& scalar1, scalar_t const& scalar2) {
  // Shortcut, also handles case where both are zero
  if (scalar1 == scalar2) {
    return true;
  }
  // Handle cases where both values are NaN or inf
  if ((utils::isnan(scalar1) && utils::isnan(scalar2)) ||
      (utils::isinf(scalar1) && utils::isinf(scalar2))) {
    return true;
  }

  const auto absolute_diff = utils::abs(scalar1 - scalar2);

  // Close to zero, the relative error doesn't work, use absolute error
  if (scalar1 == scalar_t{0} || scalar2 == scalar_t{0} ||
      absolute_diff < getAbsoluteErrorMargin<scalar_t>()) {
    return (absolute_diff < getAbsoluteErrorMargin<scalar_t>());
  }
  // Use relative error
  const auto absolute_sum = utils::abs(scalar1) + utils::abs(scalar2);
  return (absolute_diff / absolute_sum) < getRelativeErrorMargin<scalar_t>();
}

/**
 * Compare two vectors and returns false if the difference is not acceptable.
 * The second vector is considered the reference.
 */
template <typename scalar_t>
inline bool compare_vectors(std::vector<scalar_t> const& vec,
                            std::vector<scalar_t> const& ref,
                            std::ostream& err_stream = std::cerr,
                            std::string end_line = "\n") {
  if (vec.size() != ref.size()) {
    err_stream << "Error: tried to compare vectors of different sizes"
               << std::endl;
    return false;
  }

  for (int i = 0; i < vec.size(); ++i) {
    if (!almost_equal(vec[i], ref[i])) {
      err_stream << "Value mismatch at index " << i << ": " << vec[i]
                 << "; expected " << ref[i] << end_line;
      return false;
    }
  }
  return true;
}

}  // namespace utils

#endif  // UTILS_FLOAT_COMPARISON_H_
