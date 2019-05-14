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

namespace utils {

/**
 * Indicates the tolerated margin for relative differences
 */
template <typename scalar_t>
inline scalar_t getRelativeErrorMargin() {
  return static_cast<scalar_t>(0.005);
}

template <>
inline double getRelativeErrorMargin<double>() {
  return 0.000001;
}

/**
 * Indicates the tolerated margin for absolute differences
 */
template <typename scalar_t>
inline scalar_t getAbsoluteErrorMargin() {
  return 0.0001;
}

template <>
inline double getAbsoluteErrorMargin<double>() {
  return 0.0000001;
}

template <typename scalar_t>
inline bool almost_equal(scalar_t const& scalar1, scalar_t const& scalar2) {
  scalar_t absoluteErrorMargin = getAbsoluteErrorMargin<scalar_t>();
  scalar_t relativeErrorMargin = getRelativeErrorMargin<scalar_t>();
  // Shortcut, also handles case where both are zero
  if (scalar1 == scalar2) {
    return true;
  }
  // Handle cases where both values are NaN or inf
  if ((std::isnan(scalar1) && std::isnan(scalar2)) ||
      (std::isinf(scalar1) && std::isinf(scalar2))) {
    return true;
  }

  const auto absolute_diff = std::fabs(scalar1 - scalar2);

  // Close to zero, the relative error doesn't work, use absolute error
  if (scalar1 == 0 || scalar2 == 0 || absolute_diff < absoluteErrorMargin) {
    return (absolute_diff < absoluteErrorMargin);
  }
  // Use relative error
  const auto absolute_sum = std::fabs(scalar1) + std::fabs(scalar2);
  return (absolute_diff / absolute_sum) < relativeErrorMargin;
}

/**
 * Compare two vectors and returns false if the difference is not acceptable.
 * The second vector is considered the reference.
 */
template <typename scalar_t>
inline bool compare_vectors(std::vector<scalar_t>& vec,
                            std::vector<scalar_t>& ref) {
  if (vec.size() != ref.size()) {
    std::cerr << "Error: tried to compare vectors of different sizes"
              << std::endl;
    return false;
  }

  for (int i = 0; i < vec.size(); ++i) {
    if (!almost_equal(vec[i], ref[i])) {
      std::cerr << "Value mismatch at index " << i << ": " << vec[i]
                << "; expected " << ref[i] << std::endl;
      return false;
    }
  }
  return true;
}

}  // namespace utils

#endif  // UTILS_FLOAT_COMPARISON_H_
