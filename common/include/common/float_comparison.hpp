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
#ifdef BLAS_ENABLE_COMPLEX
#include <complex>
#endif

#ifdef BLAS_DATA_TYPE_HALF
#if SYCL_LANGUAGE_VERSION < 202000
#include <CL/sycl.hpp>
inline std::ostream& operator<<(std::ostream& os, const cl::sycl::half& value) {
  os << static_cast<float>(value);
  return os;
}

namespace std {
template <>
class numeric_limits<cl::sycl::half> {
 public:
  static constexpr float min() { return -65504.0f; }
  static constexpr float max() { return 65504.0f; }
};
}  // namespace std
#endif  // SYCL_LANGUAGE_VERSION
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
scalar_t abs(scalar_t value) noexcept {
  return std::abs(value);
}

#ifdef BLAS_ENABLE_COMPLEX
template <typename scalar_t>
bool isnan(std::complex<scalar_t> value) noexcept {
  return (isnan<scalar_t>(value.real()) || isnan<scalar_t>(value.imag()));
}

template <typename scalar_t>
bool isinf(std::complex<scalar_t> value) noexcept {
  return (isinf<scalar_t>(value.real()) || isinf<scalar_t>(value.imag()));
}

template <typename scalar_t>
scalar_t abs(std::complex<scalar_t> value) noexcept {
  return std::abs(value);
}
#endif

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
inline cl::sycl::half abs<cl::sycl::half>(cl::sycl::half value) noexcept {
  return std::abs(static_cast<float>(value));
}

#endif  // BLAS_DATA_TYPE_HALF

template <typename scalar_t>
scalar_t clamp_to_limits(scalar_t v) {
  constexpr auto min_value = std::numeric_limits<scalar_t>::min();
  constexpr auto max_value = std::numeric_limits<scalar_t>::max();
  if (decltype(min_value)(v) < min_value) {
    return min_value;
  } else if (decltype(max_value)(v) > max_value) {
    return max_value;
  } else {
    return v;
  }
}

/**
 * Indicates the tolerated margin for relative differences
 */
template <typename scalar_t>
inline scalar_t getRelativeErrorMargin(const bool is_trsm) {
  /* Measured empirically with gemm. The dimensions of the matrices (even k)
   * don't seem to have an impact on the observed relative differences
   * In the cases where the relative error is relevant (non close to zero),
   * relative differences of up to 0.002 were observed for float
   */
  scalar_t margin = 0.005;
  if (is_trsm) {
    const char* en_joint_matrix = std::getenv("SB_ENABLE_JOINT_MATRIX");
    if (en_joint_matrix != NULL && std::is_same<scalar_t, float>::value &&
        *en_joint_matrix == '1') {
      // increase error margin for mixed precision calculation
      // for trsm operator.
      margin = 0.009f;
    }
  }
  return margin;
}

template <>
inline double getRelativeErrorMargin<double>(const bool) {
  /* Measured empirically with gemm. The dimensions of the matrices (even k)
   * don't seem to have an impact on the observed relative differences
   * In the cases where the relative error is relevant (non close to zero),
   * relative differences of up to 10^-12 were observed for double
   */
  return 0.0000000001;  // 10^-10
}

#ifdef BLAS_DATA_TYPE_HALF

template <>
inline cl::sycl::half getRelativeErrorMargin<cl::sycl::half>(const bool) {
  // Measured empirically with gemm
  return 0.05f;
}
#endif
/**
 * Indicates the tolerated margin for absolute differences (used in case the
 * scalars are close to 0)
 */
template <typename scalar_t>
inline scalar_t getAbsoluteErrorMargin(const bool is_trsm) {
  /* Measured empirically with gemm.
   * In the cases where the relative error is irrelevant (close to zero),
   * absolute differences of up to 0.0006 were observed for float
   */
  scalar_t margin = 0.001f;
  if (is_trsm) {
    const char* en_joint_matrix = std::getenv("SB_ENABLE_JOINT_MATRIX");
    if (en_joint_matrix != NULL && std::is_same<scalar_t, float>::value &&
        *en_joint_matrix == '1') {
      // increase error margin for mixed precision calculation
      // for trsm operator.
      margin = 0.009f;
    }
  }

  return margin;
}

template <>
inline double getAbsoluteErrorMargin<double>(const bool) {
  /* Measured empirically with gemm.
   * In the cases where the relative error is irrelevant (close to zero),
   * absolute differences of up to 10^-12 were observed for double
   */
  return 0.0000000001;  // 10^-10
}
#ifdef BLAS_DATA_TYPE_HALF

template <>
inline cl::sycl::half getAbsoluteErrorMargin<cl::sycl::half>(const bool) {
  // Measured empirically with gemm.
  return 1.0f;
}
#endif

/**
 * Compare two scalars and returns false if the difference is not acceptable.
 */
template <typename scalar_t, typename epsilon_t = scalar_t>
inline bool almost_equal(scalar_t const& scalar1, scalar_t const& scalar2,
                         const bool is_trsm = false) {
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
      absolute_diff < getAbsoluteErrorMargin<epsilon_t>(is_trsm)) {
    return (absolute_diff < getAbsoluteErrorMargin<epsilon_t>(is_trsm));
  }
  // Use relative error
  const auto absolute_sum = utils::abs(scalar1) + utils::abs(scalar2);
  return (absolute_diff / absolute_sum) <
         getRelativeErrorMargin<epsilon_t>(is_trsm);
}

/**
 * Compare two vectors and returns false if the difference is not acceptable.
 * The second vector is considered the reference.
 * @tparam scalar_t the type of data present in the input vectors
 * @tparam epilon_t the type used as tolerance. Lower precision types
 * (cl::sycl::half) will have a higher tolerance for errors
 */
template <typename scalar_t, typename epsilon_t = scalar_t>
inline bool compare_vectors(std::vector<scalar_t> const& vec,
                            std::vector<scalar_t> const& ref,
                            std::ostream& err_stream = std::cerr,
                            std::string end_line = "\n",
                            const bool is_trsm = false) {
  if (vec.size() != ref.size()) {
    err_stream << "Error: tried to compare vectors of different sizes"
               << std::endl;
    return false;
  }

  for (int i = 0; i < vec.size(); ++i) {
    if (!almost_equal<scalar_t, epsilon_t>(vec[i], ref[i], is_trsm)) {
      err_stream << "Value mismatch at index " << i << ": " << vec[i]
                 << "; expected " << ref[i] << end_line;
      return false;
    }
  }
  return true;
}

#ifdef BLAS_ENABLE_COMPLEX
/**
 * Compare two vectors of complex data and returns false if the difference is
 * not acceptable. The second vector is considered the reference.
 * @tparam scalar_t the type of complex underying data present in the input
 * vectors
 * @tparam epilon_t the type used as tolerance.
 */
template <typename scalar_t, typename epsilon_t = scalar_t>
inline bool compare_vectors(std::vector<std::complex<scalar_t>> const& vec,
                            std::vector<std::complex<scalar_t>> const& ref,
                            std::ostream& err_stream = std::cerr,
                            std::string end_line = "\n", bool is_trsm = false) {
  if (vec.size() != ref.size()) {
    err_stream << "Error: tried to compare vectors of different sizes"
               << std::endl;
    return false;
  }

  for (int i = 0; i < vec.size(); ++i) {
    if (!almost_equal<std::complex<scalar_t>, epsilon_t>(vec[i], ref[i])) {
      err_stream << "Value mismatch at index " << i << ": (" << vec[i].real()
                 << "," << vec[i].imag() << "); expected (" << ref[i].real()
                 << "," << ref[i].imag() << ")" << end_line;
      return false;
    }
  }
  return true;
}
#endif

/**
 * Compare two vectors at a given stride and window (unit_vec_size) and returns
 * false if the difference is not acceptable. The second vector is considered
 * the reference.
 * @tparam scalar_t the type of data present in the input vectors
 * @tparam epsilon_t the type used as tolerance. Lower precision types
 * (cl::sycl::half) will have a higher tolerance for errors
 * @param stride is the stride between two consecutive 'windows'
 * @param window is the size of a comparison window
 */
template <typename scalar_t, typename epsilon_t = scalar_t>
inline bool compare_vectors_strided(std::vector<scalar_t> const& vec,
                                    std::vector<scalar_t> const& ref,
                                    int stride, int window,
                                    std::ostream& err_stream = std::cerr,
                                    std::string end_line = "\n") {
  if (vec.size() != ref.size()) {
    err_stream << "Error: tried to compare vectors of different sizes"
               << std::endl;
    return false;
  }

  int k = 0;

  // Loop over windows
  while (window + (k + 1) * stride < vec.size()) {
    // Loop within a window
    for (int i = 0; i < window; ++i) {
      auto index = i + k * stride;
      if (!almost_equal<scalar_t, epsilon_t>(vec[index], ref[index])) {
        err_stream << "Value mismatch at index " << index << ": " << vec[index]
                   << "; expected " << ref[index] << end_line;
        return false;
      }
    }
    k += 1;
  }

  return true;
}

#ifdef BLAS_ENABLE_COMPLEX
/**
 * Compare two vectors of complex data at a given stride and window and returns
 * false if the difference is not acceptable. The second vector is considered
 * the reference.
 * @tparam scalar_t the type of the complex underying data present in the input
 * vectors
 * @tparam epsilon_t the type used as tolerance.
 * @param stride is the stride between two consecutive 'windows'
 * @param window is the size of a comparison window
 */
template <typename scalar_t, typename epsilon_t = scalar_t>
inline bool compare_vectors_strided(
    std::vector<std::complex<scalar_t>> const& vec,
    std::vector<std::complex<scalar_t>> const& ref, int stride, int window,
    std::ostream& err_stream = std::cerr, std::string end_line = "\n") {
  if (vec.size() != ref.size()) {
    err_stream << "Error: tried to compare vectors of different sizes"
               << std::endl;
    return false;
  }

  int k = 0;

  // Loop over windows
  while (window + (k + 1) * stride < vec.size()) {
    // Loop within a window
    for (int i = 0; i < window; ++i) {
      auto index = i + k * stride;
      if (!almost_equal<std::complex<scalar_t>, epsilon_t>(vec[index],
                                                           ref[index])) {
        err_stream << "Value mismatch at index " << index << ": ("
                   << vec[index].real() << "," << vec[index].imag()
                   << "); expected (" << ref[index].real() << ","
                   << ref[index].imag() << ")" << end_line;
        return false;
      }
    }
    k += 1;
  }

  return true;
}
#endif

}  // namespace utils

#endif  // UTILS_FLOAT_COMPARISON_H_
