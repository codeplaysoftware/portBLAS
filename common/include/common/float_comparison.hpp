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

#ifdef BLAS_ENABLE_HALF
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
#endif  // BLAS_ENABLE_HALF

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

#ifdef BLAS_ENABLE_HALF
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

/**
 * Custom float/double to uint16_t cast function.
 */
template <typename T>
inline uint16_t cast_to_half(T& val) {
  static_assert(
      std::is_scalar<T>::value,
      "Value to be casted to uint16_t should be either float or double.");
  // Uint bit representation using 32 (for float) or 64 (for double)
  using uint_type = typename std::conditional<std::is_same_v<T, float>,
                                              uint32_t, uint64_t>::type;

  using scalar_t = typename std::remove_const<T>::type;

  int32_t exp;
  uint_type bin_float, sign, mant;

  // Avoid const qualifier casting cases
  scalar_t val_temp = const_cast<scalar_t&>(val);

  // Reinterpret to uint for convenient bit manipulation
  bin_float = *reinterpret_cast<uint32_t*>(&val_temp);

  if constexpr (std::is_same_v<scalar_t, float>) {
    // Single precision float data case
    // Extract sign (to half's MSB)
    sign = (bin_float & 0x80000000) >> 16;
    // Adjust for half's smaller exponent & bias
    exp = ((bin_float & 0x7F800000) >> 23) - 127 + 15;
    // Truncate to 10 MSBits used in half's mantissa
    mant = (bin_float & 0x007FFFFF) >> 13;
  } else {
    // Double precision float data case
    sign = (bin_float & 0x8000000000000000ULL) >> 48;
    exp = ((bin_float & 0x7FF0000000000000ULL) >> 52) - 1023 + 15;
    mant = (bin_float & 0x000FFFFFFFFFFFFFULL) >> 42;
  }

  // Overflow/Underflow cases
  if (exp > 31) {
    // Clamp to max exponent for half (inf)
    exp = 31;
    mant = 0;
  } else if (exp < 0) {
    // Flush to zero (subnormal not handled)
    exp = 0;
    mant = 0;
  }

  // Reconstruct binary 16-bit output
  uint16_t out = sign | (exp << 10) | mant;

  return out;
}

/**
 * Float/Double to sycl::half cast utility function using the intermediate
 * cast_to_half(T&).
 */
template <typename T>
inline cl::sycl::half cast_to_sycl_half(T& val) {
  uint16_t half_bin = cast_to_half(val);
  cl::sycl::half result = *reinterpret_cast<cl::sycl::half*>(&half_bin);
  return static_cast<cl::sycl::half>(result);
}
#endif  // BLAS_ENABLE_HALF

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

#ifdef BLAS_ENABLE_HALF

template <>
inline cl::sycl::half getRelativeErrorMargin<cl::sycl::half>() {
  // Measured empirically with gemm
  return 0.05f;
}
#endif
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
  return 0.001f;
}

template <>
inline double getAbsoluteErrorMargin<double>() {
  /* Measured empirically with gemm.
   * In the cases where the relative error is irrelevant (close to zero),
   * absolute differences of up to 10^-12 were observed for double
   */
  return 0.0000000001;  // 10^-10
}
#ifdef BLAS_ENABLE_HALF

template <>
inline cl::sycl::half getAbsoluteErrorMargin<cl::sycl::half>() {
  // Measured empirically with gemm.
  return 1.0f;
}
#endif

/**
 * Reference type of the underlying tests data aimed to match the reference
 * library in tests/benchmarks and random number generator APIs.
 */
template <typename T, typename Enable = void>
struct ReferenceType {
  using type = T;
};

// When T is sycl::half, use float as type for random generation
// and reference BLAS implementations.
template <typename T>
struct ReferenceType<T, std::enable_if_t<std::is_same_v<T, cl::sycl::half>>> {
  using type = float;
};

/**
 * Compare two scalars and returns false if the difference is not acceptable.
 */
template <typename scalar_t, typename epsilon_t = scalar_t>
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
      absolute_diff < getAbsoluteErrorMargin<epsilon_t>()) {
    return (absolute_diff < getAbsoluteErrorMargin<epsilon_t>());
  }
  // Use relative error
  const auto absolute_sum = utils::abs(scalar1) + utils::abs(scalar2);
  return (absolute_diff / absolute_sum) < getRelativeErrorMargin<epsilon_t>();
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
                            std::string end_line = "\n") {
  if (vec.size() != ref.size()) {
    err_stream << "Error: tried to compare vectors of different sizes"
               << std::endl;
    return false;
  }

  for (int i = 0; i < vec.size(); ++i) {
    if (!almost_equal<scalar_t, epsilon_t>(vec[i], ref[i])) {
      err_stream << "Value mismatch at index " << i << ": " << vec[i]
                 << "; expected " << ref[i] << end_line;
      return false;
    }
  }
  return true;
}

#ifdef BLAS_ENABLE_HALF
/**
 * Compare two vectors of cl::sycl::half and float/double types and returns
 * false if the difference is not acceptable. The second vector is considered
 * the reference (non half type as it usually results from BLAS reference
 * functions with float/double outputs).
 * @tparam scalar_t the type of data present in the reference vector
 * (float/double)
 * @tparam epilon_t the type used as tolerance. Here low precision
 * cl::sycl::half, which will have a higher tolerance for errors.
 */
template <typename scalar_t, typename epsilon_t = cl::sycl::half>
inline typename std::enable_if<!std::is_same_v<scalar_t, cl::sycl::half>,
                               bool>::type
compare_vectors(std::vector<cl::sycl::half> const& vec,
                std::vector<scalar_t> const& ref,
                std::ostream& err_stream = std::cerr,
                std::string end_line = "\n") {
  if (vec.size() != ref.size()) {
    err_stream << "Error: tried to compare vectors of different sizes"
               << std::endl;
    return false;
  }

  for (int i = 0; i < vec.size(); ++i) {
    cl::sycl::half ref_i = cast_to_sycl_half(ref[i]);
    if (!almost_equal<cl::sycl::half, epsilon_t>(vec[i], ref_i)) {
      err_stream << "Value mismatch at index " << i << ": " << vec[i]
                 << "; expected " << ref_i << end_line;
      return false;
    }
  }
  return true;
}

template <typename scalar_t, typename epsilon_t = cl::sycl::half>
inline typename std::enable_if<!std::is_same_v<scalar_t, cl::sycl::half>,
                               bool>::type
compare_vectors_strided(std::vector<cl::sycl::half> const& vec,
                        std::vector<scalar_t> const& ref, int stride,
                        int window, std::ostream& err_stream = std::cerr,
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
      cl::sycl::half ref_i = cast_to_sycl_half(ref[index]);
      if (!almost_equal<cl::sycl::half, epsilon_t>(vec[index], ref_i)) {
        err_stream << "Value mismatch at index " << index << ": " << vec[index]
                   << "; expected " << ref_i << end_line;
        return false;
      }
    }
    k += 1;
  }

  return true;
}
#endif

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
                            std::string end_line = "\n") {
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
