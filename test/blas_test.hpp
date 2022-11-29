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
 *  @filename blas_test.hpp
 *
 **************************************************************************/

#ifndef BLAS_TEST_HPP
#define BLAS_TEST_HPP

#include <climits>
#include <cmath>
#include <complex>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>

#include <gtest/gtest.h>

#include <sycl_blas.h>

#include <common/cli_device_selector.hpp>
#include <common/float_comparison.hpp>
#include <common/print_queue_information.hpp>
#include <common/system_reference_blas.hpp>

#include "blas_test_macros.hpp"

struct Args {
  std::string program_name;
  std::string device;
};
extern Args args;

using namespace blas;

// The executor type used in tests
using test_executor_t =
    blas::Executor<blas::PolicyHandler<blas::codeplay_policy>>;


using index_t = BLAS_INDEX_T;

/**
 * Construct a SYCL queue using the device specified in the command line, or
 * using the default device if not specified.
 */
inline cl::sycl::queue make_queue_impl() {
  auto async_handler = [=](cl::sycl::exception_list eL) {
    for (auto &e : eL) {
      try {
        std::rethrow_exception(e);
      } catch (cl::sycl::exception &e) {
        std::cout << "Sycl Exception " << e.what() << std::endl;
      } catch (std::exception &e) {
        std::cout << "Standard Exception " << e.what() << std::endl;
      } catch (...) {
        std::cout << "An exception " << std::endl;
      }
    }
  };

#if SYCL_LANGUAGE_VERSION >= 202002
  std::function<int(const cl::sycl::device&)> selector;
  if (args.device.empty()) {
    selector = cl::sycl::default_selector_v;
  } else {
    selector = utils::cli_device_selector(args.device);
  }
  auto q = cl::sycl::queue(selector, async_handler);
#else
  std::unique_ptr<cl::sycl::device_selector> selector;
  if (args.device.empty()) {
    selector = std::unique_ptr<cl::sycl::device_selector>(
        new cl::sycl::default_selector());
  } else {
    selector = std::unique_ptr<cl::sycl::device_selector>(
        new utils::cli_device_selector(args.device));
  }
  auto q = cl::sycl::queue(*selector, async_handler);
#endif  // HAS_SYCL2020_SELECTORS

  utils::print_queue_information(q);
  return q;
};

/**
 * Get a SYCL queue to use in tests.
 */
inline cl::sycl::queue make_queue() {
  // Provide cached SYCL queue, to avoid recompiling kernels for each test case.
  static cl::sycl::queue queue = make_queue_impl();
  return queue;
}

/**
 * @brief Generates a random scalar in the specified range
 * @param rangeMin range minimum
 * @param rangeMax range maximum
 */
template <typename scalar_t>
static inline scalar_t random_scalar(scalar_t rangeMin, scalar_t rangeMax) {
  static std::random_device rd;
  static std::default_random_engine gen(rd());
  std::uniform_real_distribution<scalar_t> dis(rangeMin, rangeMax);
  return dis(gen);
}

/**
 * @brief Generates a random vector of scalar values, using a uniform
 * distribution.
 * @param vec Input vector to fill
 * @param rangeMin Minimum value for the uniform distribution
 * @param rangeMax Maximum value for the uniform distribution
 */
template <typename scalar_t>
static inline void fill_random_with_range(std::vector<scalar_t> &vec,
                                          scalar_t rangeMin,
                                          scalar_t rangeMax) {
  for (scalar_t &e : vec) {
    e = random_scalar(rangeMin, rangeMax);
  }
}

/**
 * @brief Generates a random vector of scalar values, using a uniform
 * distribution.
 */
template <typename scalar_t>
static inline void fill_random(std::vector<scalar_t> &vec) {
  fill_random_with_range(vec, scalar_t{-2}, scalar_t{5});
}

/**
 * @brief Fills a lower or upper triangular matrix suitable for TRSM testing
 * @param A The matrix to fill. Size must be at least m * lda
 * @param m The number of rows of matrix @p A
 * @param n The number of columns of matrix @p A
 * @param lda The leading dimension of matrix @p A
 * @param uplo if 'u', @p A will be upper triangular. If 'l' @p A will be
 * lower triangular
 * @param diag Value to put in the diagonal elements
 * @param unused Value to put in the unused parts of the matrix
 */
template <typename scalar_t>
static inline void fill_trsm_matrix(std::vector<scalar_t> &A, size_t k,
                                    size_t lda, char uplo,
                                    scalar_t diag = scalar_t{1},
                                    scalar_t unused = scalar_t{0}) {
  for (size_t i = 0; i < k; ++i) {
    scalar_t sum = std::abs(diag);
    for (size_t j = 0; j < k; ++j) {
      scalar_t value = scalar_t{0};
      if (i == j) {
        value = diag;
      } else if (((uplo == 'l') && (i > j)) ||
                 ((uplo == 'u') && (i < j))) {
        if (sum >= scalar_t{1}) {
          const double limit =
              sum / std::sqrt(static_cast<double>(k) - static_cast<double>(j));
          value = random_scalar(scalar_t{-1}, scalar_t{1}) * limit;
          sum -= std::abs(value);
        }
      } else {
        value = unused;
      }
      A[i + j * lda] = value;
    }
  }
}

/**
 * @brief Helper class for dumping arguments to a stream, in a format compatible
 * with google test test names.
 *
 * @tparam T is the argument type to dump to the stream.
 * @tparam Enable is a helper for partial template specialization.
 */
template <class T, typename Enable = void>
struct dump_arg_helper {
  /** Dump the argument to the stream.
   *
   * @param ss Output stream
   * @param arg Argument to format
   **/
  inline void operator()(std::ostream &ss, T arg) { ss << arg; }
};

/** Specialization of dump_arg_helper for float and double. NB this is not a
 *  specialization for half. std::is_floating_point<cl::sycl::half>::value will
 *  return false.
 *
 *  @tparam StdFloat A standard floating point type.
 **/
template <class StdFloat>
struct dump_arg_helper<
    StdFloat,
    typename std::enable_if<std::is_floating_point<StdFloat>::value>::type> {
  /**
   * @brief Dump an argument to a stream.
   * Format floating point numbers for GTest. A test name cannot contain
   * "-" nor "." so they are replaced with "m" and "p" respectively. The
   * fractional part is ignored if null otherwise it is printed with 2 digits.
   *
   * @param ss Output stream
   * @param f Floating point number to format
   */
  inline void operator()(std::ostream &ss, StdFloat f) {
    static_assert(!std::is_same<StdFloat, cl::sycl::half>::value,
                  "std library functions will not work with half.");
    if (std::isnan(f)) {
      ss << "nan";
      return;
    }
    if (f < 0) {
      ss << "m";
      f = std::fabs(f);
    }
    StdFloat int_part;
    StdFloat frac_part = std::modf(f, &int_part);
    ss << std::fixed  << std::setprecision(0) << int_part;

    if (frac_part > 0) {
      ss << "p" << (int)(frac_part * 100);
    }
  }
};

/** Specialization of dump_arg_helper for cl::sycl::half.
 *  This is required since half will not work with standard library functions.
 **/
template <>
struct dump_arg_helper<cl::sycl::half> {
  inline void operator()(std::ostream &ss, cl::sycl::half f) {
    dump_arg_helper<float>{}(ss, f);
  }
};

/**
 * Return type of the tested api (either asynchronous (event) or
 * synchronous(result))
 */
enum class api_type : int { event = 0, result = 1 };

template <>
struct dump_arg_helper<api_type> {
  inline void operator()(std::ostream &ss, const api_type &type) {
    if (type == api_type::event) {
      ss << "event";
    } else {
      ss << "result";
    }
  }
};

/**
 * @brief Dump an argument to a stream.
 *
 * @tparam T is the type of the argument to format.
 * @param ss Output stream
 * @param arg Argument to format
 */
template <class T>
inline void dump_arg(std::ostream &ss, T arg) {
  dump_arg_helper<T>{}(ss, arg);
}

/**
 * @brief End of the recursion.
 */
inline void generate_name_helper(std::ostream &) {}

/**
 * @brief Recursively dump the list of arguments in the form
 * (__<ArgName>_<ArgValue>)*
 *
 * @param ss Output stream
 * @param arg Argument to dump
 * @param args Remaining arguments
 */
template <class T, class... Args>
inline void generate_name_helper(std::ostream &ss, T arg, Args... args) {
  auto token = strtok(nullptr, ", ");
  ss << "__" << token << "_";
  dump_arg(ss, arg);
  generate_name_helper(ss, args...);
}

/**
 * @brief Return a string of the list of arguments compatible with GTest
 * in the form <ArgName>_<ArgValue>(__<ArgName>_<ArgValue>)*
 *
 * @param str_args Writeable null-terminated string of the form
 *                 <ArgName>(, <ArgName>)*
 * @param arg First argument to dump
 * @param args List of remaining arguments to dump
 */
template <class T, class... Args>
inline std::string generate_name_helper(char *str_args, T arg, Args... args) {
  std::stringstream ss;
  auto token = strtok(str_args, ", ");
  ss << token << "_";
  dump_arg(ss, arg);
  generate_name_helper(ss, args...);
  return ss.str();
}

// Helper macro to generate tests name from a list of test parameters.
// The list of arguments is stored as a writeable null-terminated string for
// strtok.
#define BLAS_GENERATE_NAME(tuple, ...)                  \
  do {                                                  \
    char str_args[] = #__VA_ARGS__;                     \
    std::tie(__VA_ARGS__) = tuple;                      \
    return generate_name_helper(str_args, __VA_ARGS__); \
  } while (0)

#endif /* end of include guard: BLAS_TEST_HPP */
