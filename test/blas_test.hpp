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
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>

#include <gtest/gtest.h>

#include <sycl_blas.h>

#include "blas_test_macros.hpp"
#include "utils/cli_device_selector.hpp"
#include "utils/float_comparison.hpp"
#include "utils/print_queue_information.hpp"
#include "utils/quantization.hpp"
#include "utils/system_reference_blas.hpp"

struct Args {
  std::string program_name;
  std::string device;
};
extern Args args;

using namespace blas;

// The executor type used in tests
using test_executor_t =
    blas::Executor<blas::PolicyHandler<blas::codeplay_policy>>;

/**
 * Construct a SYCL queue using the device specified in the command line, or
 * using the default device if not specified.
 */
inline cl::sycl::queue make_queue_impl() {
  std::unique_ptr<cl::sycl::device_selector> selector;
  if (args.device.empty()) {
    selector = std::unique_ptr<cl::sycl::device_selector>(
        new cl::sycl::default_selector());
  } else {
    selector = std::unique_ptr<cl::sycl::device_selector>(
        new utils::cli_device_selector(args.device));
  }

  auto q = cl::sycl::queue(*selector, [=](cl::sycl::exception_list eL) {
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
  });

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
  std::random_device rd;
  std::default_random_engine gen(rd());
  std::uniform_real_distribution<scalar_t> dis(rangeMin, rangeMax);
  return dis(gen);
}

/**
 * @fn random_data
 * @brief Generates a random vector of scalar values, using a uniform
 * distribution.
 */
template <typename scalar_t>
static inline void fill_random(std::vector<scalar_t> &vec) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<> dis(-2.0, 5.0);
  for (scalar_t &e : vec) {
    e = dis(gen);
  }
}

/**
 * @breif Fills a lower or upper triangular matrix suitable for TRSM testing
 * @param A The matrix to fill. Size must be at least m * lda
 * @param m The number of rows of matrix @p A
 * @param n The number of columns of matrix @p A
 * @param lda The leading dimension of matrix @p A
 * @param triangle if 'u', @p A will be upper triangular. If 'l' @p A will be
 * lower triangular
 * @param diagonal Value to put in the diagonal elements
 * @param unused Value to put in the unused parts of the matrix
 */
template <typename scalar_t>
static inline void fill_trsm_matrix(std::vector<scalar_t> &A, size_t k,
                                    size_t lda, char triangle,
                                    scalar_t diagonal = scalar_t{1},
                                    scalar_t unused = scalar_t{0}) {
  std::random_device rd;
  std::default_random_engine gen(rd());
  std::uniform_real_distribution<scalar_t> dis(scalar_t{-1}, scalar_t{1});

  for (size_t i = 0; i < k; ++i) {
    scalar_t sum = std::abs(diagonal);
    for (size_t j = 0; j < k; ++j) {
      scalar_t value = scalar_t{0};
      if (i == j) {
        value = diagonal;
      } else if (((triangle == 'l') && (i > j)) ||
                 ((triangle == 'u') && (i < j))) {
        if (sum >= scalar_t{1}) {
          const double limit =
              sum / std::sqrt(static_cast<double>(k) - static_cast<double>(j));
          value = dis(rd) * limit;
          sum -= std::abs(value);
        }
      } else {
        value = unused;
      }
      A[i + j * lda] = value;
    }
  }
}

#endif /* end of include guard: BLAS_TEST_HPP */
