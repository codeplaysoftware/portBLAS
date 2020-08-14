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
    class blas::Executor<blas::PolicyHandler<blas::codeplay_policy>>;

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

#endif /* end of include guard: BLAS_TEST_HPP */
