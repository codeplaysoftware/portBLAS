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
#include <vector>
#include <random>

#include <gtest/gtest.h>

#include <sycl_blas.h>

#include "blas_test_macros.hpp"
#include "utils/system_reference_blas.hpp"
#include "utils/cli_device_selector.hpp"
#include "utils/print_queue_information.hpp"
#include "utils/float_comparison.hpp"

struct Args {
  std::string program_name;
  std::string device;
};
extern Args args;

using namespace blas;

template <typename ClassName>
struct option_size;
namespace {
static const size_t RANDOM_SIZE = UINT_MAX;
static const size_t RANDOM_STRD = UINT_MAX;
}  // namespace
#define REGISTER_SIZE(size, test_name)          \
  template <>                                   \
  struct option_size<class test_name> {         \
    static constexpr const size_t value = size; \
  };
template <typename ClassName>
struct option_strd;
#define REGISTER_STRD(strd, test_name)          \
  template <>                                   \
  struct option_strd<class test_name> {         \
    static constexpr const size_t value = strd; \
  };
template <typename scalar_t, typename ClassName>
struct option_prec;
#define REGISTER_PREC(type, val, test_name)   \
  template <>                                 \
  struct option_prec<type, class test_name> { \
    static constexpr const type value = val;  \
  };

// Wraps the above arguments into one template parameter.
// We will treat template-specialized blas_templ_struct as a single class
template <class ScalarT_, class MetadataT_, class ExecutorType_>
struct blas_templ_struct {
  using scalar_t = ScalarT_;
  using metadata_t = MetadataT_;
  using executor_t = ExecutorType_;
};

// "Using" shortcuts for the struct, with #ifndef guard for double

// A "default" using shortcut
template <class ScalarT_, class MetadataT_ = void,
          class ExecutorType_ = PolicyHandler<codeplay_policy>>
using blas_test_args = blas_templ_struct<ScalarT_, MetadataT_, ExecutorType_>;

// specialisation for float
template <class MetadataT_ = void,
          class ExecutorType_ = PolicyHandler<codeplay_policy>>
using blas_test_float = blas_templ_struct<float, MetadataT_, ExecutorType_>;

// specialisation (with define guard) for double
#ifdef DOUBLE_SUPPORT
template <class MetadataT_ = void,
          class ExecutorType_ = PolicyHandler<codeplay_policy>>
using blas_test_double = blas_templ_struct<double, MetadataT_, ExecutorType_>;
#else
template <class MetadataT_ = void,
          class ExecutorType_ = PolicyHandler<codeplay_policy>>
using blas_test_double = ::testing::internal::None;
#endif

// the test class itself
template <class B>
class BLAS_Test;

template <class ScalarT_, class MetadataT_, class ExecutorType_>
class BLAS_Test<blas_test_args<ScalarT_, MetadataT_, ExecutorType_>>
    : public ::testing::Test {
 public:
  using scalar_t = ScalarT_;
  using MetadataT = MetadataT_;
  using ExecutorType = ExecutorType_;

  BLAS_Test() = default;
  virtual ~BLAS_Test() = default;

  virtual void SetUp() {}
  virtual void TearDown() {}

  // 3 binary places for value, 4 needed for precision
  static size_t rand_size() {
    // make sure the generated number is not too big for a type
    // i.e. we do not want the sample size to be too big because of
    // precision/memory restrictions
    int max_size = 18 + 3 * std::log2(sizeof(scalar_t) / sizeof(float));
    int max_rand = std::log2(RAND_MAX);
    return rand() >> (max_rand - max_size);
  }

  // it is important that all tests are run with the same test size
  // so each time we access this function within the same program, we get the
  // same
  // randomly generated size
  template <typename test>
  size_t test_size() {
    if (option_size<test>::value != ::RANDOM_SIZE) {
      return option_size<test>::value;
    }
    static bool first = true;
    static size_t N;
    if (first) {
      first = false;
      N = rand_size();
    }
    return N;
  }

  // randomly generates stride when run for the first time, and keeps returning
  // the same value consecutively
  template <typename test>
  size_t test_strd() {
    if (option_strd<test>::value != ::RANDOM_STRD) {
      return option_strd<test>::value;
    }
    static bool first = true;
    static size_t N;
    if (first) {
      first = false;
      N = ((rand() & 1) * (rand() % 5)) + 1;
    }
    return N;
  }

  template <typename test>
  scalar_t test_prec() {
    return option_prec<scalar_t, test>::value;
  }

  template <typename DataType, typename value_t = typename DataType::value_type>
  static void set_rand(DataType &vec, size_t _N) {
    value_t left(-1), right(1);
    for (size_t i = 0; i < _N; ++i) {
      vec[i] = value_t(rand() % int((right - left) * 8)) * 0.125 - right;
    }
  }

  template <typename DataType, typename value_t = typename DataType::value_type>
  static void print_cont(const DataType &vec, size_t _N,
                         std::string name = "vector") {
    std::cout << name << ": ";
    for (size_t i = 0; i < _N; ++i) std::cout << vec[i] << " ";
    std::cout << std::endl;
  }

  template <typename DeviceSelector,
            typename = typename std::enable_if<std::is_same<
                ExecutorType, PolicyHandler<codeplay_policy>>::value>::type>
  static cl::sycl::queue make_queue(DeviceSelector s) {
    return cl::sycl::queue(s, [=](cl::sycl::exception_list eL) {
      for (auto &e : eL) {
        try {
          std::rethrow_exception(e);
        } catch (cl::sycl::exception &e) {
          std::cout << "E " << e.what() << std::endl;
        } catch (std::exception &e) {
          std::cout << "Standard Exception " << e.what() << std::endl;
        } catch (...) {
          std::cout << " An exception " << std::endl;
        }
      }
    });
  }
};

inline cl::sycl::queue make_queue() {
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
 * @fn random_data
 * @brief Generates a random vector of scalar values, using a uniform
 * distribution.
 */
template <typename scalar_t>
static inline void fill_random(std::vector<scalar_t> &vec) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<> dis(-1.0, 1.0);
  for (scalar_t &e : vec) {
    e = dis(gen);
  }
}

#endif /* end of include guard: BLAS_TEST_HPP */
