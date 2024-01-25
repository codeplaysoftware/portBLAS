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
 *  portBLAS: BLAS implementation using SYCL
 *
 *  @filename utils.hpp
 *
 **************************************************************************/

#ifndef PORTBLAS_TOOLS_AUTO_TUNER_UTILS_HPP_
#define PORTBLAS_TOOLS_AUTO_TUNER_UTILS_HPP_

#include "tuner_types.hpp"

#include <chrono>
#include <functional>
#include <iostream>
#include <numeric>
#include <random>

inline cl::sycl::queue make_sycl_queue() {
  cl::sycl::queue q(
      [=](cl::sycl::exception_list ex_list) {
        try {
          for (auto &e_ptr : ex_list) {
            std::rethrow_exception(e_ptr);
          }
        } catch (cl::sycl::exception &e) {
          throw std::runtime_error(e.what());
        }
      },
      {cl::sycl::property::queue::in_order()});
  std::cout << "\nDevice: "
            << q.get_device().get_info<cl::sycl::info::device::name>()
            << std::endl;

  return q;
}

template <typename T, typename RndEngine>
HostContainer<T> get_random_vector(int size, T lo, T hi, RndEngine rnd) {
  std::uniform_real_distribution<T> dst(lo, hi);
  HostContainer<T> v(size);
  for (auto &e : v) {
    e = dst(rnd);
  }
  return v;
}

template <typename T>
T relative_diff(const HostContainer<T> &ref, const HostContainer<T> &obt) {
  using std::begin;
  using std::end;
  T mag(0);
  for (auto x : ref) {
    mag += x * x;
  }
  T diff =
      std::inner_product(begin(ref), end(ref), begin(obt), T(0), std::plus<T>(),
                         [](T x, T y) { return (x - y) * (x - y); });
  return std::sqrt(diff / mag);
}

template <typename TestOperator>
static void run_tune(int rep, double flop_cnt, TestResultEntry &result,
                     TestOperator op = TestOperator()) {
  using Seconds = std::chrono::duration<double>;
  using MilliSeconds = std::chrono::duration<double, std::milli>;
  Seconds runtime_secs;
  try {
    // warmup
    for (int i = 0; i < 10; ++i) {
      op();
    }
    auto start = std::chrono::steady_clock::now();
    for (int i = 0; i < rep; ++i) {
      op();
    }
    auto end = std::chrono::steady_clock::now();
    runtime_secs = end - start;
  } catch (std::exception const &e) {
    // If an error is detected when running a kernel, return without setting the
    // time in the result.
    std::cerr << "Error detected running " << result.name << "\n"
              << e.what() << "\n";
    return;
  }
  auto seconds_per_iter = runtime_secs / rep;
  auto milliseconds =
      std::chrono::duration_cast<MilliSeconds>(seconds_per_iter);
  result.sec = milliseconds.count();
  auto gigaflop_count = flop_cnt / 1e9;
  result.gflops = gigaflop_count / seconds_per_iter.count();
}

#endif  // PORTBLAS_TOOLS_AUTO_TUNER_UTILS_HPP_
