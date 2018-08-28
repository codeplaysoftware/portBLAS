/* Copyright (c) 2015 The Khronos Group Inc.

   Permission is hereby granted, free of charge, to any person obtaining a
   copy of this software and/or associated documentation files (the
   "Materials"), to deal in the Materials without restriction, including
   without limitation the rights to use, copy, modify, merge, publish,
   distribute, sublicense, and/or sell copies of the Materials, and to
   permit persons to whom the Materials are furnished to do so, subject to
   the following conditions:

   The above copyright notice and this permission notice shall be included
   in all copies or substantial portions of the Materials.

   MODIFICATIONS TO THIS FILE MAY MEAN IT NO LONGER ACCURATELY REFLECTS
   KHRONOS STANDARDS. THE UNMODIFIED, NORMATIVE VERSIONS OF KHRONOS
   SPECIFICATIONS AND HEADER INFORMATION ARE LOCATED AT
    https://www.khronos.org/registry/

  THE MATERIALS ARE PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
  EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
  MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
  IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
  CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
  TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
  MATERIALS OR THE USE OR OTHER DEALINGS IN THE MATERIALS.
*/

// https://github.com/KhronosGroup/SyclParallelSTL/blob/master/benchmarks/benchmark.h

#ifndef BLAS_BENCHMARK_HPP
#define BLAS_BENCHMARK_HPP

#include <algorithm>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <string>
#include <unistd.h>
#include <utility>
#include <vector>

#include "range.hpp"

template <typename ScalarT>
ScalarT *new_data(size_t size, bool initialized = true) {
  ScalarT *v = new ScalarT[size];
  if (initialized) {
    for (size_t i = 0; i < size; ++i) {
      v[i] = 1e-3 * ((rand() % 2000) - 1000);
    }
  }
  return v;
}

template <typename ScalarT>
ScalarT *new_const_data(size_t size, ScalarT value = 0) {
  ScalarT *v = new ScalarT[size];
  for (size_t i = 0; i < size; ++i) {
    v[i] = value;
  }
  return v;
}

#define release_data(ptr) delete[](ptr);

template <typename ScalarT>
std::vector<ScalarT> random_data(size_t size, bool initialized = true) {
  auto default_initialiser = [](ScalarT x) -> ScalarT {
    return 1e-3 * ((rand() % 2000) - 1000);
  };
  std::vector<ScalarT> v = std::vector<ScalarT>(size);
  if (initialized) {
    std::transform(v.begin(), v.end(), v.begin(), default_initialiser);
  }
  return v;
}

template <typename ScalarT>
std::vector<ScalarT> const_data(size_t size, ScalarT const_value = 0) {
  std::vector<ScalarT> v = std::vector<ScalarT>(size);
  std::fill(v.begin(), v.end(), const_value);
  return v;
}

template <typename time_units_t_ = std::chrono::nanoseconds,
          typename ClockT = std::chrono::system_clock>
struct benchmark {
  using time_units_t = time_units_t_;

  template <typename F, typename... Args>
  static double measure(size_t numReps, size_t flops, F func, Args &&... args) {
    time_units_t dur = time_units_t::zero();

    // warm up to avoid benchmarking data transfer
    for (int i = 0; i < 5; ++i) {
      func(std::forward<Args>(args)...);
    }

    for (size_t reps = 0; reps < numReps; reps++) {
      auto start = ClockT::now();
      func(std::forward<Args>(args)...);
      auto end = ClockT::now();
      dur += end - start;
    }
    return (double(flops) * numReps) / (dur.count() * 1e-9);
  }

  static constexpr const size_t text_name_length = 30;
  static constexpr const size_t text_iterations_length = 15;
  static constexpr const size_t text_flops_length = 10;

  static std::string align_left(std::string &&text, size_t len,
                                size_t offset = 0) {
    return text + std::string((len < text.length() + offset)
                                  ? offset
                                  : len - text.length(),
                              ' ');
  }

  static void output_headers() {
    std::cout << align_left("Test", text_name_length)
              << align_left("Iterations", text_iterations_length)
              << align_left("Performance", text_flops_length) << std::endl;
  }

  static void output_data(const std::string &short_name, int size, int no_reps,
                          double flops) {
    std::cout << align_left(short_name + "_" + std::to_string(size),
                            text_name_length)
              << align_left(std::to_string(no_reps), text_iterations_length)
              << align_left(std::to_string(flops * 1e-6), text_flops_length, 1)
              << "MFlops" << std::endl;
  }
};

// why on earth do we need this??
#define BENCHMARK_FUNCTION(NAME) \
  template <class TypeParam>     \
  double NAME(size_t no_reps, size_t size)

/** BENCHMARK_MAIN.
 * The main entry point of a benchmark
 */
#define BENCHMARK_MAIN_BEGIN(RANGE_PARAM, REPS) \
  int main(int argc, char *argv[]) {            \
    benchmark<>::output_headers();              \
    auto _range = (RANGE_PARAM);                \
    const unsigned num_reps = (REPS);           \
    {
#define BENCHMARK_REGISTER_FUNCTION(NAME, FUNCTION)                \
  for (auto params = _range.yield(); !_range.finished();           \
       params = _range.yield()) {                                  \
    const std::string short_name = NAME;                           \
    auto flops = blasbenchmark.FUNCTION(num_reps, params);         \
    benchmark<>::output_data(short_name, params, num_reps, flops); \
  }
#define BENCHMARK_MAIN_END() \
  }                          \
  }

/** BENCHMARK_MAIN.
 * The main entry point of a benchmark
 */
#define BENCHMARK_MAIN(NAME, FUNCTION, RANGE_PARAM, REPS)                \
  int main(int argc, char *argv[]) {                                     \
    cl::sycl::queue q(cl::sycl::default_selector(),                      \
                      [=](cl::sycl::exception_list eL) {                 \
                        for (auto &e : eL) {                             \
                          try {                                          \
                            std::rethrow_exception(e);                   \
                          } catch (cl::sycl::exception & e) {            \
                            std::cout << " E " << e.what() << std::endl; \
                          } catch (...) {                                \
                            std::cout << " An exception " << std::endl;  \
                          }                                              \
                        }                                                \
                      }) Executor<SYCL>                                  \
        ex(q);                                                           \
    benchmark<>::output_headers();                                       \
    auto _range = (RANGE_PARAM);                                         \
    const unsigned num_reps = (REPS);                                    \
    for (auto size = _range.yield(); !_range.finished();                 \
         size = range.yield()) {                                         \
      const std::string short_name = NAME;                               \
      auto time = FUNCTION(ex, num_reps, params);                        \
      benchmark<>::output_data(short_name, params, num_reps, flops);     \
    }                                                                    \
  }

#endif /* end of include guard: BLAS_BENCHMARK_HPP */
