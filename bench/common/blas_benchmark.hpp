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

// Derived in part from:
// https://github.com/KhronosGroup/SyclParallelSTL/blob/master/benchmarks/benchmark.h

#ifndef BLAS_BENCHMARK_HPP
#define BLAS_BENCHMARK_HPP

#include <algorithm>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <regex>
#include <string>
#include <unistd.h>
#include <utility>
#include <vector>

#include <sstream>

#include "cli_benchmark_interface.hpp"
#include "cli_device_selector.hpp"
#include "clwrap.hpp"
#include "range.hpp"

/**
 * @fn type_string
 * @brief Generate a string describing the type T. Currently only supports
 * `float` and `double`.
 *
 * This function uses C++ template specialisation to dispatch the correct
 * variant of `type_string`. e.g. calling: `type_string<Float>()` will return
 * "float", while calling: `type_string<Int>()` will return "unknown"
 */
template <typename T>
const inline std::string& type_string() {
  static const std::string str = "unknown";
  return str;
}

template <>
const inline std::string& type_string<float>() {
  static const std::string str = "float";
  return str;
}

template <>
const inline std::string& type_string<double>() {
  static const std::string str = "double";
  return str;
}

template <typename TimeT = std::chrono::nanoseconds,
          typename ClockT = std::chrono::system_clock, typename FlopsT = double>
struct benchmark {
  typedef TimeT time_units_t;
  typedef FlopsT flops_units_t;

  template <typename ScalarT>
  static std::vector<ScalarT> random_data(size_t size,
                                          bool initialized = true) {
    auto default_initialiser = [](ScalarT x) -> ScalarT {
      // eeeeugh
      return 1e-3 * ((rand() % 2000) - 1000);
    };
    std::vector<ScalarT> v = std::vector<ScalarT>(size);
    if (initialized) {
      std::transform(v.begin(), v.end(), v.begin(), default_initialiser);
    }
    return v;
  }

  template <typename ScalarT>
  static std::vector<ScalarT> const_data(size_t size, ScalarT const_value = 0) {
    std::vector<ScalarT> v = std::vector<ScalarT>(size);
    std::fill(v.begin(), v.end(), const_value);
    return v;
  }

  /**
   * @fn random_scalar
   * @brief Generates a random scalar value, using an arbitrary low quality
   * algorithm.
   */
  template <typename ScalarT>
  static ScalarT random_scalar() {
    return 1e-3 * ((rand() % 2000) - 1000);
  }

  /**
   * @fn timef
   * @brief Calculates the time spent executing the function func
   */
  template <typename F, typename... Args>
  static TimeT timef(F func, Args&&... args) {
    auto start = ClockT::now();

    func(std::forward<Args>(args)...);

    return std::chrono::duration_cast<TimeT>(ClockT::now() - start);
  }

  /**
   * @fn    duration
   * @brief Returns the average number of flops across numReps runs of the
   * function F, which performs n_fl_ops floating point operations.
   */
  template <typename F, typename... Args>
  static FlopsT measure(size_t numReps, size_t n_fl_ops, F func,
                        Args&&... args) {
    TimeT dur = TimeT::zero();

    // warm up to avoid benchmarking data transfer
    for (int i = 0; i < 15; ++i) {
      func(std::forward<Args>(args)...);
    }

    for (size_t reps = 0; reps < numReps; reps++) {
      dur += benchmark<>::timef(func, std::forward<Args>(args)...);
    }

    // convert the time to flop/s based on the number of fl_ops that the
    // function performs
    // return (FlopsT(n_fl_ops) * numReps) /
    //  std::chrono::duration_cast<std::chrono::duration<double>>(dur)
    //  .count();
    // Just report the average time for now
    // std::cout
    //     << "duration : "
    //     << std::chrono::duration_cast<std::chrono::milliseconds>(dur).count()
    //     << std::endl;

    // std::cout << "numReps: " << numReps << std::endl;

    auto ttt =
        (double)(std::chrono::duration_cast<std::chrono::nanoseconds>(dur)
                     .count()) /
        (double)numReps;

    // std::cout << "ttt: " << ttt << std::endl;

    return ttt;
  }

  static constexpr const size_t text_name_length = 50;
  static constexpr const size_t text_iterations_length = 15;
  static constexpr const size_t text_flops_length = 10;

  /**
   * @fn typestr
   * @brief Print the type of a given type T. Currently only supports `float`
   * and `double`.
   */
  template <typename T>
  static std::string typestr() {
    return type_string<T>();
  }

  static std::string align_left(const std::string& text, size_t len,
                                size_t offset = 0) {
    return text + std::string((len < text.length() + offset)
                                  ? offset
                                  : len - text.length(),
                              ' ');
  }

  static void output_headers(output_type output = output_type::STDOUT) {
    if (output == output_type::STDOUT) {
      std::cerr << align_left("Benchmark", text_name_length)
                << align_left("Iterations", text_iterations_length)
                << align_left("Performance", text_flops_length) << std::endl;
    } else if (output == output_type::CSV) {
      std::cerr << "benchmark, "
                << "iterations, "
                << "performance (nanoseconds)" << std::endl;
    }
  }

  static void output_data(const std::string& name, int no_reps, double flops,
                          output_type output = output_type::STDOUT) {
    if (output == output_type::STDOUT) {
      std::cerr << align_left(name, text_name_length)
                << align_left(std::to_string(no_reps), text_iterations_length)
                << align_left(std::to_string(flops /* 1e-6*/),
                              text_flops_length, 1)
                << "nanoseconds" << std::endl;
    } else if (output == output_type::CSV) {
      std::cerr << name << ", " << std::to_string(no_reps) << ", "
                << std::to_string(flops /* 1e-6*/) << std::endl;
    } else {
      std::cerr << "Unknown output type!" << std::endl;
    }
  }
};

/** benchmark_instance
 * A given benchmark instance/type/function. This supertype defines the methods
 * that every benchmark instance needs to provice.
 */

template <typename T, typename ExecutorT, typename ParamT>
class benchmark_instance {
 public:
  virtual const std::string name() = 0;
  virtual const std::string format_name(ParamT params) = 0;
  virtual benchmark<>::flops_units_t run(ParamT params, unsigned int reps,
                                         ExecutorT ex) = 0;
};

/** BENCHMARK_NAME_FORMAT
 * Define how we want to print names for a given benchmark suite
 */

#define BENCHMARK_NAME_FORMAT(suite_name)                            \
  template <typename ElemT, typename ExecutorT, typename ParamT>     \
  class benchmark_##suite_name##_suite_class                         \
      : public benchmark_instance<ElemT, ExecutorT, ParamT> {        \
   public:                                                           \
    benchmark_##suite_name##_suite_class(){};                        \
    const std::string name() = 0;                                    \
    const std::string format_name(ParamT params);                    \
    const char* type() { return typeid(ElemT).name(); }              \
    benchmark<>::flops_units_t run(ParamT params, unsigned int reps, \
                                   ExecutorT ex) = 0;                \
  };                                                                 \
  template <typename ElemT, typename ExecutorT, typename ParamT>     \
  const std::string benchmark_##suite_name##_suite_class<            \
      ElemT, ExecutorT, ParamT>::format_name(ParamT params)

/** BENCHMARK
 * Declare a particular benchmark/instance.
 */
#define BENCHMARK(bench_name, suite_name)                             \
  template <typename ElemT, typename ExecutorT, typename ParamT>      \
  class benchmark_##bench_name##_class_                               \
      : public benchmark_##suite_name##_suite_class<ElemT, ExecutorT, \
                                                    ParamT> {         \
    const char* _name = #bench_name;                                  \
                                                                      \
   public:                                                            \
    benchmark_##bench_name##_class_(){};                              \
    const std::string name() { return std::string(_name); }           \
    const char* type() { return typeid(ElemT).name(); }               \
    benchmark<>::flops_units_t run(ParamT params, unsigned int reps,  \
                                   ExecutorT ex);                     \
  };                                                                  \
  template <typename ElemT, typename ExecutorT, typename ParamT>      \
  benchmark<>::flops_units_t                                          \
      benchmark_##bench_name##_class_<ElemT, ExecutorT, ParamT>::run( \
          ParamT params, unsigned int reps, ExecutorT ex)

/** ADD
 * Add a particular benchmark to a suite of benchmarks.
 */
#define ADD(name) (new benchmark_##name##_class_<ElemT, ExecutorT, ParamT>)

/** SUITE
 * Define a suite of benchmarks.
 */
#define SUITE(...)                                                     \
  template <typename ElemT, typename ExecutorT, typename ParamT>       \
  std::vector<benchmark_instance<ElemT, ExecutorT, ParamT>*>           \
  benchmark_suite() {                                                  \
    return std::vector<benchmark_instance<ElemT, ExecutorT, ParamT>*>( \
        {__VA_ARGS__});                                                \
  }

/** run_benchmark
 * Run a single benchmark instance, and iterate over the parameter range
 * generator to generate parameters for the instance.
 */
template <typename ElemT, typename Ex, typename ParamT>
void run_benchmark(benchmark_instance<ElemT, Ex, ParamT>* b,
                   Range<ParamT>* _range, const unsigned reps, Ex ex,
                   output_type output = output_type::STDOUT) {
  while (1) {
    auto params = _range->yield();
    auto flops = b->run(params, reps, ex);
    const std::string name = b->format_name(params);

    benchmark<>::output_data(name, reps, flops, output);

    if (_range->finished()) break;
  }
}

/** benchmark_suite
 * The definition (for a given suite) of the function that produces a list of
 * benchmarks. Forward declared so that we can use it in this header.
 */
template <typename ElemT, typename ExecutorT, typename ParamT>
std::vector<benchmark_instance<ElemT, ExecutorT, ParamT>*> benchmark_suite();

/** main_impl
 * The main implementaiton of the benchmark main. This is separated out so that
 * we can include #ifdef statements to check for double support.
 */
template <typename Ex, typename ParamT>
int main_impl(Range<ParamT>* range_param, const unsigned reps, Ex ex,
              output_type output = output_type::STDOUT) {
  benchmark<>::output_headers(output);
  auto fbenchmarks = benchmark_suite<float, Ex, ParamT>();
  for (auto b : fbenchmarks) {
    run_benchmark(b, range_param, reps, ex, output);
  }

#ifndef NO_DOUBLE_SUPPORT
  auto dbenchmarks = benchmark_suite<double, Ex, ParamT>();
  for (auto b : dbenchmarks) {
    run_benchmark(b, range_param, reps, ex, output);
  }
#endif

  return 0;
}

/** SYCL_BENCHMARK_MAIN.
 * The main entry point of a SYCL-BLAS benchmark
 */
#define SYCL_BENCHMARK_MAIN(RANGE_PARAM, REPS)                        \
  int main(int argc, char* argv[]) {                                  \
    benchmark_arguments ba(argc, argv);                               \
    if (!ba.validProgramOptions) {                                    \
      return 1;                                                       \
    }                                                                 \
    cli_device_selector cds(ba.device_vendor, ba.device_type);        \
    cl::sycl::queue q(cds);                                           \
    Executor<SYCL> ex(q);                                             \
    return main_impl((&RANGE_PARAM), (REPS), ex, ba.requestedOutput); \
  }

/** CLBLAST_BENCHMARK_MAIN
 * The main entry point of a CLBLAST benchmark
 */
#define CLBLAST_BENCHMARK_MAIN(RANGE_PARAM, REPS)                       \
  int main(int argc, char* argv[]) {                                    \
    benchmark_arguments ba(argc, argv);                                 \
    if (!ba.validProgramOptions) {                                      \
      return 1;                                                         \
    }                                                                   \
    OpenCLDeviceSelector oclds(ba.device_vendor, ba.device_type);       \
    Context ctx(oclds);                                                 \
    return main_impl((&RANGE_PARAM), (REPS), &ctx, ba.requestedOutput); \
  }

#endif /* end of include guard: BLAS_BENCHMARK_HPP */
