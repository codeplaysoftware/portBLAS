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

#ifndef BLAS_BENCHMARK2_HPP
#define BLAS_BENCHMARK2_HPP

#include <algorithm>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <regex>
#include <string>
#include <unistd.h>
#include <utility>
#include <vector>

#include <algorithm>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <string>
#include <unistd.h>
#include <utility>
#include <vector>

#include <sstream>

#include "cli_device_selector.hpp"
#include "range.hpp"

/**
 * output_type
 */
enum class output_type {
  STDOUT,  // Dumps output to standard output
  CSV      // Dumps output to standard output but separate fields with semicolon
};

struct benchmark_arguments {
  std::string program_name;
  output_type requestedOutput;
  std::string device_vendor;
  std::string device_type;
  bool validProgramOptions;

  void usage() {
    std::cout << " Usage: " << program_name << " [--output OUTPUT]"
              << std::endl;
    std::cout << "  --output  OUTPUT" << std::endl;
    std::cout << "        Changes the output of the benchmark, with OUTPUT: "
              << std::endl;
    std::cout << "         - CSV : Output to a CSV file " << std::endl;
    std::cout << "         - STDOUT: Output to stdout (default) " << std::endl;
    std::cout << "  --device  DEVICE" << std::endl;
    std::cout
        << "         Select a device (best effort) for running the benchmark."
        << std::endl;
    std::cout << "         e.g. intel:cpu, amd:gpu etc" << std::endl;
  }

  benchmark_arguments(int argc, char** argv)
      : program_name(argv[0]),
        requestedOutput(output_type::STDOUT),
        validProgramOptions(true) {
    /* Match parameters */
    std::regex output_regex("--output");
    std::regex device_regex("--device");
    /* Check if user has specified any options */
    bool match = true;
    for (int i = 1; i < argc; i++) {
      bool matchedAnything = false;
      std::string option(argv[i]);
      if (option.size() == 0) {
        std::cerr << " Incorrect parameter " << std::endl;
        match = false;
        break;
      }
      // Check for the --output parameter
      if (std::regex_match(option, output_regex)) {
        if ((i + 1) >= argc) {
          std::cerr << " Incorrect parameter " << std::endl;
          match = false;
          break;
        }
        std::string outputOption = argv[i + 1];
        std::transform(outputOption.begin(), outputOption.end(),
                       outputOption.begin(), ::tolower);
        if (outputOption == "stdout") {
          requestedOutput = output_type::STDOUT;
          matchedAnything = true;
        } else if (outputOption == "csv") {
          requestedOutput = output_type::CSV;
          matchedAnything = true;
        } else {
          match = false;
          break;
        }
        // Skip next parameter, since it was the name
        i++;
      }

      // Check for the --device parameter
      if (std::regex_match(option, device_regex)) {
        if ((i + 1) >= argc) {
          std::cerr << " Incorrect parameter " << std::endl;
          match = false;
          break;
        }
        std::string outputOption = argv[i + 1];
        std::transform(outputOption.begin(), outputOption.end(),
                       outputOption.begin(), ::tolower);
        // split the string into tokens on ':'
        std::stringstream ss(outputOption);
        std::string item;
        std::vector<std::string> tokens;
        while (std::getline(ss, item, ':')) {
          tokens.push_back(item);
        }
        if (tokens.size() != 2) {
          std::cerr << " Incorrect number of arguments to device selector "
                    << std::endl;
        } else {
          device_vendor = tokens[0];
          device_type = tokens[1];
          matchedAnything = true;
        }
        // Skip next parameter, since it was the device
        i++;
      }

      // This option is not valid
      if (!matchedAnything) {
        match = false;
        break;
      }
    }

    if (!match) {
      usage();
      validProgramOptions = false;
    }
  }
};

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
   * @fn    duration
   * @brief Returns the average number of flops across numReps runs of the
   * function F, which performs n_fl_ops floating point operations.
   */
  template <typename F, typename... Args>
  static FlopsT measure(size_t numReps, size_t n_fl_ops, F func,
                        Args&&... args) {
    TimeT dur = TimeT::zero();

    // warm up to avoid benchmarking data transfer
    for (int i = 0; i < 5; ++i) {
      func(std::forward<Args>(args)...);
    }

    for (size_t reps = 0; reps < numReps; reps++) {
      auto start = ClockT::now();

      func(std::forward<Args>(args)...);

      dur += std::chrono::duration_cast<TimeT>(ClockT::now() - start);
    }

    // convert the time to flop/s based on the number of fl_ops that the
    // function performs
    return (FlopsT(n_fl_ops) * numReps) /
           std::chrono::duration_cast<std::chrono::duration<double>>(dur)
               .count();
  }

  static constexpr const size_t text_name_length = 50;
  static constexpr const size_t text_iterations_length = 15;
  static constexpr const size_t text_flops_length = 10;

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
                << "performance (mflop/s)" << std::endl;
    }
  }

  static void output_data(const std::string& name, int no_reps, double flops,
                          output_type output = output_type::STDOUT) {
    if (output == output_type::STDOUT) {
      std::cerr << align_left(name, text_name_length)
                << align_left(std::to_string(no_reps), text_iterations_length)
                << align_left(std::to_string(flops * 1e-6), text_flops_length,
                              1)
                << "MFlops" << std::endl;
    } else if (output == output_type::CSV) {
      std::cerr << name << ", " << std::to_string(no_reps) << ", "
                << std::to_string(flops * 1e-6) << std::endl;
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
  benchmark<>::output_headers(output);
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
  auto fbenchmarks = benchmark_suite<float, Ex, ParamT>();
  for (auto b : fbenchmarks) {
    run_benchmark(b, range_param, reps, ex, output);
  }

#ifndef NO_DOUBLE_SUPPORT
  auto dbenchmarks = benchmark_suite<float, Ex, ParamT>();
  for (auto b : dbenchmarks) {
    run_benchmark(b, range_param, reps, ex, output);
  }
#endif

  return 0;
}

/** BENCHMARK_MAIN.
 * The main entry point of a benchmark
 */
#define BENCHMARK_MAIN(RANGE_PARAM, REPS)                             \
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

#endif /* end of include guard: BLAS_BENCHMARK_HPP */
