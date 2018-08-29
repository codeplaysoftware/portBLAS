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

#include "cli_device_selector.hpp"
#include "range.hpp"

template <typename ScalarT>
std::vector<ScalarT> random_data(size_t size, bool initialized = true) {
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
std::vector<ScalarT> const_data(size_t size, ScalarT const_value = 0) {
  std::vector<ScalarT> v = std::vector<ScalarT>(size);
  std::fill(v.begin(), v.end(), const_value);
  return v;
}

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

template <typename T, typename ExecutorT, typename ParamT>
class benchmark_instance {
 public:
  virtual const char* name() = 0;
  virtual double run(ParamT params, unsigned int reps, ExecutorT ex) = 0;
};

template <typename T, typename ExecutorT, typename ParamT>
class empty : public benchmark_instance<T, ExecutorT, ParamT> {
  const char* _name = "null";

 public:
  empty() {}
  const char* name() { return _name; }
  double run(ParamT params, unsigned int reps, ExecutorT ex) { return 0.0; }
};

#define BENCHMARK(bench_name)                                            \
  template <typename ElemT, typename ExecutorT, typename ParamT>         \
  class benchmark_##bench_name##_class_                                  \
      : public benchmark_instance<ElemT, ExecutorT, ParamT> {            \
    const char* _name = #bench_name;                                     \
                                                                         \
   public:                                                               \
    benchmark_##bench_name##_class_(){};                                 \
    const char* name() { return _name; };                                \
    const char* type() { return typeid(ElemT).name(); }                  \
    double run(ParamT params, unsigned int reps, ExecutorT ex);          \
  };                                                                     \
  template <typename ElemT, typename ExecutorT, typename ParamT>         \
  double benchmark_##bench_name##_class_<ElemT, ExecutorT, ParamT>::run( \
      ParamT params, unsigned int reps, ExecutorT ex)

#define ADD(name) (new benchmark_##name##_class_<ElemT, ExecutorT, ParamT>),

#define SUITE(List)                                                    \
  template <typename ElemT, typename ExecutorT, typename ParamT>       \
  std::vector<benchmark_instance<ElemT, ExecutorT, ParamT>*>           \
  benchmark_suite() {                                                  \
    return std::vector<benchmark_instance<ElemT, ExecutorT, ParamT>*>( \
        {List new empty<ElemT, ExecutorT, ParamT>});                   \
  }

template <typename TimeT = std::chrono::microseconds,
          typename ClockT = std::chrono::system_clock>
struct benchmark {
  // typedef TimeT time_units_t;

  template <typename F, typename... Args>
  static double measure(size_t numReps, size_t flops, F func, Args&&... args) {
    TimeT dur = TimeT::zero();

    // warm up to avoid benchmarking data transfer
    for (int i = 0; i < 5; ++i) {
      func(std::forward<Args>(args)...);
    }

    for (size_t reps = 0; reps < numReps; reps++) {
      auto start = ClockT::now();
      func(std::forward<Args>(args)...);
      auto end = ClockT::now();
      dur += std::chrono::duration_cast<TimeT>(end - start);
    }
    return (double(flops) * numReps) / (dur.count() * 1e-9);
  }

  /**
   * @fn    duration
   * @brief Returns the duration (in chrono's type system) of the elapsed time
   */
  // template <typename F, typename... Args>
  // static TimeT duration(unsigned numReps, F func, Args&&... args) {
  //   TimeT dur = TimeT::zero();

  //   // warm up to avoid benchmarking data transfer
  //   for (int i = 0; i < 5; ++i) {
  //     func(std::forward<Args>(args)...);
  //   }

  //   unsigned reps = 0;
  //   for (; reps < numReps; reps++) {
  //     auto start = ClockT::now();

  //     func(std::forward<Args>(args)...);

  //     dur += std::chrono::duration_cast<TimeT>(ClockT::now() - start);
  //   }
  //   return dur / reps;
  // }

  /* output_data.
   * Prints to the stderr Bench name, input size and execution time.
   */
  static void output_data(const std::string& short_name, int num_elems,
                          TimeT dur, output_type output = output_type::STDOUT) {
    if (output == output_type::STDOUT) {
      std::cerr << short_name << "  " << num_elems << " " << dur.count()
                << std::endl;
    } else if (output == output_type::CSV) {
      std::cerr << short_name << "," << num_elems << "," << dur.count()
                << std::endl;
    } else {
      std::cerr << " Incorrect output " << std::endl;
    }
  }
};

template <typename ElemT, typename Ex, typename ParamT>
void run_benchmark(benchmark_instance<ElemT, Ex, ParamT>* b,
                   Range<ParamT>* _range, const unsigned reps, Ex ex,
                   output_type output = output_type::STDOUT) {
  for (auto params = _range->yield(); !_range->finished();
       params = _range->yield()) {
    const std::string short_name = std::string(b->name());
    auto time = b->run(params, reps, ex);
    // benchmark<>::output_data(short_name, 3, time);
  }
}

// forward declare benchmark_suite
template <typename ElemT, typename ExecutorT, typename ParamT>
std::vector<benchmark_instance<ElemT, ExecutorT, ParamT>*> benchmark_suite();
/** BENCHMARK_MAIN.
 * The main entry point of a benchmark
 */

template <typename Ex, typename ParamT>
int main_impl(Range<ParamT>* range_param, const unsigned reps, Ex ex,
              output_type output = output_type::STDOUT) {
  auto benchmarks = benchmark_suite<float, Ex, ParamT>();
  for (auto b : benchmarks) {
    run_benchmark(b, range_param, reps, ex, output);
  }
  return 0;
}

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
