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

/**
 * @fn time_event
 * @brief inspects an event to calculate how long it took (start/end times).
 * Overloaded to support sycl and opencl.
 */
inline cl_ulong time_event(cl::sycl::event e) {
  // get start and ed times
  cl_ulong start_time = e.template get_profiling_info<
      cl::sycl::info::event_profiling::command_start>();

  cl_ulong end_time = e.template get_profiling_info<
      cl::sycl::info::event_profiling::command_end>();

  // return the delta
  return (end_time - start_time);
}

inline cl_ulong time_event(Event e) {
  // get start and end times
  cl_ulong start_time, end_time;
  bool all_ok = true;
  // declare a lambda to check the result of the calls
  auto check_call = [&](cl_int status) {
    switch (status) {
      case CL_SUCCESS:
        return;
        break;
      case CL_PROFILING_INFO_NOT_AVAILABLE:
        std::cerr << "The opencl queue has not been configured with profiling "
                     "information! "
                  << std::endl;
        break;
      case CL_INVALID_VALUE:
        std::cerr << "param_name is not valid, or size of param is < "
                     "param_value_size in profiling call!"
                  << std::endl;
        break;
      case CL_INVALID_EVENT:
        std::cerr << "event is invalid in profiling call " << std::endl;
        break;
      case CL_OUT_OF_RESOURCES:
        std::cerr << "cl_out_of_resources in profiling call" << std::endl;
        break;
      case CL_OUT_OF_HOST_MEMORY:
        std::cerr << "cl_out_of_host_memory in profiling call" << std::endl;
        break;
      default:
        // If we've reached this point, something's gone wrong - set the error
        // flag
        all_ok = false;
        break;
    }
  };
  check_call(clGetEventProfilingInfo(e._cl(), CL_PROFILING_COMMAND_START,
                                     sizeof(cl_ulong), &start_time, NULL));
  check_call(clGetEventProfilingInfo(e._cl(), CL_PROFILING_COMMAND_START,
                                     sizeof(cl_ulong), &end_time, NULL));
  e.release();
  // return the delta
  if (all_ok) {
    return (end_time - start_time);
  } else {
    // Return a really big number to show that we've failed.
    return 0xFFFFFFFFFFFFFFFF;
  }
}

/**
 * @fn time_events
 * @brief Times n events, and returns the aggregate time.
 */
template <typename EventT>
inline cl_ulong time_event(std::vector<EventT> es) {
  cl_ulong total_time = 0;
  for (auto e : es) {
    total_time += time_event(e);
  }
  return total_time;
}

/**
 * @struct datapoint
 * @brief Represents a datapoint for a given benchmark/parameter
 * combination.
 */

struct datapoint {
  typedef std::chrono::duration<double, std::nano> OverallTimeT;
  typedef std::chrono::duration<double, std::nano> KernelTimeT;
  typedef size_t FlopT;

  OverallTimeT _mean_overall_time;
  KernelTimeT _mean_kernel_time;
  FlopT _n_fl_ops;

  datapoint(OverallTimeT mean_exec_time, KernelTimeT mean_kernel_time,
            FlopT n_fl_ops)
      : _mean_overall_time(mean_exec_time),
        _mean_kernel_time(mean_kernel_time),
        _n_fl_ops(n_fl_ops) {}

  // Constant number of columns that we want to output.
  static constexpr const size_t columns = 7;

  static inline std::string align_left(const std::string& text, size_t len,
                                       size_t offset = 0) {
    return text + std::string((len < text.length() + offset)
                                  ? offset
                                  : len - text.length(),
                              ' ');
  }

  static inline void output_array(std::array<std::string, columns> values,
                                  output_type output = output_type::STDOUT) {
    static constexpr const size_t column_widths[columns] = {50, 20, 20, 20,
                                                            20, 20, 20};
    for (int i = 0; i < columns; i++) {
      if (output == output_type::STDOUT) {
        // If we're just printing columns, align them
        std::cerr << align_left(values[i], column_widths[i]);
      } else if (output == output_type::CSV) {
        // Otherwise, output the value, optionally prepended by a comma
        if (i != 0) std::cerr << ", ";
        std::cerr << values[i];
      } else {
        std::cerr << "Unknown output type!" << std::endl;
      }
    }
    std::cerr << std::endl;
  }

  static void output_headers(output_type output = output_type::STDOUT) {
    static const std::array<std::string, columns> column_names = {
        "Benchmark",         "Iterations",          "FP Ops",
        "Overall time (ns)", "Overall perf (gf/s)", "Kernel time (ns)",
        "Kernel perf (gf/s)"};
    output_array(column_names, output);
  }

  void output_data(const std::string& name, int no_reps,
                   output_type output = output_type::STDOUT) {
    // Calculate GFlop/s overall, and for the kernel.
    // As it turns out, Flop/Ns (i.e. floating point operations per nanosecond)
    // is the same as GFlops/S (i.e. billion floating point operations per
    // second), so we don't need to do any duration casts between time
    // intervals, or between flops and gigaflops - we can just compute Flop/Ns
    // directly.
    auto _mean_overall_flops =
        static_cast<double>(_n_fl_ops) / _mean_overall_time.count();
    auto _mean_kernel_flops =
        static_cast<double>(_n_fl_ops) / _mean_kernel_time.count();

    std::array<std::string, columns> data = {
        name,
        std::to_string(no_reps),
        std::to_string(_n_fl_ops),
        std::to_string(_mean_overall_time.count()),
        std::to_string(_mean_overall_flops),
        std::to_string(_mean_kernel_time.count()),
        std::to_string(_mean_kernel_flops)};
    output_array(data, output);
  }
};

/**
 * @struct benchmark
 * @brief Utility methods and orchestration for a benchmark suite
 */
template <typename TimeT = std::chrono::duration<double, std::nano>,
          typename ClockT = std::chrono::system_clock, typename FlopsT = double>
struct benchmark {
  typedef TimeT time_units_t;
  typedef FlopsT flops_units_t;
  typedef datapoint datapoint_t;

  /**
   * @fn random_scalar
   * @brief Generates a random scalar value, using an arbitrary low quality
   * algorithm.
   */
  template <typename ScalarT>
  static inline ScalarT random_scalar() {
    return 1e-3 * ((rand() % 2000) - 1000);
  }

  /**
   * @fn random_data
   * @brief Generates a random vector of scalar values, using an arbitrary low
   * quality algorithm.
   */
  template <typename ScalarT>
  static inline std::vector<ScalarT> random_data(size_t size,
                                                 bool initialized = true) {
    std::vector<ScalarT> v = std::vector<ScalarT>(size);
    if (initialized) {
      std::transform(v.begin(), v.end(), v.begin(), [](ScalarT x) -> ScalarT {
        return random_scalar<ScalarT>();
      });
    }
    return v;
  }

  /**
   * @fn const_data
   * @brief Generates a vector of constant values, of a given length.
   */
  template <typename ScalarT>
  static inline std::vector<ScalarT> const_data(size_t size,
                                                ScalarT const_value = 0) {
    std::vector<ScalarT> v = std::vector<ScalarT>(size);
    std::fill(v.begin(), v.end(), const_value);
    return v;
  }

  /**
   * @fn timef
   * @brief Calculates the time spent executing the function func
   */
  template <typename F, typename... Args>
  static std::tuple<TimeT, TimeT> timef(F func, Args&&... args) {
    auto start = ClockT::now();

    auto event = func(std::forward<Args>(args)...);

    auto end = ClockT::now();

    auto overall_time = end - start;

    // Cast from a clulong to a double based time interval
    TimeT event_time = static_cast<TimeT>(time_event(event));

    return std::make_tuple(overall_time, event_time);
  }

  /**
   * @fn    duration
   * @brief Returns the average number of flops across numReps runs of the
   * function F, which performs n_fl_ops floating point operations.
   */
  template <typename F, typename... Args>
  static datapoint_t measure(size_t numReps, size_t n_fl_ops, F func,
                             Args&&... args) {
    TimeT overall_duration = TimeT::zero();
    TimeT kernel_duration = TimeT::zero();

    // warm up to avoid benchmarking data transfer
    for (int i = 0; i < 5; ++i) {
      func(std::forward<Args>(args)...);
    }

    for (size_t reps = 0; reps < numReps; reps++) {
      auto exec_time = benchmark<>::timef(func, std::forward<Args>(args)...);

      overall_duration += std::get<0>(exec_time);
      kernel_duration += std::get<1>(exec_time);
    }

    // calculate the average overall time
    TimeT mean_overall_duration =
        std::chrono::duration_cast<std::chrono::nanoseconds>(overall_duration) /
        numReps;

    // calculate the average kernel event time
    TimeT mean_kernel_duration = kernel_duration / numReps;

    return datapoint(mean_overall_duration, mean_kernel_duration, n_fl_ops);
  }

  /**
   * @fn typestr
   * @brief Print the type of a given type T. Currently only supports `float`
   * and `double`.
   */
  template <typename T>
  static std::string typestr() {
    return type_string<T>();
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
  virtual benchmark<>::datapoint_t run(ParamT params, int reps,
                                       ExecutorT ex) = 0;
};

/** BENCHMARK_NAME_FORMAT
 * Define how we want to print names for a given benchmark suite
 */

#define BENCHMARK_NAME_FORMAT(suite_name)                                    \
  template <typename ElemT, typename ExecutorT, typename ParamT>             \
  class benchmark_##suite_name##_suite_class                                 \
      : public benchmark_instance<ElemT, ExecutorT, ParamT> {                \
   public:                                                                   \
    benchmark_##suite_name##_suite_class(){};                                \
    const std::string name() = 0;                                            \
    const std::string format_name(ParamT params);                            \
    const char* type() { return typeid(ElemT).name(); }                      \
    benchmark<>::datapoint_t run(ParamT params, int reps, ExecutorT ex) = 0; \
  };                                                                         \
  template <typename ElemT, typename ExecutorT, typename ParamT>             \
  const std::string benchmark_##suite_name##_suite_class<                    \
      ElemT, ExecutorT, ParamT>::format_name(ParamT params)

/** BENCHMARK
 * Declare a particular benchmark/instance.
 */
#define BENCHMARK(bench_name, suite_name)                                \
  template <typename ElemT, typename ExecutorT, typename ParamT>         \
  class benchmark_##bench_name##_class_                                  \
      : public benchmark_##suite_name##_suite_class<ElemT, ExecutorT,    \
                                                    ParamT> {            \
    const char* _name = #bench_name;                                     \
                                                                         \
   public:                                                               \
    benchmark_##bench_name##_class_(){};                                 \
    const std::string name() { return std::string(_name); }              \
    const char* type() { return typeid(ElemT).name(); }                  \
    benchmark<>::datapoint_t run(ParamT params, int reps, ExecutorT ex); \
  };                                                                     \
  template <typename ElemT, typename ExecutorT, typename ParamT>         \
  benchmark<>::datapoint_t                                               \
      benchmark_##bench_name##_class_<ElemT, ExecutorT, ParamT>::run(    \
          ParamT params, int reps, ExecutorT ex)

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
    auto res = b->run(params, reps, ex);
    const std::string name = b->format_name(params);

    res.output_data(name, reps, output);

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
  datapoint::output_headers(output);
  auto fbenchmarks = benchmark_suite<float, Ex, ParamT>();
  for (auto b : fbenchmarks) {
    run_benchmark(b, range_param, reps, ex, output);
  }

#ifdef DOUBLE_SUPPORT
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
#define SYCL_BENCHMARK_MAIN(RANGE_PARAM, REPS)                               \
  int main(int argc, char* argv[]) {                                         \
    benchmark_arguments ba(argc, argv);                                      \
    if (!ba.validProgramOptions) {                                           \
      return 1;                                                              \
    }                                                                        \
    cli_device_selector cds(ba.device_vendor, ba.device_type);               \
    cl::sycl::queue q(cds, {cl::sycl::property::queue::enable_profiling()}); \
    blas::Executor<blas::Policy_Handler<blas::BLAS_SYCL_Policy>> ex(q);      \
    return main_impl((&RANGE_PARAM), (REPS), ex, ba.requestedOutput);        \
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
