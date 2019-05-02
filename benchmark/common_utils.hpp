#ifndef UTILS_HPP
#define UTILS_HPP

#include <algorithm>
#include <benchmark/benchmark.h>
#include <chrono>
#include <climits>
#include <exception>
#include <fstream>
#include <functional>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "benchmark_cli_args.hpp"
#include "blas_meta.h"

using index_t = int;
using blas1_param_t = index_t;

template <typename scalar_t>
using blas2_param_t =
    std::tuple<std::string, index_t, index_t, scalar_t, scalar_t>;

template <typename scalar_t>
using blas3_param_t = std::tuple<std::string, std::string, index_t, index_t,
                                 index_t, scalar_t, scalar_t>;

namespace blas_benchmark {

namespace utils {

/**
 * @fn parse_csv_file
 * @brief Returns a vector containing the parameters for a benchmark as tuples,
 * read from the given csv file
 */
template <typename param_t>
std::vector<param_t> parse_csv_file(
    std::string& filepath,
    std::function<param_t(std::vector<std::string>&)> func) {
  std::vector<param_t> csv_data;
  std::ifstream data(filepath);
  std::string line;
  while (std::getline(data, line)) {
    if (line.empty()) continue;
    line.push_back({});
    std::stringstream lineStream(line);
    std::string cell;
    std::vector<std::string> csv_line;
    while (std::getline(lineStream, cell, ',')) {
      csv_line.push_back(cell);
    }
    try {
      csv_data.push_back(func(csv_line));
    } catch (std::exception& e) {
      std::cerr << "Error while parsing CSV file: " << e.what()
                << ", on line: " << line << std::endl;
      exit(1);
    }
  }
  if (csv_data.size() == 0) {
    std::cerr
        << "No data has been read. The given file might not exist or be empty."
        << std::endl;
    exit(1);
  }
  return csv_data;
}

/**
 * @fn get_range
 * @brief Prints a warning to the user if no CSV parameter file has been given
 */
inline void warning_no_csv() {
  std::cerr
      << "WARNING: no CSV parameter file has been given. Default ranges will "
         "be used."
      << std::endl;
}

/**
 * @fn str_to_int
 * @brief Converts a string to a specific integer type
 */
template <typename int_t>
inline int_t str_to_int(std::string str) {
  return static_cast<int_t>(std::stoi(str));
}

template <>
inline long int str_to_int<long int>(std::string str) {
  return std::stol(str);
}

template <>
inline long long int str_to_int<long long int>(std::string str) {
  return std::stoll(str);
}

/**
 * @fn str_to_scalar
 * @brief Converts a string to a specific scalar type
 */
template <typename scalar_t>
inline scalar_t str_to_scalar(std::string str) {
  return static_cast<scalar_t>(std::stof(str));
}

template <>
inline double str_to_scalar<double>(std::string str) {
  return std::stod(str);
}

/**
 * @fn get_blas1_params
 * @brief Returns a vector containing the blas 1 benchmark parameters, either
 * read from a file according to the command-line args, or the default ones.
 */
inline std::vector<blas1_param_t> get_blas1_params(Args& args) {
  if (args.csv_param.empty()) {
    warning_no_csv();
    std::vector<blas1_param_t> blas1_default;
    for (index_t size = 4096; size <= 1048576; size *= 2) {
      blas1_default.push_back(size);
    }
    return blas1_default;
  } else {
    return parse_csv_file<blas1_param_t>(
        args.csv_param, [&](std::vector<std::string>& v) {
          if (v.size() != 1) {
            throw std::runtime_error(
                "invalid number of parameters (1 expected)");
          }
          try {
            return str_to_int<index_t>(v[0]);
          } catch (...) {
            throw std::runtime_error("invalid parameter");
          }
        });
  }
}

/**
 * @fn get_blas2_params
 * @brief Returns a vector containing the blas 2 benchmark parameters, either
 * read from a file according to the command-line args, or the default ones.
 */
template <typename scalar_t>
inline std::vector<blas2_param_t<scalar_t>> get_blas2_params(Args& args) {
  if (args.csv_param.empty()) {
    warning_no_csv();
    std::vector<blas2_param_t<scalar_t>> blas2_default;
    constexpr index_t dmin = 64, dmax = 1024;
    scalar_t alpha = 1;
    scalar_t beta = 0;
    for (std::string t : {"n", "t"}) {
      for (index_t m = dmin; m <= dmax; m *= 2) {
        for (index_t n = dmin; n <= dmax; n *= 2) {
          blas2_default.push_back(std::make_tuple(t, m, n, alpha, beta));
        }
      }
    }
    return blas2_default;
  } else {
    return parse_csv_file<blas2_param_t<scalar_t>>(
        args.csv_param, [&](std::vector<std::string>& v) {
          if (v.size() != 5) {
            throw std::runtime_error(
                "invalid number of parameters (5 expected)");
          }
          try {
            return std::make_tuple(v[0].c_str(), str_to_int<index_t>(v[1]),
                                   str_to_int<index_t>(v[2]),
                                   str_to_scalar<scalar_t>(v[3]),
                                   str_to_scalar<scalar_t>(v[4]));
          } catch (...) {
            throw std::runtime_error("invalid parameter");
          }
        });
  }
}

/**
 * @fn get_blas3_params
 * @brief Returns a vector containing the blas 3 benchmark parameters, either
 * read from a file according to the command-line args, or the default ones.
 */
template <typename scalar_t>
inline std::vector<blas3_param_t<scalar_t>> get_blas3_params(Args& args) {
  if (args.csv_param.empty()) {
    warning_no_csv();
    std::vector<blas3_param_t<scalar_t>> blas3_default;
    constexpr index_t dmin = 64, dmax = 1024;
    std::vector<std::string> dtranspose = {"n", "t"};
    scalar_t alpha = 1;
    scalar_t beta = 0;
    for (std::string& t1 : dtranspose) {
      for (std::string& t2 : dtranspose) {
        for (index_t m = dmin; m <= dmax; m *= 2) {
          for (index_t k = dmin; k <= dmax; k *= 2) {
            for (index_t n = dmin; n <= dmax; n *= 2) {
              blas3_default.push_back(
                  std::make_tuple(t1, t2, m, k, n, alpha, beta));
            }
          }
        }
      }
    }
    return blas3_default;
  } else {
    return parse_csv_file<blas3_param_t<scalar_t>>(
        args.csv_param, [&](std::vector<std::string>& v) {
          if (v.size() != 7) {
            throw std::runtime_error(
                "invalid number of parameters (7 expected)");
          }
          try {
            return std::make_tuple(
                v[0].c_str(), v[1].c_str(), str_to_int<index_t>(v[2]),
                str_to_int<index_t>(v[3]), str_to_int<index_t>(v[4]),
                str_to_scalar<scalar_t>(v[5]), str_to_scalar<scalar_t>(v[6]));
          } catch (...) {
            throw std::runtime_error("invalid parameter");
          }
        });
  }
}

/**
 * @fn get_type_name
 * @brief Returns a string with the given type. The C++ specification doesn't
 * guarantee that typeid(T).name is human readable so we specify the template
 * for float and double.
 */
template <typename scalar_t>
inline std::string get_type_name() {
  std::string type_name(typeid(scalar_t).name());
  return type_name;
}
template <>
inline std::string get_type_name<float>() {
  return "float";
}
template <>
inline std::string get_type_name<double>() {
  return "double";
}

/**
 * @fn random_scalar
 * @brief Generates a random scalar value, using an arbitrary low quality
 * algorithm.
 */
template <typename scalar_t>
static inline scalar_t random_scalar() {
  return 1e-3 * ((rand() % 2000) - 1000);
}

/**
 * @fn random_data
 * @brief Generates a random vector of scalar values, using an arbitrary low
 * quality algorithm.
 */
template <typename scalar_t>
static inline std::vector<scalar_t> random_data(size_t size,
                                                bool initialized = true) {
  std::vector<scalar_t> v = std::vector<scalar_t>(size);
  if (initialized) {
    std::transform(v.begin(), v.end(), v.begin(), [](scalar_t x) -> scalar_t {
      return random_scalar<scalar_t>();
    });
  }
  return v;
}

/**
 * @fn const_data
 * @brief Generates a vector of constant values, of a given length.
 */
template <typename scalar_t>
static inline std::vector<scalar_t> const_data(size_t size,
                                               scalar_t const_value = 0) {
  std::vector<scalar_t> v = std::vector<scalar_t>(size);
  std::fill(v.begin(), v.end(), const_value);
  return v;
}

enum class Transposition { Normal, Transposed, Conjugate };

const std::array<Transposition, 3> possible_transpositions(
    {Transposition::Normal, Transposition::Transposed,
     Transposition::Conjugate});
/**
 * @fn to_transpose_enum
 * @brief Translates from a transposition string to an enum.
 */
static inline Transposition to_transpose_enum(std::string& t) {
  if (t == "t") {
    return Transposition::Transposed;
  } else if (t == "c") {
    return Transposition::Conjugate;
  } else if (t == "n") {
    return Transposition::Normal;
  } else {
    std::cerr << "Unrecognized transpose type: " << t << std::endl;
    exit(1);
  }
}
/**
 * @fn from_transpose_enum
 * @brief Translates from a transposition enum to a transposition string
 */
static inline std::string from_transpose_enum(Transposition t) {
  switch (t) {
    case Transposition::Transposed:
      return "t";
      break;
    case Transposition::Conjugate:
      return "c";
      break;
    case Transposition::Normal:
      return "n";
      break;
    default:
      return "n";
  }
}

/**
 * @fn time_event
 * @brief Times 1 event, and returns the aggregate time.
 */
template <typename event_t>
inline cl_ulong time_event(event_t&);
// Declared here, defined separately in the specific utils.hpp files

/**
 * @fn time_events
 * @brief Times n events, and returns the aggregate time.
 */
template <typename event_t>
inline cl_ulong time_events(std::vector<event_t> es) {
  cl_ulong total_time = 0;
  for (auto e : es) {
    total_time += time_event(e);
  }
  return total_time;
}

template <typename event_t, typename... other_events_t>
inline cl_ulong time_events(event_t first_event,
                            other_events_t... next_events) {
  return time_events<event_t>(
      blas::concatenate_vectors(first_event, next_events...));
}

/**
 * @fn timef
 * @brief Calculates the time spent executing the function func
 * (both overall and event time, returned in nanoseconds in a tuple of double)
 */
template <typename function_t, typename... args_t>
static std::tuple<double, double> timef(function_t func, args_t&&... args) {
  auto start = std::chrono::system_clock::now();
  auto event = func(std::forward<args_t>(args)...);
  auto end = std::chrono::system_clock::now();
  double overall_time = (end - start).count();

  double event_time = static_cast<double>(time_events(event));

  return std::make_tuple(overall_time, event_time);
}

// Functions to initialize and update the counters

inline void init_counters(benchmark::State& state) {
  state.counters["best_event_time"] = ULONG_MAX;
  state.counters["best_overall_time"] = ULONG_MAX;
}

inline void update_counters(benchmark::State& state,
                            std::tuple<double, double> times) {
  state.PauseTiming();
  double overall_time, event_time;
  std::tie(overall_time, event_time) = times;
  state.counters["total_event_time"] += event_time;
  state.counters["best_event_time"] =
      std::min<double>(state.counters["best_event_time"], event_time);
  state.counters["total_overall_time"] += overall_time;
  state.counters["best_overall_time"] =
      std::min<double>(state.counters["best_overall_time"], overall_time);
  state.ResumeTiming();
}

inline void calc_avg_counters(benchmark::State& state) {
  state.counters["avg_event_time"] =
      state.counters["total_event_time"] / state.iterations();
  state.counters["avg_overall_time"] =
      state.counters["total_overall_time"] / state.iterations();
}

}  // namespace utils
}  // namespace blas_benchmark

#endif
