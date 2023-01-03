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
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "benchmark_cli_args.hpp"
#include "blas_meta.h"
#include <common/float_comparison.hpp>
#include <common/system_reference_blas.hpp>

using index_t = BLAS_INDEX_T;

using blas1_param_t = index_t;

template <typename scalar_t>
using blas2_param_t =
    std::tuple<std::string, index_t, index_t, scalar_t, scalar_t>;

template <typename scalar_t>
using blas3_param_t = std::tuple<std::string, std::string, index_t, index_t,
                                 index_t, scalar_t, scalar_t>;

template <typename scalar_t>
using gemm_batched_param_t =
    std::tuple<std::string, std::string, index_t, index_t, index_t, scalar_t,
               scalar_t, index_t, int>;

using reduction_param_t = std::tuple<index_t, index_t>;

template <typename scalar_t>
using trsm_param_t =
    std::tuple<char, char, char, char, index_t, index_t, scalar_t>;

template <typename scalar_t>
using gbmv_param_t = std::tuple<std::string, index_t, index_t, index_t, index_t,
                                scalar_t, scalar_t>;

template <typename scalar_t>
using sbmv_param_t =
    std::tuple<std::string, index_t, index_t, scalar_t, scalar_t>;

namespace blas_benchmark {

namespace utils {

/**
 * @brief Print the explanatory string of an exception.
 * If the exception is nested, recurse to print the explanatory of the
 * exception it holds.
 * From https://en.cppreference.com/w/cpp/error/nested_exception.
 */
inline void print_exception(const std::exception& e, int level = 0) {
  std::cerr << std::string(level, ' ') << "Exception: " << e.what() << '\n';
  try {
    std::rethrow_if_nested(e);
  } catch (const std::exception& e) {
    print_exception(e, level + 1);
  } catch (...) {
  }
}

/**
 * @fn parse_csv_file
 * @brief Returns a vector containing the parameters for a benchmark as tuples,
 * read from the given csv file
 */
template <typename param_t>
static inline std::vector<param_t> parse_csv_file(
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
      std::cerr << "Error while parsing CSV file on line '" << line << "':\n";
      print_exception(e);
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
static inline void warning_no_csv() {
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
static inline int_t str_to_int(std::string str) {
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

inline int str_to_batch_type(std::string str) {
  // Remove any null character from str
  str.erase(std::find(str.begin(), str.end(), '\0'), str.end());
  if (str == "strided") {
    return 0;
  } else if (str == "interleaved") {
    return 1;
  } else {
    throw std::runtime_error("Unrecognized batch type: '" + str + "'");
  }
  return -1;
}

inline std::string batch_type_to_str(int batch_type) {
  switch (batch_type) {
    case 0:
      return "strided";

    case 1:
      return "interleaved";

    default:
      throw std::runtime_error("Unrecognized batch type: " +
                               std::to_string(batch_type));
  }
  return "";
}

/**
 * @fn str_to_scalar
 * @brief Converts a string to a specific scalar type
 */
template <typename scalar_t>
static inline scalar_t str_to_scalar(std::string str) {
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
static inline std::vector<blas1_param_t> get_blas1_params(Args& args) {
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
static inline std::vector<blas2_param_t<scalar_t>> get_blas2_params(
    Args& args) {
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
static inline std::vector<blas3_param_t<scalar_t>> get_blas3_params(
    Args& args) {
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
 * @fn get_gemm_batched_params
 * @brief Returns a vector containing the gemm_batched benchmark parameters,
 * either read from a file according to the command-line args, or the default
 * ones.
 */
template <typename scalar_t>
inline std::vector<gemm_batched_param_t<scalar_t>> get_gemm_batched_params(
    Args& args) {
  if (args.csv_param.empty()) {
    warning_no_csv();
    std::vector<gemm_batched_param_t<scalar_t>> gemm_batched_default;
    constexpr index_t dmin = 64, dmax = 1024;
    std::vector<std::string> dtranspose = {"n", "t"};
    scalar_t alpha = 1;
    scalar_t beta = 0;
    index_t batch_size = 8;
    int batch_type = 0;
    for (std::string& t1 : dtranspose) {
      for (std::string& t2 : dtranspose) {
        for (index_t m = dmin; m <= dmax; m *= 2) {
          for (index_t k = dmin; k <= dmax; k *= 2) {
            for (index_t n = dmin; n <= dmax; n *= 2) {
              gemm_batched_default.push_back(std::make_tuple(
                  t1, t2, m, k, n, alpha, beta, batch_size, batch_type));
            }
          }
        }
      }
    }
    return gemm_batched_default;
  } else {
    return parse_csv_file<gemm_batched_param_t<scalar_t>>(
        args.csv_param, [&](std::vector<std::string>& v) {
          if (v.size() != 9) {
            throw std::runtime_error(
                "invalid number of parameters (9 expected)");
          }
          try {
            return std::make_tuple(
                v[0].c_str(), v[1].c_str(), str_to_int<index_t>(v[2]),
                str_to_int<index_t>(v[3]), str_to_int<index_t>(v[4]),
                str_to_scalar<scalar_t>(v[5]), str_to_scalar<scalar_t>(v[6]),
                str_to_int<index_t>(v[7]), str_to_batch_type(v[8]));
          } catch (...) {
            std::throw_with_nested(std::runtime_error("invalid parameter"));
          }
        });
  }
}

/**
 * @fn get_reduction_params
 * @brief Returns a vector containing the reduction benchmark parameters, either
 * read from a file according to the command-line args, or the default ones.
 */
template <typename scalar_t>
static inline std::vector<reduction_param_t> get_reduction_params(Args& args) {
  if (args.csv_param.empty()) {
    warning_no_csv();
    std::vector<reduction_param_t> reduction_default;
    constexpr index_t dmin = 64, dmax = 1024;

    for (index_t rows = dmin; rows <= dmax; rows *= 2) {
      for (index_t cols = dmin; cols <= dmax; cols *= 2) {
        reduction_default.push_back(std::make_tuple(rows, cols));
      }
    }

    return reduction_default;
  } else {
    return parse_csv_file<reduction_param_t>(
        args.csv_param, [&](std::vector<std::string>& v) {
          if (v.size() != 2) {
            throw std::runtime_error(
                "invalid number of parameters (2 expected)");
          }
          try {
            return std::make_tuple(str_to_int<index_t>(v[0]),
                                   str_to_int<index_t>(v[1]));
          } catch (...) {
            throw std::runtime_error("invalid parameter");
          }
        });
  }
}

/**
 * @fn get_trsm_params
 * @brief Returns a vector containing the trsm benchmark parameters, either
 * read from a file according to the command-line args, or the default ones.
 */
template <typename scalar_t>
static inline std::vector<trsm_param_t<scalar_t>> get_trsm_params(Args& args) {
  if (args.csv_param.empty()) {
    warning_no_csv();
    std::vector<trsm_param_t<scalar_t>> trsm_default;
    constexpr index_t dmin = 64, dmax = 1024;
    for (char side : {'l', 'r'}) {
      for (char uplo : {'u', 'l'}) {
        for (char trans : {'n', 't'}) {
          for (char diag : {'u', 'n'}) {
            for (index_t m = dmin; m <= dmax; m *= 2) {
              for (index_t n = dmin; n <= dmax; n *= 2) {
                trsm_default.push_back(std::make_tuple(side, uplo, trans, diag,
                                                       m, n, scalar_t{1}));
              }
            }
          }
        }
      }
    }
    return trsm_default;
  } else {
    return parse_csv_file<trsm_param_t<scalar_t>>(
        args.csv_param, [&](std::vector<std::string>& v) {
          if (v.size() != 7) {
            throw std::runtime_error(
                "invalid number of parameters (7 expected)");
          }
          try {
            return std::make_tuple(
                v[0][0], v[1][0], v[2][0], v[3][0], str_to_int<index_t>(v[4]),
                str_to_int<index_t>(v[5]), str_to_scalar<scalar_t>(v[6]));
          } catch (...) {
            throw std::runtime_error("invalid parameter");
          }
        });
  }
}

/**
 * @fn get_gbmv_params
 * @brief Returns a vector containing the gbmv benchmark parameters, either
 * read from a file according to the command-line args, or the default ones.
 */
template <typename scalar_t>
static inline std::vector<gbmv_param_t<scalar_t>> get_gbmv_params(Args& args) {
  if (args.csv_param.empty()) {
    warning_no_csv();
    std::vector<gbmv_param_t<scalar_t>> gbmv_default;
    constexpr index_t dmin = 64, dmax = 1024;
    constexpr index_t kmin = 1;
    scalar_t alpha = 1;
    scalar_t beta = 0;
    for (std::string t : {"n", "t"}) {
      for (index_t m = dmin; m <= dmax; m *= 2) {
        for (index_t n = dmin; n <= dmax; n *= 2) {
          for (index_t kl = kmin; kl <= m / 4; kl *= 2) {
            for (index_t ku = kmin; ku <= n / 4; ku *= 2) {
              gbmv_default.push_back(
                  std::make_tuple(t, m, n, kl, ku, alpha, beta));
            }
          }
        }
      }
    }
    return gbmv_default;
  } else {
    return parse_csv_file<gbmv_param_t<scalar_t>>(
        args.csv_param, [&](std::vector<std::string>& v) {
          if (v.size() != 7) {
            throw std::runtime_error(
                "invalid number of parameters (7 expected)");
          }
          try {
            return std::make_tuple(
                v[0].c_str(), str_to_int<index_t>(v[1]),
                str_to_int<index_t>(v[2]), str_to_int<index_t>(v[3]),
                str_to_int<index_t>(v[4]), str_to_scalar<scalar_t>(v[5]),
                str_to_scalar<scalar_t>(v[6]));
          } catch (...) {
            throw std::runtime_error("invalid parameter");
          }
        });
  }
}

/**
 * @fn get_sbmv_params
 * @brief Returns a vector containing the sbmv benchmark parameters, either
 * read from a file according to the command-line args, or the default ones.
 */
template <typename scalar_t>
static inline std::vector<sbmv_param_t<scalar_t>> get_sbmv_params(Args& args) {
  if (args.csv_param.empty()) {
    warning_no_csv();
    std::vector<sbmv_param_t<scalar_t>> sbmv_default;
    constexpr index_t dmin = 64, dmax = 1024;
    constexpr index_t kmin = 1;
    scalar_t alpha = 1;
    scalar_t beta = 0;
    for (std::string ul : {"u", "l"}) {
      for (index_t n = dmin; n <= dmax; n *= 2) {
        for (index_t k = kmin; k <= n / 4; k *= 2) {
          sbmv_default.push_back(std::make_tuple(ul, n, k, alpha, beta));
        }
      }
    }
    return sbmv_default;
  } else {
    return parse_csv_file<sbmv_param_t<scalar_t>>(
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
 * @fn get_type_name
 * @brief Returns a string with the given type. The C++ specification doesn't
 * guarantee that typeid(T).name is human readable so we specify the template
 * for float and double.
 */
template <typename scalar_t>
static inline std::string get_type_name() {
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
 * @brief Generates a random scalar in the specified range
 * @param rangeMin range minimum
 * @param rangeMax range maximum
 */
template <typename scalar_t>
static inline scalar_t random_scalar(scalar_t rangeMin, scalar_t rangeMax) {
  static std::random_device rd;
  static std::default_random_engine gen(rd());
  std::uniform_real_distribution<scalar_t> dis(rangeMin, rangeMax);
  return dis(gen);
}

/**
 * @fn random_data
 * @brief Generates a random vector of scalar values, using a uniform
 * distribution.
 */
template <typename scalar_t>
static inline std::vector<scalar_t> random_data(size_t size) {
  std::vector<scalar_t> v = std::vector<scalar_t>(size);

  for (scalar_t& e : v) {
    e = random_scalar(scalar_t{-2}, scalar_t{5});
  }
  return v;
}

/**
 * @breif Fills a lower or upper triangular matrix suitable for TRSM testing
 * @param A The matrix to fill. Size must be at least m * lda
 * @param m The number of rows of matrix @p A
 * @param n The number of columns of matrix @p A
 * @param lda The leading dimension of matrix @p A
 * @param uplo if 'u', @p A will be upper triangular. If 'l' @p A will be
 * lower triangular
 * @param diag Value to put in the diagonal elements
 * @param unused Value to put in the unused parts of the matrix
 */
template <typename scalar_t>
static inline void fill_trsm_matrix(std::vector<scalar_t>& A, size_t k,
                                    size_t lda, char uplo,
                                    scalar_t diag = scalar_t{1},
                                    scalar_t unused = scalar_t{0}) {
  for (size_t i = 0; i < k; ++i) {
    scalar_t sum = std::abs(diag);
    for (size_t j = 0; j < k; ++j) {
      scalar_t value = scalar_t{0};
      if (i == j) {
        value = diag;
      } else if (((uplo == 'l') && (i > j)) || ((uplo == 'u') && (i < j))) {
        if (sum >= scalar_t{1}) {
          const double limit =
              sum / std::sqrt(static_cast<double>(k) - static_cast<double>(j));
          value = random_scalar(scalar_t{-1}, scalar_t{1}) * limit;
          sum -= std::abs(value);
        }
      } else {
        value = unused;
      }
      A[i + j * lda] = value;
    }
  }
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
 * @fn warmup
 * @brief Warm up to avoid benchmarking data transfer
 */
template <typename function_t, typename... args_t>
static inline void warmup(function_t func, args_t&&... args) {
  for (int i = 0; i < 10; ++i) {
    func(std::forward<args_t>(args)...);
  }
}

/**
 * @fn time_event
 * @brief Times 1 event, and returns the aggregate time.
 */
template <typename event_t>
static inline double time_event(event_t&);
// Declared here, defined separately in the specific utils.hpp files

/**
 * @fn time_events
 * @brief Times n events, and returns the aggregate time.
 */
template <typename event_t>
static inline double time_events(std::vector<event_t> es) {
  double total_time = 0;
  for (auto e : es) {
    total_time += time_event(e);
  }
  return total_time;
}

template <typename event_t, typename... other_events_t>
static inline double time_events(event_t first_event,
                                 other_events_t... next_events) {
  return time_events(blas::concatenate_vectors(first_event, next_events...));
}
/**
 * @fn timef
 * @brief Calculates the time spent executing the function func
 * (both overall and event time, returned in nanoseconds in a tuple of double)
 */
template <typename function_t, typename... args_t>
static inline std::tuple<double, double> timef(function_t func,
                                               args_t&&... args) {
  auto start = std::chrono::system_clock::now();
  auto event = func(std::forward<args_t>(args)...);
  auto end = std::chrono::system_clock::now();
  double overall_time = (end - start).count();

  double event_time = time_events(event);

  return std::make_tuple(overall_time, event_time);
}

// Functions to initialize and update the counters

static inline void init_counters(benchmark::State& state) {
  state.counters["best_event_time"] = double(ULONG_MAX);
  state.counters["best_overall_time"] = double(ULONG_MAX);
}

static inline void update_counters(benchmark::State& state,
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

static inline void calc_avg_counters(benchmark::State& state) {
  state.counters["avg_event_time"] =
      state.counters["total_event_time"] / state.iterations();
  state.counters["avg_overall_time"] =
      state.counters["total_overall_time"] / state.iterations();
}

}  // namespace utils
}  // namespace blas_benchmark

/** Registers benchmark for the float data type
 * @see BLAS_REGISTER_BENCHMARK
 */
#define BLAS_REGISTER_BENCHMARK_FLOAT(args, sb_handle_ptr, success) \
  register_benchmark<float>(args, sb_handle_ptr, success)

#ifdef BLAS_DATA_TYPE_DOUBLE
/** Registers benchmark for the double data type
 * @see BLAS_REGISTER_BENCHMARK
 */
#define BLAS_REGISTER_BENCHMARK_DOUBLE(args, sb_handle_ptr, success) \
  register_benchmark<double>(args, sb_handle_ptr, success)
#else
#define BLAS_REGISTER_BENCHMARK_DOUBLE(args, sb_handle_ptr, success)
#endif  // BLAS_DATA_TYPE_DOUBLE

#ifdef BLAS_DATA_TYPE_HALF
/** Registers benchmark for the cl::sycl::half data type
 * @see BLAS_REGISTER_BENCHMARK
 */
#define BLAS_REGISTER_BENCHMARK_HALF(args, sb_handle_ptr, success) \
  register_benchmark<cl::sycl::half>(args, sb_handle_ptr, success)
#else
#define BLAS_REGISTER_BENCHMARK_HALF(args, sb_handle_ptr, success)
#endif  // BLAS_DATA_TYPE_HALF

/** Registers benchmark for all supported data types.
 *  Expects register_benchmark<scalar_t> to exist.
 * @param args Reference to blas_benchmark::Args
 * @param sb_handle_ptr Pointer to blas::SB_Handle
 * @param[out] success Pointer to boolean indicating success
 */
#define BLAS_REGISTER_BENCHMARK(args, sb_handle_ptr, success)     \
  do {                                                            \
    BLAS_REGISTER_BENCHMARK_FLOAT(args, sb_handle_ptr, success);  \
    BLAS_REGISTER_BENCHMARK_DOUBLE(args, sb_handle_ptr, success); \
    BLAS_REGISTER_BENCHMARK_HALF(args, sb_handle_ptr, success);   \
  } while (false)

#endif
