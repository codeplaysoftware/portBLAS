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

using index_t = BLAS_INDEX_T;

#include "benchmark_cli_args.hpp"
#include "blas_meta.h"
#include <common/benchmark_identifier.hpp>
#include <common/benchmark_names.hpp>
#include <common/blas1_state_counters.hpp>
#include <common/blas2_state_counters.hpp>
#include <common/blas3_state_counters.hpp>
#include <common/blas_extension_state_counters.hpp>
#include <common/float_comparison.hpp>
#include <common/set_benchmark_label.hpp>
#include <common/system_reference_blas.hpp>

using blas1_param_t = index_t;

template <typename scalar_t>
using blas2_param_t =
    std::tuple<std::string, index_t, index_t, scalar_t, scalar_t>;

template <typename scalar_t>
using copy_param_t = std::tuple<index_t, index_t, index_t, scalar_t>;

template <typename scalar_t>
using blas3_param_t = std::tuple<std::string, std::string, index_t, index_t,
                                 index_t, scalar_t, scalar_t>;

template <typename scalar_t>
using gemm_batched_param_t =
    std::tuple<std::string, std::string, index_t, index_t, index_t, scalar_t,
               scalar_t, index_t, int>;

template <typename scalar_t>
using gemm_batched_strided_param_t =
    std::tuple<std::string, std::string, index_t, index_t, index_t, scalar_t,
               scalar_t, index_t, index_t, index_t, index_t>;

#ifdef BLAS_ENABLE_COMPLEX
template <typename scalar_t>
using blas3_cplx_param_t =
    std::tuple<std::string, std::string, index_t, index_t, index_t, scalar_t,
               scalar_t, scalar_t, scalar_t>;

template <typename scalar_t>
using gemm_batched_strided_cplx_param_t =
    std::tuple<std::string, std::string, index_t, index_t, index_t, scalar_t,
               scalar_t, scalar_t, scalar_t, index_t, index_t, index_t,
               index_t>;

template <typename scalar_t>
using gemm_batched_cplx_param_t =
    std::tuple<std::string, std::string, index_t, index_t, index_t, scalar_t,
               scalar_t, scalar_t, scalar_t, index_t, int>;
#endif

using reduction_param_t = std::tuple<index_t, index_t>;

template <typename scalar_t>
using trsm_param_t =
    std::tuple<char, char, char, char, index_t, index_t, scalar_t>;

template <typename scalar_t>
using symm_param_t =
    std::tuple<char, char, index_t, index_t, scalar_t, scalar_t>;

template <typename scalar_t>
using syrk_param_t =
    std::tuple<char, char, index_t, index_t, scalar_t, scalar_t>;

template <typename scalar_t>
using gbmv_param_t = std::tuple<std::string, index_t, index_t, index_t, index_t,
                                scalar_t, scalar_t>;

template <typename scalar_t>
using sbmv_param_t =
    std::tuple<std::string, index_t, index_t, scalar_t, scalar_t>;

template <typename scalar_t>
using symv_param_t = std::tuple<std::string, index_t, scalar_t, scalar_t>;

template <typename scalar_t>
using syr_param_t = std::tuple<std::string, index_t, scalar_t>;

template <typename scalar_t>
using ger_param_t = std::tuple<index_t, index_t, scalar_t>;

template <typename scalar_t>
using spr_param_t = std::tuple<std::string, index_t, scalar_t, index_t>;

template <typename scalar_t>
using spr2_param_t =
    std::tuple<std::string, index_t, scalar_t, index_t, index_t>;

using tbmv_param_t =
    std::tuple<std::string, std::string, std::string, index_t, index_t>;

using trsv_param_t = std::tuple<std::string, std::string, std::string, index_t>;

template <typename scalar_t>
using trsm_batched_param_t =
    std::tuple<char, char, char, char, index_t, index_t, scalar_t, index_t,
               index_t, index_t>;

template <typename scalar_t>
using matcopy_param_t =
    std::tuple<char, index_t, index_t, scalar_t, index_t, index_t>;

template <typename scalar_t>
using omatcopy2_param_t = std::tuple<char, index_t, index_t, scalar_t, index_t,
                                     index_t, index_t, index_t>;

template <typename scalar_t>
using omatadd_param_t = std::tuple<char, char, index_t, index_t, scalar_t,
                                   scalar_t, index_t, index_t, index_t>;

template <typename scalar_t>
using matcopy_batch_param_t =
    std::tuple<char, index_t, index_t, scalar_t, index_t, index_t, index_t,
               index_t, index_t>;

template <typename scalar_t>
using omatadd_batch_param_t =
    std::tuple<char, char, index_t, index_t, scalar_t, scalar_t, index_t,
               index_t, index_t, index_t, index_t, index_t, index_t>;

template <typename scalar_t>
using axpy_batch_param_t =
    std::tuple<index_t, scalar_t, index_t, index_t, index_t, index_t, index_t>;

namespace blas_benchmark {

namespace utils {

inline constexpr char MEM_TYPE_BUFFER[] = "buffer";
inline constexpr char MEM_TYPE_USM[] = "usm";

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
    for (index_t size = 1L << 10; size <= 1L << 22; size *= 2) {
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
    constexpr index_t dmin = 32, dmax = 8192;
    scalar_t alpha = 1;
    scalar_t beta = 1;
    for (std::string t : {"n", "t"}) {
      for (index_t n = dmin; n <= dmax; n *= 4) {
        blas2_default.push_back(std::make_tuple(t, n, n, alpha, beta));
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
 * @fn get_copy_params
 * @brief Returns a vector containing the blas1 copy benchmark parameters,
 * either read from a file according to the command-line args, or the default
 * ones.
 */
template <typename scalar_t>
static inline std::vector<copy_param_t<scalar_t>> get_copy_params(Args& args) {
  if (args.csv_param.empty()) {
    warning_no_csv();
    std::vector<copy_param_t<scalar_t>> default_values;
    for (index_t incx = 1; incx <= 2; incx *= 2) {
      for (index_t incy = 1; incy <= 2; incy *= 2) {
        for (index_t size = 1L << 12; size <= 1L << 20; size *= 16) {
          default_values.push_back(
              std::make_tuple(size, incx, incy, scalar_t(0)));
        }
      }
    }
    return default_values;
  } else {
    return parse_csv_file<copy_param_t<scalar_t>>(
        args.csv_param, [&](std::vector<std::string>& v) {
          if (v.size() != 3) {
            throw std::runtime_error(
                "invalid number of parameters (3 expected)");
          }
          try {
            return std::make_tuple(str_to_int<index_t>(v[0]),
                                   str_to_int<index_t>(v[1]),
                                   str_to_int<index_t>(v[2]), scalar_t(0));
          } catch (...) {
            throw std::runtime_error("invalid parameter");
          }
        });
  }
}

/**
 * @fn get_spr_params
 * @brief Returns a vector containing the spr benchmark parameters, either
 * read from a file according to the command-line args, or the default ones.
 */
template <typename scalar_t>
static inline std::vector<spr_param_t<scalar_t>> get_spr_params(Args& args) {
  if (args.csv_param.empty()) {
    warning_no_csv();
    std::vector<spr_param_t<scalar_t>> spr_default;
    constexpr index_t dmin = 32, dmax = 8192;
    const index_t incX = 1;
    const scalar_t alpha = 1;
    for (std::string uplo : {"u", "l"}) {
      for (index_t m = dmin; m <= dmax; m *= 4) {
        spr_default.push_back(std::make_tuple(uplo, m, alpha, incX));
      }
    }
    return spr_default;
  } else {
    return parse_csv_file<spr_param_t<scalar_t>>(
        args.csv_param, [&](std::vector<std::string>& v) {
          if (v.size() != 4) {
            throw std::runtime_error(
                "invalid number of parameters (4 expected)");
          }
          try {
            return std::make_tuple(v[0].c_str(), str_to_int<index_t>(v[1]),
                                   str_to_scalar<scalar_t>(v[2]),
                                   str_to_int<index_t>(v[3]));
          } catch (...) {
            throw std::runtime_error("invalid parameter");
          }
        });
  }
}

/**
 * @fn get_spr2_params
 * @brief Returns a vector containing the spr2 benchmark parameters, either
 * read from a file according to the command-line args, or the default ones.
 */
template <typename scalar_t>
static inline std::vector<spr2_param_t<scalar_t>> get_spr2_params(Args& args) {
  if (args.csv_param.empty()) {
    warning_no_csv();
    std::vector<spr2_param_t<scalar_t>> spr2_default;
    constexpr index_t dmin = 32, dmax = 8192;
    const index_t incX = 1;
    const index_t incY = 1;
    const scalar_t alpha = 1;
    for (std::string uplo : {"u", "l"}) {
      for (index_t m = dmin; m <= dmax; m *= 4) {
        spr2_default.push_back(std::make_tuple(uplo, m, alpha, incX, incY));
      }
    }
    return spr2_default;
  } else {
    return parse_csv_file<spr2_param_t<scalar_t>>(
        args.csv_param, [&](std::vector<std::string>& v) {
          if (v.size() != 5) {
            throw std::runtime_error(
                "invalid number of parameters (5 expected)");
          }
          try {
            return std::make_tuple(v[0].c_str(), str_to_int<index_t>(v[1]),
                                   str_to_scalar<scalar_t>(v[2]),
                                   str_to_int<index_t>(v[3]),
                                   str_to_int<index_t>(v[4]));
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
    constexpr index_t dmin = 32, dmax = 8192;
    std::vector<std::string> dtranspose = {"n", "t"};
    scalar_t alpha = 1;
    scalar_t beta = 1;
    for (std::string& t1 : dtranspose) {
      for (std::string& t2 : dtranspose) {
        for (index_t m = dmin; m <= dmax; m *= 8) {
          for (index_t k = dmin; k <= dmax; k *= 8) {
            for (index_t n = dmin; n <= dmax; n *= 8) {
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

#ifdef BLAS_ENABLE_COMPLEX
/**
 * @fn get_blas3_cplx_params for complex data type
 * @brief Returns a vector containing the blas 3 benchmark cplx parameters,
 * either read from a file according to the command-line args, or the default
 * ones. So far only used/supported for GEMM & its batched extensions.
 */
template <typename scalar_t>
static inline std::vector<blas3_cplx_param_t<scalar_t>> get_blas3_cplx_params(
    Args& args) {
  if (args.csv_param.empty()) {
    warning_no_csv();
    std::vector<blas3_cplx_param_t<scalar_t>> blas3_default;
    constexpr index_t dmin = 32, dmax = 8192;
    std::vector<std::string> dtranspose = {"n", "t"};
    std::complex<scalar_t> alpha{1, 1};
    std::complex<scalar_t> beta{1, 1};
    for (std::string& t1 : dtranspose) {
      for (std::string& t2 : dtranspose) {
        for (index_t m = dmin; m <= dmax; m *= 8) {
          for (index_t k = dmin; k <= dmax; k *= 8) {
            for (index_t n = dmin; n <= dmax; n *= 8) {
              blas3_default.push_back(
                  std::make_tuple(t1, t2, m, k, n, alpha.real(), alpha.imag(),
                                  beta.real(), beta.imag()));
            }
          }
        }
      }
    }
    return blas3_default;
  } else {
    return parse_csv_file<blas3_cplx_param_t<scalar_t>>(
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
                str_to_scalar<scalar_t>(v[7]), str_to_scalar<scalar_t>(v[8]));
          } catch (...) {
            throw std::runtime_error("invalid parameter");
          }
        });
  }
}

/**
 * @fn get_gemm_batched_strided_cplx_params for complex data type
 * @brief Returns a vector containing the gemm_batched_strided cplx benchmark
 * parameters, either read from a file according to the command-line args, or
 * the default ones.
 */
template <typename scalar_t>
inline std::vector<gemm_batched_strided_cplx_param_t<scalar_t>>
get_gemm_batched_strided_cplx_params(Args& args) {
  if (args.csv_param.empty()) {
    warning_no_csv();
    std::vector<gemm_batched_strided_cplx_param_t<scalar_t>>
        gemm_batched_strided_default;
    constexpr index_t dmin = 128, dmax = 8192;
    std::vector<std::string> dtranspose = {"n", "t"};
    std::complex<scalar_t> alpha{1, 1};
    std::complex<scalar_t> beta{1, 1};
    index_t batch_size = 8;
    for (std::string& t1 : dtranspose) {
      for (std::string& t2 : dtranspose) {
        for (index_t m = dmin; m <= dmax; m *= 8) {
          gemm_batched_strided_default.push_back(
              std::make_tuple(t1, t2, m, m, m, alpha.real(), alpha.imag(),
                              beta.real(), beta.imag(), batch_size, 2, 2, 2));
        }
      }
    }
    return gemm_batched_strided_default;
  } else {
    return parse_csv_file<gemm_batched_strided_cplx_param_t<scalar_t>>(
        args.csv_param, [&](std::vector<std::string>& v) {
          if (v.size() != 13) {
            throw std::runtime_error(
                "invalid number of parameters (13 expected)");
          }
          try {
            return std::make_tuple(
                v[0].c_str(), v[1].c_str(), str_to_int<index_t>(v[2]),
                str_to_int<index_t>(v[3]), str_to_int<index_t>(v[4]),
                str_to_scalar<scalar_t>(v[5]), str_to_scalar<scalar_t>(v[6]),
                str_to_scalar<scalar_t>(v[7]), str_to_scalar<scalar_t>(v[8]),
                str_to_int<index_t>(v[9]), str_to_int<index_t>(v[10]),
                str_to_int<index_t>(v[11]), str_to_int<index_t>(v[12]));
          } catch (...) {
            std::throw_with_nested(std::runtime_error("invalid parameter"));
          }
        });
  }
}

/**
 * @fn get_gemm_cplx_batched_params
 * @brief Returns a vector containing the gemm_batched cplx benchmark
 * parameters, either read from a file according to the command-line args, or
 * the default ones.
 */
template <typename scalar_t>
inline std::vector<gemm_batched_cplx_param_t<scalar_t>>
get_gemm_cplx_batched_params(Args& args) {
  if (args.csv_param.empty()) {
    warning_no_csv();
    std::vector<gemm_batched_cplx_param_t<scalar_t>> gemm_batched_default;
    constexpr index_t dmin = 128, dmax = 8192;
    std::vector<std::string> dtranspose = {"n", "t"};
    std::complex<scalar_t> alpha{1, 1};
    std::complex<scalar_t> beta{1, 1};
    index_t batch_size = 8;
    int batch_type = 0;
    for (std::string& t1 : dtranspose) {
      for (std::string& t2 : dtranspose) {
        for (index_t n = dmin; n <= dmax; n *= 8) {
          gemm_batched_default.push_back(std::make_tuple(
              t1, t2, n, n, n, alpha.real(), alpha.imag(), beta.real(),
              beta.imag(), batch_size, batch_type));
        }
      }
    }
    return gemm_batched_default;
  } else {
    return parse_csv_file<gemm_batched_cplx_param_t<scalar_t>>(
        args.csv_param, [&](std::vector<std::string>& v) {
          if (v.size() != 11) {
            throw std::runtime_error(
                "invalid number of parameters (11 expected)");
          }
          try {
            return std::make_tuple(
                v[0].c_str(), v[1].c_str(), str_to_int<index_t>(v[2]),
                str_to_int<index_t>(v[3]), str_to_int<index_t>(v[4]),
                str_to_scalar<scalar_t>(v[5]), str_to_scalar<scalar_t>(v[6]),
                str_to_scalar<scalar_t>(v[7]), str_to_scalar<scalar_t>(v[8]),
                str_to_int<index_t>(v[9]), str_to_batch_type(v[10]));
          } catch (...) {
            std::throw_with_nested(std::runtime_error("invalid parameter"));
          }
        });
  }
}
#endif

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
    constexpr index_t dmin = 128, dmax = 8192;
    std::vector<std::string> dtranspose = {"n", "t"};
    scalar_t alpha = 1;
    scalar_t beta = 1;
    index_t batch_size = 8;
    int batch_type = 0;
    for (std::string& t1 : dtranspose) {
      for (std::string& t2 : dtranspose) {
        for (index_t n = dmin; n <= dmax; n *= 8) {
          gemm_batched_default.push_back(std::make_tuple(
              t1, t2, n, n, n, alpha, beta, batch_size, batch_type));
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
 * @fn get_gemm_batched_strided_params
 * @brief Returns a vector containing the gemm_batched_strided benchmark
 * parameters, either read from a file according to the command-line args, or
 * the default ones.
 */
template <typename scalar_t>
inline std::vector<gemm_batched_strided_param_t<scalar_t>>
get_gemm_batched_strided_params(Args& args) {
  if (args.csv_param.empty()) {
    warning_no_csv();
    std::vector<gemm_batched_strided_param_t<scalar_t>>
        gemm_batched_strided_default;
    constexpr index_t dmin = 128, dmax = 8192;
    std::vector<std::string> dtranspose = {"n", "t"};
    scalar_t alpha = 1;
    scalar_t beta = 1;
    index_t batch_size = 8;
    for (std::string& t1 : dtranspose) {
      for (std::string& t2 : dtranspose) {
        for (index_t m = dmin; m <= dmax; m *= 8) {
          gemm_batched_strided_default.push_back(std::make_tuple(
              t1, t2, m, m, m, alpha, beta, batch_size, 2, 2, 2));
        }
      }
    }
    return gemm_batched_strided_default;
  } else {
    return parse_csv_file<gemm_batched_strided_param_t<scalar_t>>(
        args.csv_param, [&](std::vector<std::string>& v) {
          if (v.size() != 11) {
            throw std::runtime_error(
                "invalid number of parameters (11 expected)");
          }
          try {
            return std::make_tuple(
                v[0].c_str(), v[1].c_str(), str_to_int<index_t>(v[2]),
                str_to_int<index_t>(v[3]), str_to_int<index_t>(v[4]),
                str_to_scalar<scalar_t>(v[5]), str_to_scalar<scalar_t>(v[6]),
                str_to_int<index_t>(v[7]), str_to_int<index_t>(v[8]),
                str_to_int<index_t>(v[9]), str_to_int<index_t>(v[10]));
          } catch (...) {
            std::throw_with_nested(std::runtime_error("invalid parameter"));
          }
        });
  }
}

/**
 * @fn get_trsm_batched_params
 * @brief Returns a vector containing the trsm_batched benchmark parameters,
 * either read from a file according to the command-line args, or the default
 * ones.
 */
template <typename scalar_t>
static inline std::vector<trsm_batched_param_t<scalar_t>>
get_trsm_batched_params(Args& args) {
  if (args.csv_param.empty()) {
    warning_no_csv();
    std::vector<trsm_batched_param_t<scalar_t>> trsm_batched_default;
    constexpr index_t dmin = 512, dmax = 8192;
    // Stride Multipliers are set by default and correspond to default striding
    constexpr index_t stride_a_mul = 1;
    constexpr index_t stride_b_mul = 1;
    constexpr index_t batch_size = 8;
    constexpr scalar_t alpha = 1;
    for (char side : {'l', 'r'}) {
      for (char uplo : {'u', 'l'}) {
        for (char trans : {'n', 't'}) {
          for (char diag : {'u', 'n'}) {
            for (index_t n = dmin; n <= dmax; n *= 4) {
              trsm_batched_default.push_back(
                  std::make_tuple(side, uplo, trans, diag, n, n, alpha,
                                  batch_size, stride_a_mul, stride_b_mul));
            }
          }
        }
      }
    }
    return trsm_batched_default;
  } else {
    return parse_csv_file<trsm_batched_param_t<scalar_t>>(
        args.csv_param, [&](std::vector<std::string>& v) {
          if (v.size() != 10) {
            throw std::runtime_error(
                "invalid number of parameters (10 expected)");
          }
          try {
            return std::make_tuple(
                v[0][0], v[1][0], v[2][0], v[3][0], str_to_int<index_t>(v[4]),
                str_to_int<index_t>(v[5]), str_to_scalar<scalar_t>(v[6]),
                str_to_int<index_t>(v[7]), str_to_int<index_t>(v[8]),
                str_to_int<index_t>(v[9]));
          } catch (...) {
            throw std::runtime_error("invalid parameter");
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
    constexpr index_t dmin = 256, dmax = 8192;
    for (index_t rows = dmin; rows <= dmax; rows *= 2) {
      reduction_default.push_back(std::make_tuple(rows, rows));
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
 * @fn get_symm_params
 * @brief Returns a vector containing the symm benchmark parameters, either
 * read from a file according to the command-line args, or the default ones.
 */
template <typename scalar_t>
static inline std::vector<symm_param_t<scalar_t>> get_symm_params(Args& args) {
  if (args.csv_param.empty()) {
    utils::warning_no_csv();
    std::vector<symm_param_t<scalar_t>> symm_default;
    constexpr index_t dmin = 32, dmax = 8192;
    constexpr scalar_t alpha{1};
    constexpr scalar_t beta{1};
    for (char side : {'l', 'r'}) {
      for (char uplo : {'u', 'l'}) {
        for (index_t m = dmin; m <= dmax; m *= 4) {
          symm_default.push_back(
              std::make_tuple(side, uplo, m, m, alpha, beta));
        }
      }
    }
    return symm_default;
  } else {
    return parse_csv_file<symm_param_t<scalar_t>>(
        args.csv_param, [&](std::vector<std::string>& v) {
          if (v.size() != 6) {
            throw std::runtime_error(
                "invalid number of parameters (6 expected)");
          }
          try {
            return std::make_tuple(
                v[0][0], v[1][0], utils::str_to_int<index_t>(v[2]),
                utils::str_to_int<index_t>(v[3]), str_to_scalar<scalar_t>(v[4]),
                str_to_scalar<scalar_t>(v[5]));
          } catch (...) {
            throw std::runtime_error("invalid parameter");
          }
        });
  }
}

/**
 * @fn get_syrk_params
 * @brief Returns a vector containing the syrk benchmark parameters, either
 * read from a file according to the command-line args, or the default ones.
 */
template <typename scalar_t>
static inline std::vector<syrk_param_t<scalar_t>> get_syrk_params(Args& args) {
  if (args.csv_param.empty()) {
    warning_no_csv();
    std::vector<syrk_param_t<scalar_t>> syrk_default;
    constexpr index_t dmin = 512, dmax = 8192;
    constexpr scalar_t alpha{1};
    constexpr scalar_t beta{1};
    for (char uplo : {'u', 'l'}) {
      for (char trans : {'n', 't'}) {
        for (index_t n = dmin; n <= dmax; n *= 4) {
          syrk_default.push_back(
              std::make_tuple(uplo, trans, n, n, alpha, beta));
        }
      }
    }
    return syrk_default;
  } else {
    return parse_csv_file<syrk_param_t<scalar_t>>(
        args.csv_param, [&](std::vector<std::string>& v) {
          if (v.size() != 6) {
            throw std::runtime_error(
                "invalid number of parameters (6 expected)");
          }
          try {
            return std::make_tuple(v[0][0], v[1][0], str_to_int<index_t>(v[2]),
                                   str_to_int<index_t>(v[3]),
                                   str_to_scalar<scalar_t>(v[4]),
                                   str_to_scalar<scalar_t>(v[5]));
          } catch (...) {
            throw std::runtime_error("invalid parameter");
          }
        });
  }
}
/**
 * @fn get_trsm_params
 * @brief Returns a vector containing the trsm benchmark parameters (also valid
 * for trmm), either read from a file according to the command-line args, or the
 * default ones.
 */
template <typename scalar_t>
static inline std::vector<trsm_param_t<scalar_t>> get_trsm_params(Args& args) {
  if (args.csv_param.empty()) {
    warning_no_csv();
    std::vector<trsm_param_t<scalar_t>> trsm_default;
    constexpr index_t dmin = 512, dmax = 8192;
    for (char side : {'l', 'r'}) {
      for (char uplo : {'u', 'l'}) {
        for (char trans : {'n', 't'}) {
          for (char diag : {'u', 'n'}) {
            for (index_t m = dmin; m <= dmax; m *= 4) {
              trsm_default.push_back(
                  std::make_tuple(side, uplo, trans, diag, m, m, scalar_t{1}));
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
    constexpr index_t dmin = 512, dmax = 8192;
    scalar_t alpha = 1;
    scalar_t beta = 0;
    for (std::string t : {"n", "t"}) {
      for (index_t m = dmin; m <= dmax; m *= 4) {
        for (index_t kl = m / 32; kl <= m / 4; kl *= 2) {
          gbmv_default.push_back(std::make_tuple(t, m, m, kl, kl, alpha, beta));
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
    constexpr index_t dmin = 512, dmax = 8192;
    scalar_t alpha = 1;
    scalar_t beta = 1;
    for (std::string ul : {"u", "l"}) {
      for (index_t n = dmin; n <= dmax; n *= 4) {
        for (index_t k = n / 32; k <= n / 4; k *= 2) {
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
 * @fn get_symv_params
 * @brief Returns a vector containing the symv (also valid for spmv) benchmark
 * parameters, either read from a file according to the command-line args, or
 * the default ones.
 */
template <typename scalar_t>
static inline std::vector<symv_param_t<scalar_t>> get_symv_params(Args& args) {
  if (args.csv_param.empty()) {
    warning_no_csv();
    std::vector<symv_param_t<scalar_t>> symv_default;
    constexpr index_t dmin = 32, dmax = 8192;
    scalar_t alpha = 1;
    scalar_t beta = 1;
    for (std::string uplo : {"u", "l"}) {
      for (index_t n = dmin; n <= dmax; n *= 4) {
        symv_default.push_back(std::make_tuple(uplo, n, alpha, beta));
      }
    }
    return symv_default;
  } else {
    return parse_csv_file<symv_param_t<scalar_t>>(
        args.csv_param, [&](std::vector<std::string>& v) {
          if (v.size() != 4) {
            throw std::runtime_error(
                "invalid number of parameters (4 expected)");
          }
          try {
            return std::make_tuple(v[0].c_str(), str_to_int<index_t>(v[1]),
                                   str_to_scalar<scalar_t>(v[2]),
                                   str_to_scalar<scalar_t>(v[3]));
          } catch (...) {
            throw std::runtime_error("invalid parameter");
          }
        });
  }
}

/**
 * @fn get_syr_params
 * @brief Returns a vector containing the syr (also valid for syr2 and spr2)
 * benchmark parameters, either read from a file according to the command-line
 * args, or the default ones.
 */
template <typename scalar_t>
static inline std::vector<syr_param_t<scalar_t>> get_syr_params(Args& args) {
  if (args.csv_param.empty()) {
    warning_no_csv();
    std::vector<syr_param_t<scalar_t>> syr_default;
    constexpr index_t dmin = 32, dmax = 8192;
    scalar_t alpha = 1;
    for (std::string uplo : {"u", "l"}) {
      for (index_t n = dmin; n <= dmax; n *= 4) {
        syr_default.push_back(std::make_tuple(uplo, n, alpha));
      }
    }
    return syr_default;
  } else {
    return parse_csv_file<syr_param_t<scalar_t>>(
        args.csv_param, [&](std::vector<std::string>& v) {
          if (v.size() != 3) {
            throw std::runtime_error(
                "invalid number of parameters (3 expected)");
          }
          try {
            return std::make_tuple(v[0].c_str(), str_to_int<index_t>(v[1]),
                                   str_to_scalar<scalar_t>(v[2]));
          } catch (...) {
            throw std::runtime_error("invalid parameter");
          }
        });
  }
}

/**
 * @fn get_ger_params
 * @brief Returns a vector containing the ger benchmark parameters, either
 * read from a file according to the command-line args, or the default ones.
 */
template <typename scalar_t>
static inline std::vector<ger_param_t<scalar_t>> get_ger_params(Args& args) {
  if (args.csv_param.empty()) {
    warning_no_csv();
    std::vector<ger_param_t<scalar_t>> ger_default;
    constexpr index_t dmin = 32, dmax = 8192;
    scalar_t alpha = 1;
    for (index_t m = dmin; m <= dmax; m *= 2) {
      ger_default.push_back(std::make_tuple(m, m, alpha));
    }
    return ger_default;
  } else {
    return parse_csv_file<ger_param_t<scalar_t>>(
        args.csv_param, [&](std::vector<std::string>& v) {
          if (v.size() != 3) {
            throw std::runtime_error(
                "invalid number of parameters (3 expected)");
          }
          try {
            return std::make_tuple(str_to_int<index_t>(v[0]),
                                   str_to_int<index_t>(v[1]),
                                   str_to_scalar<scalar_t>(v[2]));
          } catch (...) {
            throw std::runtime_error("invalid parameter");
          }
        });
  }
}

/**
 * @fn get_tbmv_params
 * @brief Returns a vector containing the tbmv (also valid for tbsv) benchmark
 * parameters, either read from a file according to the command-line args, or
 * the default ones.
 */
static inline std::vector<tbmv_param_t> get_tbmv_params(Args& args) {
  if (args.csv_param.empty()) {
    warning_no_csv();
    std::vector<tbmv_param_t> tbmv_default;
    constexpr index_t dmin = 128, dmax = 8192;
    for (std::string t : {"n", "t"}) {
      for (std::string ul : {"u", "l"}) {
        for (std::string diag : {"n", "u"}) {
          for (index_t n = dmin; n <= dmax; n *= 8) {
            for (index_t k = n / 32; k <= n / 4; k *= 2) {
              tbmv_default.push_back(std::make_tuple(ul, t, diag, n, k));
            }
          }
        }
      }
    }
    return tbmv_default;
  } else {
    return parse_csv_file<tbmv_param_t>(
        args.csv_param, [&](std::vector<std::string>& v) {
          if (v.size() != 5) {
            throw std::runtime_error(
                "invalid number of parameters (5 expected)");
          }
          try {
            return std::make_tuple(v[0].c_str(), v[1].c_str(), v[2].c_str(),
                                   str_to_int<index_t>(v[3]),
                                   str_to_int<index_t>(v[4]));
          } catch (...) {
            throw std::runtime_error("invalid parameter");
          }
        });
  }
}

/**
 * @fn get_trsv_params
 * @brief Returns a vector containing the trsv (also valid for trmv, tpmv and
 * tpmv) benchmark parameters, either read from a file according to the
 * command-line args, or the default ones.
 */
static inline std::vector<trsv_param_t> get_trsv_params(Args& args) {
  if (args.csv_param.empty()) {
    warning_no_csv();
    std::vector<trsv_param_t> trsv_default;
    constexpr index_t dmin = 128, dmax = 8192;
    for (std::string t : {"n", "t"}) {
      for (std::string ul : {"u", "l"}) {
        for (std::string diag : {"n", "u"}) {
          for (index_t n = dmin; n <= dmax; n *= 8) {
            trsv_default.push_back(std::make_tuple(ul, t, diag, n));
          }
        }
      }
    }
    return trsv_default;
  } else {
    return parse_csv_file<trsv_param_t>(
        args.csv_param, [&](std::vector<std::string>& v) {
          if (v.size() != 4) {
            throw std::runtime_error(
                "invalid number of parameters (4 expected)");
          }
          try {
            return std::make_tuple(v[0].c_str(), v[1].c_str(), v[2].c_str(),
                                   str_to_int<index_t>(v[3]));
          } catch (...) {
            throw std::runtime_error("invalid parameter");
          }
        });
  }
}

/**
 * @fn get_matcopy_params
 * @brief Returns a vector containing the matcopy benchmark parameters, either
 * read from a file according to the command-line args, or the default ones.
 */
template <typename scalar_t>
static inline std::vector<matcopy_param_t<scalar_t>> get_matcopy_params(
    Args& args) {
  if (args.csv_param.empty()) {
    warning_no_csv();
    std::vector<matcopy_param_t<scalar_t>> matcopy_default;
    constexpr index_t dmin = 64, dmax = 8192;
    constexpr scalar_t alpha{2};
    for (char trans : {'n', 't'}) {
      for (index_t m = dmin; m <= dmax; m *= 2) {
        for (index_t n = dmin; n <= dmax; n *= 2) {
          for (index_t lda_mul = 1; lda_mul < 2; ++lda_mul) {
            for (index_t ldb_mul = 1; ldb_mul < 2; ++ldb_mul) {
              matcopy_default.push_back(
                  std::make_tuple(trans, m, n, alpha, lda_mul, ldb_mul));
            }
          }
        }
      }
    }
    return matcopy_default;
  } else {
    return parse_csv_file<matcopy_param_t<scalar_t>>(
        args.csv_param, [&](std::vector<std::string>& v) {
          if (v.size() != 6) {
            throw std::runtime_error(
                "invalid number of parameters (6 expected)");
          }
          try {
            return std::make_tuple(
                v[0][0], str_to_int<index_t>(v[1]), str_to_int<index_t>(v[2]),
                str_to_scalar<scalar_t>(v[3]), str_to_int<index_t>(v[4]),
                str_to_int<index_t>(v[5]));
          } catch (...) {
            throw std::runtime_error("invalid parameter");
          }
        });
  }
}

/**
 * @fn get_omatcopy2_params
 * @brief Returns a vector containing the omatcopy2 benchmark parameters, either
 * read from a file according to the command-line args, or the default ones.
 */
template <typename scalar_t>
static inline std::vector<omatcopy2_param_t<scalar_t>> get_omatcopy2_params(
    Args& args) {
  if (args.csv_param.empty()) {
    warning_no_csv();
    std::vector<omatcopy2_param_t<scalar_t>> omatcopy2_default;
    constexpr index_t dmin = 1024, dmax = 8192;
    constexpr scalar_t alpha{2};
    for (char trans : {'n', 't'}) {
      for (index_t m = dmin; m <= dmax; m *= 2) {
        for (index_t n = dmin; n <= dmax; n *= 2) {
          for (index_t lda_mul = 1; lda_mul < 2; ++lda_mul) {
            for (index_t inc_a = 1; inc_a < 3; ++inc_a) {
              for (index_t ldb_mul = 1; ldb_mul < 2; ++ldb_mul) {
                for (index_t inc_b = 1; inc_b < 3; ++inc_b) {
                  omatcopy2_default.push_back(std::make_tuple(
                      trans, m, n, alpha, lda_mul, ldb_mul, inc_a, inc_b));
                }
              }
            }
          }
        }
      }
    }
    return omatcopy2_default;
  } else {
    return parse_csv_file<omatcopy2_param_t<scalar_t>>(
        args.csv_param, [&](std::vector<std::string>& v) {
          if (v.size() != 8) {
            throw std::runtime_error(
                "invalid number of parameters (8 expected)");
          }
          try {
            return std::make_tuple(
                v[0][0], str_to_int<index_t>(v[1]), str_to_int<index_t>(v[2]),
                str_to_scalar<scalar_t>(v[3]), str_to_int<index_t>(v[4]),
                str_to_int<index_t>(v[5]), str_to_int<index_t>(v[6]),
                str_to_int<index_t>(v[7]));
          } catch (...) {
            throw std::runtime_error("invalid parameter");
          }
        });
  }
}

/**
 * @fn get_matcopy_batch_params
 * @brief Returns a vector containing the matcopy_batch benchmark parameters,
 * either read from a file according to the command-line args, or the default
 * ones.
 */
template <typename scalar_t>
static inline std::vector<matcopy_batch_param_t<scalar_t>>
get_matcopy_batch_params(Args& args) {
  if (args.csv_param.empty()) {
    warning_no_csv();
    std::vector<matcopy_batch_param_t<scalar_t>> matcopy_batch_default;
    constexpr index_t dmin = 256, dmax = 8192;
    constexpr scalar_t alpha{2};
    constexpr index_t batch_size{3};
    constexpr index_t stride_a_mul{1};
    constexpr index_t stride_b_mul{1};
    for (char trans : {'n', 't'}) {
      for (index_t m = dmin; m <= dmax; m *= 2) {
        for (index_t n = dmin; n <= dmax; n *= 2) {
          for (index_t lda_mul = 1; lda_mul < 2; ++lda_mul) {
            for (index_t ldb_mul = 1; ldb_mul < 2; ++ldb_mul) {
              matcopy_batch_default.push_back(
                  std::make_tuple(trans, m, n, alpha, lda_mul, ldb_mul,
                                  stride_a_mul, stride_b_mul, batch_size));
            }
          }
        }
      }
    }
    return matcopy_batch_default;
  } else {
    return parse_csv_file<matcopy_batch_param_t<scalar_t>>(
        args.csv_param, [&](std::vector<std::string>& v) {
          if (v.size() != 9) {
            throw std::runtime_error(
                "invalid number of parameters (9 expected)");
          }
          try {
            return std::make_tuple(
                v[0][0], str_to_int<index_t>(v[1]), str_to_int<index_t>(v[2]),
                str_to_scalar<scalar_t>(v[3]), str_to_int<index_t>(v[4]),
                str_to_int<index_t>(v[5]), str_to_int<index_t>(v[6]),
                str_to_int<index_t>(v[7]), str_to_int<index_t>(v[8]));
          } catch (...) {
            throw std::runtime_error("invalid parameter");
          }
        });
  }
}

/**
 * @fn get_omatadd_params
 * @brief Returns a vector containing the omatadd benchmark parameters, either
 * read from a file according to the command-line args, or the default ones.
 */
template <typename scalar_t>
static inline std::vector<omatadd_param_t<scalar_t>> get_omatadd_params(
    Args& args) {
  if (args.csv_param.empty()) {
    warning_no_csv();
    std::vector<omatadd_param_t<scalar_t>> omatadd_default;
    constexpr index_t dmin = 64, dmax = 8192;
    constexpr scalar_t alpha{2};
    constexpr scalar_t beta{2};
    for (char trans_a : {'n', 't'}) {
      for (char trans_b : {'n', 't'}) {
        for (index_t m = dmin; m <= dmax; m *= 2) {
          for (index_t n = dmin; n <= dmax; n *= 2) {
            for (index_t lda_mul = 1; lda_mul < 2; ++lda_mul) {
              for (index_t ldb_mul = 1; ldb_mul < 2; ++ldb_mul) {
                for (index_t ldc_mul = 1; ldc_mul < 2; ++ldc_mul) {
                  omatadd_default.push_back(
                      std::make_tuple(trans_a, trans_b, m, n, alpha, beta,
                                      lda_mul, ldb_mul, ldc_mul));
                }
              }
            }
          }
        }
      }
    }
    return omatadd_default;
  } else {
    return parse_csv_file<omatadd_param_t<scalar_t>>(
        args.csv_param, [&](std::vector<std::string>& v) {
          if (v.size() != 9) {
            throw std::runtime_error(
                "invalid number of parameters (9 expected)");
          }
          try {
            return std::make_tuple(
                v[0][0], v[1][0], str_to_int<index_t>(v[2]),
                str_to_int<index_t>(v[3]), str_to_scalar<scalar_t>(v[4]),
                str_to_scalar<scalar_t>(v[5]), str_to_int<index_t>(v[6]),
                str_to_int<index_t>(v[7]), str_to_int<index_t>(v[8]));
          } catch (...) {
            throw std::runtime_error("invalid parameter");
          }
        });
  }
}

/**
 * @fn get_omatadd_batch_params
 * @brief Returns a vector containing the omatadd_batch benchmark parameters,
 * either read from a file according to the command-line args, or the default
 * ones.
 */
template <typename scalar_t>
static inline std::vector<omatadd_batch_param_t<scalar_t>>
get_omatadd_batch_params(Args& args) {
  if (args.csv_param.empty()) {
    warning_no_csv();
    std::vector<omatadd_batch_param_t<scalar_t>> omatadd_batch_default;
    constexpr index_t dmin = 256, dmax = 8192;
    constexpr scalar_t alpha{2};
    constexpr scalar_t beta{2};
    constexpr index_t batch_size{3};
    constexpr index_t stride_a_mul{1};
    constexpr index_t stride_b_mul{1};
    constexpr index_t stride_c_mul{1};
    for (char trans_a : {'n', 't'}) {
      for (char trans_b : {'n', 't'}) {
        for (index_t m = dmin; m <= dmax; m *= 2) {
          for (index_t n = dmin; n <= dmax; n *= 2) {
            for (index_t lda_mul = 1; lda_mul < 2; ++lda_mul) {
              for (index_t ldb_mul = 1; ldb_mul < 2; ++ldb_mul) {
                for (index_t ldc_mul = 1; ldc_mul < 2; ++ldc_mul) {
                  omatadd_batch_default.push_back(
                      std::make_tuple(trans_a, trans_b, m, n, alpha, beta,
                                      lda_mul, ldb_mul, ldc_mul, stride_a_mul,
                                      stride_b_mul, stride_c_mul, batch_size));
                }
              }
            }
          }
        }
      }
    }
    return omatadd_batch_default;
  } else {
    return parse_csv_file<omatadd_batch_param_t<scalar_t>>(
        args.csv_param, [&](std::vector<std::string>& v) {
          if (v.size() != 13) {
            throw std::runtime_error(
                "invalid number of parameters (13 expected)");
          }
          try {
            return std::make_tuple(
                v[0][0], v[1][0], str_to_int<index_t>(v[2]),
                str_to_int<index_t>(v[3]), str_to_scalar<scalar_t>(v[4]),
                str_to_scalar<scalar_t>(v[5]), str_to_int<index_t>(v[6]),
                str_to_int<index_t>(v[7]), str_to_int<index_t>(v[8]),
                str_to_int<index_t>(v[9]), str_to_int<index_t>(v[10]),
                str_to_int<index_t>(v[11]), str_to_int<index_t>(v[12]));
          } catch (...) {
            throw std::runtime_error("invalid parameter");
          }
        });
  }
}

/**
 * @fn get_axpy_batch_params
 * @brief Returns a vector containing the axpy_batch benchmark parameters,
 * either read from a file according to the command-line args, or the default
 * ones.
 */
template <typename scalar_t>
static inline std::vector<axpy_batch_param_t<scalar_t>> get_axpy_batch_params(
    Args& args) {
  if (args.csv_param.empty()) {
    warning_no_csv();
    std::vector<axpy_batch_param_t<scalar_t>> axpy_batch_default;
    constexpr index_t dmin = 1 << 10, dmax = 1 << 22;
    constexpr index_t batch_size{5};
    constexpr index_t incX{1};
    constexpr index_t incY{1};
    constexpr index_t stride_x_mul{1};
    constexpr index_t stride_y_mul{1};
    constexpr scalar_t alpha{1};
    for (auto n = dmin; n <= dmax; n *= 2) {
      axpy_batch_default.push_back(std::make_tuple(
          n, alpha, incX, incY, stride_x_mul, stride_y_mul, batch_size));
    }
    return axpy_batch_default;
  } else {
    return parse_csv_file<axpy_batch_param_t<scalar_t>>(
        args.csv_param, [&](std::vector<std::string>& v) {
          if (v.size() != 7) {
            throw std::runtime_error(
                "invalid number of parameters (7 expected)");
          }
          try {
            return std::make_tuple(
                str_to_int<index_t>(v[0]), str_to_scalar<scalar_t>(v[1]),
                str_to_int<index_t>(v[2]), str_to_int<index_t>(v[3]),
                str_to_int<index_t>(v[4]), str_to_int<index_t>(v[5]),
                str_to_int<index_t>(v[6]));
          } catch (...) {
            throw std::runtime_error("invalid parameter");
          }
        });
  }
}
/**
 * @fn get_type_name
 * @brief Returns a string with the given type. The C++ specification
 * doesn't guarantee that typeid(T).name is human readable so we specify the
 * template for float and double.
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

#ifdef BLAS_ENABLE_COMPLEX
template <>
inline std::string get_type_name<std::complex<float>>() {
  return "complex<float>";
}
template <>
inline std::string get_type_name<std::complex<double>>() {
  return "complex<double>";
}
#endif

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

#ifdef BLAS_ENABLE_COMPLEX
/**
 * @fn random_cplx_scalar
 * @brief Generates a random complex value, using an arbitrary low quality
 * algorithm.
 */
template <typename scalar_t>
static inline std::complex<scalar_t> random_cplx_scalar() {
  scalar_t rl = 1e-3 * ((rand() % 2000) - 1000);
  scalar_t im = 1e-3 * ((rand() % 2000) - 1000);
  return std::complex<scalar_t>(rl, im);
}

/**
 * @brief Generates a random complex in the specified range of its underlying
 * data elements (real & imag)
 * @param rangeMin range minimum
 * @param rangeMax range maximum
 */
template <typename scalar_t>
static inline std::complex<scalar_t> random_cplx_scalar(scalar_t rangeMin,
                                                        scalar_t rangeMax) {
  static std::random_device rd;
  static std::default_random_engine gen(rd());
  std::uniform_real_distribution<scalar_t> disRl(rangeMin, rangeMax);
  std::uniform_real_distribution<scalar_t> disIm(rangeMin, rangeMax);

  return std::complex<scalar_t>(disRl(gen), disIm(gen));
}

/**
 * @fn random_cplx_data
 * @brief Generates a random vector of complex values, using a uniform
 * distribution of the underlying data elements (real & imag).
 */
template <typename scalar_t>
static inline std::vector<std::complex<scalar_t>> random_cplx_data(
    size_t size) {
  std::vector<std::complex<scalar_t>> v(size);

  for (std::complex<scalar_t>& e : v) {
    e = random_cplx_scalar<scalar_t>(scalar_t{-2}, scalar_t{5});
  }
  return v;
}

/**
 * @fn const_cplx_data
 * @brief Generates a vector of constant complex values, of a given length.
 */
template <typename scalar_t>
static inline std::vector<std::complex<scalar_t>> const_cplx_data(
    size_t size, scalar_t const_value = 0) {
  std::vector<std::complex<scalar_t>> v(size);
  std::complex<scalar_t> const_cplx_value{const_value, const_value};
  std::fill(v.begin(), v.end(), const_cplx_value);
  return v;
}

#endif  // BLAS_ENABLE_COMPLEX

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
 * (both overall and event time, returned in nanoseconds in a tuple of
 * double)
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

#ifdef BLAS_ENABLE_COMPLEX
/** Registers benchmark for the float complex data type
 * @see BLAS_REGISTER_BENCHMARK
 */
#define BLAS_REGISTER_BENCHMARK_CPLX_FLOAT(args, sb_handle_ptr, success) \
  register_cplx_benchmark<float>(args, sb_handle_ptr, success)
#else
#define BLAS_REGISTER_BENCHMARK_CPLX_FLOAT(args, sb_handle_ptr, success)
#endif

#if defined(BLAS_ENABLE_COMPLEX) & defined(BLAS_DATA_TYPE_DOUBLE)
/** Registers benchmark for the double complex data type
 * @see BLAS_REGISTER_BENCHMARK
 */
#define BLAS_REGISTER_BENCHMARK_CPLX_DOUBLE(args, sb_handle_ptr, success) \
  register_cplx_benchmark<double>(args, sb_handle_ptr, success)
#else
#define BLAS_REGISTER_BENCHMARK_CPLX_DOUBLE(args, sb_handle_ptr, success)
#endif

/** Registers benchmark for all supported data types.
 *  Expects register_benchmark<scalar_t> to exist.
 * @param args Reference to blas_benchmark::Args
 * @param sb_handle_ptr Pointer to blas::SB_Handle
 * @param[out] success Pointer to boolean indicating success
 */
#define BLAS_REGISTER_BENCHMARK(args, sb_handle_ptr, success)          \
  do {                                                                 \
    BLAS_REGISTER_BENCHMARK_FLOAT(args, sb_handle_ptr, success);       \
    BLAS_REGISTER_BENCHMARK_DOUBLE(args, sb_handle_ptr, success);      \
    BLAS_REGISTER_BENCHMARK_HALF(args, sb_handle_ptr, success);        \
    BLAS_REGISTER_BENCHMARK_CPLX_FLOAT(args, sb_handle_ptr, success);  \
    BLAS_REGISTER_BENCHMARK_CPLX_DOUBLE(args, sb_handle_ptr, success); \
  } while (false)

#endif
