/*
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
 *  @filename benchmark_names.hpp
 *
 */

#ifndef SYCL_BLAS_BENCHMARK_NAMES_HPP
#define SYCL_BLAS_BENCHMARK_NAMES_HPP

#include <common/common_utils.hpp>

namespace blas_benchmark {
namespace utils {

template <typename scalar_t>
static inline std::string get_type_name();

namespace internal {

template <typename T>
std::string get_parameters_as_string(T arg) {
  std::ostringstream str{};
  str << arg;
  return str.str();
}

template <typename T, typename... Args>
std::string get_parameters_as_string(T arg, Args... args) {
  std::ostringstream str{};
  str << arg << "/" << get_parameters_as_string(args...);
  return str.str();
}

template <typename scalar_t>
inline std::string get_benchmark_name(const std::string &operator_name) {
  std::ostringstream str{};
  str << "BM_" << operator_name << "<" << get_type_name<scalar_t>() << ">";
  return str.str();
}

template <Level1Op op, typename scalar_t, typename... Args>
inline std::string get_name() {
  return get_benchmark_name<scalar_t>(get_operator_name<op>());
}

template <Level1Op op, typename scalar_t, typename... Args>
inline std::string get_name(Args... args) {
  std::ostringstream str{};
  str << get_name<op, scalar_t>() << "/";
  str << get_parameters_as_string(args...);
  return str.str();
}
}  // namespace internal

template <Level1Op op, typename scalar_t>
inline typename std::enable_if<op == Level1Op::rotg || op == Level1Op::rotmg,
                               std::string>::type
get_name(std::string mem_type) {
  return internal::get_name<op, scalar_t>(mem_type);
}

template <Level1Op op, typename scalar_t, typename index_t>
inline
    typename std::enable_if<op == Level1Op::asum || op == Level1Op::axpy ||
                                op == Level1Op::dot || op == Level1Op::iamax ||
                                op == Level1Op::iamin || op == Level1Op::nrm2 ||
                                op == Level1Op::rotm || op == Level1Op::scal ||
                                op == Level1Op::sdsdot,
                            std::string>::type
    get_name(index_t size, std::string mem_type) {
  return internal::get_name<op, scalar_t>(size, mem_type);
}

template <Level1Op op, typename scalar_t, typename index_t>
inline typename std::enable_if<op == Level1Op::copy, std::string>::type
get_name(index_t size, index_t incx, index_t incy, std::string mem_type) {
  return internal::get_name<op, scalar_t>(size, incx, incy, mem_type);
}

template <Level2Op op, typename scalar_t, typename... Args>
inline std::string get_name(Args... args) {
  std::ostringstream str{};
  str << internal::get_benchmark_name<scalar_t>(get_operator_name<op>()) << "/";
  str << internal::get_parameters_as_string(args...);
  return str.str();
}

}  // namespace utils
}  // namespace blas_benchmark

#endif  // SYCL_BLAS_BENCHMARK_NAMES_HPP
