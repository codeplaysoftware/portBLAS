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
 *  portBLAS: BLAS implementation using SYCL
 *
 *  @filename benchmark_names.hpp
 *
 */

#ifndef PORTBLAS_BENCHMARK_NAMES_HPP
#define PORTBLAS_BENCHMARK_NAMES_HPP

#include <common/common_utils.hpp>

namespace blas_benchmark {
namespace utils {

template <typename scalar_t>
static inline std::string get_type_name();
inline std::string batch_type_to_str(int batch_type);

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

template <Level2Op op, typename scalar_t, typename... Args>
inline std::string get_name(Args... args) {
  std::ostringstream str{};
  str << get_benchmark_name<scalar_t>(get_operator_name<op>()) << "/";
  str << get_parameters_as_string(args...);
  return str.str();
}

template <Level3Op op, typename scalar_t, typename... Args>
inline std::string get_name(Args... args) {
  std::ostringstream str{};
  str << get_benchmark_name<scalar_t>(get_operator_name<op>()) << "/";
  str << get_parameters_as_string(args...);
  return str.str();
}

template <ExtensionOp op, typename scalar_t, typename... Args>
inline std::string get_name(Args... args) {
  std::ostringstream str{};
  str << get_benchmark_name<scalar_t>(get_operator_name<op>()) << "/";
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

template <Level2Op op, typename scalar_t, typename index_t>
inline typename std::enable_if<op == Level2Op::gbmv, std::string>::type
get_name(std::string t, index_t m, index_t n, index_t kl, index_t ku,
         std::string mem_type) {
  return internal::get_name<op, scalar_t>(t, m, n, kl, ku, mem_type);
}

template <Level2Op op, typename scalar_t, typename index_t>
inline typename std::enable_if<op == Level2Op::gemv || op == Level2Op::sbmv,
                               std::string>::type
get_name(std::string t, index_t m, index_t n, std::string mem_type) {
  return internal::get_name<op, scalar_t>(t, m, n, mem_type);
}

template <Level2Op op, typename scalar_t, typename index_t>
inline typename std::enable_if<op == Level2Op::ger, std::string>::type get_name(
    index_t m, index_t n, std::string mem_type) {
  return internal::get_name<op, scalar_t>(m, n, mem_type);
}

template <Level2Op op, typename scalar_t, typename index_t>
inline typename std::enable_if<op == Level2Op::syr || op == Level2Op::syr2,
                               std::string>::type
get_name(std::string uplo, index_t n, scalar_t alpha, std::string mem_type) {
  return internal::get_name<op, scalar_t>(uplo, n, alpha, mem_type);
}

template <Level2Op op, typename scalar_t, typename index_t>
inline typename std::enable_if<op == Level2Op::spr, std::string>::type get_name(
    std::string uplo, index_t n, scalar_t alpha, index_t incx,
    std::string mem_type) {
  return internal::get_name<op, scalar_t>(uplo, n, alpha, incx, mem_type);
}

template <Level2Op op, typename scalar_t, typename index_t>
inline typename std::enable_if<op == Level2Op::spmv || op == Level2Op::symv,
                               std::string>::type
get_name(std::string uplo, index_t n, scalar_t alpha, scalar_t beta,
         std::string mem_type) {
  return internal::get_name<op, scalar_t>(uplo, n, alpha, beta, mem_type);
}

template <Level2Op op, typename scalar_t, typename index_t>
inline typename std::enable_if<op == Level2Op::spr2, std::string>::type
get_name(std::string uplo, index_t n, scalar_t alpha, index_t incx,
         index_t incy, std::string mem_type) {
  return internal::get_name<op, scalar_t>(uplo, n, alpha, incx, incy, mem_type);
}

template <Level2Op op, typename scalar_t, typename index_t>
inline typename std::enable_if<op == Level2Op::tbmv || op == Level2Op::tbsv,
                               std::string>::type
get_name(std::string uplo, std::string t, std::string diag, index_t n,
         index_t k, std::string mem_type) {
  return internal::get_name<op, scalar_t>(uplo, t, diag, n, k, mem_type);
}

template <Level2Op op, typename scalar_t, typename index_t>
inline typename std::enable_if<op == Level2Op::tpmv || op == Level2Op::trmv ||
                                   op == Level2Op::trsv || op == Level2Op::tpsv,
                               std::string>::type
get_name(std::string uplo, std::string t, std::string diag, index_t n,
         std::string mem_type) {
  return internal::get_name<op, scalar_t>(uplo, t, diag, n, mem_type);
}

template <Level3Op op, typename scalar_t, typename index_t>
inline typename std::enable_if<op == Level3Op::gemm, std::string>::type
get_name(std::string t1, std::string t2, index_t m, index_t k, index_t n,
         std::string mem_type) {
  return internal::get_name<op, scalar_t>(t1, t2, m, k, n, mem_type);
}

template <Level3Op op, typename scalar_t, typename index_t>
inline typename std::enable_if<op == Level3Op::gemm_batched, std::string>::type
get_name(std::string t1, std::string t2, index_t m, index_t k, index_t n,
         index_t batch_size, int batch_type, std::string mem_type) {
  return internal::get_name<op, scalar_t>(
      t1, t2, m, k, n, batch_size, batch_type_to_str(batch_type), mem_type);
}

template <Level3Op op, typename scalar_t, typename index_t>
inline typename std::enable_if<op == Level3Op::gemm_batched_strided,
                               std::string>::type
get_name(std::string t1, std::string t2, index_t m, index_t k, index_t n,
         index_t batch_size, index_t stride_a_mul, index_t stride_b_mul,
         index_t stride_c_mul, std::string mem_type) {
  return internal::get_name<op, scalar_t>(t1, t2, m, k, n, batch_size,
                                          stride_a_mul, stride_b_mul,
                                          stride_c_mul, mem_type);
}

template <Level3Op op, typename scalar_t, typename index_t>
inline typename std::enable_if<op == Level3Op::symm || op == Level3Op::syr2k ||
                                   op == Level3Op::syrk,
                               std::string>::type
get_name(std::string s1, std::string s2, index_t m, index_t n, scalar_t alpha,
         scalar_t beta, std::string mem_type) {
  return internal::get_name<op, scalar_t>(s1, s2, m, n, alpha, beta, mem_type);
}

template <Level3Op op, typename scalar_t, typename index_t>
inline typename std::enable_if<op == Level3Op::trsm || op == Level3Op::trmm,
                               std::string>::type
get_name(char side, char uplo, char trans, char diag, index_t m, index_t n,
         std::string mem_type) {
  return internal::get_name<op, scalar_t>(side, uplo, trans, diag, m, n,
                                          mem_type);
}

template <Level3Op op, typename scalar_t, typename index_t>
inline typename std::enable_if<op == Level3Op::trsm_batched, std::string>::type
get_name(char side, char uplo, char trans, char diag, index_t m, index_t n,
         index_t batch_size, index_t stride_a_mul, index_t stride_b_mul,
         std::string mem_type) {
  return internal::get_name<op, scalar_t>(side, uplo, trans, diag, m, n,
                                          batch_size, stride_a_mul,
                                          stride_b_mul, mem_type);
}

template <ExtensionOp op, typename scalar_t, typename index_t>
inline typename std::enable_if<op == ExtensionOp::omatcopy, std::string>::type
get_name(std::string trans, int m, int n, scalar_t alpha, index_t lda_mul,
         index_t ldb_mul, std::string mem_type) {
  return internal::get_name<op, scalar_t>(trans, m, n, alpha, lda_mul, ldb_mul,
                                          mem_type);
}

template <ExtensionOp op, typename scalar_t, typename index_t>
inline typename std::enable_if<op == ExtensionOp::omatcopy2, std::string>::type
get_name(std::string trans, int m, int n, scalar_t alpha, index_t lda_mul,
         index_t ldb_mul, index_t inc_a, index_t inc_b, std::string mem_type) {
  return internal::get_name<op, scalar_t>(trans, m, n, alpha, lda_mul, ldb_mul,
                                          inc_a, inc_b, mem_type);
}

template <ExtensionOp op, typename scalar_t, typename index_t>
inline typename std::enable_if<op == ExtensionOp::omatadd, std::string>::type
get_name(std::string trans_a, std::string trans_b, int m, int n, scalar_t alpha,
         scalar_t beta, index_t lda_mul, index_t ldb_mul, index_t ldc_mul,
         std::string mem_type) {
  return internal::get_name<op, scalar_t>(trans_a, trans_b, m, n, alpha, beta,
                                          lda_mul, ldb_mul, ldc_mul, mem_type);
}

template <ExtensionOp op, typename scalar_t, typename index_t>
inline typename std::enable_if<op == ExtensionOp::omatcopy_batch,
                               std::string>::type
get_name(std::string trans, int m, int n, scalar_t alpha, index_t lda_mul,
         index_t ldb_mul, index_t stride_a_mul, index_t stride_b_mul,
         index_t batch_size, std::string mem_type) {
  return internal::get_name<op, scalar_t>(trans, m, n, alpha, lda_mul, ldb_mul,
                                          stride_a_mul, stride_b_mul,
                                          batch_size, mem_type);
}

template <ExtensionOp op, typename scalar_t, typename index_t>
inline
    typename std::enable_if<op == ExtensionOp::omatadd_batch, std::string>::type
    get_name(std::string trans_a, std::string trans_b, int m, int n,
             scalar_t alpha, scalar_t beta, index_t lda_mul, index_t ldb_mul,
             index_t ldc_mul, index_t stride_a_mul, index_t stride_b_mul,
             index_t stride_c_mul, index_t batch_size, std::string mem_type) {
  return internal::get_name<op, scalar_t>(
      trans_a, trans_b, m, n, alpha, beta, lda_mul, ldb_mul, ldc_mul,
      stride_a_mul, stride_b_mul, stride_c_mul, batch_size, mem_type);
}

template <ExtensionOp op, typename scalar_t, typename index_t>
inline typename std::enable_if<op == ExtensionOp::reduction, std::string>::type
get_name(index_t rows, index_t cols, std::string reduction_dim,
         std::string mem_type) {
  return internal::get_name<op, scalar_t>(rows, cols, reduction_dim, mem_type);
}

template <ExtensionOp op, typename scalar_t, typename index_t>
inline typename std::enable_if<op == ExtensionOp::axpy_batch, std::string>::type
get_name(index_t n, scalar_t alpha, index_t inc_x, index_t inc_y,
         index_t stride_x_mul, index_t stride_y_mul, index_t batch_size,
         std::string mem_type) {
  return internal::get_name<op, scalar_t>(n, alpha, inc_x, inc_y, stride_x_mul,
                                          stride_y_mul, batch_size, mem_type);
}

}  // namespace utils
}  // namespace blas_benchmark

#endif  // PORTBLAS_BENCHMARK_NAMES_HPP
