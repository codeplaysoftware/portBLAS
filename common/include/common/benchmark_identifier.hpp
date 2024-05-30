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
 *  @filename benchmark_identifier.hpp
 *
 */

#ifndef PORTBLAS_BENCHMARK_IDENTIFIER_HPP
#define PORTBLAS_BENCHMARK_IDENTIFIER_HPP

namespace blas_benchmark {
namespace utils {

enum class Level1Op : int {
  asum = 0,
  axpy = 1,
  dot = 2,
  iamax = 3,
  iamin = 4,
  nrm2 = 5,
  rotm = 6,
  rotmg = 7,
  scal = 8,
  sdsdot = 9,
  copy = 10,
  rotg = 11
};

enum class Level2Op : int {
  gbmv = 0,
  gemv = 1,
  ger = 2,
  sbmv = 3,
  spmv = 4,
  spr = 5,
  spr2 = 6,
  symv = 7,
  syr = 8,
  syr2 = 9,
  tbmv = 10,
  tbsv = 11,
  tpmv = 12,
  tpsv = 13,
  trmv = 14,
  trsv = 15
};

enum class Level3Op : int {
  gemm_batched_strided = 0,
  gemm_batched = 1,
  gemm = 2,
  symm = 3,
  syr2k = 4,
  syrk = 5,
  trmm = 6,
  trsm_batched = 7,
  trsm = 8
};

enum class ExtensionOp : int {
  omatcopy = 0,
  imatcopy = 1,
  omatadd = 2,
  omatcopy_batch = 3,
  imatcopy_batch = 4,
  omatadd_batch = 5,
  omatcopy2 = 6,
  reduction = 7,
  axpy_batch = 8
};

template <Level1Op op>
std::string get_operator_name() {
  if constexpr (op == Level1Op::asum)
    return "Asum";
  else if constexpr (op == Level1Op::axpy)
    return "Axpy";
  else if constexpr (op == Level1Op::dot)
    return "Dot";
  else if constexpr (op == Level1Op::iamax)
    return "Iamax";
  else if constexpr (op == Level1Op::iamin)
    return "Iamin";
  else if constexpr (op == Level1Op::nrm2)
    return "Nrm2";
  else if constexpr (op == Level1Op::rotm)
    return "Rotm";
  else if constexpr (op == Level1Op::rotmg)
    return "Rotmg";
  else if constexpr (op == Level1Op::scal)
    return "Scal";
  else if constexpr (op == Level1Op::sdsdot)
    return "Sdsdot";
  else if constexpr (op == Level1Op::copy)
    return "Copy";
  else if constexpr (op == Level1Op::rotg)
    return "Rotg";
  else
    throw std::runtime_error("Unknown BLAS 1 operator");
}

template <Level2Op op>
std::string get_operator_name() {
  if constexpr (op == Level2Op::gbmv)
    return "Gbmv";
  else if constexpr (op == Level2Op::gemv)
    return "Gemv";
  else if constexpr (op == Level2Op::ger)
    return "Ger";
  else if constexpr (op == Level2Op::sbmv)
    return "Sbmv";
  else if constexpr (op == Level2Op::spmv)
    return "Spmv";
  else if constexpr (op == Level2Op::spr)
    return "Spr";
  else if constexpr (op == Level2Op::spr2)
    return "Spr2";
  else if constexpr (op == Level2Op::symv)
    return "Symv";
  else if constexpr (op == Level2Op::syr)
    return "Syr";
  else if constexpr (op == Level2Op::syr2)
    return "Syr2";
  else if constexpr (op == Level2Op::tbmv)
    return "Tbmv";
  else if constexpr (op == Level2Op::tbsv)
    return "Tbsv";
  else if constexpr (op == Level2Op::tpmv)
    return "Tpmv";
  else if constexpr (op == Level2Op::tpsv)
    return "Tpsv";
  else if constexpr (op == Level2Op::trmv)
    return "Trmv";
  else if constexpr (op == Level2Op::trsv)
    return "Trsv";
  else
    throw std::runtime_error("Unknown BLAS 2 operator");
}

template <Level3Op op>
std::string get_operator_name() {
  if constexpr (op == Level3Op::gemm_batched_strided)
    return "Gemm_batched_strided";
  else if constexpr (op == Level3Op::gemm_batched)
    return "Gemm_batched";
  else if constexpr (op == Level3Op::gemm)
    return "Gemm";
  else if constexpr (op == Level3Op::symm)
    return "Symm";
  else if constexpr (op == Level3Op::syr2k)
    return "Syr2k";
  else if constexpr (op == Level3Op::syrk)
    return "Syrk";
  else if constexpr (op == Level3Op::trmm)
    return "Trmm";
  else if constexpr (op == Level3Op::trsm_batched)
    return "Trsm_batched";
  else if constexpr (op == Level3Op::trsm)
    return "Trsm";
  else
    throw std::runtime_error("Unknown BLAS 3 operator");
}

template <ExtensionOp op>
std::string get_operator_name() {
  if constexpr (op == ExtensionOp::omatcopy)
    return "Omatcopy";
  else if constexpr (op == ExtensionOp::imatcopy)
    return "Imatcopy";
  else if constexpr (op == ExtensionOp::omatadd)
    return "Omatadd";
  else if constexpr (op == ExtensionOp::omatcopy_batch)
    return "Omatcopy_batch";
  else if constexpr (op == ExtensionOp::imatcopy_batch)
    return "Imatcopy_batch";
  else if constexpr (op == ExtensionOp::omatadd_batch)
    return "Omatadd_batch";
  else if constexpr (op == ExtensionOp::omatcopy2)
    return "Omatcopy2";
  else if constexpr (op == ExtensionOp::reduction)
    return "Reduction";
  else if constexpr (op == ExtensionOp::axpy_batch)
    return "Axpy_batch";
  else
    throw std::runtime_error("Unknown BLAS extension operator");
}

}  // namespace utils
}  // namespace blas_benchmark

#endif  // PORTBLAS_BENCHMARK_IDENTIFIER_HPP
