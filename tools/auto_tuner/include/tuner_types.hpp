/***************************************************************************
 *
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
 *  @filename tuner_types.hpp
 *
 **************************************************************************/

#ifndef SYCLBLAS_TOOLS_AUTO_TUNER_TUNER_TYPES_HPP_
#define SYCLBLAS_TOOLS_AUTO_TUNER_TUNER_TYPES_HPP_

#include <iostream>
#include <vector>

#include "sycl_blas.hpp"

using sycl_blas_handle_t = ::blas::SB_Handle;

template <typename DataType>
using HostContainer = std::vector<DataType>;

template <typename DataType>
using DeviceContainer = typename ::blas::BufferIterator<DataType>;

template <typename DataType>
using MatrixContainer =
    typename ::blas::MatrixViewTypeFactory<DataType, int,
                                           ::blas::col_major>::output_t;

struct TestResultEntry {
  std::string name;
  double sec;
  double gflops;
  double error;

  TestResultEntry(std::string name) : name(name) {}

  void print() const {
    std::cout << gflops << " gflops: " << name << " - Time: " << sec
              << " ms, Error: " << error << "\n";
  }

  bool operator<(const TestResultEntry &other) const {
    return gflops < other.gflops;
  }
  bool operator>(const TestResultEntry &other) const {
    return gflops > other.gflops;
  }
};

class TestResult : public std::vector<TestResultEntry> {
 public:
  void print_all() const {
    std::cout << "== Performance Results ==\n";
    for (auto &r : *this) {
      if (r.error < 0.1) {
        r.print();
      }
    }
  }
};

template <bool _TransA, bool _TransB, ::blas::gemm_memory_t _MemoryMode,
          ::blas::gemm_algorithm_t _ShapeMode,
          ::blas::gemm_batch_type_t _BatchType,
          ::blas::gemm_vectorization_t _VecType>
struct GemmConfig {
  static constexpr auto TransA = _TransA;
  static constexpr auto TransB = _TransB;
  static constexpr auto MemoryMode = _MemoryMode;
  static constexpr auto ShapeMode = _ShapeMode;
  static constexpr auto BatchType = _BatchType;
  static constexpr auto VecType = _VecType;
};

template <typename element_t>
struct GemmArgs {
  int m;
  int n;
  int k;
  element_t alpha;
  const DeviceContainer<element_t> &a;
  int lda;
  const DeviceContainer<element_t> &b;
  int ldb;
  element_t beta;
  const HostContainer<element_t> &init_c;
  DeviceContainer<element_t> &c;
  HostContainer<element_t> &output_c;
  int ldc;
  int batch_size;
  const HostContainer<element_t> &expected_c;
};

#endif  // SYCLBLAS_TOOLS_AUTO_TUNER_TUNER_TYPES_HPP_
