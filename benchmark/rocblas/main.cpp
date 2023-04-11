/**************************************************************************
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
 *  @filename main.cpp
 *
 **************************************************************************/

#include "utils.hpp"

int main(int argc, char** argv) {
  rocblas_status rstatus = rocblas_status_success;

  // Read the command-line arguments
  auto args = blas_benchmark::utils::parse_args(argc, argv);

  // Initialize googlebench
  benchmark::Initialize(&argc, argv);

  // Using rocblas API to create a handle
  rocblas_handle rb_handle;
  rstatus = rocblas_create_handle(&rb_handle);

  CHECK_ROCBLAS_STATUS(rstatus);

  // This will be set to false by a failing benchmark
  bool success = true;

  // Create the benchmarks
  blas_benchmark::create_benchmark(args, rb_handle, &success);

  // Run
  benchmark::RunSpecifiedBenchmarks();

  rstatus = rocblas_destroy_handle(rb_handle);

  CHECK_ROCBLAS_STATUS(rstatus);

  return !success;
}
