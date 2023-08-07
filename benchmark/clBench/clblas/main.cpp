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
 *  portBLAS: BLAS implementation using SYCL
 *
 *  @filename main.cpp
 *
 **************************************************************************/

#include "utils.hpp"
#include <common/extract_vendor_type.hpp>

int main(int argc, char** argv) {
  // Read the command-line arguments
  auto args = blas_benchmark::utils::parse_args(argc, argv);

  std::string vendor{};
  std::string type{};
  if (!args.device.empty()) {
    bool valid;
    std::tie(valid, vendor, type) = utils::extract_vendor_type(args.device);

    if (!valid) {
      // A message was already printed by extract_vendor_type, just exit
      std::exit(1);
    }
  }

  OpenCLDeviceSelector oclds(vendor, type);

  blas_benchmark::utils::print_device_information(oclds.device());

  // Initialize googlebench
  benchmark::Initialize(&argc, argv);

  ExecutorType executor(oclds);

  // This will be set to false by a failing benchmark
  bool success = true;

  // Create the benchmarks
  blas_benchmark::create_benchmark(args, &executor, &success);

  benchmark::RunSpecifiedBenchmarks();

  return !success;
}
