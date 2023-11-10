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
#include <common/cli_device_selector.hpp>
#include <common/print_queue_information.hpp>

// Create a shared pointer to a portBLAS Handle, so that we don't keep
// reconstructing it each time (which is slow). Although this won't be
// cleaned up if RunSpecifiedBenchmarks exits badly, that's okay, as those
// are presumably exceptional circumstances.
std::unique_ptr<utils::cli_device_selector> cdsp;
void free_device_selector() { cdsp.reset(); }

int main(int argc, char** argv) {
  // Read the command-line arguments
  auto args = blas_benchmark::utils::parse_args(argc, argv);

  // Initialize googlebench
  benchmark::Initialize(&argc, argv);

  cl::sycl::queue q;

  if (!args.device.empty()) {
    // Initialise the command line device selector in a unique pointer so that
    // we can register an `atexit` handler to delete the command line device
    // selector. If not, then if benchmark::Initialize calls exit() (for
    // example, if the flag `--help` is passed), the cds will not be freed
    // before the sycl runtime tries to exit, leading to an exception, as the
    // cds will still hold some sycl objects.
    cdsp = std::unique_ptr<utils::cli_device_selector>(
        new utils::cli_device_selector(args.device));
    std::atexit(free_device_selector);

    // Create a queue from the device selector - do this after initialising
    // googlebench, as otherwise we may not be able to delete the queue before
    // we exit (if Initialise calls exit(0)), and dump some information about it
    q = cl::sycl::queue(*cdsp.get(),
                        {cl::sycl::property::queue::enable_profiling()});
  } else {
    q = cl::sycl::queue(cl::sycl::default_selector(),
                        {cl::sycl::property::queue::enable_profiling()});
  }

  utils::print_queue_information(q);

#ifdef BLAS_MEMPOOL_BENCHMARK
  blas::Temp_Mem_Pool mp(q);
  // Create a portBLAS sb_handle from the memory pool
  blas::SB_Handle sb_handle(&mp);
#else
  // Create a portBLAS sb_handle from the queue
  blas::SB_Handle sb_handle(q);
#endif

  // This will be set to false by a failing benchmark
  bool success = true;

  // Create the benchmarks
  blas_benchmark::create_benchmark(args, &sb_handle, &success);

  // Run the benchmarks
  benchmark::RunSpecifiedBenchmarks();

  return !success;
}
