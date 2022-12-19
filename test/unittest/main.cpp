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
 *  @filename main.cpp
 *
 **************************************************************************/

#include <gtest/gtest.h>

#include "blas_test.hpp"
#include "blas_test_macros.hpp"
#include "clara.hpp"

struct Args args {};

namespace {
/**
 * @fn parse_args
 * @brief Returns a structure containing the data extracted from the
 * command-line arguments.
 */
void parse_args(int argc, char** argv) {
  // Updates the global args
  args.program_name = std::string(argv[0]);
  bool show_help = false;

  auto parser = clara::Help(show_help) |
                clara::Opt(args.device, "device")["--device"](
                    "Select a device (best effort) for running the benchmark.");

  auto res = parser.parse(clara::Args(argc, argv));
  if (!res) {
    std::cerr << "Error in command line: " << res.errorMessage() << std::endl;
    exit(1);
  } else if (show_help) {
    std::cout << parser << std::endl;
  }
};
}  // namespace

int main(int argc, char* argv[]) {
  int seed = 12345;
  srand(seed);

  // Process google test's arguments
  ::testing::InitGoogleTest(&argc, argv);

  // Do our own argument processing
  parse_args(argc, argv);

  auto exit_code = RUN_ALL_TESTS();
  // Explicitly wait just before returning from main to ensure that any SYCL
  // work in the queue has been completed.
  make_queue().wait_and_throw();
  return exit_code;
}
