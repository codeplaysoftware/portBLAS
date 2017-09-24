/***************************************************************************
 *
 *  @license
 *  Copyright (C) 2017 Codeplay Software Limited
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
 *  @filename gemm_test_tb.cpp
 *
 **************************************************************************/

#include <cstdlib>


#include "gemm_utils.hpp"


int main(int argc, char *argv[]) {
  if (argc != 5) {
    std::cerr << "Usage: " << argv[0] << " M K N rep" << std::endl;
    return -1;
  }

  const bool transA = true;
  const bool transB = true;

  const int seed = 42;
  const int m = std::atoi(argv[1]);
  const int k = std::atoi(argv[2]);
  const int n = std::atoi(argv[3]);
  const int rep = std::atoi(argv[4]);

  run_gemm_tests<!transA, transB, float>(seed, m, k, n, rep);

  return 0;
}

