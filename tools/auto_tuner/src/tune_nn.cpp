/***************************************************************************
 *
 *  @license
 *  Copyright (C) 2018 Codeplay Software Limited
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
 *  @filename tune_nn.cpp
 *
 **************************************************************************/

#include <cstdlib>

#include "gemm_tuner.hpp"

int main(int argc, char *argv[]) {
  if (argc < 6) {
    std::cerr << "Usage: " << argv[0] << " M K N bs rep" << std::endl;
    return -1;
  }

  const bool transA = false;
  const bool transB = false;

  const int seed = 42;
  const int m = std::atoi(argv[1]);
  const int k = std::atoi(argv[2]);
  const int n = std::atoi(argv[3]);
  const int batch_size = std::atoi(argv[4]);
  const int rep = std::atoi(argv[5]);
  ::blas::gemm_batch_type_t batch_type = gemm_batch_type_t::strided;
  if (argc == 7) {
    auto b_t = std::string(argv[6]);
    std::transform(b_t.begin(), b_t.end(), b_t.begin(),
                   [](unsigned char c) { return std::tolower(c); });
    if (b_t.compare(std::string("interleaved")) == 0) {
      batch_type = gemm_batch_type_t::interleaved;
    } else if (b_t.compare(std::string("strided")) == 0) {
      batch_type = gemm_batch_type_t::strided;
    } else {
      std::cerr << "batch type can be either Interleaved or strided \n";
      return -1;
    }
  }

#ifdef BLAS_ENABLE_AUTO_TUNER_MEMPOOL
  Temp_Mem_Pool mem_pool(make_sycl_queue());
  portblas_handle_t sb_handle(&mem_pool);
#else
  portblas_handle_t sb_handle(make_sycl_queue());
#endif

  run_tune_gemm<transA, transB, float>(sb_handle, seed, m, k, n, batch_size,
                                       rep, batch_type);

  return 0;
}
