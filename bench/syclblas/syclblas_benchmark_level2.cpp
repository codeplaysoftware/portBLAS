/***************************************************************************
 *
 *  @license
 *  Copyright (C) 2016 Codeplay Software Limited
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
 *  @filename syclblas_benchmark.cpp
 *
 **************************************************************************/

#include "../blas_benchmark.hpp"

#include <interface/blas1_interface.hpp>
#include <interface/blas2_interface.hpp>

using namespace blas;

template <typename ExecutorType = SYCL>
class SyclBlasBenchmarker {
  cl::sycl::queue q;
  Executor<ExecutorType> ex;

 public:
  SyclBlasBenchmarker()
      : q(cl::sycl::default_selector(),
          [=](cl::sycl::exception_list eL) {
            for (auto &e : eL) {
              try {
                std::rethrow_exception(e);
              } catch (cl::sycl::exception &e) {
                std::cout << " E " << e.what() << std::endl;
              } catch (...) {
                std::cout << " An exception " << std::endl;
              }
            }
          }),
        ex(q) {}

};

BENCHMARK_MAIN_BEGIN(range(1 << 1, 1 << 13, 1<<1), 10);
SyclBlasBenchmarker<SYCL> blasbenchmark;

BENCHMARK_MAIN_END();
