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
 *  @filename utils.hpp
 *
 **************************************************************************/

#ifndef SYCL_UTILS_HPP
#define SYCL_UTILS_HPP

#include <CL/sycl.hpp>
#include <chrono>
#include <tuple>

#include "portblas.h"
#include <common/common_utils.hpp>

// Forward declare methods that we use in `benchmark.cpp`, but define in
// `main.cpp`

namespace blas_benchmark {

// Forward-declaring the function that will create the benchmark
void create_benchmark(Args& args, blas::SB_Handle* sb_handle_ptr,
                      bool* success);

namespace utils {

/**
 * @fn time_event
 * @brief Get the overall run time (start -> end) of a cl::sycl::event enqueued
 * on a queue with profiling.
 */
template <>
inline double time_event<cl::sycl::event>(cl::sycl::event& e) {
  // get start and end times
  auto start_time = e.template get_profiling_info<
      cl::sycl::info::event_profiling::command_start>();

  auto end_time = e.template get_profiling_info<
      cl::sycl::info::event_profiling::command_end>();

  // return the delta
  return static_cast<double>(end_time - start_time);
}

}  // namespace utils
}  // namespace blas_benchmark

#endif
