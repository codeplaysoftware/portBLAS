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
 *  @filename utils.hpp
 *
 **************************************************************************/

#ifndef ROCBLAS_UTILS_HPP
#define ROCBLAS_UTILS_HPP

#include "sycl_blas.h"
#include <common/common_utils.hpp>

#include <hip/hip_runtime.h>
#include <rocblas.h>

#ifndef CHECK_HIP_ERROR
#define CHECK_HIP_ERROR(error)                                    \
  if (error != hipSuccess) {                                      \
    fprintf(stderr, "hip error: '%s'(%d) at %s:%d\n",             \
            hipGetErrorString(error), error, __FILE__, __LINE__); \
  }
#endif

#ifndef CHECK_ROCBLAS_STATUS
#define CHECK_ROCBLAS_STATUS(status)                                       \
  if (status != rocblas_status_success) {                                  \
    fprintf(stderr, "rocBLAS error: ");                                    \
    fprintf(stderr, "rocBLAS error: '%s'(%d) at %s:%d\n",                  \
            rocblas_status_to_string(status), status, __FILE__, __LINE__); \
  }
#endif

namespace blas_benchmark {

void create_benchmark(blas_benchmark::Args& args, rocblas_handle& rb_handle_ptr,
                      bool* success);

namespace utils {

// base-class to allocate/deallocate hip device memory
template <typename T>
class DeviceVectorMemory {
 protected:
  size_t mSize, mBytes;

  DeviceVectorMemory(size_t s) : mSize(s), mBytes(s * sizeof(T)) {}

  T* setup() {
    T* d;
    if ((hipMalloc)(&d, mBytes) != hipSuccess) {
      fprintf(stderr, "Error allocating %zu mBytes\n", mBytes);
      d = nullptr;
    }
    return d;
  }

  void teardown(T* d) {
    if (d != nullptr) {
      // Free device memory
      CHECK_HIP_ERROR((hipFree)(d));
    }
  }
};

// pseudo-vector subclass which uses device memory
template <typename T>
class DeviceVector : private DeviceVectorMemory<T> {
 public:
  explicit DeviceVector(size_t s) : DeviceVectorMemory<T>(s) {
    mData = this->setup();
  }

  ~DeviceVector() { this->teardown(mData); }

  // Decay into pointer wherever pointer is expected
  operator T*() { return mData; }

  operator const T*() const { return mData; }

  T* data() const { return mData; }

  // Tell whether malloc failed
  explicit operator bool() const { return mData != nullptr; }

  // Disallow copying or assigning
  DeviceVector(const DeviceVector&) = delete;
  DeviceVector& operator=(const DeviceVector&) = delete;

 private:
  T* mData;
};

template <typename function_t, typename... args_t>
static inline std::tuple<double, double> timef_hip(function_t func,
                                                   args_t&&... args) {
  auto start = std::chrono::system_clock::now();
  std::vector<hipEvent_t> events = func(std::forward<args_t>(args)...);
  auto end = std::chrono::system_clock::now();

  double overall_time = (end - start).count();

  float elapsed_time;
  CHECK_HIP_ERROR(
      hipEventElapsedTime(&elapsed_time, events.at(0), events.at(1)));

  return std::make_tuple(overall_time, static_cast<double>(elapsed_time) * 1E6);
}

}  // namespace utils
}  // namespace blas_benchmark

#endif
