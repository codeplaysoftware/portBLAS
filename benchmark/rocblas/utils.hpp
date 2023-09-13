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

#ifndef ROCBLAS_UTILS_HPP
#define ROCBLAS_UTILS_HPP

#include "portblas.h"
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

/**
 * @class HIPDeviceMemory
 * @brief Base-class to allocate/deallocate hip device memory.
 * @tparam T is the type of the underlying data
 */
template <typename T>
class HIPDeviceMemory {
 protected:
  size_t size_;
  size_t bytes_;

  HIPDeviceMemory() : size_(1), bytes_(sizeof(T)) {}

  HIPDeviceMemory(size_t s) : size_(s), bytes_(s * sizeof(T)) {}

  // Allocate on Device
  T* alloc() {
    T* d;
    if ((hipMalloc)(&d, bytes_) != hipSuccess) {
      fprintf(stderr, "Error allocating %zu bytes_\n", bytes_);
      d = nullptr;
    }
    return d;
  }

  // Copy from Host to Device
  void copyH2D(const T* hPtr, T* dPtr) {
    if (dPtr != nullptr) {
      CHECK_HIP_ERROR(hipMemcpy(dPtr, hPtr, bytes_, hipMemcpyHostToDevice));
    }
    return;
  }

  // Copy from Device to Host
  void copyD2H(const T* dPtr, T* hPtr) {
    if (hPtr != nullptr) {
      CHECK_HIP_ERROR(hipMemcpy(hPtr, dPtr, bytes_, hipMemcpyDeviceToHost));
    }
    return;
  }

  // Free device memory
  void free(T* d) {
    if (d != nullptr) {
      CHECK_HIP_ERROR(hipFree(d));
    }
  }
};

// Pseudo-vector subclass which uses device memory
template <typename T, bool CopyToHost = false>
class HIPVector : private HIPDeviceMemory<T> {
 public:
  explicit HIPVector(size_t s) : HIPDeviceMemory<T>(s) {
    d_data_ = this->alloc();
  }

  // Constructor using host pointer copies data to device
  HIPVector(size_t s, T* hPtr) : HIPVector<T, CopyToHost>(s) {
    h_data_ = hPtr;
    this->copyH2D(h_data_, d_data_);
  }

  // Destructor copies data back to host if specified & valid
  // & free-up device memory
  ~HIPVector() {
    if constexpr (CopyToHost) {
      this->copyD2H(d_data_, h_data_);
    }
    this->free(d_data_);
  }

  // Decay into device pointer wherever pointer is expected
  operator T*() { return d_data_; }
  operator const T*() const { return d_data_; }

  // Disallow copying or assigning
  HIPVector(const HIPVector&) = delete;
  HIPVector& operator=(const HIPVector&) = delete;

 private:
  T* d_data_;
  T* h_data_ = nullptr;
};

// Pseudo-scalar subclass which uses device memory
template <typename T, bool CopyToHost = false>
class HIPScalar : private HIPDeviceMemory<T> {
 public:
  explicit HIPScalar() : HIPDeviceMemory<T>() { d_data_ = this->alloc(); }

  // Constructor using host scalar reference copies value to device
  HIPScalar(T& hValue) : HIPScalar<T, CopyToHost>() {
    h_data_ = &hValue;
    this->copyH2D(h_data_, d_data_);
  }

  // Destructor copies data back to host if specified & valid
  // & free-up device memory
  ~HIPScalar() {
    if constexpr (CopyToHost) {
      this->copyD2H(d_data_, h_data_);
    }
    this->free(d_data_);
  }

  // Decay into device pointer wherever pointer is expected
  operator T*() { return d_data_; }
  operator const T*() const { return d_data_; }
  T* data() const { return d_data_; }

  // Disallow copying or assigning
  HIPScalar(const HIPScalar&) = delete;
  HIPScalar& operator=(const HIPScalar&) = delete;

 private:
  T* d_data_;
  T* h_data_ = nullptr;
};

// Pseudo batch of vectors subclass which uses device memory
template <typename T, bool CopyToHost = false>
class HIPVectorBatched : private HIPDeviceMemory<T*> {
 public:
  explicit HIPVectorBatched(size_t vector_size, size_t batch_size)
      : HIPDeviceMemory<T*>(batch_size),
        vector_size_(vector_size),
        vector_bytes_(vector_size * sizeof(T)) {
    // Initiate array of batch_size pointers
    d_ptr_array_ = (T**)malloc(HIPDeviceMemory<T*>::bytes_);

    // Allocate vector_bytes_ device memory for each pointers in the array
    for (int i = 0; i < HIPDeviceMemory<T*>::size_; i++) {
      CHECK_HIP_ERROR(hipMalloc(&d_ptr_array_[i], vector_bytes_));
    }

    // Allocate pointers array on device & copy H2D pointers
    d_batch_data_ = this->alloc();
    this->copyH2D(d_ptr_array_, d_batch_data_);
  }

  // Constructor using host array of pointer copies batches of data to device
  HIPVectorBatched(size_t vector_size, size_t batch_size, T* h_v[])
      : HIPVectorBatched<T, CopyToHost>(vector_size, batch_size) {
    auto size = HIPDeviceMemory<T*>::size_;
    if constexpr (CopyToHost) {
      h_batch_data_ = (T**)malloc(HIPDeviceMemory<T*>::bytes_);
      for (int i = 0; i < size; i++) {
        h_batch_data_[i] = h_v[i];
      }
    }

    for (int i = 0; i < size; i++) {
      CHECK_HIP_ERROR(hipMemcpy(d_ptr_array_[i], h_v[i], vector_bytes_,
                                hipMemcpyHostToDevice));
    }
  }

  // Constructor using host pointer of contiguous arrays copies batched data to
  // device
  HIPVectorBatched(size_t vector_size, size_t batch_size, T* h_v)
      : HIPVectorBatched<T, CopyToHost>(vector_size, batch_size) {
    auto size = HIPDeviceMemory<T*>::size_;
    if constexpr (CopyToHost) {
      h_batch_data_ = (T**)malloc(HIPDeviceMemory<T*>::bytes_);
      for (int i = 0; i < size; i++) {
        h_batch_data_[i] = &h_v[i * vector_size_];
      }
    }

    for (int i = 0; i < size; i++) {
      CHECK_HIP_ERROR(hipMemcpy(d_ptr_array_[i], &h_v[i * vector_size_],
                                vector_bytes_, hipMemcpyHostToDevice));
    }
  }

  // Destructor copies data back to host if specified & valid
  // & free-up device memories
  ~HIPVectorBatched() {
    auto size = HIPDeviceMemory<T*>::size_;
    if constexpr (CopyToHost) {
      for (int i = 0; i < size; i++) {
        if (h_batch_data_[i] != nullptr) {
          CHECK_HIP_ERROR(hipMemcpy(h_batch_data_[i], d_ptr_array_[i],
                                    vector_bytes_, hipMemcpyDeviceToHost));
        }
      }
    }
    for (int i = 0; i < size; i++) {
      if (d_ptr_array_[i] != nullptr) {
        CHECK_HIP_ERROR(hipFree(d_ptr_array_[i]));
      }
    }
    this->free(d_batch_data_);
    free(d_ptr_array_);
  }

  // Decay into device array pointer wherever pointer is expected
  operator T* *() { return d_batch_data_; }
  operator const T* *() const { return d_batch_data_; }

  // Disallow copying or assigning
  HIPVectorBatched(const HIPVectorBatched&) = delete;
  HIPVectorBatched& operator=(const HIPVectorBatched&) = delete;

 private:
  T** d_batch_data_;
  T** d_ptr_array_;
  T** h_batch_data_ = nullptr;

  size_t vector_size_;
  size_t vector_bytes_;
};

// Strided batched pseudo-vector subclass which uses device memory
template <typename T, bool CopyToHost = false>
class HIPVectorBatchedStrided : private HIPDeviceMemory<T> {
 public:
  // Construct using explicit vector, batch and stride sizes
  explicit HIPVectorBatchedStrided(size_t vector_size, size_t batch_size,
                                   size_t stride)
      : HIPDeviceMemory<T>(vector_size + (batch_size - 1) * stride),
        vector_size_{vector_size},
        batch_size_{batch_size},
        stride_{stride},
        vector_bytes_{vector_size * sizeof(T)} {
    d_data_ = this->alloc();
  }

  // Construct using explicit vector and batch sizes, and use vector size as
  // default stride
  explicit HIPVectorBatchedStrided(size_t vector_size, size_t batch_size)
      : HIPVectorBatchedStrided<T, CopyToHost>(vector_size, batch_size,
                                               vector_size) {}

  // Constructor using host pointer copies data to device
  HIPVectorBatchedStrided(size_t vector_size, size_t batch_size, size_t stride,
                          T* hPtr)
      : HIPVectorBatchedStrided<T, CopyToHost>(vector_size, batch_size,
                                               stride) {
    h_data_ = hPtr;
    if (stride_ <= vector_size_) {
      // Full copy
      this->copyH2D(h_data_, d_data_);
    } else {
      // Interleaved copy
      if (d_data_ != nullptr) {
        for (int i = 0; i < batch_size_; i++) {
          CHECK_HIP_ERROR(hipMemcpy(d_data_ + i * stride_,
                                    h_data_ + i * stride_, vector_bytes_,
                                    hipMemcpyHostToDevice));
        }
      }
    }
  }

  // Destructor copies data back to host if specified & valid
  // & free-up device memory
  ~HIPVectorBatchedStrided() {
    if constexpr (CopyToHost) {
      if (stride_ <= vector_size_) {
        // Full copy
        this->copyD2H(d_data_, h_data_);
      } else {
        // Interleaved copy
        if (h_data_ != nullptr) {
          for (int i = 0; i < batch_size_; i++) {
            CHECK_HIP_ERROR(hipMemcpy(h_data_ + i * stride_,
                                      d_data_ + i * stride_, vector_bytes_,
                                      hipMemcpyDeviceToHost));
          }
        }
      }
    }
    this->free(d_data_);
  }

  // Decay into device pointer wherever pointer is expected
  operator T*() { return d_data_; }
  operator const T*() const { return d_data_; }

  // Disallow copying or assigning
  HIPVectorBatchedStrided(const HIPVectorBatchedStrided&) = delete;
  HIPVectorBatchedStrided& operator=(const HIPVectorBatchedStrided&) = delete;

 private:
  T* d_data_;
  T* h_data_ = nullptr;

  size_t vector_size_;
  size_t vector_bytes_;
  size_t stride_;
  size_t batch_size_;
};

/**
 * @fn timef_hip
 * @brief Calculates the time spent executing the function func returning 2
 * hipEvents (both overall and HIP events time, returned in nanoseconds in a
 * tuple of double)
 */
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
