#/***************************************************************************
# *
# *  @license
# *  Copyright (C) Codeplay Software Limited
# *  Licensed under the Apache License, Version 2.0 (the "License");
# *  you may not use this file except in compliance with the License.
# *  You may obtain a copy of the License at
# *
# *      http://www.apache.org/licenses/LICENSE-2.0
# *
# *  For your convenience, a copy of the License has been included in this
# *  repository.
# *
# *  Unless required by applicable law or agreed to in writing, software
# *  distributed under the License is distributed on an "AS IS" BASIS,
# *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# *  See the License for the specific language governing permissions and
# *  limitations under the License.
# *
# *  portBLAS: BLAS implementation using SYCL
# *
# *  @filename utils.hpp 
# *
# **************************************************************************/

#ifndef SYCL_UTILS_HPP
#define SYCL_UTILS_HPP

#include <chrono>
#include <tuple>

#include "benchmark/benchmark.h"
#include "portblas.h"
#include <common/common_utils.hpp>

#include <cublas_v2.h>
#include <cuda.h>
#include <cuda_runtime.h>
// Forward declare methods that we use in `benchmark.cpp`, but define in
// `main.cpp`

#define CUDA_CHECK(err)                                                  \
  {                                                                      \
    cudaError_t err_ = (err);                                            \
    if (err_ != cudaSuccess) {                                           \
      std::printf("CUDA error %d at %s:%d\n", err_, __FILE__, __LINE__); \
      throw std::runtime_error("CUDA error");                            \
    }                                                                    \
  }

// cublas API error checking
#define CUBLAS_CHECK(err)                                                  \
  {                                                                        \
    cublasStatus_t err_ = (err);                                           \
    if (err_ != CUBLAS_STATUS_SUCCESS) {                                   \
      std::printf("cublas error %d at %s:%d\n", err_, __FILE__, __LINE__); \
      throw std::runtime_error("cublas error");                            \
    }                                                                      \
  }

namespace blas_benchmark {

// Forward-declaring the function that will create the benchmark
void create_benchmark(Args& args, cublasHandle_t* cuda_handle_ptr,
                      bool* success);

namespace utils {

/**
 * @class CUDADeviceMemory
 * @brief Base-class to allocate/deallocate cuda device memory.
 * @tparam T is the type of the underlying data
 * @tparam CopyToHost whether to copy back
 */
template <typename T, bool CopyToHost = false>
class CUDADeviceMemory {
 protected:
  size_t size_;
  size_t bytes_;

  CUDADeviceMemory() : size_(1), bytes_(sizeof(T)) {}

  CUDADeviceMemory(size_t s) : size_(s), bytes_(s * sizeof(T)) {}

  // Allocate on Device
  T* alloc() {
    T* d;
    if (cudaMalloc(&d, bytes_) != cudaSuccess) {
      fprintf(stderr, "Error allocating %zu bytes_\n", bytes_);
      d = nullptr;
    }
    return d;
  }

  // Copy from Host to Device
  void copyH2D(const T* hPtr, T* dPtr) {
    if (dPtr != nullptr) {
      CUDA_CHECK(cudaMemcpy(dPtr, hPtr, bytes_, cudaMemcpyHostToDevice));
    }
    return;
  }

  // Copy from Device to Host
  void copyD2H(const T* dPtr, T* hPtr) {
    if (hPtr != nullptr) {
      CUDA_CHECK(cudaMemcpy(hPtr, dPtr, bytes_, cudaMemcpyDeviceToHost));
    }
    return;
  }

  // Free device memory
  void free(T* d) {
    if (d != nullptr) {
      CUDA_CHECK(cudaFree(d));
    }
  }
};

// Pseudo-vector subclass which uses device memory
template <typename T, bool CopyToHost = false>
class CUDAVector : private CUDADeviceMemory<T> {
 public:
  explicit CUDAVector(size_t s) : CUDADeviceMemory<T>(s) {
    d_data_ = this->alloc();
  }

  // Constructor using host pointer copies data to device
  CUDAVector(size_t s, T* hPtr) : CUDAVector<T, CopyToHost>(s) {
    h_data_ = hPtr;
    this->copyH2D(h_data_, d_data_);
  }

  // Destructor copies data back to host if specified & valid
  // & free-up device memory
  ~CUDAVector() {
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
  CUDAVector(const CUDAVector&) = delete;
  CUDAVector& operator=(const CUDAVector&) = delete;

 private:
  T* d_data_;
  T* h_data_ = nullptr;
};

template <typename T, bool CopyToHost = false>
class CUDAVectorBatched : private CUDADeviceMemory<T> {
 public:
  explicit CUDAVectorBatched(size_t matrix_size, size_t batch_count)
      : CUDADeviceMemory<T>(matrix_size),
        c_batch_count(batch_count),
        c_matrix_size(matrix_size) {
    d_data = std::vector<T*>(batch_count, nullptr);
    for (int i = 0; i < batch_count; ++i) {
      CUDA_CHECK(cudaMalloc(&d_data[i], sizeof(T) * matrix_size));
    }
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&batch_Array),
                          sizeof(T*) * batch_count));
    CUDA_CHECK(cudaMemcpy(batch_Array, d_data.data(), sizeof(T*) * batch_count,
                          cudaMemcpyHostToDevice));
  }

  CUDAVectorBatched(size_t matrix_size, size_t batch_count, std::vector<T>& h_v)
      : CUDAVectorBatched<T, CopyToHost>(matrix_size, batch_count) {
    if constexpr (CopyToHost) h_data = h_v.data();
    for (int i = 0; i < batch_count; ++i) {
      CUDA_CHECK(cudaMemcpy(d_data[i], &h_v[matrix_size * i],
                            sizeof(T) * c_matrix_size, cudaMemcpyHostToDevice));
    }
  }

  ~CUDAVectorBatched() {
    if constexpr (CopyToHost) {
      for (int i = 0; i < c_batch_count; ++i) {
        this->copyD2H(d_data[i], (h_data + c_matrix_size * i));
      }
    }
    for (int i = 0; i < c_batch_count; ++i) {
      this->free(d_data[i]);
    }
    free_batch_dpointer(batch_Array);
  }

  T** get_batch_array() const { return batch_Array; }

  // Disallow copying or assigning
  CUDAVectorBatched(const CUDAVectorBatched&) = delete;
  CUDAVectorBatched& operator=(const CUDAVectorBatched&) = delete;

 private:
  const int c_matrix_size;
  const int c_batch_count;
  T** batch_Array;
  std::vector<T*> d_data;
  T* h_data = nullptr;

  void free_batch_dpointer(T** batch_pointer) {
    if (batch_pointer != nullptr) CUDA_CHECK(cudaFree(batch_pointer));
  }
};

// Pseudo-scalar subclass which uses device memory
template <typename T, bool CopyToHost = false>
class CUDAScalar : private CUDADeviceMemory<T> {
 public:
  explicit CUDAScalar() : CUDADeviceMemory<T>() { d_data_ = this->alloc(); }

  // Constructor using host scalar reference copies value to device
  CUDAScalar(T& hValue) : CUDAScalar<T, CopyToHost>() {
    h_data_ = &hValue;
    this->copyH2D(h_data_, d_data_);
  }

  // Destructor copies data back to host if specified & valid
  // & free-up device memory
  ~CUDAScalar() {
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
  CUDAScalar(const CUDAScalar&) = delete;
  CUDAScalar& operator=(const CUDAScalar&) = delete;

 private:
  T* d_data_;
  T* h_data_ = nullptr;
};
/**
 * @fn timef_cuda
 * @brief Calculates the time spent executing the function func returning 2
 * cudaEvents (both overall and CUDA events time, returned in nanoseconds in a
 * tuple of double)
 */
template <typename function_t, typename... args_t>
static inline std::tuple<double, double> timef_cuda(function_t func,
                                                    args_t&&... args) {
  auto start = std::chrono::system_clock::now();
  std::vector<cudaEvent_t> events = func(std::forward<args_t>(args)...);
  auto end = std::chrono::system_clock::now();

  double overall_time = (end - start).count();

  float elapsed_time;
  CUDA_CHECK(cudaEventElapsedTime(&elapsed_time, events.at(0), events.at(1)));

  return std::make_tuple(overall_time, static_cast<double>(elapsed_time) * 1E6);
}

}  // namespace utils
}  // namespace blas_benchmark

#endif
