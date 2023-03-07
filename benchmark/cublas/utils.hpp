#ifndef SYCL_UTILS_HPP
#define SYCL_UTILS_HPP

#include <chrono>
#include <tuple>

#include "benchmark/benchmark.h"
#include "sycl_blas.h"
#include <common/common_utils.hpp>

#include <cublas_v2.h>
#include <cuda.h>
#include <cuda_runtime.h>
// Forward declare methods that we use in `benchmark.cpp`, but define in
// `main.cpp`

namespace blas_benchmark {

// Forward-declaring the function that will create the benchmark
void create_benchmark(Args& args, cublasHandle_t* cuda_handle_ptr,
                      bool* success);

namespace utils {

/**
 * @fn time_event
 * @brief Get the overall run time (start -> end) of a CUDA operator using
 *  CudaEvent_t
 */
template <>
inline double time_event<std::vector<cudaEvent_t>>(
    std::vector<cudaEvent_t>& cuda_events) {
  float elapsed_time;
  cudaEventElapsedTime(&elapsed_time, cuda_events[0], cuda_events[1]);
  // convert result from ms to ns
  return static_cast<double>(elapsed_time) * 1'000'000.;
}

template <typename scalar_t>
inline void init_level_1_counters(benchmark::State& state, index_t size) {
  // Google-benchmark counters are double.
  double size_d = static_cast<double>(size);
  state.counters["size"] = size_d;
  state.counters["n_fl_ops"] = 2.0 * size_d;
  state.counters["bytes_processed"] = size_d * sizeof(scalar_t);
  return;
}

}  // namespace utils
}  // namespace blas_benchmark

#define CUDA_CHECK(err)                                                  \
  do {                                                                   \
    cudaError_t err_ = (err);                                            \
    if (err_ != cudaSuccess) {                                           \
      std::printf("CUDA error %d at %s:%d\n", err_, __FILE__, __LINE__); \
      throw std::runtime_error("CUDA error");                            \
    }                                                                    \
  } while (0)

// cublas API error checking
#define CUBLAS_CHECK(err)                                                  \
  do {                                                                     \
    cublasStatus_t err_ = (err);                                           \
    if (err_ != CUBLAS_STATUS_SUCCESS) {                                   \
      std::printf("cublas error %d at %s:%d\n", err_, __FILE__, __LINE__); \
      throw std::runtime_error("cublas error");                            \
    }                                                                      \
  } while (0)
#endif
