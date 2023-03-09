#include "cublas_v2.h"
#include "cuda_runtime_api.h"
#include "utils.hpp"
#include <common/cli_device_selector.hpp>
#include <common/print_queue_information.hpp>

// Create a shared pointer to a sycl blas Handle, so that we don't keep
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

  // Create a sycl blas sb_handle from the queue
  cublasHandle_t cublas_handle = NULL;
  cublasCreate(&cublas_handle);

  // This will be set to false by a failing benchmark
  bool success = true;

  // Create the benchmarks
  blas_benchmark::create_benchmark(args, &cublas_handle, &success);

  // Run the benchmarks
  benchmark::RunSpecifiedBenchmarks();

  cublasDestroy(cublas_handle);
  cudaDeviceReset();

  return !success;
}
