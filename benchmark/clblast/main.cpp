#include "cli_device_selector.hpp"
#include "utils.hpp"

int main(int argc, char** argv) {
  // Read the command-line arguments
  auto args = blas_benchmark::utils::parse_args(argc, argv);

  cli_device_selector cds(args);
  OpenCLDeviceSelector oclds(cds.vendor_name, cds.device_type);

  // Initialize googlebench
  benchmark::Initialize(&argc, argv);

  ExecutorType executor(oclds);

  // Create the benchmarks
  blas_benchmark::create_benchmark(args, &executor);

  benchmark::RunSpecifiedBenchmarks();
}
