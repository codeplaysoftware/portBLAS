#include "utils.hpp"
#include <common/extract_vendor_type.hpp>"

int main(int argc, char** argv) {
  // Read the command-line arguments
  auto args = blas_benchmark::utils::parse_args(argc, argv);

  std::string vendor{};
  std::string type{};
  if (!args.device.empty()) {
    bool valid;
    std::tie(valid, vendor, type) = utils::extract_vendor_type(args.device);

    if (!valid) {
      // A message was already printed by extract_vendor_type, just exit
      std::exit(1);
    }
  }

  OpenCLDeviceSelector oclds(vendor, type);

  blas_benchmark::utils::print_device_information(oclds.device());

  // Initialize googlebench
  benchmark::Initialize(&argc, argv);

  ExecutorType executor(oclds);

  // This will be set to false by a failing benchmark
  bool success = true;

  // Create the benchmarks
  blas_benchmark::create_benchmark(args, &executor, &success);

  benchmark::RunSpecifiedBenchmarks();

  return !success;
}
