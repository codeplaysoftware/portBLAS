#include "cli_device_selector.hpp"
#include "utils.hpp"

namespace Private {
ExecutorPtr ex;
}  // namespace Private

ExecutorPtr Global::executorInstancePtr;

int main(int argc, char** argv) {
  // Read the command-line arguments
  auto args = blas_benchmark::utils::parse_args(argc, argv);

  cli_device_selector cds(args);
  OpenCLDeviceSelector oclds(cds.vendor_name, cds.device_type);

  // Register the benchmark and initialize googlebench
  blas_benchmark::create_benchmark(args);
  benchmark::Initialize(&argc, argv);

  Context ctx(oclds);
  Global::executorInstancePtr = std::unique_ptr<ExecutorType>(&ctx);

  benchmark::RunSpecifiedBenchmarks();
}
