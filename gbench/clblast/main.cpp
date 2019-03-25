#include "cli_device_selector.hpp"
#include "utils.hpp"

namespace Private {
ExecutorPtr ex;
}  // namespace Private

ExecutorPtr Global::executorInstancePtr;

int main(int argc, char** argv) {
  cli_device_selector cds(argc, argv);
  OpenCLDeviceSelector oclds(cds.device_vendor, cds.device_type);

  benchmark::Initialize(&argc, argv);

  Context ctx(oclds);
  Global::executorInstancePtr = std::unique_ptr<ExecutorType>(&ctx);

  benchmark::RunSpecifiedBenchmarks();
}
