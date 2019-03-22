#include "cli_device_selector.hpp"
#include "utils.hpp"

namespace Private {
ExecutorPtr ex;
}  // namespace Private

ExecutorPtr getExecutor() { return Private::ex; }

int main(int argc, char** argv) {
  cli_device_selector cds(argc, argv);
  OpenCLDeviceSelector oclds(cds.device_vendor, cds.device_type);

  benchmark::Initialize(&argc, argv);

  Context ctx(oclds);
  Private::ex = std::make_shared<ExecutorType>(std::move(ctx));

  benchmark::RunSpecifiedBenchmarks();
}
