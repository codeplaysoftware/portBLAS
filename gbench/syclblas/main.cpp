#include "utils.hpp"

ExecutorPtr ex;

ExecutorPtr getExecutor() { return ex; }

int main(int argc, char** argv) {
  cl::sycl::default_selector device_selector;
  cl::sycl::queue queue(device_selector,
                        {cl::sycl::property::queue::enable_profiling()});
  {
    ex = std::make_shared<blas::Executor<SYCL>>(queue);

    benchmark::Initialize(&argc, argv);
    benchmark::RunSpecifiedBenchmarks();

    ex.reset();
  }
}