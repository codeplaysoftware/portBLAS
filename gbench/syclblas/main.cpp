#include "cli_device_selector.hpp"
#include "utils.hpp"

namespace Private {
ExecutorPtr ex;
// Declare the queue globally so that it is properly destroyed if exit() is
// called. If not, then it will not be destroyed before the sycl runtime tries
// to exit, which will cause an exception.
cl::sycl::queue q;
}  // namespace Private

ExecutorPtr getExecutor() { return Private::ex; }

int main(int argc, char** argv) {
  {
    // Initialise the command line device selector and queue in a nested scope
    // so that the cds is destroyed before benchmark::Initialize is called. If
    // not, then if benchmark::Initialize calls exit(), the cds will not be
    // freed before the sycl runtime tries to exit, leading to an exception, as
    // the cds will still hold some sycl objects.
    cli_device_selector cds(argc, argv);
    Private::q =
        cl::sycl::queue(cds, {cl::sycl::property::queue::enable_profiling()});
  }

  // Initialise googlebench
  benchmark::Initialize(&argc, argv);

// Print out some information about the device that we're running on.
#if 1
  std::cout << "Device vendor: "
            << Private::q.get_device()
                   .template get_info<cl::sycl::info::device::vendor>()
            << std::endl;
  std::cout << "Device name: "
            << Private::q.get_device()
                   .template get_info<cl::sycl::info::device::name>()
            << std::endl;
  std::cout << "Device type: ";
  switch (Private::q.get_device()
              .template get_info<cl::sycl::info::device::device_type>()) {
    case cl::sycl::info::device_type::cpu:
      std::cout << "cpu";
    case cl::sycl::info::device_type::gpu:
      std::cout << "gpu";
    case cl::sycl::info::device_type::accelerator:
      std::cout << "accelerator";
    case cl::sycl::info::device_type::custom:
      std::cout << "custom";
    case cl::sycl::info::device_type::automatic:
      std::cout << "automatic";
    case cl::sycl::info::device_type::host:
      std::cout << "host";
    case cl::sycl::info::device_type::all:
      std::cout << "all";
    default:
      std::cout << "unknown";
  };
  std::cout << std::endl;
#endif

  // Create a shared pointer to a sycl blas executor, so that we don't keep
  // reconstructing it each time (which is slow). Although this won't be cleaned
  // up if RunSpecifiedBenchmarks exits badly, that's okay, as those are
  // presumably exceptional circumstances.
  Private::ex = std::make_shared<blas::Executor<SYCL>>(Private::q);

  benchmark::RunSpecifiedBenchmarks();
}