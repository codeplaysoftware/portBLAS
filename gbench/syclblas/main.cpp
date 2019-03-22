#include "cli_device_selector.hpp"
#include "utils.hpp"

namespace Private {
// Create a shared pointer to a sycl blas executor, so that we don't keep
// reconstructing it each time (which is slow). Although this won't be
// cleaned up if RunSpecifiedBenchmarks exits badly, that's okay, as those
// are presumably exceptional circumstances.
ExecutorPtr ex;
std::unique_ptr<cli_device_selector> cdsp;
void free_device_selector() { cdsp.reset(); }
}  // namespace Private

ExecutorPtr getExecutor() { return Private::ex; }

int main(int argc, char** argv) {
  // Initialise the command line device selector in a unique pointer so that we
  // can register an `atexit` handler to delete the command line device
  // selector. If not, then if benchmark::Initialize calls exit() (for example,
  // if the flag `--help` is passed), the cds will not be freed before the sycl
  // runtime tries to exit, leading to an exception, as the cds will still hold
  // some sycl objects.
  Private::cdsp =
      std::unique_ptr<cli_device_selector>(new cli_device_selector(argc, argv));
  std::atexit(Private::free_device_selector);

  // Initialise googlebench
  benchmark::Initialize(&argc, argv);

  // Create a queue from the device selector - do this after initialising
  // googlebench, as otherwise we may not be able to delete the queue before we
  // exit (if Initialise calls exit(0))
  cl::sycl::queue q = cl::sycl::queue(
      *Private::cdsp.get(), {cl::sycl::property::queue::enable_profiling()});

// Print out some information about the device that we're running on.
#if 1
  std::cout
      << "Device vendor: "
      << q.get_device().template get_info<cl::sycl::info::device::vendor>()
      << std::endl;
  std::cout << "Device name: "
            << q.get_device().template get_info<cl::sycl::info::device::name>()
            << std::endl;
  std::cout << "Device type: ";
  switch (
      q.get_device().template get_info<cl::sycl::info::device::device_type>()) {
    case cl::sycl::info::device_type::cpu:
      std::cout << "cpu";
      break;
    case cl::sycl::info::device_type::gpu:
      std::cout << "gpu";
      break;
    case cl::sycl::info::device_type::accelerator:
      std::cout << "accelerator";
      break;
    case cl::sycl::info::device_type::custom:
      std::cout << "custom";
      break;
    case cl::sycl::info::device_type::automatic:
      std::cout << "automatic";
      break;
    case cl::sycl::info::device_type::host:
      std::cout << "host";
      break;
    case cl::sycl::info::device_type::all:
      std::cout << "all";
      break;
    default:
      std::cout << "unknown";
      break;
  };
  std::cout << std::endl;
#endif

  // Create a sycl blas executor from the queue
  Private::ex = std::make_shared<SyclExecutorType>(q);
  benchmark::RunSpecifiedBenchmarks();
  // Delete the sycl blas executor.
  Private::ex.reset();
}
