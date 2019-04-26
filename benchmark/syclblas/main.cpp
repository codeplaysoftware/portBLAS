#include "cli_device_selector.hpp"
#include "utils.hpp"

// Create a shared pointer to a sycl blas executor, so that we don't keep
// reconstructing it each time (which is slow). Although this won't be
// cleaned up if RunSpecifiedBenchmarks exits badly, that's okay, as those
// are presumably exceptional circumstances.
std::unique_ptr<cli_device_selector> cdsp;
void free_device_selector() { cdsp.reset(); }

int main(int argc, char** argv) {
  // Read the command-line arguments
  auto args = blas_benchmark::utils::parse_args(argc, argv);

  // Initialize googlebench
  benchmark::Initialize(&argc, argv);

  // Initialise the command line device selector in a unique pointer so that we
  // can register an `atexit` handler to delete the command line device
  // selector. If not, then if benchmark::Initialize calls exit() (for example,
  // if the flag `--help` is passed), the cds will not be freed before the sycl
  // runtime tries to exit, leading to an exception, as the cds will still hold
  // some sycl objects.
  cdsp = std::unique_ptr<cli_device_selector>(new cli_device_selector(args));
  std::atexit(free_device_selector);

  // Create a queue from the device selector - do this after initialising
  // googlebench, as otherwise we may not be able to delete the queue before we
  // exit (if Initialise calls exit(0)), and dump some information about it
  cl::sycl::queue q = cl::sycl::queue(
      *cdsp.get(), {cl::sycl::property::queue::enable_profiling()});
  blas_benchmark::utils::print_queue_information(q);

  // Create a sycl blas executor from the queue
  ExecutorType executor(q);

  // Create the benchmarks
  blas_benchmark::create_benchmark(args, &executor);

  // Run the benchmarks
  benchmark::RunSpecifiedBenchmarks();
}
