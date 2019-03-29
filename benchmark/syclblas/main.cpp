#include "cli_device_selector.hpp"
#include "utils.hpp"

// Create a shared pointer to a sycl blas executor, so that we don't keep
// reconstructing it each time (which is slow). Although this won't be
// cleaned up if RunSpecifiedBenchmarks exits badly, that's okay, as those
// are presumably exceptional circumstances.
std::unique_ptr<cli_device_selector> cdsp;
void free_device_selector() { cdsp.reset(); }

// Declare the executorInstancePtr (it is already `extern` declared in utils.hpp
// but it must be explicitly declared at least once.)
ExecutorPtr Global::executorInstancePtr;

int main(int argc, char** argv) {
  // Initialise the command line device selector in a unique pointer so that we
  // can register an `atexit` handler to delete the command line device
  // selector. If not, then if benchmark::Initialize calls exit() (for example,
  // if the flag `--help` is passed), the cds will not be freed before the sycl
  // runtime tries to exit, leading to an exception, as the cds will still hold
  // some sycl objects.
  cdsp =
      std::unique_ptr<cli_device_selector>(new cli_device_selector(argc, argv));
  std::atexit(free_device_selector);

  // Initialise googlebench
  benchmark::Initialize(&argc, argv);

  // Create a queue from the device selector - do this after initialising
  // googlebench, as otherwise we may not be able to delete the queue before we
  // exit (if Initialise calls exit(0)), and dump some information about it
  cl::sycl::queue q = cl::sycl::queue(
      *cdsp.get(), {cl::sycl::property::queue::enable_profiling()});
  benchmark::utils::print_queue_information(q);

  // Create a sycl blas executor from the queue
  Global::executorInstancePtr = std::unique_ptr<SyclExecutorType>(
      new SyclExecutorType(q));  // std::make_unique<SyclExecutorType>(q);
  benchmark::RunSpecifiedBenchmarks();

  // We need to explicitly reset/delete the executor instance pointer so that
  // the executor (and the the queue) is properly deleted before the sycl
  // runtime shuts down. If we don't, the runtime will complain about objects
  // not being properly destroyed, and the benchmark will exit without a
  // successful return code
  Global::executorInstancePtr.reset();
}
