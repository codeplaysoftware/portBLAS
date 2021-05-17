#ifndef SYCL_UTILS_HPP
#define SYCL_UTILS_HPP

#include <CL/sycl.hpp>
#include <chrono>
#include <tuple>

#include <common/common_utils.hpp>
#include <common/quantization.hpp>

#ifdef SYCL_BLAS_FPGA
#include <common/cli_device_selector.hpp>
#include <common/print_queue_information.hpp>
#endif

#ifdef BLAS_HEADER_ONLY
#include <sycl_blas.hpp>
#else
#include <sycl_blas.h>
#endif

// Forward declare methods that we use in `benchmark.cpp`, but define in
// `main.cpp`
typedef blas::Executor<blas::PolicyHandler<blas::codeplay_policy>> ExecutorType;

#ifdef SYCL_BLAS_FPGA
static std::unique_ptr<utils::cli_device_selector> cdsp;
inline void free_device_selector() { cdsp.reset(); }

extern Args args;

#endif
namespace blas_benchmark {

// Forward-declaring the function that will create the benchmark
#ifdef SYCL_BLAS_FPGA
void create_benchmark(Args& args, bool* success);
#else
void create_benchmark(Args& args, ExecutorType* exPtr, bool* success);
#endif


namespace utils {

/**
 * @fn time_event
 * @brief Get the overall run time (start -> end) of a cl::sycl::event enqueued
 * on a queue with profiling.
 */
template <>
inline double time_event<cl::sycl::event>(cl::sycl::event& e) {
  // get start and end times
  cl_ulong start_time = e.template get_profiling_info<
      cl::sycl::info::event_profiling::command_start>();

  cl_ulong end_time = e.template get_profiling_info<
      cl::sycl::info::event_profiling::command_end>();

  // return the delta
  return static_cast<double>(end_time - start_time);
}

#ifdef SYCL_BLAS_FPGA
inline cl::sycl::queue make_queue_impl() {

  cl::sycl::queue q;

  if (!args.device.empty()) {
    // Initialise the command line device selector in a unique pointer so that
    // we can register an `atexit` handler to delete the command line device
    // selector. If not, then if benchmark::Initialize calls exit() (for
    // example, if the flag `--help` is passed), the cds will not be freed
    // before the sycl runtime tries to exit, leading to an exception, as the
    // cds will still hold some sycl objects.
    cdsp = std::unique_ptr<::utils::cli_device_selector>(
        new ::utils::cli_device_selector(args.device));
    std::atexit(free_device_selector);

    // Create a queue from the device selector - do this after initialising
    // googlebench, as otherwise we may not be able to delete the queue before
    // we exit (if Initialise calls exit(0)), and dump some information about it
    q = cl::sycl::queue(*cdsp.get(),
                        {cl::sycl::property::queue::enable_profiling()});
  } else {
    q = cl::sycl::queue(cl::sycl::default_selector(),
                        {cl::sycl::property::queue::enable_profiling()});
  }
  ::utils::print_queue_information(q);

  return q;
}


inline cl::sycl::queue make_queue(){
  // Provide cached SYCL queue, to avoid recompiling kernels for each test case.
  static cl::sycl::queue queue = make_queue_impl(); 

  return queue;
}
#endif // SYCL_BLAS_FPGA

}  // namespace utils
}  // namespace blas_benchmark

#endif
