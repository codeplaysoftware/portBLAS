#ifndef SYCL_UTILS_HPP
#define SYCL_UTILS_HPP

#include <CL/sycl.hpp>
#include <chrono>
#include <tuple>

#include "sycl_blas.h"
#include "common_utils.hpp"

// Forward declare methods that we use in `benchmark.cpp`, but define in
// `main.cpp`
typedef blas::Executor<blas::PolicyHandler<blas::codeplay_policy>>
    SyclExecutorType;
typedef std::unique_ptr<SyclExecutorType> ExecutorPtr;

// Declare the global executor pointer
namespace Global {
extern ExecutorPtr executorInstancePtr;
}

namespace benchmark {
namespace utils {

/**
 * @fn time_event
 * @brief Get the overall run time (start -> end) of a cl::sycl::event enqueued
 * on a queue with profiling.
 */
template<>
inline cl_ulong time_event<cl::sycl::event>(cl::sycl::event& e) {
  // get start and end times
  cl_ulong start_time = e.template get_profiling_info<
      cl::sycl::info::event_profiling::command_start>();

  cl_ulong end_time = e.template get_profiling_info<
      cl::sycl::info::event_profiling::command_end>();

  // return the delta
  return (end_time - start_time);
}

inline void print_queue_information(cl::sycl::queue q) {
  std::cerr
      << "Device vendor: "
      << q.get_device().template get_info<cl::sycl::info::device::vendor>()
      << std::endl;
  std::cerr << "Device name: "
            << q.get_device().template get_info<cl::sycl::info::device::name>()
            << std::endl;
  std::cerr << "Device type: ";
  switch (
      q.get_device().template get_info<cl::sycl::info::device::device_type>()) {
    case cl::sycl::info::device_type::cpu:
      std::cerr << "cpu";
      break;
    case cl::sycl::info::device_type::gpu:
      std::cerr << "gpu";
      break;
    case cl::sycl::info::device_type::accelerator:
      std::cerr << "accelerator";
      break;
    case cl::sycl::info::device_type::custom:
      std::cerr << "custom";
      break;
    case cl::sycl::info::device_type::automatic:
      std::cerr << "automatic";
      break;
    case cl::sycl::info::device_type::host:
      std::cerr << "host";
      break;
    case cl::sycl::info::device_type::all:
      std::cerr << "all";
      break;
    default:
      std::cerr << "unknown";
      break;
  };
  std::cerr << std::endl;
}

}  // namespace utils
}  // namespace benchmark

#endif
