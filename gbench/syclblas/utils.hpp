#ifndef SYCL_UTILS_HPP
#define SYCL_UTILS_HPP

#include "common_utils.hpp"
#include <CL/sycl.hpp>
#include <chrono>
#include <tuple>

#include "sycl_blas.h"

// Forward declare methods that we use in `benchmark.cpp`, but define in
// `main.cpp`
typedef blas::Executor<blas::PolicyHandler<blas::codeplay_policy>>
    SyclExecutorType;
typedef std::shared_ptr<SyclExecutorType> ExecutorPtr;
ExecutorPtr getExecutor();

namespace benchmark {
namespace utils {

/**
 * @fn time_event
 * @brief Get the overall run time (start -> end) of a cl::sycl::event enqueued
 * on a queue with profiling.
 */
inline cl_ulong time_event(cl::sycl::event e) {
  // get start and end times
  cl_ulong start_time = e.template get_profiling_info<
      cl::sycl::info::event_profiling::command_start>();

  cl_ulong end_time = e.template get_profiling_info<
      cl::sycl::info::event_profiling::command_end>();

  // return the delta
  return (end_time - start_time);
}

/**
 * @fn time_events
 * @brief Times n events, and returns the aggregate time.
 */
template <typename EventT>
inline cl_ulong time_events(std::vector<EventT> es) {
  cl_ulong total_time = 0;
  for (auto e : es) {
    total_time += time_event(e);
  }
  return total_time;
}

/**
 * @fn timef
 * @brief Calculates the time spent executing the function func
 * (both overall and event time, returned in nanoseconds in a tuple of double)
 */
template <typename F, typename... Args>
static std::tuple<double, double> timef(F func, Args&&... args) {
  auto start = std::chrono::system_clock::now();
  auto event = func(std::forward<Args>(args)...);
  auto end = std::chrono::system_clock::now();
  double overall_time = (end - start).count();

  double event_time = static_cast<double>(time_events(event));

  return std::make_tuple(overall_time, event_time);
}


template <typename EventT, typename... OtherEvents>
inline cl_ulong time_events(EventT first_event, OtherEvents... next_events) {
  return time_events<EventT>(
      blas::concatenate_vectors(first_event, next_events...));
}

}  // namespace utils
}  // namespace benchmark

#endif
