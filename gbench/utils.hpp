#ifndef UTILS_HPP
#define UTILS_HPP

#include <CL/sycl.hpp>
#include <benchmark/benchmark.h>
#include <chrono>

#include <memory>

#include <interface/blas1_interface.hpp>
#include <interface/blas2_interface.hpp>
#include <interface/blas3_interface.hpp>

// Forward declare methods that we use in `benchmark.cpp`, but define in
// `main.cpp`
typedef std::shared_ptr<blas::Executor<SYCL>> ExecutorPtr;

ExecutorPtr getExecutor();

namespace benchmark {
namespace utils {

/**
 * @fn time_event
 * @brief Get the overall run time (start -> end) of a cl::sycl::event enqueued
 * on a queue with profiling.
 */
inline cl_ulong time_event(cl::sycl::event e) {
  // get start and ed times
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

template <typename EventT, typename... OtherEvents>
inline cl_ulong time_events(EventT first_event, OtherEvents... next_events) {
  return time_events<EventT>({first_event, next_events...});
}

/**
 * @fn random_scalar
 * @brief Generates a random scalar value, using an arbitrary low quality
 * algorithm.
 */
template <typename ScalarT>
static inline ScalarT random_scalar() {
  return 1e-3 * ((rand() % 2000) - 1000);
}

/**
 * @fn random_data
 * @brief Generates a random vector of scalar values, using an arbitrary low
 * quality algorithm.
 */
template <typename ScalarT>
static inline std::vector<ScalarT> random_data(size_t size,
                                               bool initialized = true) {
  std::vector<ScalarT> v = std::vector<ScalarT>(size);
  if (initialized) {
    std::transform(v.begin(), v.end(), v.begin(), [](ScalarT x) -> ScalarT {
      return random_scalar<ScalarT>();
    });
  }
  return v;
}

/**
 * @fn const_data
 * @brief Generates a vector of constant values, of a given length.
 */
template <typename ScalarT>
static inline std::vector<ScalarT> const_data(size_t size,
                                              ScalarT const_value = 0) {
  std::vector<ScalarT> v = std::vector<ScalarT>(size);
  std::fill(v.begin(), v.end(), const_value);
  return v;
}

enum class Transposition { Normal, Transposed, Conjugate };
/**
 * @fn to_transpose_enum
 * @brief Translates from a transposition string to an enum.
 */
static inline Transposition to_transpose_enum(const char* t) {
  if (t[0] == 't') {
    return Transposition::Transposed;
  } else if (t[0] == 'c') {
    return Transposition::Conjugate;
  } else {
    return Transposition::Normal;
  }
}
/**
 * @fn from_transpose_enum
 * @brief Translates from a transposition enum to a transposition string
 */
static inline const char* from_transpose_enum(Transposition t) {
  switch (t) {
    case Transposition::Transposed:
      return "t";
      break;
    case Transposition::Conjugate:
      return "c";
      break;
    case Transposition::Normal:
      return "n";
      break;
    default:
      return "n";
  }
}
}  // namespace utils
}  // namespace benchmark

#endif