#ifndef CLBLAST_UTILS_HPP
#define CLBLAST_UTILS_HPP

#include "clwrap.h"
#include "common_utils.hpp"
#include <clblast.h>

typedef Context ExecutorType;
typedef std::unique_ptr<ExecutorType> ExecutorPtr;

namespace Global {
extern ExecutorPtr executorInstancePtr;
}

namespace benchmark {
namespace utils {

/**
 * @fn translate_transposition
 * @brief Helper function to translate transposition information from netlib
 * blas style strings into clblast types.
 */
inline clblast::Transpose translate_transposition(const char *t_str) {
  if (t_str[0] == 'n') {
    return clblast::Transpose::kNo;
  } else if (t_str[0] == 't') {
    return clblast::Transpose::kYes;
  } else if (t_str[0] == 'c') {
    return clblast::Transpose::kConjugate;
  } else {
    throw std::runtime_error("Got invalid transpose parameter!");
  }
}

/**
 * @fn time_event
 * @brief Get the overall run time (start -> end) of the given event (see
 * CLEventHandler class in clwrap.h)
 */
template <>
inline cl_ulong time_event<cl_event>(cl_event &e) {
  cl_ulong start_time, end_time;
  bool all_ok = true;
  // Declare a lambda to check the result of the calls
  auto check_call = [&all_ok](cl_int status) {
    switch (status) {
      case CL_SUCCESS:
        return;
        break;
      case CL_PROFILING_INFO_NOT_AVAILABLE:
        std::cerr << "The opencl queue has not been configured with profiling "
                     "information! "
                  << std::endl;
        break;
      case CL_INVALID_VALUE:
        std::cerr << "param_name is not valid, or size of param is < "
                     "param_value_size in profiling call!"
                  << std::endl;
        break;
      case CL_INVALID_EVENT:
        std::cerr << "event is invalid in profiling call " << std::endl;
        break;
      case CL_OUT_OF_RESOURCES:
        std::cerr << "cl_out_of_resources in profiling call" << std::endl;
        break;
      case CL_OUT_OF_HOST_MEMORY:
        std::cerr << "cl_out_of_host_memory in profiling call" << std::endl;
        break;
      default:
        // If we've reached this point, something's gone wrong - set the error
        // flag
        all_ok = false;
        break;
    }
  };
  check_call(clGetEventProfilingInfo(e, CL_PROFILING_COMMAND_START,
                                     sizeof(cl_ulong), &start_time, NULL));
  check_call(clGetEventProfilingInfo(e, CL_PROFILING_COMMAND_END,
                                     sizeof(cl_ulong), &end_time, NULL));

  CLEventHandler::release(e);

  // Return the delta
  if (all_ok) {
    return (end_time - start_time);
  } else {
    // Return a really big number to show that we've failed.
    return 0xFFFFFFFFFFFFFFFF;
  }
}

/**
 * @fn warmup
 * @brief Warm up to avoid benchmarking data transfer
 */
template <typename function_t, typename... args_t>
inline void warmup(function_t func, args_t &&... args) {
  for (int i = 0; i < 10; ++i) {
    func(std::forward<args_t>(args)...);
  }
}

}  // namespace utils
}  // namespace benchmark

#endif
