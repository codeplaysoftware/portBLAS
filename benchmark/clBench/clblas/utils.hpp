#ifndef CLBLAST_UTILS_HPP
#define CLBLAST_UTILS_HPP

#include <clBLAS.h>
#include <clBench/clwrap.hpp>
#include <common/common_utils.hpp>

typedef Context ExecutorType;

namespace blas_benchmark {

void create_benchmark(blas_benchmark::Args &args, ExecutorType *exPtr,
                      bool *success);

namespace utils {

static inline void print_device_information(cl_device_id device) {
  cl_platform_id platform;
  clGetDeviceInfo(device, CL_DEVICE_PLATFORM, sizeof(cl_platform_id), &platform,
                  nullptr);
  size_t platform_name_length = 0;
  clGetPlatformInfo(platform, CL_PLATFORM_NAME, 0, nullptr,
                    &platform_name_length);
  std::string platform_name(platform_name_length, '\0');
  clGetPlatformInfo(platform, CL_PLATFORM_NAME, platform_name_length,
                    &platform_name[0], nullptr);
  std::cerr << "Platform name: " << platform_name << std::endl;

  size_t device_name_length = 0;
  clGetDeviceInfo(device, CL_DEVICE_NAME, 0, nullptr, &device_name_length);
  if (!device_name_length) {
    return;
  }
  std::string device_name(device_name_length, '\0');
  clGetDeviceInfo(device, CL_DEVICE_NAME, device_name.size(), &device_name[0],
                  nullptr);
  cl_device_type device_type;
  clGetDeviceInfo(device, CL_DEVICE_TYPE, sizeof(cl_device_type), &device_type,
                  nullptr);
  std::cerr << "Device name: " << device_name << std::endl;
  std::cerr << "Device type: ";
  switch (device_type) {
    case CL_DEVICE_TYPE_CPU:
      std::cerr << "cpu";
      break;
    case CL_DEVICE_TYPE_GPU:
      std::cerr << "gpu";
      break;
    case CL_DEVICE_TYPE_ACCELERATOR:
      std::cerr << "accelerator";
      break;
    case CL_DEVICE_TYPE_DEFAULT:
      std::cerr << "default";
      break;
    default:
      std::cerr << "unknown";
      break;
  };
  std::cerr << std::endl;
}

/**
 * @fn translate_transposition
 * @brief Helper function to translate transposition information from netlib
 * blas style strings into clblast types.
 */
static inline clblasTranspose translate_transposition(const char *t_str) {
  if (t_str[0] == 'n') {
    return clblasNoTrans;
  } else if (t_str[0] == 't') {
    return clblasTrans;
  } else if (t_str[0] == 'c') {
    return clblasConjTrans;
  } else {
    throw std::runtime_error("Got invalid transpose parameter!");
  }
}

static inline clblasSide translate_side(const char *side) {
  if (side[0] == 'r') {
    return clblasRight;
  } else if (side[0] == 'l') {
    return clblasLeft;
  } else {
    throw std::runtime_error("Got invalid side parameter!");
  }
}

static inline clblasUplo translate_triangle(const char *triangle) {
  if (triangle[0] == 'u') {
    return clblasUpper;
  } else if (triangle[0] == 'l') {
    return clblasLower;
  } else {
    throw std::runtime_error("Got invalid triangle parameter!");
  }
}

static inline clblasDiag translate_diagonal(const char *diag) {
  if (diag[0] == 'u') {
    return clblasUnit;
  }
  if (diag[0] == 'n') {
    return clblasNonUnit;
  } else {
    throw std::runtime_error("Got invalid diagonal parameter!");
  }
}

/**
 * @fn time_event
 * @brief Get the overall run time (start -> end) of the given event (see
 * CLEventHandler class in clwrap.h)
 */
template <>
inline double time_event<cl_event>(cl_event &e) {
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
    return static_cast<double>(end_time - start_time);
  } else {
    // Return a really big number to show that we've failed.
    return static_cast<double>(0xFFFFFFFFFFFFFFFF);
  }
}

}  // namespace utils
}  // namespace blas_benchmark

#endif
