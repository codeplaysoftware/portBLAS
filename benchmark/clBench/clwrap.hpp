/***************************************************************************
 *
 *  @license
 *  Copyright (C) 2018 Codeplay Software Limited
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  For your convenience, a copy of the License has been included in this
 *  repository.
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 *
 *  SYCL-BLAS: BLAS implementation using SYCL
 *
 *  @filename clwrap.h
 *
 **************************************************************************/

#ifndef CLWRAP_H
#define CLWRAP_H

#include <iostream>
#include <stdexcept>
#include <vector>

#include <CL/cl.h>
#include <algorithm>
#include <memory>

/* We don't want to return exceptions in destructors. #define them out for now.
 */
static inline void show_error(std::string err_str) {
  std::cerr << "Got error that we would otherwise have thrown: " << err_str
            << std::endl;
}

#ifdef THROW_EXCEPTIONS
#define do_error throw std::runtime_error
#else
#define do_error show_error
#endif

static char *getCLErrorString(cl_int err) {
  switch (err) {
    case CL_SUCCESS:
      return (char *)"Success!";
    case CL_DEVICE_NOT_FOUND:
      return (char *)"Device not found.";
    case CL_DEVICE_NOT_AVAILABLE:
      return (char *)"Device not available";
    case CL_COMPILER_NOT_AVAILABLE:
      return (char *)"Compiler not available";
    case CL_MEM_OBJECT_ALLOCATION_FAILURE:
      return (char *)"Memory object allocation failure";
    case CL_OUT_OF_RESOURCES:
      return (char *)"Out of resources";
    case CL_OUT_OF_HOST_MEMORY:
      return (char *)"Out of host memory";
    case CL_PROFILING_INFO_NOT_AVAILABLE:
      return (char *)"Profiling information not available";
    case CL_MEM_COPY_OVERLAP:
      return (char *)"Memory copy overlap";
    case CL_IMAGE_FORMAT_MISMATCH:
      return (char *)"Image format mismatch";
    case CL_IMAGE_FORMAT_NOT_SUPPORTED:
      return (char *)"Image format not supported";
    case CL_BUILD_PROGRAM_FAILURE:
      return (char *)"Program build failure";
    case CL_MAP_FAILURE:
      return (char *)"Map failure";
    case CL_INVALID_VALUE:
      return (char *)"Invalid value";
    case CL_INVALID_DEVICE_TYPE:
      return (char *)"Invalid device type";
    case CL_INVALID_PLATFORM:
      return (char *)"Invalid platform";
    case CL_INVALID_DEVICE:
      return (char *)"Invalid device";
    case CL_INVALID_CONTEXT:
      return (char *)"Invalid context";
    case CL_INVALID_QUEUE_PROPERTIES:
      return (char *)"Invalid queue properties";
    case CL_INVALID_COMMAND_QUEUE:
      return (char *)"Invalid command queue";
    case CL_INVALID_HOST_PTR:
      return (char *)"Invalid host pointer";
    case CL_INVALID_MEM_OBJECT:
      return (char *)"Invalid memory object";
    case CL_INVALID_IMAGE_FORMAT_DESCRIPTOR:
      return (char *)"Invalid image format descriptor";
    case CL_INVALID_IMAGE_SIZE:
      return (char *)"Invalid image size";
    case CL_INVALID_SAMPLER:
      return (char *)"Invalid sampler";
    case CL_INVALID_BINARY:
      return (char *)"Invalid binary";
    case CL_INVALID_BUILD_OPTIONS:
      return (char *)"Invalid build options";
    case CL_INVALID_PROGRAM:
      return (char *)"Invalid program";
    case CL_INVALID_PROGRAM_EXECUTABLE:
      return (char *)"Invalid program executable";
    case CL_INVALID_KERNEL_NAME:
      return (char *)"Invalid kernel name";
    case CL_INVALID_KERNEL_DEFINITION:
      return (char *)"Invalid kernel definition";
    case CL_INVALID_KERNEL:
      return (char *)"Invalid kernel";
    case CL_INVALID_ARG_INDEX:
      return (char *)"Invalid argument index";
    case CL_INVALID_ARG_VALUE:
      return (char *)"Invalid argument value";
    case CL_INVALID_ARG_SIZE:
      return (char *)"Invalid argument size";
    case CL_INVALID_KERNEL_ARGS:
      return (char *)"Invalid kernel arguments";
    case CL_INVALID_WORK_DIMENSION:
      return (char *)"Invalid work dimension";
    case CL_INVALID_WORK_GROUP_SIZE:
      return (char *)"Invalid work group size";
    case CL_INVALID_WORK_ITEM_SIZE:
      return (char *)"Invalid work item size";
    case CL_INVALID_GLOBAL_OFFSET:
      return (char *)"Invalid global offset";
    case CL_INVALID_EVENT_WAIT_LIST:
      return (char *)"Invalid event wait list";
    case CL_INVALID_EVENT:
      return (char *)"Invalid event";
    case CL_INVALID_OPERATION:
      return (char *)"Invalid operation";
    case CL_INVALID_GL_OBJECT:
      return (char *)"Invalid OpenGL object";
    case CL_INVALID_BUFFER_SIZE:
      return (char *)"Invalid buffer size";
    case CL_INVALID_MIP_LEVEL:
      return (char *)"Invalid mip-map level";
    default:
      return (char *)"Unknown";
  }
}
class OpenCLDeviceSelector {
  cl_device_id best_device = NULL;
  cl_platform_id best_platform = NULL;
  int best_score = 0;

  static cl_device_type match_device_type(std::string requested) {
    if (requested.empty()) return CL_DEVICE_TYPE_ALL;
    std::transform(requested.begin(), requested.end(), requested.begin(),
                   ::tolower);
    if (requested == "gpu") return CL_DEVICE_TYPE_GPU;
    if (requested == "cpu") return CL_DEVICE_TYPE_CPU;
    if (requested == "accel") return CL_DEVICE_TYPE_ACCELERATOR;
    if (requested == "*" || requested == "any") return CL_DEVICE_TYPE_ALL;

    return CL_DEVICE_TYPE_DEFAULT;
  }

  static int score_platform(std::string requested, cl_platform_id platform) {
    cl_int err = CL_SUCCESS;

    size_t platformNameSize = 0;
    err = clGetPlatformInfo(platform, CL_PLATFORM_NAME, 0, nullptr,
                            &platformNameSize);
    if (err != CL_SUCCESS) {
      do_error("Error acquiring OpenCL platform name");
      return -10;
    }

    std::string name(platformNameSize, '\0');

    err = clGetPlatformInfo(platform, CL_PLATFORM_NAME, platformNameSize,
                            &name[0], nullptr);
    if (err != CL_SUCCESS) {
      do_error("Failure in clGetPlatformInfo");
      return -10;
    }

    std::transform(name.begin(), name.end(), name.begin(), ::tolower);
    int score;
    if (name.find(requested) != std::string::npos && !requested.empty()) {
      score = 2;
    } else if (requested == "*" || requested.empty()) {
      score = 1;
    } else {
      score = -2;
    }
    return score;
  }

  static int score_device(std::string requested, cl_device_id device) {
    // Get the requested device type:
    cl_device_type req_type = match_device_type(requested);
    cl_device_type dev_type;
    cl_int status = clGetDeviceInfo(device, CL_DEVICE_TYPE,
                                    sizeof(cl_device_type), &dev_type, NULL);
    if (status != CL_SUCCESS) {
      do_error("Failure in clGetDeviceInfo");
      return -10;
    }
    int score;
    if (req_type == dev_type || req_type == CL_DEVICE_TYPE_ALL) {
      score = 2;
    } else if (req_type == CL_DEVICE_TYPE_DEFAULT) {
      score = 1;
    } else {
      score = -2;
    }
    return score;
  }

  static cl_uint get_platform_count() {
    cl_uint num_platforms;
    cl_int status = clGetPlatformIDs(0, NULL, &num_platforms);
    if (status != CL_SUCCESS) {
      do_error("failure in clGetPlatformIDs");
    }
    return num_platforms;
  }

  static cl_uint get_device_count(cl_platform_id plat) {
    cl_uint num_devices;
    cl_int status =
        clGetDeviceIDs(plat, CL_DEVICE_TYPE_ALL, 0, NULL, &num_devices);
    if (status != CL_SUCCESS) {
      do_error("failure in clGetDeviceIDs");
    }
    return num_devices;
  }

 public:
  OpenCLDeviceSelector(std::string vendor, std::string type) {
    // Get the number of platforms, and a list of IDs
    cl_uint num_platforms = get_platform_count();
    std::unique_ptr<cl_platform_id[]> platforms(
        new cl_platform_id[num_platforms]);
    cl_int status = clGetPlatformIDs(num_platforms, platforms.get(), NULL);
    if (status != CL_SUCCESS) {
      do_error("failure in clGetPlatformIDs");
    }

    // Iterate over the platforms, and then over each of the devices.
    for (cl_uint platform_id = 0; platform_id < num_platforms; platform_id++) {
      // get the specific ID, and score it.
      cl_platform_id platform = platforms[platform_id];
      int platform_score = score_platform(vendor, platform);

      // Get devices, etc.
      cl_uint num_devices = get_device_count(platform);
      std::unique_ptr<cl_device_id[]> devices(new cl_device_id[num_devices]);
      cl_int status = clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, num_devices,
                                     devices.get(), NULL);
      if (status != CL_SUCCESS) {
        do_error("failure in clGetDeviceIDs");
      }

      // Iterate over the device_ids, and score the combo:
      for (cl_uint device_id = 0; device_id < num_devices; device_id++) {
        cl_device_id device = devices[device_id];
        int device_score = score_device(type, device);
        // SCORE!
        int score = platform_score + device_score;
        if (score > best_score) {
          best_score = score;
          best_device = device;
          best_platform = platform;
        }
      }
    }

    if (best_platform == NULL || best_device == NULL) {
      do_error("No platform or device selected, maybe none match?");
    }
  }
  cl_device_id device() { return best_device; }
  cl_platform_id platform() { return best_platform; }
};

class Context {
  cl_platform_id platform = NULL;
  cl_device_id device = NULL;
  cl_context context = NULL;
  cl_command_queue command_queue = NULL;
  bool is_active = false;

 public:
  // Delete the copy constructor so that we don't accidentally leak references
  // to the underlying opencl context
  Context(const Context &) = delete;

  Context(Context &&c)
      : platform(c.plt()),
        device(c.dev()),
        context(c.ctx()),
        is_active(c.active()),
        command_queue(c.queue()) {}

  Context(OpenCLDeviceSelector oclds = OpenCLDeviceSelector("*", "*")) {
    platform = oclds.platform();
    device = oclds.device();
    create();
  }

  void create() {
    if (is_active) do_error("context is already active");
    cl_int status;
    context = clCreateContext(NULL, 1, &device, NULL, NULL, &status);
    if (status != CL_SUCCESS) {
      do_error("failure to create context");
    }
    command_queue = clCreateCommandQueue(context, device,
                                         CL_QUEUE_PROFILING_ENABLE, &status);
    if (status != CL_SUCCESS) {
      do_error("failure to create command queue");
    }
    is_active = true;
  }

  bool active() { return is_active; }

  cl_context ctx() { return context; }

  cl_device_id dev() { return device; }

  cl_platform_id plt() { return platform; }

  cl_command_queue *_queue() { return &command_queue; }

  cl_command_queue queue() const { return command_queue; }

  ~Context() {
    if (is_active) {
      cl_int status = clReleaseCommandQueue(command_queue);
      if (status != CL_SUCCESS) {
        do_error("failure to release command queue");
      }
      status = clReleaseContext(context);
      if (status != CL_SUCCESS) {
        do_error("failure to release context");
      }
      is_active = false;
    }
  }

  operator cl_context() const { return context; }
};

class CLEventHandler {
 public:
  static void wait(cl_event event) {
    cl_int status = clWaitForEvents(1, &event);
    if (status != CL_SUCCESS) {
      do_error("failure in clWaitForEvents");
    }
  }

  static void wait(std::vector<cl_event> &&events) {
    for (auto &event : events) {
      wait(event);
    }
  }

  static void release(cl_event event) {
    cl_int status = clReleaseEvent(event);
    if (status != CL_SUCCESS) {
      do_error("failure in clReleaseEvent");
    }
  }

  static void release(std::vector<cl_event> &&events) {
    for (auto &&event : events) {
      release(event);
    }
  }
};

template <typename scalar_t, int Options = CL_MEM_READ_WRITE>
class MemBuffer {
 public:
  static constexpr const bool to_write =
      (Options == CL_MEM_READ_WRITE || Options == CL_MEM_WRITE_ONLY);
  static constexpr const bool to_read =
      (Options == CL_MEM_READ_WRITE || Options == CL_MEM_READ_ONLY);

 private:
  Context *context;
  size_t size = 0;
  cl_mem dev_ptr = NULL;
  scalar_t *host_ptr = NULL;
  bool is_active = false;

  void init() {
    if (is_active) {
      do_error("buffer is already active");
    }
    cl_int status;

    dev_ptr = clCreateBuffer(context->ctx(), Options, size * sizeof(scalar_t),
                             NULL, &status);
    if (status != CL_SUCCESS) {
      do_error("failure to create buffer");
    }

    is_active = true;
    if (to_write) {
      status = clEnqueueWriteBuffer(context->queue(), dev_ptr, CL_TRUE, 0,
                                    size * sizeof(scalar_t), host_ptr, 0, NULL,
                                    NULL);
      if (status != CL_SUCCESS) {
        do_error("failure in clEnqueueWriteBuffer");
      }
    }
  }

 public:
  MemBuffer(Context *ctx, scalar_t *ptr, size_t size)
      : context(ctx), host_ptr(ptr), size(size) {
    init();
  }

  scalar_t operator[](size_t i) const { return host_ptr[i]; }

  scalar_t &operator[](size_t i) { return host_ptr[i]; }

  cl_mem dev() { return dev_ptr; }

  scalar_t *host() { return host_ptr; }

  ~MemBuffer() {
    if (is_active) {
      if (to_read) {
        cl_int status;
        status = clEnqueueReadBuffer(context->queue(), dev_ptr, CL_TRUE, 0,
                                     size * sizeof(scalar_t), host_ptr, 0, NULL,
                                     NULL);
        if (status != CL_SUCCESS) {
          do_error("failure in clEnqueueReadBuffer");
        }
      }
      if (!is_active) {
        do_error("cannot release inactive buffer");
      }
      cl_int status = clReleaseMemObject(dev_ptr);

      if (status != CL_SUCCESS) {
        do_error("failure to release memobject");
      }
      is_active = false;
    }
  }
};

#endif /* end of include guard: CLWRAP_HPP */
