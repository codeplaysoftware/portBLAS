/***************************************************************************
 *
 *  @license
 *  Copyright (C) 2016 Codeplay Software Limited
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
 *  @filename clwrap.cpp
 *
 **************************************************************************/

#ifndef CLWRAP_HPP
#define CLWRAP_HPP

#include <stdexcept>

#include <CL/cl.h>

/* We don't want to return exceptions in destructors. #define them out for now. */
void show_error(std::string err_str) {
  std::cerr << "Got error that we would otherwise have thrown: " << err_str << std::endl;
}

#ifdef THROW_EXCEPTIONS
#define do_error throw std::runtime_error 
#else 
#define do_error show_error 
#endif

char *getCLErrorString(cl_int err) {
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

class DeviceSelector {
  std::string m_vendor_name;
  std::string m_device_type;

  static cl_device_type match_device_type(std::string requested) {
    if (requested.empty()) return CL_DEVICE_TYPE_ALL;
    std::transform(requested.begin(), requested.end(), requested.begin(),
                   ::tolower);
    if (requested == "gpu") return CL_DEVICE_TYPE_GPU;
    if (requested == "cpu") return CL_DEVICE_TYPE_CPU;
    if (requested == "accel") return CL_DEVICE_TYPE_ACCELERATOR;
    if (requested == "*" || requested == "any") return CL_DEVICE_TYPE_ALL;

    return CL_DEVICE_TYPE_ALL;
  }

 public:
  // cli_device_selector(std::string vendor_name, std::string device_type)
  //     : cl::sycl::device_selector(),
  //       m_vendor_name(vendor_name),
  //       m_device_type(device_type) {}

  // int operator()(const cl::sycl::device &device) const {
  //   int score = 0;

  //   // Score the device type...
  //   cl::sycl::info::device_type dtype =
  //       device.get_info<cl::sycl::info::device::device_type>();
  //   cl::sycl::info::device_type rtype = match_device_type(m_device_type);
  //   if (rtype == dtype || rtype == cl::sycl::info::device_type::all) {
  //     score += 2;
  //   } else if (rtype == cl::sycl::info::device_type::automatic) {
  //     score += 1;
  //   } else {
  //     score -= 2;
  //   }

  //   // score the vendor name
  //   cl::sycl::platform plat = device.get_platform();
  //   std::string name = plat.template
  //   get_info<cl::sycl::info::platform::name>(); std::transform(name.begin(),
  //   name.end(), name.begin(), ::tolower); if (name.find(m_vendor_name) !=
  //   std::string::npos &&
  //       !m_vendor_name.empty()) {
  //     score += 2;
  //   } else if (m_vendor_name == "*" || m_vendor_name.empty()) {
  //     score += 1;
  //   } else {
  //     score -= 2;
  //   }
  //   return score;
  // }
};

class Context {
  cl_platform_id platform = NULL;
  cl_device_id device = NULL;
  cl_context context = NULL;
  cl_command_queue command_queue = NULL;
  bool is_active = false;

  static cl_uint get_platform_count() {
    cl_uint num_platforms;
    cl_int status = clGetPlatformIDs(0, NULL, &num_platforms);
    if (status != CL_SUCCESS) {
      do_error("failure in clGetPlatformIDs");
    }
    return num_platforms;
  }

  static cl_platform_id get_platform_id(size_t platform_id = 0) {
    cl_uint num_platforms = get_platform_count();
    cl_platform_id platforms[num_platforms];
    cl_int status = clGetPlatformIDs(num_platforms, platforms, NULL);
    if (status != CL_SUCCESS) {
      do_error("failure in clGetPlatformIDs");
    }
    cl_platform_id platform = platforms[platform_id];
    return platform;
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

  static cl_device_id get_device_id(cl_platform_id plat, size_t device_id = 0) {
    cl_uint num_devices = get_device_count(plat);
    cl_device_id devices[num_devices];
    cl_int status =
        clGetDeviceIDs(plat, CL_DEVICE_TYPE_ALL, num_devices, devices, NULL);
    if (status != CL_SUCCESS) {
      do_error("failure in clGetDeviceIDs");
    }
    cl_device_id device = devices[device_id];
    return device;
  }

 public:
  Context(size_t plat_id = 0, size_t dev_id = 0) {
    platform = get_platform_id(plat_id);
    device = get_device_id(platform, dev_id);
    create();
  }

  void create() {
    if (is_active) do_error("context is already active");
    cl_int status;
    context = clCreateContext(NULL, 1, &device, NULL, NULL, &status);
    if (status != CL_SUCCESS) {
      do_error("failure to create context");
    }
    command_queue = clCreateCommandQueue(context, device, 0, &status);
    if (status != CL_SUCCESS) {
      do_error("failure to create command queue");
    }
    is_active = true;
  }

  bool active() { return is_active; }

  cl_context ctx() { return context; }

  cl_device_id dev() { return device; }

  cl_platform_id plt() { return platform; }

  operator cl_context() const { return context; }

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
};

class Event {
  cl_event event;

 public:
  Event() {}

  cl_event &_cl() { return event; }

  void wait() {
    cl_int status = clWaitForEvents(1, &event);
    if (status != CL_SUCCESS) {
      do_error("failure in clWaitForEvents");
    }
  }

  static void wait(std::vector<Event> &&events) {
    for (auto &ev : events) {
      ev.wait();
    }
  }

  void release() {
    cl_int status = clReleaseEvent(event);
    if (status != CL_SUCCESS) {
      do_error("failure to release an event");
    }
  }

  template <typename... EVs>
  static void release(std::vector<Event> &&events) {
    for (auto &&ev : events) {
      ev.wait();
    }
  }
};

template <typename ScalarT, int Options = CL_MEM_READ_WRITE>
class MemBuffer {
 public:
  static constexpr const bool to_write =
      (Options == CL_MEM_READ_WRITE || Options == CL_MEM_WRITE_ONLY);
  static constexpr const bool to_read =
      (Options == CL_MEM_READ_WRITE || Options == CL_MEM_READ_ONLY);

 private:
  Context &context;
  size_t size = 0;
  cl_mem dev_ptr = NULL;
  ScalarT *host_ptr = NULL;
  bool is_active = false;

  void init() {
    if (is_active) {
      do_error("buffer is already active");
    }
    cl_int status;
    dev_ptr =
        clCreateBuffer(context, Options, size * sizeof(ScalarT), NULL, &status);
    if (status != CL_SUCCESS) {
      do_error("failure to create buffer");
    }
    is_active = true;
    if (to_write) {
      status =
          clEnqueueWriteBuffer(context.queue(), dev_ptr, CL_TRUE, 0,
                               size * sizeof(ScalarT), host_ptr, 0, NULL, NULL);
      if (status != CL_SUCCESS) {
        do_error("failure in clEnqueueWriteBuffer");
      }
    }
  }

 public:
  MemBuffer(Context &ctx, ScalarT *ptr, size_t size)
      : context(ctx), host_ptr(ptr), size(size) {
    init();
  }

  ScalarT operator[](size_t i) const { return host_ptr[i]; }

  ScalarT &operator[](size_t i) { return host_ptr[i]; }

  cl_mem dev() { return dev_ptr; }

  ScalarT *host() { return host_ptr; }

  ~MemBuffer() {
    if (is_active) {
      if (to_read) {
        cl_int status;
        status = clEnqueueReadBuffer(context.queue(), dev_ptr, CL_TRUE, 0,
                                     size * sizeof(ScalarT), host_ptr, 0, NULL,
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
