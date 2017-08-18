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
      throw std::runtime_error("failure in clGetPlatformIDs");
    }
    return num_platforms;
  }

  static cl_platform_id get_platform_id(size_t platform_id = 0) {
    cl_uint num_platforms = get_platform_count();
    cl_platform_id platforms[num_platforms];
    cl_int status = clGetPlatformIDs(num_platforms, platforms, NULL);
    if (status != CL_SUCCESS) {
      throw std::runtime_error("failure in clGetPlatformIDs");
    }
    cl_platform_id platform = platforms[platform_id];
    return platform;
  }

  static cl_uint get_device_count(cl_platform_id plat) {
    cl_uint num_devices;
    cl_int status =
        clGetDeviceIDs(plat, CL_DEVICE_TYPE_ALL, 0, NULL, &num_devices);
    if (status != CL_SUCCESS) {
      throw std::runtime_error("failure in clGetDeviceIDs");
    }
    return num_devices;
  }

  static cl_device_id get_device_id(cl_platform_id plat, size_t device_id = 0) {
    cl_uint num_devices = get_device_count(plat);
    cl_device_id devices[num_devices];
    cl_int status =
        clGetDeviceIDs(plat, CL_DEVICE_TYPE_ALL, num_devices, devices, NULL);
    if (status != CL_SUCCESS) {
      throw std::runtime_error("failure in clGetDeviceIDs");
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
    if (is_active) throw std::runtime_error("context is already active");
    cl_int status;
    context = clCreateContext(NULL, 1, &device, NULL, NULL, &status);
    if (status != CL_SUCCESS) {
      throw std::runtime_error("failure to create context");
    }
    command_queue = clCreateCommandQueue(context, device, 0, &status);
    if (status != CL_SUCCESS) {
      throw std::runtime_error("failure to create command queue");
    }
    is_active = true;
  }

  void release() {
    cl_int status = clReleaseCommandQueue(command_queue);
    if (status != CL_SUCCESS) {
      throw std::runtime_error("failure to release command queue");
    }
    status = clReleaseContext(context);
    if (status != CL_SUCCESS) {
      throw std::runtime_error("failure to release context");
    }
    is_active = false;
  }

  operator cl_context() const { return context; }

  cl_command_queue *_queue() { return &command_queue; }

  cl_command_queue queue() const { return command_queue; }

  ~Context() {
    if (is_active) release();
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
      throw std::runtime_error("failure in clWaitForEvents");
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
      throw std::runtime_error("failure to release an event");
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
  bool private_host_ptr = false;
  bool is_active = false;

  void init() {
    if (is_active) {
      throw std::runtime_error("buffer is already active");
    }
    cl_int status;
    dev_ptr =
        clCreateBuffer(context, Options, size * sizeof(ScalarT), NULL, &status);
    if (status != CL_SUCCESS) {
      throw std::runtime_error("failure to create buffer");
    }
    is_active = true;
    if (to_write) {
      status =
          clEnqueueWriteBuffer(context.queue(), dev_ptr, CL_TRUE, 0,
                               size * sizeof(ScalarT), host_ptr, 0, NULL, NULL);
      if (status != CL_SUCCESS) {
        throw std::runtime_error("failure in clEnqueueWriteBuffer");
      }
    }
  }

 public:
  MemBuffer(Context &ctx, ScalarT *ptr, size_t size)
      : context(ctx), host_ptr(ptr), size(size) {
    init();
  }

  MemBuffer(Context &ctx, size_t size, bool initialized = true)
      : context(ctx), size(size) {
    private_host_ptr = true;
    host_ptr = new_data<ScalarT>(size, initialized);
    init();
  }

  ScalarT operator[](size_t i) const { return host_ptr[i]; }

  ScalarT &operator[](size_t i) { return host_ptr[i]; }

  cl_mem dev() { return dev_ptr; }

  ScalarT *host() { return host_ptr; }

  void release() {
    if (to_read) {
      cl_int status;
      status =
          clEnqueueReadBuffer(context.queue(), dev_ptr, CL_TRUE, 0,
                              size * sizeof(ScalarT), host_ptr, 0, NULL, NULL);
      if (status != CL_SUCCESS) {
        throw std::runtime_error("failure in clEnqueueReadBuffer");
      }
    }
    if (!is_active) {
      throw std::runtime_error("cannot release inactive buffer");
    }
    cl_int status = clReleaseMemObject(dev_ptr);
    if (status != CL_SUCCESS) {
      throw std::runtime_error("failure to release memobject");
    }
    is_active = false;
  }

  ~MemBuffer() {
    if (is_active) {
      release();
    }
    if (private_host_ptr) {
      release_data(host_ptr);
    }
  }
};

#endif /* end of include guard: CLWRAP_HPP */
