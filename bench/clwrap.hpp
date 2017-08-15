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
    clGetPlatformIDs(0, NULL, &num_platforms);
    return num_platforms;
  }

  static cl_platform_id get_platform_id(size_t platform_id = 0) {
    cl_uint num_platforms = get_platform_count();
    cl_platform_id platforms[num_platforms];
    clGetPlatformIDs(num_platforms, platforms, NULL);
    cl_platform_id platform = platforms[platform_id];
    return platform;
  }

  static cl_uint get_device_count(cl_platform_id plat) {
    cl_uint num_devices;
    clGetDeviceIDs(plat, CL_DEVICE_TYPE_ALL, 0, NULL, &num_devices);
    return num_devices;
  }

  static cl_device_id get_device_id(cl_platform_id plat, size_t device_id = 0) {
    cl_uint num_devices = get_device_count(plat);
    cl_device_id devices[num_devices];
    clGetDeviceIDs(plat, CL_DEVICE_TYPE_ALL, num_devices, devices, NULL);
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
    context = clCreateContext(NULL, 1, &device, NULL, NULL, NULL);
    command_queue = clCreateCommandQueue(context, device, 0, NULL);
    is_active = true;
  }

  void release() {
    clReleaseCommandQueue(command_queue);
    clReleaseContext(context);
    is_active = false;
  }

  operator cl_context() const { return context; }

  cl_command_queue *_queue() { return &command_queue; }

  cl_command_queue queue() const { return command_queue; }

  ~Context() {
    if (is_active) release();
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
    if (is_active) throw std::runtime_error("buffer is already active");
    dev_ptr =
        clCreateBuffer(context, Options, size * sizeof(ScalarT), NULL, NULL);
    is_active = true;
    if (to_write)
      clEnqueueWriteBuffer(context.queue(), dev_ptr, CL_TRUE, 0,
                           size * sizeof(ScalarT), host_ptr, 0, NULL, NULL);
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
    if (to_read)
      clEnqueueReadBuffer(context.queue(), dev_ptr, CL_TRUE, 0,
                          size * sizeof(ScalarT), host_ptr, 0, NULL, NULL);
    if (!is_active) throw std::runtime_error("cannot release inactive buffer");
    clReleaseMemObject(dev_ptr);
    is_active = false;
  }

  ~MemBuffer() {
    if (is_active) release();
    if (private_host_ptr) release_data(host_ptr);
  }
};

#endif /* end of include guard: CLWRAP_HPP */
