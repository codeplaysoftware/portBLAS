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

/* We don't want to return exceptions in destructors. #define them out for now.
 */
inline void show_error(std::string err_str) {
  std::cerr << "Got error that we would otherwise have thrown: " << err_str
            << std::endl;
}

#ifdef THROW_EXCEPTIONS
#define do_error throw std::runtime_error
#else
#define do_error show_error
#endif

char *getCLErrorString(cl_int err);

class OpenCLDeviceSelector {
  cl_device_id best_device = NULL;
  cl_platform_id best_platform = NULL;
  int best_score = 0;

  static cl_device_type match_device_type(std::string requested);

  static int score_platform(std::string requested, cl_platform_id platform);

  static int score_device(std::string requested, cl_device_id device);

  static cl_uint get_platform_count();

  static cl_uint get_device_count(cl_platform_id plat);

 public:
  OpenCLDeviceSelector(std::string vendor, std::string type);

  cl_device_id device();
  cl_platform_id platform();
};

class Context {
  cl_platform_id platform = NULL;
  cl_device_id device = NULL;
  cl_context context = NULL;
  cl_command_queue command_queue = NULL;
  bool is_active = false;

 public:
  Context(OpenCLDeviceSelector oclds = OpenCLDeviceSelector("*", "*"));

  // Delete the copy constructor so that we don't accidentally leak references
  // to the underlying opencl context
  Context(const Context &) = delete;

  Context(Context &&c)
      : platform(c.plt()),
        device(c.dev()),
        context(c.ctx()),
        is_active(c.active()),
        command_queue(c.queue()) {}

  void create();

  bool active();

  cl_context ctx();

  cl_device_id dev();

  cl_platform_id plt();

  operator cl_context() const { return context; }

  cl_command_queue *_queue();

  cl_command_queue queue() const;

  ~Context();
};

class CLEventHandler {
 public:
  static void wait(cl_event event);

  static void wait(std::vector<cl_event> &&events);

  static void release(cl_event event);

  static void release(std::vector<cl_event> &&events);
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
