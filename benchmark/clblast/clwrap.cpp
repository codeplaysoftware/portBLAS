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
 *  @filename clwrap.cpp
 *
 **************************************************************************/

#include "clwrap.h"

#include <algorithm>

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

/* Class OpenCLDeviceSelector */

cl_device_type OpenCLDeviceSelector::match_device_type(std::string requested) {
  if (requested.empty()) return CL_DEVICE_TYPE_ALL;
  std::transform(requested.begin(), requested.end(), requested.begin(),
                 ::tolower);
  if (requested == "gpu") return CL_DEVICE_TYPE_GPU;
  if (requested == "cpu") return CL_DEVICE_TYPE_CPU;
  if (requested == "accel") return CL_DEVICE_TYPE_ACCELERATOR;
  if (requested == "*" || requested == "any") return CL_DEVICE_TYPE_ALL;

  return CL_DEVICE_TYPE_DEFAULT;
}

int OpenCLDeviceSelector::score_platform(std::string requested,
                                         cl_platform_id platform) {
  const size_t str_size = 1024 * sizeof(char);
  char *str = (char *)malloc(str_size);
  std::string name;

  cl_int status =
      clGetPlatformInfo(platform, CL_PLATFORM_NAME, str_size, str, NULL);
  if (status != CL_SUCCESS) {
    free(str);
    do_error("Failure in clGetPlatformInfo");
  } else {
    name = std::string(str);
    free(str);
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

int OpenCLDeviceSelector::score_device(std::string requested,
                                       cl_device_id device) {
  // Get the requested device type:
  cl_device_type req_type = match_device_type(requested);
  cl_device_type dev_type;
  cl_int status = clGetDeviceInfo(device, CL_DEVICE_TYPE,
                                  sizeof(cl_device_type), &dev_type, NULL);
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

cl_uint OpenCLDeviceSelector::get_platform_count() {
  cl_uint num_platforms;
  cl_int status = clGetPlatformIDs(0, NULL, &num_platforms);
  if (status != CL_SUCCESS) {
    do_error("failure in clGetPlatformIDs");
  }
  return num_platforms;
}

cl_uint OpenCLDeviceSelector::get_device_count(cl_platform_id plat) {
  cl_uint num_devices;
  cl_int status =
      clGetDeviceIDs(plat, CL_DEVICE_TYPE_ALL, 0, NULL, &num_devices);
  if (status != CL_SUCCESS) {
    do_error("failure in clGetDeviceIDs");
  }
  return num_devices;
}

OpenCLDeviceSelector::OpenCLDeviceSelector(std::string vendor,
                                           std::string type) {
  // Get the number of platforms, and a list of IDs
  cl_uint num_platforms = get_platform_count();
  cl_platform_id platforms[num_platforms];
  cl_int status = clGetPlatformIDs(num_platforms, platforms, NULL);
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
    cl_device_id devices[num_devices];
    cl_int status = clGetDeviceIDs(platform, CL_DEVICE_TYPE_ALL, num_devices,
                                   devices, NULL);
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

cl_device_id OpenCLDeviceSelector::device() { return best_device; }
cl_platform_id OpenCLDeviceSelector::platform() { return best_platform; }

/* Class Context */

Context::Context(OpenCLDeviceSelector oclds) {
  platform = oclds.platform();
  device = oclds.device();
  create();
}

void Context::create() {
  if (is_active) do_error("context is already active");
  cl_int status;
  context = clCreateContext(NULL, 1, &device, NULL, NULL, &status);
  if (status != CL_SUCCESS) {
    do_error("failure to create context");
  }
  command_queue =
      clCreateCommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &status);
  if (status != CL_SUCCESS) {
    do_error("failure to create command queue");
  }
  is_active = true;
}

bool Context::active() { return is_active; }

cl_context Context::ctx() { return context; }

cl_device_id Context::dev() { return device; }

cl_platform_id Context::plt() { return platform; }

cl_command_queue *Context::_queue() { return &command_queue; }

cl_command_queue Context::queue() const { return command_queue; }

Context::~Context() {
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

/* Class Event */

void CLEventHandler::wait(cl_event event) {
  cl_int status = clWaitForEvents(1, &event);
  if (status != CL_SUCCESS) {
    do_error("failure in clWaitForEvents");
  }
}

void CLEventHandler::wait(std::vector<cl_event> &&events) {
  for (auto &event : events) {
    CLEventHandler::wait(event);
  }
}

void CLEventHandler::release(cl_event event) {
  cl_int status = clReleaseEvent(event);
  if (status != CL_SUCCESS) {
    do_error("failure in clReleaseEvent");
  }
}

void CLEventHandler::release(std::vector<cl_event> &&events) {
  for (auto &&event : events) {
    CLEventHandler::release(event);
  }
}
