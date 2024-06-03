/* Copyright (c) 2015-2018 The Khronos Group Inc.

   Permission is hereby granted, free of charge, to any person obtaining a
   copy of this software and/or associated documentation files (the
   "Materials"), to deal in the Materials without restriction, including
   without limitation the rights to use, copy, modify, merge, publish,
   distribute, sublicense, and/or sell copies of the Materials, and to
   permit persons to whom the Materials are furnished to do so, subject to
   the following conditions:

   The above copyright notice and this permission notice shall be included
   in all copies or substantial portions of the Materials.

   MODIFICATIONS TO THIS FILE MAY MEAN IT NO LONGER ACCURATELY REFLECTS
   KHRONOS STANDARDS. THE UNMODIFIED, NORMATIVE VERSIONS OF KHRONOS
   SPECIFICATIONS AND HEADER INFORMATION ARE LOCATED AT
    https://www.khronos.org/registry/

  THE MATERIALS ARE PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
  EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
  MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
  IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
  CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
  TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
  MATERIALS OR THE USE OR OTHER DEALINGS IN THE MATERIALS.

*/

#ifndef PRINT_QUEUE_INFORMATION_HPP
#define PRINT_QUEUE_INFORMATION_HPP

#include <sycl/sycl.hpp>
#include <iostream>
#include <regex>
#include <string>

namespace utils {

inline void print_queue_information(sycl::queue q) {
  std::cerr
      << "Device vendor: "
      << q.get_device().template get_info<sycl::info::device::vendor>()
      << std::endl;
  std::cerr << "Device name: "
            << q.get_device().template get_info<sycl::info::device::name>()
            << std::endl;
  std::cerr << "Device type: ";
  switch (
      q.get_device().template get_info<sycl::info::device::device_type>()) {
    case sycl::info::device_type::cpu:
      std::cerr << "cpu";
      break;
    case sycl::info::device_type::gpu:
      std::cerr << "gpu";
      break;
    case sycl::info::device_type::accelerator:
      std::cerr << "accelerator";
      break;
    case sycl::info::device_type::custom:
      std::cerr << "custom";
      break;
    case sycl::info::device_type::automatic:
      std::cerr << "automatic";
      break;
    case sycl::info::device_type::host:
      std::cerr << "host";
      break;
    case sycl::info::device_type::all:
      std::cerr << "all";
      break;
    default:
      std::cerr << "unknown";
      break;
  };
  std::cerr << std::endl;
}

}  // namespace utils

#endif  // CLI_DEVICE_SELECTOR_HPP
