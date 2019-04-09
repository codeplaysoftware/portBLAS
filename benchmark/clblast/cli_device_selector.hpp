/* Copyright (c) 2015 The Khronos Group Inc.

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

#ifndef CLI_DEVICE_SELECTOR_HPP
#define CLI_DEVICE_SELECTOR_HPP

#include <algorithm>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <regex>
#include <string>
#include <unistd.h>
#include <utility>
#include <vector>

#include "cli_args.hpp"

class cli_device_selector {
  std::string program_name;

 public:
  std::string vendor_name;
  std::string device_type;

  cli_device_selector(blas_benchmark::Args& args)
      : program_name(args.program_name) {
    if (!args.device.empty()) {
      std::string device = args.device;
      std::transform(device.begin(), device.end(), device.begin(), ::tolower);
      // split the string into tokens on ':'
      std::stringstream ss(device);
      std::string item;
      std::vector<std::string> tokens;
      while (std::getline(ss, item, ':')) {
        tokens.push_back(item);
      }
      if (tokens.size() != 2) {
        std::cerr << "Device selector should be a colon-separated string (e.g "
                     "intel:gpu)"
                  << std::endl;
      } else {
        vendor_name = tokens[0];
        device_type = tokens[1];
      }
    }
  }
};

#endif  // CLI_DEVICE_SELECTOR_HPP
