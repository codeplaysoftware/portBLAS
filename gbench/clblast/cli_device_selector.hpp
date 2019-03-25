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

class cli_device_selector {
  std::string program_name;

  void usage() {
    std::cout << "Usage: " << program_name << " [google-benchmark args]"
              << std::endl;
    std::cout << " --device  DEVICE" << std::endl
              << "    Select a device (best effort) for running the benchmark."
              << std::endl;
    std::cout << "    e.g. intel:cpu, amd:gpu etc" << std::endl;
  }

 public:
  std::string device_vendor;
  std::string device_type;

  cli_device_selector(int argc, char** argv) : program_name(argv[0]) {
    /* Match parameters */
    std::regex help_regex("--help");
    std::regex device_regex("--device");
    /* Check if user has specified any options */
    for (int i = 1; i < argc; i++) {
      std::string option(argv[i]);
      if (option.size() == 0) {
        std::cerr << " Incorrect parameter " << std::endl;
        break;
      }

      // Check for the --help parameter
      if (std::regex_match(option, help_regex)) {
        usage();
        break;
      }

      // Check for the --device parameter
      if (std::regex_match(option, device_regex)) {
        if ((i + 1) >= argc) {
          std::cerr << " Incorrect parameter " << std::endl;
          break;
        }
        std::string outputOption = argv[i + 1];
        std::transform(outputOption.begin(), outputOption.end(),
                       outputOption.begin(), ::tolower);
        // split the string into tokens on ':'
        std::stringstream ss(outputOption);
        std::string item;
        std::vector<std::string> tokens;
        while (std::getline(ss, item, ':')) {
          tokens.push_back(item);
        }
        if (tokens.size() != 2) {
          std::cerr << " Incorrect number of arguments to device selector "
                    << std::endl;
        } else {
          device_vendor = tokens[0];
          device_type = tokens[1];
        }
        // Skip next parameter, since it was the device
        i++;
      }
    }
  }
};

#endif  // CLI_DEVICE_SELECTOR_HPP
