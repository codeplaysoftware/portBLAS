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
 *  @filename blas_device_sycl.hpp
 *
 **************************************************************************/

#ifndef BLAS_DEVICE_SYCL_HPP
#define BLAS_DEVICE_SYCL_HPP

#include <CL/sycl.hpp>
#include <executors/blas_device.hpp>

namespace blas {

class SYCLDevice {
  /*!
   * @brief SYCL queue for execution of trees.
   */
  cl::sycl::queue m_queue;

 public:
  SYCLDevice()
      : m_queue(cl::sycl::queue([&](cl::sycl::exception_list l) {
          for (const auto &e : l) {
            try {
              if (e) {
                std::rethrow_exception(e);
              }
            } catch (cl::sycl::exception e) {
              std::cerr << e.what() << std::endl;
            }
          }
        })) {}

  static void parallel_for_setup(size_t &localSize, size_t &nWG,
                                 size_t &globalSize, size_t N) {
    localSize = 256;
    globalSize = N;
    nWG = (globalSize + localSize - 1) / localSize;
  }

  cl::sycl::queue sycl_queue() { return m_queue; }
  cl::sycl::device sycl_device() { return m_queue.get_device(); }
};
}  // namespace BLAS

#endif /* end of include guard: BLAS_DEVICE_SYCL_HPP */
