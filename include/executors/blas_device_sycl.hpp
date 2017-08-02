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
#include <executors/blas_pointer_struct.hpp>

namespace blas {

/*!
 * SYCLDevice.
 * @brief Device which uses SYCL to execute the expression trees.
 */
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

  /*!
   * @brief Sets the parameters for simple case of execution.
   * e.g. axpy/scal kernels.
   * @param [out] localsize Local size.
   * @param [out] nwg Number of work groups.
   * @param [out] globalsize Global size.
   * @param [in] N Number of elements to be processed.
   */
  static void parallel_for_setup(size_t &localsize, size_t &nwg,
                                 size_t &globalsize, size_t N) {
    localsize = 256;
    globalsize = N;
    nwg = (globalsize + localsize - 1) / localsize;
    globalsize = nwg * localsize;
  }

  /*!
   * @brief Gets cl::sycl::queue attached to this device instance.
   */
  cl::sycl::queue sycl_queue() { return m_queue; }
  /*!
   * @brief Gets a cl::sycl::device from cl::sycl::queue attached to this device
   * instance.
   */
  cl::sycl::device sycl_device() { return m_queue.get_device(); }

  /*!
   * @brief Allocates a global memory buffer on heap.
   * @tparam T Type of the objects that the buffer refers to.
   * @param N Number of elements to be allocated in the global memory.
   * @returns Pointer to the new cl::sycl::buffer<T, 1>.
   */
  template <typename T>
  cl::sycl::buffer<T, 1> *allocate(size_t N) {
    return new cl::sycl::buffer<T, 1>{N};
  }

  /*!
   * @brief deallocates a buffer (causes data movement).
   * @tparam T Type of the objects that buffer refers to.
   * @param buffer Pointer to buffer allocated for this device.
   */
  template <typename T>
  void deallocate(cl::sycl::buffer<T, 1> *buffer) {
    delete buffer;
  }
};

}  // namespace BLAS

#endif /* end of include guard: BLAS_DEVICE_SYCL_HPP */
