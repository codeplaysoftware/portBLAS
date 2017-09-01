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

#include <functional>

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

  enum VENDOR {
    INTEL,
    AMD,
    UNKNOWN_VENDOR
  };

 public:
  const VENDOR vendor;
  const cl::sycl::info::device_type devtype;
  const size_t MAX_LOCALSIZE;
  const size_t MAX_COMPUTE_UNITS;
  const size_t MAX_LOCAL_MEM;

 private:
  VENDOR get_vendor(cl::sycl::device d) {
    std::string plat_name = d.get_platform().template get_info<cl::sycl::info::platform::name>();
    if(plat_name.find("intel")) {
      return INTEL;
    } else if(plat_name.find("amd")) {
      return AMD;
    }
    return UNKNOWN_VENDOR;
  }

 public:
  template <typename DeviceSelector = cl::sycl::default_selector>
  SYCLDevice(DeviceSelector selector = cl::sycl::default_selector()):
    SYCLDevice(cl::sycl::queue(selector, [&](cl::sycl::exception_list l) {
      for (const auto &e : l) {
        try {
          if (e) {
            std::rethrow_exception(e);
          }
        } catch (cl::sycl::exception e) {
          std::cerr << e.what() << std::endl;
        }
      }
    }))
  {}

  SYCLDevice(std::function<void (cl::sycl::exception_list)> &&queue_lambda):
    SYCLDevice(cl::sycl::queue(cl::sycl::default_selector(), queue_lambda))
  {}


  template <typename DeviceSelector = cl::sycl::default_selector>
  SYCLDevice(DeviceSelector selector, std::function<void (cl::sycl::exception_list)> &&queue_lambda):
    SYCLDevice((cl::sycl::queue(selector, queue_lambda)))
  {}

  SYCLDevice(cl::sycl::queue q):
    m_queue(q),
    vendor(get_vendor(q.get_device())),
    MAX_LOCALSIZE(sycl_device().template get_info<cl::sycl::info::device::max_work_group_size>()),
    MAX_COMPUTE_UNITS(sycl_device().template get_info<cl::sycl::info::device::max_compute_units>()),
    MAX_LOCAL_MEM(sycl_device().template get_info<cl::sycl::info::device::local_mem_size>()),
    devtype(sycl_device().template get_info<cl::sycl::info::device::device_type>())
  {}

  /*!
   * @brief Sets the parameters for simple case of execution.
   * e.g. axpy/scal kernels.
   * @param [out] localsize Local size.
   * @param [out] nwg Number of work groups.
   * @param [out] globalsize Global size.
   * @param [in] N Number of elements to be processed.
   */
  void parallel_for_setup(size_t &localsize, size_t &nwg, size_t &globalsize, size_t N) {
    if(sycl_device().is_gpu()) {
      localsize = 256;
    } else if(sycl_device().is_cpu() && vendor == INTEL) {
      localsize = std::min<size_t>((N + 128 - 1) / 128 * 128, MAX_LOCALSIZE / 2);
    } else {
      throw std::runtime_error("unsupported device type");
    }
    nwg = (N + localsize - 1) / localsize;
    globalsize = nwg * localsize;
  }

  /*!
   * @brief Sets up the execution parameters for generic reduction.
   * e.g. asum/dot/nrm2 ...
   * @param [out] localsize Local size.
   * @param [out] nwg Number of work groups.
   * @param [out] globalsize Global size.
   * @param [in] N Number of elements to be processed.
   */
  template <typename T>
  void generic_reduction_setup(size_t &localsize, size_t &nwg, size_t &globalsize, size_t N) {
    if(sycl_device().is_gpu()) {
      localsize = 256;
      nwg = 256;
    } else if(sycl_device().is_cpu() && vendor == INTEL) {
      size_t max_sharedsize = MAX_LOCAL_MEM / sizeof(T) / 2;
      localsize = std::min<size_t>((N + 128 - 1) / 128 * 128, std::min<size_t>(MAX_LOCALSIZE / 4, max_sharedsize));
      nwg = MAX_COMPUTE_UNITS;
    } else {
      throw std::runtime_error("unsupported device type");
    }
    globalsize = localsize * nwg;
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
