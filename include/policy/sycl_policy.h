/***************************************************************************
 *  @license
 *  Copyright (C) Codeplay Software Limited
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
 *  @filename sycl_policy.h
 *
 **************************************************************************/

#ifndef SYCL_BLAS_SYCL_POLICY_H
#define SYCL_BLAS_SYCL_POLICY_H

#include "blas_meta.h"
#include <CL/sycl.hpp>
#include <stdexcept>

namespace blas {

struct BLAS_SYCL_Policy {
  template <typename ScalarT, int dim = 1>
  using buffer_t = cl::sycl::buffer<ScalarT, dim>;
  template <typename ScalarT,
            cl::sycl::access::mode AcM = cl::sycl::access::mode::read_write,
            cl::sycl::access::target AcT =
                cl::sycl::access::target::global_buffer,
            cl::sycl::access::placeholder AcP =
                cl::sycl::access::placeholder::false_t>
  using SyclAccessor = cl::sycl::accessor<ScalarT, 1, AcM, AcT, AcP>;
  template <
      typename ScalarT,
      cl::sycl::access::mode AcM = cl::sycl::access::mode::read_write,
      cl::sycl::access::target AcT = cl::sycl::access::target::global_buffer,
      cl::sycl::access::placeholder AcP = cl::sycl::access::placeholder::true_t>
  using placeholder_accessor_t = cl::sycl::accessor<ScalarT, 1, AcM, AcT, AcP>;
  using access_mode_type = cl::sycl::access::mode;
  using queue_type = cl::sycl::queue;
  template <typename T,
            cl::sycl::access::mode AcM = cl::sycl::access::mode::read_write>
  using access_type = placeholder_accessor_t<T, AcM>;
  using event_type = std::vector<cl::sycl::event>;

  enum class device_type {
    SYCL_CPU,
    SYCL_HOST,
    SYCL_UNSUPPORTED_DEVICE,
    SYCL_INTEL_GPU,
    SYCL_AMD_GPU,
    SYCL_ARM_GPU,
    SYCL_RCAR_CVENGINE,
    SYCL_POWERVR_GPU,
    SYCL_RCAR_HOST_CPU
  };
  static inline bool has_local_memory(cl::sycl::queue &q_) {
    return (q_.get_device()
                .template get_info<cl::sycl::info::device::local_mem_type>() ==
            cl::sycl::info::local_mem_type::local);
  }
  // Force the system not to set this to bigger than 256. Using work group size
  // bigger than 256 may cause out of resource error on different platforms.
  static inline size_t get_work_group_size(cl::sycl::queue &q_) {
    return std::min(
        size_t(256),
        q_.get_device()
            .template get_info<cl::sycl::info::device::max_work_group_size>());
  }

  static inline size_t get_num_compute_units(cl::sycl::queue &q_) {
    return q_.get_device()
        .template get_info<cl::sycl::info::device::max_compute_units>();
  }

  static device_type find_chosen_device_type(cl::sycl::queue &q_) {
    auto dev = q_.get_device();
    auto platform = dev.get_platform();
    auto plat_name =
        platform.template get_info<cl::sycl::info::platform::name>();
    auto dev_type =
        dev.template get_info<cl::sycl::info::device::device_type>();
    std::transform(plat_name.begin(), plat_name.end(), plat_name.begin(),
                   ::tolower);
    if (plat_name.find("amd") != std::string::npos &&
        dev_type == cl::sycl::info::device_type::gpu) {
      return device_type::SYCL_AMD_GPU;
    } else if (plat_name.find("intel") != std::string::npos &&
               dev_type == cl::sycl::info::device_type::gpu) {
      return device_type::SYCL_INTEL_GPU;
    } else if (plat_name.find("computeaorta") != std::string::npos &&
               dev_type == cl::sycl::info::device_type::accelerator) {
      return device_type::SYCL_RCAR_CVENGINE;
    } else if (plat_name.find("arm") != std::string::npos &&
               dev_type == cl::sycl::info::device_type::gpu) {
      return device_type::SYCL_ARM_GPU;
    } else if (plat_name.find("powervr") != std::string::npos &&
               dev_type == cl::sycl::info::device_type::gpu) {
      return device_type::SYCL_POWERVR_GPU;
    } else if (plat_name.find("computeaorta") != std::string::npos &&
               dev_type == cl::sycl::info::device_type::cpu) {
      return device_type::SYCL_RCAR_HOST_CPU;
    } else {
      return device_type::SYCL_UNSUPPORTED_DEVICE;
    }
    throw std::runtime_error("couldn't find device");
  }
};  // namespace blas

}  // namespace blas
#endif  // QUEUE_SYCL_HPP
