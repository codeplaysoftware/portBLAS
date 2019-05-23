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

#include "../blas_meta.h"
#include <CL/sycl.hpp>
#include <stdexcept>

namespace blas {

struct codeplay_policy {
  template <typename scalar_t, int dim = 1>
  using buffer_t = cl::sycl::buffer<scalar_t, dim>;
  template <typename scalar_t,
            cl::sycl::access::mode acc_md_t =
                cl::sycl::access::mode::read_write,
            cl::sycl::access::target access_t =
                cl::sycl::access::target::global_buffer,
            cl::sycl::access::placeholder place_holder_t =
                cl::sycl::access::placeholder::false_t>
  using accessor_t =
      cl::sycl::accessor<scalar_t, 1, acc_md_t, access_t, place_holder_t>;
  template <typename scalar_t,
            cl::sycl::access::mode acc_md_t =
                cl::sycl::access::mode::read_write,
            cl::sycl::access::target access_t =
                cl::sycl::access::target::global_buffer,
            cl::sycl::access::placeholder place_holder_t =
                cl::sycl::access::placeholder::true_t>
  using placeholder_accessor_t =
      cl::sycl::accessor<scalar_t, 1, acc_md_t, access_t, place_holder_t>;
  using access_mode_t = cl::sycl::access::mode;
  using queue_t = cl::sycl::queue;
  template <typename value_t,
            access_mode_t acc_md_t = cl::sycl::access::mode::read_write>
  using default_accessor_t = placeholder_accessor_t<value_t, acc_md_t>;
  using event_t = std::vector<cl::sycl::event>;

  enum class device_type : int {
    cpu,
    host,
    intel_gpu,
    amd_gpu,
    arm_gpu,
    rcar_cvengine,
    rcar_cpu,
    unsupported
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
      return device_type::amd_gpu;
    } else if (plat_name.find("intel") != std::string::npos &&
               dev_type == cl::sycl::info::device_type::gpu) {
      return device_type::intel_gpu;
    } else if (plat_name.find("computeaorta") != std::string::npos &&
               dev_type == cl::sycl::info::device_type::accelerator) {
      return device_type::rcar_cvengine;
    } else if (plat_name.find("arm") != std::string::npos &&
               dev_type == cl::sycl::info::device_type::gpu) {
      return device_type::arm_gpu;
    } else if (plat_name.find("computeaorta") != std::string::npos &&
               dev_type == cl::sycl::info::device_type::cpu) {
      return device_type::rcar_cpu;
    } else {
      return device_type::unsupported;
    }
    throw std::runtime_error("couldn't find device");
  }
};  // namespace blas

}  // namespace blas
#endif  // QUEUE_SYCL_HPP
