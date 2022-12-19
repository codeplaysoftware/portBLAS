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
 *  @filename sycl_blas_helper.h
 *
 **************************************************************************/

#ifndef SYCL_BLAS_HELPER_H
#define SYCL_BLAS_HELPER_H

#include "blas_meta.h"
#include "container/sycl_iterator.h"
#include <CL/sycl.hpp>

namespace blas {
namespace helper {

inline bool has_local_memory(cl::sycl::queue &q) {
  return (q.get_device()
              .template get_info<cl::sycl::info::device::local_mem_type>() ==
          cl::sycl::info::local_mem_type::local);
}
// Force the system not to set this to bigger than 256. Using work group size
// bigger than 256 may cause out of resource error on different platforms.
inline size_t get_work_group_size(cl::sycl::queue &q) {
  return std::min(
      size_t(256),
      q.get_device()
          .template get_info<cl::sycl::info::device::max_work_group_size>());
}

inline size_t get_num_compute_units(cl::sycl::queue &q) {
  return q.get_device()
      .template get_info<cl::sycl::info::device::max_compute_units>();
}

/* @brief Copying the data back to device
  @tparam element_t is the type of the data
  @param src is the host pointer we want to copy from.
  @param dst is the BufferIterator we want to copy to.
  @param size is the number of elements to be copied
*/
template <typename element_t>
inline cl::sycl::event copy_to_device(cl::sycl::queue q, const element_t *src,
                                      BufferIterator<element_t> dst,
                                      size_t size) {
  auto event = q.submit([&](cl::sycl::handler &cgh) {
    auto acc = dst.template get_range_accessor<cl::sycl::access::mode::write>(
        cgh, size);
    cgh.copy(src, acc);
  });
  return event;
}

/*  @brief Copying the data back to device
  @tparam element_t is the type of the data
  @param src is the BufferIterator we want to copy from.
  @param dst is the host pointer we want to copy to.
  @param size is the number of elements to be copied
*/
template <typename element_t>
inline cl::sycl::event copy_to_host(cl::sycl::queue q,
                                    BufferIterator<element_t> src,
                                    element_t *dst, size_t size) {
  auto event = q.submit([&](cl::sycl::handler &cgh) {
    auto acc = src.template get_range_accessor<cl::sycl::access::mode::read>(
        cgh, size);
    cgh.copy(acc, dst);
  });
  return event;
}

template <typename element_t>
inline cl::sycl::event fill(cl::sycl::queue q, BufferIterator<element_t> buff,
                            element_t value, size_t size) {
  auto event = q.submit([&](cl::sycl::handler &cgh) {
    auto acc = buff.template get_range_accessor<cl::sycl::access::mode::write>(
        cgh, size);
    cgh.fill(acc, value);
  });
  return event;
}
}  // end namespace helper
}  // end namespace blas
#endif  // SYCL_BLAS_HELPER_H
