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

/**
 * Allocation type for tests and benchmarks
 */
enum class AllocType : int { usm = 0, buffer = 1 };

template <typename value_t, AllocType mem_alloc>
struct AllocHelper;

template <typename value_t>
struct AllocHelper<value_t, AllocType::usm> {
  using type = value_t *;
  using tuple_type = std::tuple<value_t *, cl::sycl::event>;
};

template <typename value_t>
struct AllocHelper<value_t, AllocType::buffer> {
  using type = blas::BufferIterator<value_t>;
  using tuple_type = std::tuple<BufferIterator<value_t>, cl::sycl::event>;
};

#ifdef SB_ENABLE_USM
template <AllocType alloc, typename value_t>
typename std::enable_if<alloc == AllocType::usm,
                        typename AllocHelper<value_t, alloc>::type>::type
allocate(int size, cl::sycl::queue q) {
  return cl::sycl::malloc_device<value_t>(size, q);
}
#endif

template <AllocType alloc, typename value_t>
typename std::enable_if<alloc == AllocType::buffer,
                        typename AllocHelper<value_t, alloc>::type>::type
allocate(int size, cl::sycl::queue q) {
  return make_sycl_iterator_buffer<value_t>(size);
}

#ifdef SB_ENABLE_USM
template <AllocType alloc, typename container_t>
typename std::enable_if<alloc == AllocType::usm>::type deallocate(
    container_t mem, cl::sycl::queue q) {
  if (mem != NULL) {
    cl::sycl::free(reinterpret_cast<void *>(mem), q);
  }
}
#endif

template <AllocType alloc, typename container_t>
typename std::enable_if<alloc == AllocType::buffer>::type deallocate(
    container_t mem, cl::sycl::queue q) {}

// Need to add this guard since the enqueue_deallocate
// function requires a host_task which throws a runtime
// exception when used with the enable_profiling{} queue
// property. We need to create all intermediate memory
// before launching the kernel to avoid running into this
// issue.
// Enabling this code only for DEFAULT_CPU backend to get the
// CI to pass.
#ifdef DEFAULT_CPU
template <AllocType alloc, typename container_t>
typename std::enable_if<alloc == AllocType::usm, cl::sycl::event>::type
enqueue_deallocate(std::vector<cl::sycl::event> dependencies, container_t mem,
                   cl::sycl::queue q) {
  cl::sycl::event event;
#ifdef SB_ENABLE_USM
  event = q.submit([&](cl::sycl::handler &cgh) {
    cgh.depends_on(dependencies);
    cgh.host_task([=]() { cl::sycl::free(mem, q); });
  });
#endif
  return event;
}

template <AllocType alloc, typename container_t>
typename std::enable_if<alloc == AllocType::buffer, cl::sycl::event>::type
enqueue_deallocate(std::vector<cl::sycl::event>, container_t mem,
                   cl::sycl::queue q) {
  cl::sycl::event event;
#ifdef SB_ENABLE_USM
  event = q.submit([&](cl::sycl::handler &cgh) { cgh.host_task([=]() {}); });
#endif
  return event;
}
#endif

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

#ifdef SB_ENABLE_USM
template <typename element_t>
inline cl::sycl::event copy_to_device(cl::sycl::queue q, const element_t *src,
                                      element_t *dst, size_t size) {
  auto event = q.memcpy(dst, src, size * sizeof(element_t));
  return event;
}
#endif

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

#ifdef SB_ENABLE_USM
template <typename element_t>
inline cl::sycl::event copy_to_host(cl::sycl::queue q, element_t *src,
                                    element_t *dst, size_t size) {
  auto event = q.memcpy(dst, src, size * sizeof(element_t));
  return event;
}
#endif

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
