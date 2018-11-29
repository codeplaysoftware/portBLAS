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
 *  @filename queue_sycl.hpp
 *
 **************************************************************************/

#ifndef QUEUE_SYCL_HPP
#define QUEUE_SYCL_HPP

#include <CL/sycl.hpp>
#include <queue/queue_base.hpp>
#include <queue/sycl_iterator.hpp>
#include <stdexcept>
#include <types/sycl_types.hpp>
#include <vptr/virtual_ptr.hpp>

namespace blas {

template <>
class Queue_Interface<SYCL> {
  /*!
   * @brief SYCL queue for execution of trees.
   */
  cl::sycl::queue q_;
  std::shared_ptr<cl::sycl::codeplay::PointerMapper> pointerMapperPtr_;
  bool pointer_mapper_owner;
  using generic_buffer_data_type = cl::sycl::codeplay::buffer_data_type_t;

 public:
  enum device_type {
    SYCL_CPU,
    SYCL_HOST,
    SYCL_UNSUPPORTED_DEVICE,
    SYCL_INTEL_GPU,
    SYCL_AMD_GPU,
    SYCL_RCAR_CVENGINE,
    SYCL_RCAR_HOST_CPU
  };

  explicit Queue_Interface(cl::sycl::queue q)
      : q_(q),
        pointerMapperPtr_(std::shared_ptr<cl::sycl::codeplay::PointerMapper>(
            new cl::sycl::codeplay::PointerMapper(),
            [](cl::sycl::codeplay::PointerMapper *p) {
              p->clear();
              delete p;
            })),
        pointer_mapper_owner(true) {}

  const device_type get_device_type() const {
    auto dev = q_.get_device();
    auto platform = dev.get_platform();
    auto plat_name =
        platform.template get_info<cl::sycl::info::platform::name>();
    auto device_type =
        dev.template get_info<cl::sycl::info::device::device_type>();
    std::transform(plat_name.begin(), plat_name.end(), plat_name.begin(),
                   ::tolower);
    if (plat_name.find("amd") != std::string::npos &&
        device_type == cl::sycl::info::device_type::gpu) {
      return SYCL_AMD_GPU;
    } else if (plat_name.find("intel") != std::string::npos &&
               device_type == cl::sycl::info::device_type::gpu) {
      return SYCL_INTEL_GPU;
    } else if (plat_name.find("computeaorta") != std::string::npos &&
               device_type == cl::sycl::info::device_type::accelerator) {
      return SYCL_RCAR_CVENGINE;
    } else if (plat_name.find("computeaorta") != std::string::npos &&
               device_type == cl::sycl::info::device_type::cpu) {
      return SYCL_RCAR_HOST_CPU;
    } else {
      return SYCL_UNSUPPORTED_DEVICE;
    }
    throw std::runtime_error("couldn't find device");
  }
  inline bool has_local_memory() const {
    return (q_.get_device()
                .template get_info<cl::sycl::info::device::local_mem_type>() ==
            cl::sycl::info::local_mem_type::local);
  }
  template <typename T>
  inline T *allocate(size_t num_elements) const {
    return static_cast<T *>(cl::sycl::codeplay::SYCLmalloc(
        num_elements * sizeof(T), *pointerMapperPtr_));
  }

  template <typename T>
  inline void deallocate(T *p) const {
    cl::sycl::codeplay::SYCLfree(static_cast<void *>(p), *pointerMapperPtr_);
  }
  cl::sycl::queue get_queue() const { return q_; }

  // This function returns the nearest power of 2 Work-group size which is <=
  // maximum device workgroup size.
  inline size_t get_rounded_power_of_two_work_group_size() const {
    return get_power_of_two(get_work_group_size(), false);
  }
  // Force the systme not to set this to bigger than 256. As it can be
  inline size_t get_work_group_size() const {
    return std::min(
        size_t(256),
        q_.get_device()
            .template get_info<cl::sycl::info::device::max_work_group_size>());
  }

  // This function returns the nearest power of 2
  // if roundup is ture returns result>=wgsize
  // else it return result <= wgsize
  inline size_t get_power_of_two(size_t wGSize, bool rounUp) const {
    if (rounUp) --wGSize;
    wGSize |= (wGSize >> 1);
    wGSize |= (wGSize >> 2);
    wGSize |= (wGSize >> 4);
    wGSize |= (wGSize >> 8);
    wGSize |= (wGSize >> 16);
#if defined(__x86_64__) || defined(_M_X64) || defined(__amd64) || \
    defined(__aarch64__) || defined(_WIN64)
    wGSize |= (wGSize >> 32);
#endif
    return ((!rounUp) ? (wGSize - (wGSize >> 1)) : ++wGSize);
  }

  /*
  @brief this class is to return the dedicated buffer to the user
  @ tparam T is the type of the pointer
  @tparam bufferT<T> is the type of the buffer points to the data. on the host
  side buffer<T> and T are the same
  */
  template <typename T>
  inline buffer_t<T> get_buffer(T *ptr) const {
    using pointer_t = typename std::remove_const<T>::type *;
    auto original_buffer = pointerMapperPtr_->get_buffer(
        static_cast<void *>(const_cast<pointer_t>(ptr)));
    auto typed_size = original_buffer.get_size() / sizeof(T);
    auto buff = original_buffer.reinterpret<T>(cl::sycl::range<1>(typed_size));
    return buff;
  }

  /*
  @brief this class is to return the dedicated buffer to the user
  @ tparam T is the type of the buffer
  @tparam buffer_iterator<T> is the type of the buffer that user can apply
  arithmetic operation on the host side
  */
  template <typename T>
  inline buffer_iterator<T> get_buffer(buffer_iterator<T> buff) const {
    return buff;
  }

  /*  @brief Getting range accessor from the buffer created by virtual pointer
      @tparam T is the type of the data
      @tparam AcM is the access mode
      @param container is the  data we want to get range accessor
  */
  template <cl::sycl::access::mode AcM = cl::sycl::access::mode::read_write,
            typename T>
  placeholder_accessor_t<T, AcM> get_range_access(T *vptr) {
    auto buff_t = get_buffer(vptr);
    auto offset = get_offset(vptr);
    return placeholder_accessor_t<T, AcM>(
        buff_t, cl::sycl::range<1>(buff_t.get_count() - offset),
        cl::sycl::id<1>(offset));
  }

  /*  @brief Getting range accessor from the buffer created by buffer iterator
      @tparam T is the type of the data
      @tparam AcM is the access mode
      @param container is the  data we want to get range accessor
  */
  template <cl::sycl::access::mode AcM = cl::sycl::access::mode::read_write,
            typename T>
  placeholder_accessor_t<T, AcM> get_range_access(buffer_iterator<T> buff) {
    return blas::get_range_accessor<AcM>(buff);
  }

  /*
  @brief this function is to get the offset from the actual pointer
  @tparam T is the type of the pointer
  */
  template <typename T>
  inline ptrdiff_t get_offset(const T *ptr) const {
    return (pointerMapperPtr_->get_offset(static_cast<const void *>(ptr)) /
            sizeof(T));
  }
  /*
  @brief this function is to get the offset from the actual pointer
  @tparam T is the type of the buffer_iterator<T>
  */
  template <typename T>
  inline ptrdiff_t get_offset(buffer_iterator<T> buff) const {
    return buff.get_offset();
  }
  /*  @brief Copying the data back to device
      @tparam T is the type of the data
      @param src is the host pointer we want to copy from.
      @param dst is the device pointer we want to copy to.
      @param size is the number of elements to be copied
  */
  template <typename T>
  cl::sycl::event copy_to_device(const T *src, T *dst, size_t size) {
    auto buffer = pointerMapperPtr_->get_buffer(static_cast<void *>(dst));
    auto offset = pointerMapperPtr_->get_offset(static_cast<void *>(dst));
    auto event = q_.submit([&](cl::sycl::handler &cgh) {
      auto write_acc =
          buffer.template get_access<cl::sycl::access::mode::write,
                                     cl::sycl::access::target::global_buffer>(
              cgh, cl::sycl::range<1>(size * sizeof(T)),
              cl::sycl::id<1>(offset));
      cgh.copy(static_cast<generic_buffer_data_type *>(
                   static_cast<void *>(const_cast<T *>(src))),
               write_acc);
    });

    return event;
  }

  /*  @brief Copying the data back to device
    @tparam T is the type of the data
    @param src is the host pointer we want to copy from.
    @param dst is the buffer_iterator we want to copy to.
    @param size is the number of elements to be copied
*/
  template <typename T>
  inline cl::sycl::event copy_to_device(T *src, buffer_iterator<T> dst,
                                        size_t = 0) {
    auto event = q_.submit([&](cl::sycl::handler &cgh) {
      auto acc =
          blas::get_range_accessor<cl::sycl::access::mode::write>(dst, cgh);
      cgh.copy(src, acc);
    });

    return event;
  }
  /*  @brief Copying the data back to device
      @tparam T is the type of the data
      @param src is the device pointer we want to copy from.
      @param dst is the host pointer we want to copy to.
      @param size is the number of elements to be copied
  */
  template <typename T>
  cl::sycl::event copy_to_host(T *src, T *dst, size_t size) {
    auto buffer = pointerMapperPtr_->get_buffer(static_cast<void *>(src));
    auto offset = pointerMapperPtr_->get_offset(static_cast<void *>(src));

    auto event = q_.submit([&](cl::sycl::handler &cgh) {
      auto read_acc =
          buffer.template get_access<cl::sycl::access::mode::read,
                                     cl::sycl::access::target::global_buffer>(
              cgh, cl::sycl::range<1>(size * sizeof(T)),
              cl::sycl::id<1>(offset));
      cgh.copy(read_acc, static_cast<generic_buffer_data_type *>(
                             static_cast<void *>(dst)));
    });

    return event;
  }
  /*  @brief Copying the data back to device
    @tparam T is the type of the data
    @param src is the buffer_iterator we want to copy from.
    @param dst is the host pointer we want to copy to.
    @param size is the number of elements to be copied
*/
  template <typename T>
  inline cl::sycl::event copy_to_host(buffer_iterator<T> src, T *dst,
                                      size_t = 0) {
    auto event = q_.submit([&](cl::sycl::handler &cgh) {
      auto acc =
          blas::get_range_accessor<cl::sycl::access::mode::read>(src, cgh);
      cgh.copy(acc, dst);
    });

    return event;
  }

  /*  @brief waiting for a list of sycl events
    @param first_event  and next_events are instances of sycl::sycl::event
*/
  template <typename first_event_t, typename... next_event_t>
  void inline wait_for_events(first_event_t first_event,
                              next_event_t... next_events) {
    cl::sycl::event::wait({first_event, next_events...});
  }

  /*  @brief waiting for a sycl::queue.wait()
   */
  void inline wait() { q_.wait(); }

};  // class Queue_Interface
}  // namespace blas
#endif  // QUEUE_SYCL_HPP
