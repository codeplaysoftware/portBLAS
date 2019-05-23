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
 *  @filename sycl_policy_handler.h
 *
 **************************************************************************/

#ifndef SYCL_BLAS_SYCL_POLICY_HANDLER_H
#define SYCL_BLAS_SYCL_POLICY_HANDLER_H

#include "../blas_meta.h"
#include "../container/sycl_iterator.h"
#include "default_policy_handler.h"
#include "sycl_policy.h"
#include <CL/sycl.hpp>
#include <stdexcept>
#include <vptr/virtual_ptr.hpp>

namespace blas {

template <>
class PolicyHandler<codeplay_policy> {
 public:
  using policy_t = codeplay_policy;

  explicit PolicyHandler(cl::sycl::queue q)
      : q_(q),
        pointerMapperPtr_(std::shared_ptr<cl::sycl::codeplay::PointerMapper>(
            new cl::sycl::codeplay::PointerMapper(),
            [](cl::sycl::codeplay::PointerMapper *p) {
              p->clear();
              delete p;
            })),
        workGroupSize_(codeplay_policy::get_work_group_size(q)),
        selectedDeviceType_(codeplay_policy::find_chosen_device_type(q)),
        localMemorySupport_(codeplay_policy::has_local_memory(q)),
        computeUnits_(codeplay_policy::get_num_compute_units(q)) {}

  template <typename element_t>
  element_t *allocate(size_t num_elements) const;

  template <typename element_t>
  void deallocate(element_t *p) const;
  /*
  @brief this class is to return the dedicated buffer to the user
  @ tparam element_t is the type of the pointer
  @tparam bufferT<element_t> is the type of the buffer points to the data. on
  the host side buffer<element_t> and element_t are the same
  */

  template <typename element_t>
  BufferIterator<element_t, policy_t> get_buffer(element_t *ptr) const;
  /*
  @brief this class is to return the dedicated buffer to the user
  @ tparam element_t is the type of the buffer
  @tparam BufferIterator<element_t, policy_t> is the type of the buffer that
  user can apply arithmetic operation on the host side
  */

  template <typename element_t>
  BufferIterator<element_t, policy_t> get_buffer(
      BufferIterator<element_t, policy_t> buff) const;

  /*  @brief Getting range accessor from the buffer created by virtual pointer
      @tparam element_t is the type of the data
      @tparam acc_md_t is the access mode
      @param container is the  data we want to get range accessor
  */

  template <typename policy_t::access_mode_t acc_md_t, typename element_t>
  typename policy_t::default_accessor_t<typename ValueType<element_t>::type,
                                        acc_md_t>
  get_range_access(element_t *vptr);

  /*  @brief Getting range accessor from the buffer created by buffer iterator
      @tparam element_t is the type of the data
      @tparam acc_md_t is the access mode
      @param container is the  data we want to get range accessor
  */

  template <typename element_t, typename policy_t::access_mode_t acc_md_t>
  typename policy_t::default_accessor_t<typename ValueType<element_t>::type,
                                        acc_md_t>
  get_range_access(BufferIterator<element_t, policy_t> buff);

  /*
  @brief this function is to get the offset from the actual pointer
  @tparam element_t is the type of the pointer
  */

  template <typename element_t>
  ptrdiff_t get_offset(const element_t *ptr) const;
  /*
  @brief this function is to get the offset from the actual pointer
  @tparam element_t is the type of the BufferIterator<element_t, policy_t>
  */

  template <typename element_t>
  ptrdiff_t get_offset(BufferIterator<element_t, policy_t> buff) const;

  /*  @brief Copying the data back to device
      @tparam element_t is the type of the data
      @param src is the host pointer we want to copy from.
      @param dst is the device pointer we want to copy to.
      @param size is the number of elements to be copied
  */

  template <typename element_t>
  typename policy_t::event_t copy_to_device(const element_t *src,
                                            element_t *dst, size_t size);
  /*  @brief Copying the data back to device
    @tparam element_t is the type of the data
    @param src is the host pointer we want to copy from.
    @param dst is the BufferIterator we want to copy to.
    @param size is the number of elements to be copied
  */

  template <typename element_t>
  typename policy_t::event_t copy_to_device(
      const element_t *src, BufferIterator<element_t, policy_t> dst, size_t);
  /*  @brief Copying the data back to device
      @tparam element_t is the type of the data
      @param src is the device pointer we want to copy from.
      @param dst is the host pointer we want to copy to.
      @param size is the number of elements to be copied
  */

  template <typename element_t>
  typename policy_t::event_t copy_to_host(element_t *src, element_t *dst,
                                          size_t size);

  template <typename element_t>
  typename policy_t::event_t copy_to_host(
      BufferIterator<element_t, policy_t> src, element_t *dst, size_t);

  inline const policy_t::device_type get_device_type() const {
    return selectedDeviceType_;
  };
  inline bool has_local_memory() const { return localMemorySupport_; }
  typename policy_t::queue_t get_queue() const { return q_; }

  inline size_t get_work_group_size() const { return workGroupSize_; }

  inline size_t get_num_compute_units() const { return computeUnits_; }

  inline void wait() { q_.wait(); }

  inline void wait(policy_t::event_t evs) { cl::sycl::event::wait(evs); }

  /*  @brief waiting for a list of sycl events
 @param first_event  and next_events are instances of sycl::sycl::event
*/
  // this must be in header as the number of event is controlled by user and we
  // dont know howmany permutation can be used by a user
  template <typename first_event_t, typename... next_event_t>
  void inline wait(first_event_t first_event, next_event_t... next_events) {
    cl::sycl::event::wait(concatenate_vectors(first_event, next_events...));
  }

 private:
  typename policy_t::queue_t q_;
  std::shared_ptr<cl::sycl::codeplay::PointerMapper> pointerMapperPtr_;
  const size_t workGroupSize_;
  const policy_t::device_type selectedDeviceType_;
  const bool localMemorySupport_;
  const size_t computeUnits_;
};

}  // namespace blas
#endif  // QUEUE_SYCL_HPP
