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
 *  @filename usm_policy_handler.h
 *
 **************************************************************************/

#ifndef SYCL_BLAS_USM_POLICY_HANDLER_H
#define SYCL_BLAS_USM_POLICY_HANDLER_H

#include "blas_meta.h"
#include "policy/default_policy_handler.h"
#include "policy/sycl_policy.h"
#include <CL/sycl.hpp>
#include <stdexcept>

namespace blas {

template <>
class PolicyHandler<usm_policy> {
 public:
  using policy_t = usm_policy;

  explicit PolicyHandler(cl::sycl::queue q)
      : q_(q),
        workGroupSize_(usm_policy::get_work_group_size(q)),
        selectedDeviceType_(usm_policy::find_chosen_device_type(q)),
        localMemorySupport_(usm_policy::has_local_memory(q)),
        computeUnits_(usm_policy::get_num_compute_units(q)) {}

  // template <typename element_t>
  // element_t *allocate(size_t num_elements) const;

  // template <typename element_t>
  // void deallocate(element_t *p) const;

  // template <typename element_t>
  // ptrdiff_t get_offset(const element_t *ptr) const;

  template <typename element_t>
  typename policy_t::event_t copy_to_device(const element_t *src,
                                            element_t *dst, size_t size = 0);

  template <typename element_t>
  typename policy_t::event_t copy_to_host(element_t *src, element_t *dst,
                                          size_t size = 0);

  // template<typename element_t>
  // typename policy_t::event_t fill(element_t *buf,
  //                                 element_t value = element_t{0},
  //                                 size_t size = 0);

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
  const size_t workGroupSize_;
  const policy_t::device_type selectedDeviceType_;
  const bool localMemorySupport_;
  const size_t computeUnits_;
};

}  // namespace blas
#endif  // QUEUE_SYCL_HPP
