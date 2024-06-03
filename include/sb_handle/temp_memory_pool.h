/***************************************************************************
 *
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
 *  portBLAS: BLAS implementation using SYCL
 *
 *  @filename temp_memory_pool.h
 *
 **************************************************************************/
#ifndef TEMP_MEMORY_POOL_H
#define TEMP_MEMORY_POOL_H

#ifndef __ADAPTIVECPP__
#include <map>
#include <mutex>

namespace blas {
class Temp_Mem_Pool {
  using queue_t = sycl::queue;
  using event_t = std::vector<sycl::event>;
  using temp_usm_map_t = std::multimap<size_t, void*>;
  using temp_usm_size_map_t = std::map<void*, size_t>;
  using temp_buffer_map_t = std::multimap<size_t, sycl::buffer<int8_t, 1>>;

 public:
  Temp_Mem_Pool(queue_t q)
      : q_(q),
#ifdef SB_ENABLE_USM
        temp_usm_map_tot_byte_size_(0),
#endif
        temp_buffer_map_tot_byte_size_(0) {
  }
  Temp_Mem_Pool(const Temp_Mem_Pool& h) = delete;
  Temp_Mem_Pool operator=(Temp_Mem_Pool) = delete;

  ~Temp_Mem_Pool() {
    // Wait for the completion of all the host tasks
    q_.wait();

#ifdef VERBOSE
    std::cout << "# buffers destroyed on memory pool destruction: "
              << temp_buffer_map_.size() << " ("
              << temp_buffer_map_tot_byte_size_ << " bytes)" << std::endl;
#endif

#ifdef SB_ENABLE_USM
#ifdef VERBOSE
    std::cout << "# USM allocations freed on memory pool destruction: "
              << temp_usm_map_.size() << " (" << temp_usm_map_tot_byte_size_
              << " bytes)" << std::endl;
#endif
    for (const temp_usm_map_t::value_type& p : temp_usm_map_)
      sycl::free(p.second, q_);
#endif
  }

  inline queue_t get_queue() const { return q_; }

  template <typename value_t>
  typename helper::AllocHelper<value_t, helper::AllocType::buffer>::type
  acquire_buff_mem(size_t size);

  template <typename container_t>
  typename Temp_Mem_Pool::event_t release_buff_mem(
      const typename Temp_Mem_Pool::event_t&, const container_t&);

#ifdef SB_ENABLE_USM
  template <typename value_t>
  typename helper::AllocHelper<value_t, helper::AllocType::usm>::type
  acquire_usm_mem(size_t size);

  template <typename container_t>
  typename Temp_Mem_Pool::event_t release_usm_mem(
      const typename Temp_Mem_Pool::event_t&, const container_t&);
#endif

 private:
  static_assert(sizeof(temp_buffer_map_t::mapped_type::value_type) == 1);

  static constexpr size_t max_size_temp_mem_ = 1e9;
  queue_t q_;

  std::mutex temp_buffer_map_mutex_;
  size_t temp_buffer_map_tot_byte_size_;
  temp_buffer_map_t temp_buffer_map_;

  template <typename container_t>
  void release_buff_mem_(const container_t& mem);

#ifdef SB_ENABLE_USM
  std::mutex temp_usm_map_mutex_;
  size_t temp_usm_map_tot_byte_size_;
  temp_usm_map_t temp_usm_map_;
  temp_usm_size_map_t temp_usm_size_map_;

  template <typename container_t>
  void release_usm_mem_(const container_t& mem);
#endif  // SB_ENABLE_USM
};
}  // namespace blas

#endif  // __ADAPTIVECPP__

#endif
