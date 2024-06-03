#ifndef TEMP_MEMORY_POOL_HPP
#define TEMP_MEMORY_POOL_HPP
#ifndef __ADAPTIVECPP__
#include "portblas_helper.h"

namespace blas {
template <typename value_t>
typename helper::AllocHelper<value_t, helper::AllocType::buffer>::type
Temp_Mem_Pool::acquire_buff_mem(size_t size) {
  const size_t pad = sizeof(double) / sizeof(value_t);
  // Adjust the requested size in order to reinterpret for double's
  size += (pad - size % pad);
  const size_t byteSize = size * sizeof(value_t);
  temp_buffer_map_mutex_.lock();  // lock
  auto found = temp_buffer_map_.lower_bound(byteSize);
  if (found != temp_buffer_map_.end()) {
    sycl::buffer<temp_buffer_map_t::mapped_type::value_type, 1> buff =
        found->second;
    temp_buffer_map_tot_byte_size_ -= found->first;
    temp_buffer_map_.erase(found);
    temp_buffer_map_mutex_.unlock();  // unlock
    return blas::BufferIterator<value_t>{buff.reinterpret<value_t>(
        sycl::range<1>(buff.byte_size() / sizeof(value_t)))};
  } else {
    temp_buffer_map_mutex_.unlock();  // unlock
#ifdef VERBOSE
    std::cout << "Create a temporary buffer of " << byteSize << " bytes."
              << std::endl;
#endif
    return make_sycl_iterator_buffer<value_t>(size);
  }
}

template <typename container_t>
void Temp_Mem_Pool::release_buff_mem_(const container_t& mem) {
  const size_t byteSize = mem.get_buffer().byte_size();
  auto rebuff =
      mem.get_buffer()
          .template reinterpret<temp_buffer_map_t::mapped_type::value_type>(
              sycl::range<1>(
                  byteSize /
                  sizeof(temp_buffer_map_t::mapped_type::value_type)));
  temp_buffer_map_mutex_.lock();  // lock
  if (temp_buffer_map_tot_byte_size_ + byteSize <= max_size_temp_mem_) {
    temp_buffer_map_tot_byte_size_ += byteSize;
    temp_buffer_map_.emplace(byteSize, rebuff);
  }
  temp_buffer_map_mutex_.unlock();  // unlock
}

template <typename container_t>
typename Temp_Mem_Pool::event_t Temp_Mem_Pool::release_buff_mem(
    const typename Temp_Mem_Pool::event_t& dependencies,
    const container_t& mem) {
  return {q_.submit([&](sycl::handler& cgh) {
    cgh.depends_on(dependencies);
    cgh.host_task([&, mem]() { release_buff_mem_(mem); });
  })};
}

#ifdef SB_ENABLE_USM
template <typename value_t>
typename helper::AllocHelper<value_t, helper::AllocType::usm>::type
Temp_Mem_Pool::acquire_usm_mem(size_t size) {
  const size_t byteSize = size * sizeof(value_t);
  temp_usm_map_mutex_.lock();  // lock
  auto found = temp_usm_map_.lower_bound(byteSize);
  if (found != temp_usm_map_.end()) {
    temp_usm_map_tot_byte_size_ -= found->first;
    value_t* tmp = reinterpret_cast<value_t*>(found->second);
    temp_usm_map_.erase(found);
    temp_usm_map_mutex_.unlock();  // unlock
    return tmp;
  } else {
    temp_usm_map_mutex_.unlock();  // unlock
#ifdef VERBOSE
    std::cout << "Create a temporary USM allocation of " << byteSize
              << " bytes." << std::endl;
#endif
    value_t* tmp = sycl::malloc_device<value_t>(size, q_);
    temp_usm_map_mutex_.lock();  // lock
    temp_usm_size_map_.emplace(
        reinterpret_cast<temp_usm_size_map_t::key_type>(tmp), byteSize);
    temp_usm_map_mutex_.unlock();  // unlock
    return tmp;
  }
}

template <typename container_t>
void Temp_Mem_Pool::release_usm_mem_(const container_t& mem) {
  temp_usm_map_mutex_.lock();  // lock
  auto found = temp_usm_size_map_.find(
      reinterpret_cast<temp_usm_size_map_t::key_type>(mem));
  const size_t byteSize = found->second;
  if (temp_usm_map_tot_byte_size_ + byteSize > max_size_temp_mem_) {
    temp_usm_size_map_.erase(found);
    temp_usm_map_mutex_.unlock();  // unlock
    sycl::free(mem, q_);
  } else {
    temp_usm_map_tot_byte_size_ += byteSize;
    temp_usm_map_.emplace(byteSize,
                          reinterpret_cast<temp_usm_map_t::mapped_type>(mem));
    temp_usm_map_mutex_.unlock();  // unlock
  }
}

template <typename container_t>
typename Temp_Mem_Pool::event_t Temp_Mem_Pool::release_usm_mem(
    const typename Temp_Mem_Pool::event_t& dependencies,
    const container_t& mem) {
  return {q_.submit([&](sycl::handler& cgh) {
    cgh.depends_on(dependencies);
    cgh.host_task([&, mem]() { release_usm_mem_(mem); });
  })};
}
}
#endif  // SB_ENABLE_USM
#endif  // __ADAPTIVECPP__
#endif
