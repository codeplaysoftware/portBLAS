#ifndef QUEUE_SYCL_HPP
#define QUEUE_SYCL_HPP

#include <CL/sycl.hpp>
#include <queue/pointer_mapper.hpp>
#include <queue/queue_base.hpp>
#include <stdexcept>
namespace blas {

template <>
class Queue_Interface<SYCL> {
  /*!
   * @brief SYCL queue for execution of trees.
   */
  cl::sycl::queue &q_;
  mutable cl::sycl::codeplay::PointerMapper pointer_mapper;
  bool pointer_mapper_owner;
  using generic_buffer_data_type = cl::sycl::codeplay::buffer_data_type_t;

 public:
  enum device_type { UNSUPPORTED_DEVICE, INTELGPU, AMDGPU };

  explicit Queue_Interface(cl::sycl::queue &q)
      : q_(q), pointer_mapper_owner(true) {}
  // FIXME : The assignment constructor for pointer mapper is deleted
  /*  Queue_Interface(cl::sycl::queue q,
                    cl::sycl::codeplay::PointerMapper& pointer_mapper_)
        : q_(q), pointer_mapper(pointer_mapper_), pointer_mapper_owner(false)
     {}*/

  device_type get_device_type() {
    auto dev = q_.get_device();
    auto platform = dev.get_platform();
    auto plat_name =
        platform.template get_info<cl::sycl::info::platform::name>();
    std::transform(plat_name.begin(), plat_name.end(), plat_name.begin(),
                   ::tolower);
    if (plat_name.find("amd") != std::string::npos && dev.is_gpu()) {
      return AMDGPU;
    } else if (plat_name.find("intel") != std::string::npos && dev.is_gpu()) {
      return INTELGPU;
    } else {
      return UNSUPPORTED_DEVICE;
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
        num_elements * sizeof(T), pointer_mapper));
  }

  template <typename T>
  inline void deallocate(T *p) const {
    cl::sycl::codeplay::SYCLfree(static_cast<void *>(p), pointer_mapper);
  }
  cl::sycl::queue &sycl_queue() const { return q_; }
  ~Queue_Interface() {
    if (pointer_mapper_owner) {
      pointer_mapper.clear();
    }
  }
  /*
  @brief this class is to return the dedicated buffer to the user
  @ tparam T is the type of the pointer
  @tparam bufferT<T> is the type of the buffer points to the data. on the host
  side buffer<T> and T are the same
  */
  template <typename T>
  inline bufferT<T> get_buffer(T *ptr) const {
    auto original_buffer = pointer_mapper.get_buffer(static_cast<void *>(ptr));
    auto typed_size = original_buffer.get_size() / sizeof(T);
    auto buff = original_buffer.reinterpret<T>(cl::sycl::range<1>(typed_size));
    return buff;
  }
  /*
  @brief this function is to get the offset from the actual pointer
  @tparam T is the type of the pointer
  */
  template <typename T>
  inline ptrdiff_t get_offset(T *ptr) const {
    return (pointer_mapper.get_offset(static_cast<void *>(ptr)) / sizeof(T));
  }
  /*  @brief Copying the data back to device
      @tparam T is the type of the data
      @param src is the host pointer we want to copy from.
      @param dst is the device pointer we want to copy to.
      @param size is the number of elements to be copied
  */
  template <typename T>
  void copy_to_device(T *src, T *dst, size_t size) {
    auto buffer = pointer_mapper.get_buffer(static_cast<void *>(dst));
    auto offset = pointer_mapper.get_offset(static_cast<void *>(dst));
    q_.submit([&](cl::sycl::handler &cgh) {
      auto write_acc =
          buffer.template get_access<cl::sycl::access::mode::write,
                                     cl::sycl::access::target::global_buffer>(
              cgh, cl::sycl::range<1>(size * sizeof(T)),
              cl::sycl::id<1>(offset));
      cgh.copy(
          static_cast<generic_buffer_data_type *>(static_cast<void *>(src)),
          write_acc);
    });
    q_.wait();
  }
  /*  @brief Copying the data back to device
      @tparam T is the type of the data
      @param src is the device pointer we want to copy from.
      @param dst is the host pointer we want to copy to.
      @param size is the number of elements to be copied
  */
  template <typename T>
  void copy_to_host(T *src, T *dst, size_t size) {
    auto buffer = pointer_mapper.get_buffer(static_cast<void *>(src));
    auto offset = pointer_mapper.get_offset(static_cast<void *>(src));
    q_.wait();  // FIXME: we should not have that when the size of the buffer is
                // 1 there is an issue

    q_.submit([&](cl::sycl::handler &cgh) {
      auto read_acc =
          buffer.template get_access<cl::sycl::access::mode::read,
                                     cl::sycl::access::target::global_buffer>(
              cgh, cl::sycl::range<1>(size * sizeof(T)),
              cl::sycl::id<1>(offset));
      cgh.copy(read_acc, static_cast<generic_buffer_data_type *>(
                             static_cast<void *>(dst)));
    });
    q_.wait();
  }
};  // namespace blas
}  // namespace blas
#endif  //
