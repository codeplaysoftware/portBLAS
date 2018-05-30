#ifndef SYCL_MEMEORY
#define SYCL_MEMEORY
#include <types/sycl_types.hpp>
namespace blas {
template <typename T>
class sycl_buffer {
 public:
  using ScalarT = T;
  using Self = sycl_buffer<ScalarT>;
  using buff_t = buffer_t<ScalarT, 1, cl::sycl::default_allocator<ScalarT>>;
  template <typename IndexType>
  sycl_buffer(ScalarT* data, IndexType size)
      : m_buffer(data, cl::sycl::range<1>{size}), offset(0) {}

  template <typename IndexType>
  sycl_buffer(std::vector<ScalarT>& data, IndexType size)
      : m_buffer(data.data(), cl::sycl::range<1>{size}), offset(0) {}

  template <typename IndexType>
  sycl_buffer(std::shared_ptr<ScalarT> data, IndexType size)
      : m_buffer(data, cl::sycl::range<1>{size}), offset(0) {}

  template <typename IndexType>
  sycl_buffer(IndexType size) : m_buffer(cl::sycl::range<1>{size}), offset(0) {}

  sycl_buffer(buff_t& buff_) : m_buffer(buff_), offset(0) {}

  sycl_buffer(buff_t buff_) : m_buffer(buff_) {}

  inline Self& operator+=(std::ptrdiff_t offset_) {
    offset += offset_;
    return *this;
  }

  inline Self operator+(std::ptrdiff_t offset_) const {
    return Self(m_buffer, offset + offset_);
  }

  inline Self operator-(std::ptrdiff_t offset_) const {
    return Self(m_buffer, offset - offset_);
  }

  inline Self& operator-=(std::ptrdiff_t offset_) {
    offset -= offset_;
    return *this;
  }
  template <typename Executor>
  inline cl::sycl::event copy_to_device(Executor& ex, ScalarT* data) {
    auto event = ex.sycl_queue().submit([&](cl::sycl::handler& cgh) {
      auto acc = get_range_access<cl::sycl::access::mode::write>(cgh);
      cgh.copy(data, acc);
    });
    //event.wait();
    return event;
  }
  template <typename Executor>
  inline cl::sycl::event copy_to_host(Executor& ex, ScalarT* data) {
    auto event = ex.sycl_queue().submit([&](cl::sycl::handler& cgh) {
      auto acc = get_range_access<cl::sycl::access::mode::read>(cgh);
      cgh.copy(acc, data);
    });
    //event.wait();
    return event;
  }

  template <typename Executor>
  cl::sycl::event copy_to_device(Executor& ex, std::vector<ScalarT>& data) {
    auto event = ex.sycl_queue().submit([&](cl::sycl::handler& cgh) {
      auto acc = get_range_access<cl::sycl::access::mode::write>(cgh);
      cgh.copy(data.data(), acc);
    });
   // event.wait();
    return event;
  }
  template <typename Executor>
  cl::sycl::event copy_to_host(Executor& ex, std::vector<ScalarT>& data) {
    auto event = ex.sycl_queue().submit([&](cl::sycl::handler& cgh) {
      auto acc = get_range_access<cl::sycl::access::mode::read>(cgh);
      cgh.copy(acc, data.data());
    });
  //  event.wait();
    return event;
  }
  template <cl::sycl::access::mode AcM = cl::sycl::access::mode::read_write>
  SyclAccessor<ScalarT, AcM> get_range_access(cl::sycl::handler& cgh) {
    return SyclAccessor<ScalarT, AcM>(
        m_buffer, cgh, cl::sycl::range<1>(m_buffer.get_count() - offset),
        cl::sycl::id<1>(offset));
  }

  template <cl::sycl::access::mode AcM = cl::sycl::access::mode::read_write>
  placeholder_accessor_t<ScalarT, AcM> get_range_access() {
    return placeholder_accessor_t<ScalarT, AcM>(
        m_buffer, cl::sycl::range<1>(m_buffer.get_count() - offset),
        cl::sycl::id<1>(offset));
  }
  inline ptrdiff_t get_offset() { return offset; }

 private:
  sycl_buffer(buff_t buff_, std::ptrdiff_t offset_)
      : m_buffer(buff_), offset(offset_) {}
  buff_t m_buffer;
  std::ptrdiff_t offset;
};
template <typename T>
struct ScalrType<sycl_buffer<T>> {
  using ScalarT = T;
};

template <typename T, typename U>
struct Reconstruct_Container<T, sycl_buffer<U>> {
  using type = sycl_buffer<T>;
};

}  // end namespace blas
#endif  // SYCL_MEMEORY
