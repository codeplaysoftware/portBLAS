#ifndef SYCL_MEMEORY
#define SYCL_MEMEORY
template <typename T>
class sycl_buffer {
 public:
  using Self = sycl_buffer<T>;
  using SyclBuffer = cl::sycl::buffer<T, 1>;
  template <cl::sycl::access::mode AcM,
            cl::sycl::access::target AcT =
                cl::sycl::access::target::global_buffer,
            cl::sycl::access::placeholder AcP =
                cl::sycl::access::placeholder::false_t>
  using SyclAccessor = cl::sycl::accessor<T, 1, AcM, AcT, AcP>;
  template <
      cl::sycl::access::mode AcM,
      cl::sycl::access::target AcT = cl::sycl::access::target::global_buffer,
      cl::sycl::access::placeholder AcP = cl::sycl::access::placeholder::true_t>
  using SyclPAccessor = cl::sycl::accessor<T, 1, AcM, AcT, AcP>;
  template <typename IndexType>
  sycl_buffer(T* data, IndexType size)
      : buff_t(data, cl::sycl::range<1>{size}), offset(0) {}

  template <typename IndexType>
  sycl_buffer(std::vector<T> data, IndexType size)
      : buff_t(data.data(), cl::sycl::range<1>{size}), offset(0) {}

  template <typename IndexType>
  sycl_buffer(std::shared_ptr<T> data, IndexType size)
      : buff_t(data.data(), cl::sycl::range<1>{size}), offset(0) {}

  template <typename IndexType>
  sycl_buffer(IndexType size) : buff_t(cl::sycl::range<1>{size}), offset(0) {}

  sycl_buffer(SyclBuffer& sycl_buffer) : buff_t(sycl_buffer), offset(0) {}

  inline Self& operator+=(std::ptrdiff_t offset_) {
    offset += offset_;
    return *this;
  }
  //  template <typename IndexType>
  inline Self operator+(std::ptrdiff_t offset_) {
    return Self(buff_t, offset + offset_);
  }

  inline Self operator-(std::ptrdiff_t offset_) {
    return Self(buff_t, offset - offset_);
  }

  inline Self& operator-=(std::ptrdiff_t offset_) {
    offset -= offset_;
    return *this;
  }
  template <typename Executor>
  inline cl::sycl::event copy_to_device(Executor& ex, T* data) {
    auto acc = get_range_access<cl::sycl::access::mode::write>();
    auto event = ex.sycl_queue().submit([&](cl::sycl::handler& cgh) {
      cgh.require(acc);
      cgh.copy(data, acc);
    });
    event.wait();
    return event;
  }
  template <typename Executor>
  inline cl::sycl::event copy_to_host(Executor& ex, T* data) {
    auto acc = get_range_access<cl::sycl::access::mode::read>();
    auto event = ex.sycl_queue().submit([&](cl::sycl::handler& cgh) {
      cgh.require(acc);
      cgh.copy(acc, data);
    });
    event.wait();
    return event;
  }

  template <typename Executor>
  cl::sycl::event copy_to_device(Executor& ex, std::shared_ptr<T> data) {
    auto acc = get_range_access<cl::sycl::access::mode::write>();
    auto event = ex.sycl_queue().submit([&](cl::sycl::handler& cgh) {
      cgh.require(acc);
      cgh.copy(data, acc);
    });
    event.wait();
    return event;
  }
  template <typename Executor>
  cl::sycl::event copy_to_host(Executor& ex, std::vector<T> data) {
    auto acc = get_range_access<cl::sycl::access::mode::read>();
    auto event = ex.sycl_queue().submit([&](cl::sycl::handler& cgh) {
      cgh.require(acc);
      cgh.copy(acc, data.data());
    });
    event.wait();
    return event;
  }
  template <cl::sycl::access::mode AcM>
  SyclAccessor<AcM> get_range_access(cl::sycl::handler& cgh) {
    return SyclAccessor<AcM>(buff_t, cgh,
                             cl::sycl::range<1>(buff_t.get_count() - offset),
                             cl::sycl::id<1>(offset));
  }

  template <cl::sycl::access::mode AcM>
  SyclPAccessor<AcM> get_range_access() {
    return SyclPAccessor<AcM>(buff_t,
                              cl::sycl::range<1>(buff_t.get_count() - offset),
                              cl::sycl::id<1>(offset));
  }

 private:
  sycl_buffer(SyclBuffer buff_, std::ptrdiff_t offset_)
      : buff_t(buff_), offset(offset_) {}
  SyclBuffer buff_t;
  std::ptrdiff_t offset;
};
#endif  // SYCL_MEMEORY
