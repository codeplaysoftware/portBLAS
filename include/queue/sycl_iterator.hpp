#ifndef BLAS_SYCL_ITERATOR_HPP
#define BLAS_SYCL_ITERATOR_HPP
#include <types/sycl_types.hpp>
namespace blas {
template <typename T>
class buffer_iterator {
 public:
  using scalar_t = T;
  using self_t = buffer_iterator<scalar_t>;
  using buff_t = buffer_t<scalar_t, 1>;

  template <cl::sycl::access::mode AcM, typename scal_t>
  friend inline SyclAccessor<scal_t, AcM> get_range_accessor(
      buffer_iterator<scal_t> buff_iterator, cl::sycl::handler& cgh);

  template <cl::sycl::access::mode AcM, typename scal_t>
  friend inline placeholder_accessor_t<scal_t, AcM> get_range_accessor(
      buffer_iterator<scal_t> buff_iterator);

  buffer_iterator(const buff_t& buff_, std::ptrdiff_t offset_)
      : m_offset(offset_), m_buffer(buff_) {}

  buffer_iterator(const buff_t& buff_) : buffer_iterator(buff_, 0) {}

  template <typename other_scalar_t>
  buffer_iterator(const buffer_iterator<other_scalar_t>& other)
      : buffer_iterator(other.get_buffer(), other.get_offset()) {}

  inline self_t& operator+=(std::ptrdiff_t offset_) {
    m_offset += offset_;
    return *this;
  }

  inline self_t operator+(std::ptrdiff_t offset_) const {
    return self_t(m_buffer, m_offset + offset_);
  }

  inline self_t operator-(std::ptrdiff_t offset_) const {
    return self_t(m_buffer, m_offset - offset_);
  }

  inline self_t& operator-=(std::ptrdiff_t offset_) {
    m_offset -= offset_;
    return *this;
  }

  // Prefix operator (Increment and return value)
  self_t& operator++() {
    m_offset++;
    return (*this);
  }

  // Postfix operator (Return value and increment)
  self_t operator++(int i) {
    self_t temp_iterator(*this);
    m_offset += 1;
    return temp_iterator;
  }

  inline std::ptrdiff_t get_size() const {
    return (m_buffer.get_count() - m_offset);
  }

  inline std::ptrdiff_t get_offset() const { return m_offset; }

  inline void set_offset(std::ptrdiff_t offset_) { m_offset = offset_; }

  scalar_t& operator*() = delete;

  scalar_t* operator->() = delete;

  const buff_t& get_buffer() const { return m_buffer; }

 private:
  std::ptrdiff_t m_offset;
  buff_t m_buffer;
};

template <typename T>
struct scalar_type<buffer_iterator<T>> {
  using type = T;
};

template <typename T, typename U>
struct rebind_type<T, buffer_iterator<U>> {
  using type = buffer_iterator<T>;
};

template <cl::sycl::access::mode AcM = cl::sycl::access::mode::read_write,
          typename scalar_t>
inline SyclAccessor<scalar_t, AcM> get_range_accessor(
    buffer_iterator<scalar_t> buff_iterator, cl::sycl::handler& cgh) {
  return SyclAccessor<scalar_t, AcM>(
      buff_iterator.m_buffer, cgh, cl::sycl::range<1>(buff_iterator.get_size()),
      cl::sycl::id<1>(buff_iterator.get_offset()));
}

template <cl::sycl::access::mode AcM = cl::sycl::access::mode::read_write,
          typename scalar_t>
inline placeholder_accessor_t<scalar_t, AcM> get_range_accessor(
    buffer_iterator<scalar_t> buff_iterator) {
  return placeholder_accessor_t<scalar_t, AcM>(
      buff_iterator.m_buffer, cl::sycl::range<1>(buff_iterator.get_size()),
      cl::sycl::id<1>(buff_iterator.get_offset()));
}
}  // end namespace blas
#endif  // BLAS_SYCL_ITERATOR_HPP
