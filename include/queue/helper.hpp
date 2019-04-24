#ifndef BLAS_HELPER_HPP
#define BLAS_HELPER_HPP
#include <queue/sycl_iterator.hpp>
#include <types/sycl_types.hpp>

namespace blas {
namespace helper {
template <typename scalar_t, typename index_t>
inline blas::buffer_iterator<scalar_t> make_sycl_iteator_buffer(scalar_t* data,
                                                                index_t size) {
  using buff_t = blas::buffer_t<scalar_t, 1, cl::sycl::default_allocator>;
  return blas::buffer_iterator<scalar_t>{
      buff_t{data, cl::sycl::range<1>{size}}};
}

template <typename scalar_t, typename index_t>
inline buffer_iterator<scalar_t> make_sycl_iteator_buffer(
    std::vector<scalar_t>& data, index_t size) {
  using buff_t = blas::buffer_t<scalar_t, 1, cl::sycl::default_allocator>;
  return blas::buffer_iterator<scalar_t>{
      buff_t{data.data(), cl::sycl::range<1>{size}}};
}

template <typename scalar_t, typename index_t>
inline blas::buffer_iterator<scalar_t> make_sycl_iteator_buffer(index_t size) {
  using buff_t = blas::buffer_t<scalar_t, 1, cl::sycl::default_allocator>;
  return blas::buffer_iterator<scalar_t>{buff_t{cl::sycl::range<1>{size}}};
}

template <typename scalar_t, typename index_t>
inline blas::buffer_iterator<scalar_t> make_sycl_iteator_buffer(
    blas::buffer_t<scalar_t, 1, cl::sycl::default_allocator> buff_) {
  return blas::buffer_iterator<scalar_t>{buff_};
}

}  // namespace helper
}  // namespace blas
#endif  // BLAS_HELPER_HPP
