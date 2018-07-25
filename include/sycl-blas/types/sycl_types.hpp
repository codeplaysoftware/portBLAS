#ifndef SYCL_TYPES
#define SYCL_TYPES
#include <CL/sycl.hpp>

namespace blas {

template <typename ScalarT, int dim = 1,
          typename Allocator = cl::sycl::default_allocator<uint8_t>>
using buffer_t = cl::sycl::buffer<ScalarT, dim, Allocator>;
template <
    typename ScalarT,
    cl::sycl::access::mode AcM = cl::sycl::access::mode::read_write,
    cl::sycl::access::target AcT = cl::sycl::access::target::global_buffer,
    cl::sycl::access::placeholder AcP = cl::sycl::access::placeholder::false_t>
using SyclAccessor = cl::sycl::accessor<ScalarT, 1, AcM, AcT, AcP>;
template <
    typename ScalarT,
    cl::sycl::access::mode AcM = cl::sycl::access::mode::read_write,
    cl::sycl::access::target AcT = cl::sycl::access::target::global_buffer,
    cl::sycl::access::placeholder AcP = cl::sycl::access::placeholder::true_t>
using placeholder_accessor_t = cl::sycl::accessor<ScalarT, 1, AcM, AcT, AcP>;

/// \struct RemoveAll
/// \brief These methods are used to remove all the & const and * from  a type.
/// template parameters
/// \tparam T : the type we are interested in
template <typename T>
struct RemoveAll {
  typedef T Type;
};
/// specialisation of RemoveAll when the type contains &
template <typename T>
struct RemoveAll<T &> {
  typedef typename RemoveAll<T>::Type Type;
};
/// specialisation of RemoveAll when the type contains *
template <typename T>
struct RemoveAll<T *> {
  typedef typename RemoveAll<T>::Type Type;
};
/// specialisation of RemoveAll when the type contains const
template <typename T>
struct RemoveAll<const T> {
  typedef typename RemoveAll<T>::Type Type;
};

/// specialisation of RemoveAll when the type contains const and &
template <typename T>
struct RemoveAll<const T &> {
  typedef typename RemoveAll<T>::Type Type;
};
/// specialisation of RemoveAll when the type contains volatile
template <typename T>
struct RemoveAll<T volatile> {
  typedef typename RemoveAll<T>::Type Type;
};
/// specialisation of RemoveAll when the type contains const volatile
template <typename T>
struct RemoveAll<T const volatile> {
  typedef typename RemoveAll<T>::Type Type;
};
/// specialisation of RemoveAll when the type contains const and *
template <typename T>
struct RemoveAll<const T *> {
  typedef typename RemoveAll<T>::Type Type;
};

template <typename ContainerT>
struct scalar_type {
  using ScalarT = typename RemoveAll<ContainerT>::Type;
};

template <typename T, typename ContainerT>
struct rebind_type {
  using type = RemoveAll<T> *;
};

}  // namespace blas

#endif  // sycl types
