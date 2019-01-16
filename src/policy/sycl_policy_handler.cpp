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
 *  @filename sycl_policy_handler.cpp
 *
 **************************************************************************/

#include "operations/blas_constants.h"
// the templated methods
#include "policy/sycl_policy_handler.hpp"

namespace blas {

#define INSTANTIATE_TEMPLATE_METHODS(T)                                       \
  template T *Policy_Handler<BLAS_SYCL_Policy>::allocate<T>(                  \
      size_t num_elements) const;                                             \
  template void Policy_Handler<BLAS_SYCL_Policy>::deallocate<T>(T * p) const; \
  template buffer_iterator<T, BLAS_SYCL_Policy>                               \
      Policy_Handler<BLAS_SYCL_Policy>::get_buffer<T>(T * ptr) const;         \
  template buffer_iterator<T, BLAS_SYCL_Policy>                               \
  Policy_Handler<BLAS_SYCL_Policy>::get_buffer<T>(                            \
      buffer_iterator<T, BLAS_SYCL_Policy> buff) const;                       \
  template typename BLAS_SYCL_Policy::access_type<                            \
      typename scalar_type<T>::type, cl::sycl::access::mode::read_write>      \
      Policy_Handler<BLAS_SYCL_Policy>::get_range_access<                     \
          cl::sycl::access::mode::read_write, T>(T * vptr);                   \
                                                                              \
  template typename BLAS_SYCL_Policy::access_type<                            \
      typename scalar_type<T>::type, cl::sycl::access::mode::read_write>      \
  Policy_Handler<BLAS_SYCL_Policy>::get_range_access<                         \
      T, cl::sycl::access::mode::read_write>(                                 \
      buffer_iterator<T, BLAS_SYCL_Policy> buff);                             \
  template typename BLAS_SYCL_Policy::event_type                              \
  Policy_Handler<BLAS_SYCL_Policy>::copy_to_device<T>(const T *src, T *dst,   \
                                                      size_t size);           \
                                                                              \
  template typename BLAS_SYCL_Policy::event_type                              \
  Policy_Handler<BLAS_SYCL_Policy>::copy_to_device<T>(                        \
      const T *src, buffer_iterator<T, BLAS_SYCL_Policy> dst,                 \
      size_t size = 0);                                                       \
  template typename BLAS_SYCL_Policy::event_type                              \
  Policy_Handler<BLAS_SYCL_Policy>::copy_to_host<T>(T * src, T * dst,         \
                                                    size_t size);             \
                                                                              \
  template typename BLAS_SYCL_Policy::event_type                              \
  Policy_Handler<BLAS_SYCL_Policy>::copy_to_host<T>(                          \
      buffer_iterator<T, BLAS_SYCL_Policy> src, T * dst, size_t size = 0);    \
  template ptrdiff_t Policy_Handler<BLAS_SYCL_Policy>::get_offset<T>(         \
      const T *ptr) const;                                                    \
                                                                              \
  template ptrdiff_t Policy_Handler<BLAS_SYCL_Policy>::get_offset<T>(         \
      buffer_iterator<T, BLAS_SYCL_Policy> ptr) const;

INSTANTIATE_TEMPLATE_METHODS(float)
INSTANTIATE_TEMPLATE_METHODS(double)

#define INSTANTIATE_TEMPLATE_METHODS_SPECIAL(ind, val)                         \
  template IndexValueTuple<ind, val>                                           \
      *Policy_Handler<BLAS_SYCL_Policy>::allocate<IndexValueTuple<ind, val>>(  \
          size_t num_elements) const;                                          \
  template void                                                                \
      Policy_Handler<BLAS_SYCL_Policy>::deallocate<IndexValueTuple<ind, val>>( \
          IndexValueTuple<ind, val> * p) const;                                \
  template buffer_iterator<IndexValueTuple<ind, val>, BLAS_SYCL_Policy>        \
      Policy_Handler<BLAS_SYCL_Policy>::get_buffer<IndexValueTuple<ind, val>>( \
          IndexValueTuple<ind, val> * ptr) const;                              \
  template buffer_iterator<IndexValueTuple<ind, val>, BLAS_SYCL_Policy>        \
  Policy_Handler<BLAS_SYCL_Policy>::get_buffer<IndexValueTuple<ind, val>>(     \
      buffer_iterator<IndexValueTuple<ind, val>, BLAS_SYCL_Policy> buff)       \
      const;                                                                   \
  template typename BLAS_SYCL_Policy::access_type<                             \
      typename scalar_type<IndexValueTuple<ind, val>>::type,                   \
      cl::sycl::access::mode::read_write>                                      \
      Policy_Handler<BLAS_SYCL_Policy>::get_range_access<                      \
          cl::sycl::access::mode::read_write, IndexValueTuple<ind, val>>(      \
          IndexValueTuple<ind, val> * vptr);                                   \
                                                                               \
  template typename BLAS_SYCL_Policy::access_type<                             \
      typename scalar_type<IndexValueTuple<ind, val>>::type,                   \
      cl::sycl::access::mode::read_write>                                      \
  Policy_Handler<BLAS_SYCL_Policy>::get_range_access<                          \
      IndexValueTuple<ind, val>, cl::sycl::access::mode::read_write>(          \
      buffer_iterator<IndexValueTuple<ind, val>, BLAS_SYCL_Policy> buff);      \
  template typename BLAS_SYCL_Policy::event_type                               \
  Policy_Handler<BLAS_SYCL_Policy>::copy_to_device<IndexValueTuple<ind, val>>( \
      const IndexValueTuple<ind, val> *src, IndexValueTuple<ind, val> *dst,    \
      size_t size);                                                            \
                                                                               \
  template typename BLAS_SYCL_Policy::event_type                               \
  Policy_Handler<BLAS_SYCL_Policy>::copy_to_device<IndexValueTuple<ind, val>>( \
      const IndexValueTuple<ind, val> *src,                                    \
      buffer_iterator<IndexValueTuple<ind, val>, BLAS_SYCL_Policy> dst,        \
      size_t size = 0);                                                        \
  template typename BLAS_SYCL_Policy::event_type                               \
  Policy_Handler<BLAS_SYCL_Policy>::copy_to_host<IndexValueTuple<ind, val>>(   \
      IndexValueTuple<ind, val> * src, IndexValueTuple<ind, val> * dst,        \
      size_t size);                                                            \
                                                                               \
  template typename BLAS_SYCL_Policy::event_type                               \
  Policy_Handler<BLAS_SYCL_Policy>::copy_to_host<IndexValueTuple<ind, val>>(   \
      buffer_iterator<IndexValueTuple<ind, val>, BLAS_SYCL_Policy> src,        \
      IndexValueTuple<ind, val> * dst, size_t size = 0);                       \
  template ptrdiff_t                                                           \
  Policy_Handler<BLAS_SYCL_Policy>::get_offset<IndexValueTuple<ind, val>>(     \
      const IndexValueTuple<ind, val> *ptr) const;                             \
                                                                               \
  template ptrdiff_t                                                           \
  Policy_Handler<BLAS_SYCL_Policy>::get_offset<IndexValueTuple<ind, val>>(     \
      buffer_iterator<IndexValueTuple<ind, val>, BLAS_SYCL_Policy> ptr) const;

INSTANTIATE_TEMPLATE_METHODS_SPECIAL(float, int)
INSTANTIATE_TEMPLATE_METHODS_SPECIAL(float, long)
INSTANTIATE_TEMPLATE_METHODS_SPECIAL(float, long long)
INSTANTIATE_TEMPLATE_METHODS_SPECIAL(double, int)
INSTANTIATE_TEMPLATE_METHODS_SPECIAL(double, long)
INSTANTIATE_TEMPLATE_METHODS_SPECIAL(double, long long)

}  // namespace blas
