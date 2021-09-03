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
 *  @filename usm_policy_handler.cpp
 *
 **************************************************************************/

#ifndef SYCL_BLAS_USM_POLICY_HANDLER_CPP
#define SYCL_BLAS_USM_POLICY_HANDLER_CPP
#include "operations/blas_constants.h"
// the templated methods
#include "policy/usm_policy_handler.hpp"
namespace blas {

#define INSTANTIATE_TEMPLATE_METHODS(element_t)                                \
  template typename codeplay_policy::event_t                                   \
  PolicyHandler<usm_policy>::copy_to_device<element_t>(                   \
      const element_t *src, element_t *dst, size_t size);                      \
                                                                               \
  template typename codeplay_policy::event_t                                   \
  PolicyHandler<usm_policy>::copy_to_host<element_t>(                     \
      element_t * src, element_t * dst, size_t size);                          \
                                                                               \

INSTANTIATE_TEMPLATE_METHODS(float)

#ifdef BLAS_DATA_TYPE_DOUBLE
INSTANTIATE_TEMPLATE_METHODS(double)
#endif  // BLAS_DATA_TYPE_DOUBLE

#ifdef BLAS_DATA_TYPE_HALF
INSTANTIATE_TEMPLATE_METHODS(cl::sycl::half)
#endif  // BLAS_DATA_TYPE_HALF

#define INSTANTIATE_TEMPLATE_METHODS_SPECIAL(ind, val)                        \
  template typename codeplay_policy::event_t                                  \
  PolicyHandler<usm_policy>::copy_to_device<IndexValueTuple<ind, val>>(  \
      const IndexValueTuple<ind, val> *src, IndexValueTuple<ind, val> *dst,   \
      size_t size);                                                           \
  template typename codeplay_policy::event_t                                  \
  PolicyHandler<usm_policy>::copy_to_host<IndexValueTuple<ind, val>>(    \
      IndexValueTuple<ind, val> * src, IndexValueTuple<ind, val> * dst,       \
      size_t size);                                                           \

INSTANTIATE_TEMPLATE_METHODS_SPECIAL(int, float)
INSTANTIATE_TEMPLATE_METHODS_SPECIAL(long, float)
INSTANTIATE_TEMPLATE_METHODS_SPECIAL(long long, float)

#ifdef BLAS_DATA_TYPE_DOUBLE
INSTANTIATE_TEMPLATE_METHODS_SPECIAL(int, double)
INSTANTIATE_TEMPLATE_METHODS_SPECIAL(long, double)
INSTANTIATE_TEMPLATE_METHODS_SPECIAL(long long, double)
#endif  // BLAS_DATA_TYPE_DOUBLE

#ifdef BLAS_DATA_TYPE_HALF
INSTANTIATE_TEMPLATE_METHODS_SPECIAL(int, cl::sycl::half)
INSTANTIATE_TEMPLATE_METHODS_SPECIAL(long, cl::sycl::half)
INSTANTIATE_TEMPLATE_METHODS_SPECIAL(long long, cl::sycl::half)
#endif  // BLAS_DATA_TYPE_HALF

}  // namespace blas
#endif
