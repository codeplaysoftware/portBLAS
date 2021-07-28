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
 *  SYCL-BLAS: BLAS implementation using SYCL
 *
 *  @filename sycl_usm.h
 *
 **************************************************************************/
#ifndef SYCL_BLAS_USM_H
#define SYCL_BLAS_USM_H
#include "blas_meta.h"
#include "container/blas_iterator.h"
#include "policy/sycl_policy.h"
#include <CL/sycl.hpp>

namespace blas {
template <typename scalar_t, typename queue_t>
inline void *
sycl_usm_malloc_device(scalar_t numBytes, const queue_t& syclQueue) {
  return cl::sycl::malloc_device(numBytes, syclQueue);
}

template <typename pointer_t, typename queue_t>
inline void
sycl_usm_free(pointer_t *ptr, queue_t& syclQueue) {
  return cl::sycl::free(ptr, syclQueue);
}
}  // end namespace blas

#endif  // SYCL_BLAS_USM_H
