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
 *  @filename usm_policy_handler.hpp
 *
 **************************************************************************/

#ifndef SYCL_BLAS_USM_POLICY_HANDLER_HPP
#define SYCL_BLAS_USM_POLICY_HANDLER_HPP

#include "policy/usm_policy_handler.h"

namespace blas {

/*  @brief Copying the data back to device
    @tparam element_t is the type of the data
    @param src is the host pointer we want to copy from.
    @param dst is the device pointer we want to copy to.
    @param size is the number of elements to be copied
*/
template <typename element_t>
inline typename usm_policy::event_t
PolicyHandler<usm_policy>::copy_to_device(const element_t *src,
                                               element_t *dst, size_t size) {
  return {q_.memcpy(dst, src, size)};
}

/*  @brief Copying the data back to device
    @tparam element_t is the type of the data
    @param src is the device pointer we want to copy from.
    @param dst is the host pointer we want to copy to.
    @param size is the number of elements to be copied
*/
template <typename element_t>
inline typename usm_policy::event_t
PolicyHandler<usm_policy>::copy_to_host(element_t *src, element_t *dst,
                                             size_t size) {
  return {q_.memcpy(dst, src, size)};
}

}  // namespace blas
#endif  // SYCL_BLAS_USM_POLICY_HANDLER_HPPQUEUE_SYCL_HPP
