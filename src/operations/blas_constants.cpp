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
 *  @filename blas_constants.cpp
 *
 **************************************************************************/
#ifdef SYCL_BLAS_CONSTANT_CPP
#define SYCL_BLAS_CONSTANT_CPP
#include "operations/blas_constants.h"
namespace blas {
#define CPP_INSTANTIATE_TYPE(data_t, indicator) \
  constexpr const type constant<data_t, indicator>::value;

#define CPP_INSTANTIATE_INDEX_VALUE_TYPE(data_t, index_t, indicator) \
  constexpr const IndexValueTuple<data_t, index_t>                   \
      constant<IndexValueTuple<data_t, index_t>, indicator>::value;

CPP_INSTANTIATE_TYPE(int, const_val::zero)
CPP_INSTANTIATE_TYPE(int, const_val::one)
CPP_INSTANTIATE_TYPE(int, const_val::m_one)
CPP_INSTANTIATE_TYPE(int, const_val::two)
CPP_INSTANTIATE_TYPE(int, const_val::m_two)
CPP_INSTANTIATE_TYPE(int, const_val::max)
CPP_INSTANTIATE_TYPE(int, const_val::min)
CPP_INSTANTIATE_TYPE(float, const_val::zero)
CPP_INSTANTIATE_TYPE(float, const_val::one)
CPP_INSTANTIATE_TYPE(float, const_val::m_one)
CPP_INSTANTIATE_TYPE(float, const_val::two)
CPP_INSTANTIATE_TYPE(float, const_val::m_two)
CPP_INSTANTIATE_TYPE(float, const_val::max)

CPP_INSTANTIATE_TYPE(float, const_val::min)
CPP_INSTANTIATE_TYPE(double, const_val::zero)
CPP_INSTANTIATE_TYPE(double, const_val::one)
CPP_INSTANTIATE_TYPE(double, const_val::m_one)
CPP_INSTANTIATE_TYPE(double, const_val::two)
CPP_INSTANTIATE_TYPE(double, const_val::m_two)
CPP_INSTANTIATE_TYPE(double, const_val::max)
CPP_INSTANTIATE_TYPE(double, const_val::min)
CPP_INSTANTIATE_TYPE(std::complex<float>, const_val::zero)
CPP_INSTANTIATE_TYPE(std::complex<float>, const_val::one)
CPP_INSTANTIATE_TYPE(std::complex<float>, const_val::m_one)
CPP_INSTANTIATE_TYPE(std::complex<float>, const_val::two)
CPP_INSTANTIATE_TYPE(std::complex<float>, const_val::m_two)
CPP_INSTANTIATE_TYPE(std::complex<float>, const_val::max)
CPP_INSTANTIATE_TYPE(std::complex<float>, const_val::min)
CPP_INSTANTIATE_TYPE(std::complex<double>, const_val::zero)
CPP_INSTANTIATE_TYPE(std::complex<double>, const_val::one)
CPP_INSTANTIATE_TYPE(std::complex<double>, const_val::m_one)
CPP_INSTANTIATE_TYPE(std::complex<double>, const_val::two)
CPP_INSTANTIATE_TYPE(std::complex<double>, const_val::m_two)
CPP_INSTANTIATE_TYPE(std::complex<double>, const_val::max)
CPP_INSTANTIATE_TYPE(std::complex<double>, const_val::min)
#undef CPP_INSTANTIATE_TYPE

CPP_INSTANTIATE_INDEX_VALUE_TYPE(float, int, const_val::imax)
CPP_INSTANTIATE_INDEX_VALUE_TYPE(float, int, const_val::imin)

CPP_INSTANTIATE_INDEX_VALUE_TYPE(float, long, const_val::imax)
CPP_INSTANTIATE_INDEX_VALUE_TYPE(float, long, const_val::imin)
CPP_INSTANTIATE_INDEX_VALUE_TYPE(float, long long, const_val::imax)
CPP_INSTANTIATE_INDEX_VALUE_TYPE(float, long long, const_val::imin)
CPP_INSTANTIATE_INDEX_VALUE_TYPE(double, long, const_val::imax)
CPP_INSTANTIATE_INDEX_VALUE_TYPE(double, long, const_val::imin)
CPP_INSTANTIATE_INDEX_VALUE_TYPE(double, int, const_val::imax)
CPP_INSTANTIATE_INDEX_VALUE_TYPE(double, int, const_val::imin)
CPP_INSTANTIATE_INDEX_VALUE_TYPE(double, long long, const_val::imax)
CPP_INSTANTIATE_INDEX_VALUE_TYPE(double, long long, const_val::imin)
#undef CPP_INSTANTIATE_INDEX_VALUE_TYPE
}  // namespace blas
#endif
