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
 *  @filename sycl_iterator.cpp
 *
 **************************************************************************/
#include "container/sycl_iterator.hpp"
#include "operations/blas_constants.h"
namespace blas {
template class BufferIterator<float, codeplay_policy>;
template class BufferIterator<double, codeplay_policy>;
template class BufferIterator<Indexvalue_tuple<float, int>, codeplay_policy>;
template class BufferIterator<Indexvalue_tuple<float, long>, codeplay_policy>;
template class BufferIterator<Indexvalue_tuple<float, long long>,
                              codeplay_policy>;
template class BufferIterator<Indexvalue_tuple<double, int>, codeplay_policy>;
template class BufferIterator<Indexvalue_tuple<double, long>, codeplay_policy>;
template class BufferIterator<Indexvalue_tuple<double, long long>,
                              codeplay_policy>;
}  // end namespace blas
