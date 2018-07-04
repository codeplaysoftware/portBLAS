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
 *  @filename blas1_interface.hpp
 *
 **************************************************************************/

#ifndef BLAS_INTERFACE_SYCL_HPP
#define BLAS_INTERFACE_SYCL_HPP

#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>

#include <types/sycl_types.hpp>

#include <executors/executor_sycl.hpp>
#include <views/view_sycl.hpp>

namespace blas {

template <typename Executor, typename T, typename IncrementType,
          typename IndexType>
inline typename ViewTypeTrace<Executor, T>::VectorView make_vector_view(
    Executor &ex, T *vptr, IncrementType inc, IndexType sz) {
  auto container = ex.get_range_access(vptr);
  IndexType offset = ex.get_offset(vptr);
  using LeafNode = typename ViewTypeTrace<Executor, T>::VectorView;
  return LeafNode{container, offset, inc, sz};
}

template <typename Executor, typename T, typename IncrementType,
          typename IndexType>
inline typename ViewTypeTrace<Executor, T>::VectorView make_vector_view(
    Executor &, buffer_iterator<T> buff, IncrementType inc, IndexType sz) {
  using LeafNode = typename ViewTypeTrace<Executor, T>::VectorView;
  return LeafNode{buff, inc, sz};
}

template <typename Executor, typename T, typename IndexType, typename Opertype>
inline typename ViewTypeTrace<Executor, T>::MatrixView make_matrix_view(
    Executor &ex, T *vptr, IndexType m, IndexType n, IndexType lda,
    Opertype accessOpr) {
  using LeafNode = typename ViewTypeTrace<Executor, T>::MatrixView;
  auto container = ex.get_range_access(vptr);
  IndexType offset = ex.get_offset(vptr);
  return LeafNode{container, m, n, accessOpr, lda, offset};
}

template <typename Executor, typename T, typename IndexType, typename Opertype>
inline typename ViewTypeTrace<Executor, T>::MatrixView make_matrix_view(
    Executor &ex, buffer_iterator<T> buff, IndexType m, IndexType n,
    IndexType lda, Opertype accessOpr) {
  using LeafNode = typename ViewTypeTrace<Executor, T>::MatrixView;
  return LeafNode{buff, m, n, accessOpr, lda};
}

}  // namespace blas

#endif  // BLAS_INTERFACE_SYCL_HPP
