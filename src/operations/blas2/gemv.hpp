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
 *  @filename gemv.hpp
 *
 **************************************************************************/

#ifndef GEMV_HPP
#define GEMV_HPP
#include <operations/blas2_trees.h>
#include <operations/blas_operators.hpp>
#include <stdexcept>
#include <vector>
#include <views/view_sycl.hpp>
namespace blas {

template <class Output_t, class Matrix_t, class Vector_t>
Gemv<Output_t, Matrix_t, Vector_t>::Gemv(Output_t &_l, Matrix_t &_matrix,
                                         Vector_t &_vector)
    : l(_l), matrix(_matrix), vector(_vector) {}

template <class Output_t, class Matrix_t, class Vector_t>
sycl_blas_inline typename Gemv<Output_t, Matrix_t, Vector_t>::IndexType
Gemv<Output_t, Matrix_t, Vector_t>::getSize() const {
  return vector.getSizeR();
}
template <class Output_t, class Matrix_t, class Vector_t>
sycl_blas_inline bool Gemv<Output_t, Matrix_t, Vector_t>::valid_thread(
    cl::sycl::nd_item<1> ndItem) const {
  return ndItem.get_global_id(0) < matrix.getSizeR();
}

template <class Output_t, class Matrix_t, class Vector_t>
sycl_blas_inline typename Gemv<Output_t, Matrix_t, Vector_t>::value_type
Gemv<Output_t, Matrix_t, Vector_t>::eval(
    typename Gemv<Output_t, Matrix_t, Vector_t>::IndexType i) {
  auto dim = vector.getSize();

  typename Gemv<Output_t, Matrix_t, Vector_t>::value_type acc = 0;
  for (typename Gemv<Output_t, Matrix_t, Vector_t>::IndexType j = 0;
       j < vector.getSize(); j++) {
    // acc += vector.eval(j) * matrix.eval(i, j);
    acc = cl::sycl::mad(vector.eval(j), matrix.eval(i, j), acc);
  }
  return l.eval(i) = acc;
}

template <class Output_t, class Matrix_t, class Vector_t>
sycl_blas_inline typename Gemv<Output_t, Matrix_t, Vector_t>::value_type
Gemv<Output_t, Matrix_t, Vector_t>::eval(cl::sycl::nd_item<1> ndItem) {
  return Gemv<Output_t, Matrix_t, Vector_t>::eval(ndItem.get_global_id(0));
}

template <class Output_t, class Matrix_t, class Vector_t>
sycl_blas_inline void Gemv<Output_t, Matrix_t, Vector_t>::bind(
    cl::sycl::handler &h) {
  l.bind(h);
  matrix.bind(h);
  vector.bind(h);
}

}  // namespace blas

#endif  // GEMV_HPP