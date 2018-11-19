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

#include <operations/blas_operators.hpp>
#include <stdexcept>
#include <vector>
#include <views/view_sycl.hpp>

namespace blas {

template <class Output_t, class Matrix_t, class Vector_t>
struct Gemv {
  using value_type = typename Vector_t::value_type;
  using IndexType = typename Vector_t::IndexType;

  Output_t l;
  Matrix_t matrix;
  Vector_t vector;

  Gemv(Output_t &_l, Matrix_t &_matrix, Vector_t &_vector)
      : l(_l), matrix(_matrix), vector(_vector){};

  inline IndexType getSize() const { return vector.getSizeR(); }

  inline bool valid_thread(cl::sycl::nd_item<1> ndItem) const {
    return ndItem.get_global_id(0) < matrix.getSizeR();
  }

  value_type eval(IndexType i) {
    auto dim = vector.getSize();

    value_type acc = 0;
    for (IndexType j = 0; j < vector.getSize(); j++) {
      // acc += vector.eval(j) * matrix.eval(i, j);
      acc = cl::sycl::mad(vector.eval(j), matrix.eval(i, j), acc);
    }
    return l.eval(i) = acc;
  }

  value_type eval(cl::sycl::nd_item<1> ndItem) {
    IndexType globalId = ndItem.get_global_id(0);
    return eval(globalId);
  }

  void bind(cl::sycl::handler &h) {
    l.bind(h);
    matrix.bind(h);
    vector.bind(h);
  }
};

template <typename Output_t, typename Matrix_t, typename Vector_t>
Gemv<Output_t, Matrix_t, Vector_t> make_gemv(Output_t &l, Matrix_t &matrix,
                                             Vector_t &vector) {
  return Gemv<Output_t, Matrix_t, Vector_t>(l, matrix, vector);
}

}  // namespace blas

#endif  // GEMV_HPP