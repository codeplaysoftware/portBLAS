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

template <typename output_t, typename matrxi_t, typename vector_t>
Gemv<output_t, matrxi_t, vector_t>::Gemv(output_t &_l, matrxi_t &_matrix,
                                         vector_t &_vector)
    : lhs_(_l), matrix_(_matrix), vector_(_vector) {}

template <typename output_t, typename matrxi_t, typename vector_t>
SYCL_BLAS_INLINE typename Gemv<output_t, matrxi_t, vector_t>::index_t
Gemv<output_t, matrxi_t, vector_t>::get_size() const {
  return vector_.get_size_row();
}
template <typename output_t, typename matrxi_t, typename vector_t>
SYCL_BLAS_INLINE bool Gemv<output_t, matrxi_t, vector_t>::valid_thread(
    cl::sycl::nd_item<1> ndItem) const {
  return ndItem.get_global_id(0) < matrix_.get_size_row();
}

template <typename output_t, typename matrxi_t, typename vector_t>
SYCL_BLAS_INLINE typename Gemv<output_t, matrxi_t, vector_t>::value_t
Gemv<output_t, matrxi_t, vector_t>::eval(
    typename Gemv<output_t, matrxi_t, vector_t>::index_t i) {
  auto dim = vector_.get_size();

  typename Gemv<output_t, matrxi_t, vector_t>::value_t acc = 0;
  for (typename Gemv<output_t, matrxi_t, vector_t>::index_t j = 0;
       j < vector_.get_size(); j++) {
    // acc += vector_.eval(j) * matrix_.eval(i, j);
    acc = cl::sycl::mad(vector_.eval(j), matrix_.eval(i, j), acc);
  }
  return lhs_.eval(i) = acc;
}

template <typename output_t, typename matrxi_t, typename vector_t>
SYCL_BLAS_INLINE typename Gemv<output_t, matrxi_t, vector_t>::value_t
Gemv<output_t, matrxi_t, vector_t>::eval(cl::sycl::nd_item<1> ndItem) {
  return Gemv<output_t, matrxi_t, vector_t>::eval(ndItem.get_global_id(0));
}

template <typename output_t, typename matrxi_t, typename vector_t>
SYCL_BLAS_INLINE void Gemv<output_t, matrxi_t, vector_t>::bind(
    cl::sycl::handler &h) {
  lhs_.bind(h);
  matrix_.bind(h);
  vector_.bind(h);
}

}  // namespace blas

#endif  // GEMV_HPP
