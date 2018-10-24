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
 *  @filename gemv_naive.hpp
 *
 **************************************************************************/

#ifndef GEMV_NAIVE_HPP
#define GEMV_NAIVE_HPP

#include <operations/blas_operators.hpp>
#include <stdexcept>
#include <vector>
#include <views/view_sycl.hpp>

namespace blas {

template <class Output_t, class Matrix_t, class Vector_t>
struct NaiveGemv {
  using value_type = typename Vector_t::value_type;
  using IndexType = typename Vector_t::IndexType;

  Output_t l;
  Matrix_t matrix;
  Vector_t vector;

  NaiveGemv(Output_t &_l, Matrix_t &_matrix, Vector_t &_vector, IndexType &_nWgRow, IndexType &_nWgCol, IndexType &_sharedMemSize):
    l(_l), matrix(_matrix), vector(_vector) {};

    value_type eval(IndexType i) { 
        auto dim = vector.getSize();

        // initialise val to the correct type.
        auto val = iniAddOp1_struct::eval(vector.eval(0));

        // value_type val = {};

        for (IndexType j = 0; j < dim; j++) { 
            auto prod = prdOp2_struct::eval(matrix.eval(i,j), vector.eval(j));
            val = addOp2_struct::eval(val, prod); 
        }
        return l.eval(i) = val; 
    }
};

template <typename Output_t, typename Matrix_t, typename Vector_t> make_naive_gemm(Output_t &l, Matrix_t &matrix, Vector_t &vector, typename )

}  // namespace blas

#endif  // GEMV_NAIVE_HPP