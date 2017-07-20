/***************************************************************************
 *
 *  @license
 *  Copyright (C) 2017 Codeplay Software Limited
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
 *  @filename blas3_trees_gemm.hpp
 *
 **************************************************************************/

#ifndef BLAS3_TREES_GEMM_HPP
#define BLAS3_TREES_GEMM_HPP


#include <CL/sycl.hpp>


#include <utils/vec.hpp>


namespace blas {


namespace vector {


template <int vec_size, typename VecType>
inline void _gemv(const VecType *A, const VecType &B, VecType &C) {
  using element_type = typename VecType::element_type;
  for_vec_elem<0, vec_size>::map(B, [&] (int i, element_type el) {
    C += el * A[i];
  });
}


}  // namespace vector


namespace thread {


template <int scratch_width, int thr_size, int vec_size,
          typename VecType, typename ScratchType>
inline void _gemm(const VecType *A, ScratchType B, VecType *C) {
  VecType vecB;
  #pragma unroll
  for (int i = 0; i < scratch_width; ++i) {
    #pragma unroll
    for (int j = 0; j < thr_size; ++j) {
      vecB.load(i*thr_size + j, B);
      vector::_gemv<vec_size>(A + j*vec_size, vecB, C[i]);
    }
  }
}


}  // namespace thread


}  // namespace blas


#endif  // BLAS3_TREES_GEMM_HPP

